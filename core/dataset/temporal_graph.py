"""
Temporal heterogeneous graph of software-delivery activity, and the
"will this PR be merged within H hours?" prediction task.

Input: the event table written by ``core/dataset/gharchive_ingest.py`` (or the
synthetic generator in ``core/dataset/synthetic_events.py``, which produces
the same schema).

Graph
-----
Node types
    pr    one node per (repo, number). Features are computed *at the moment the
          PR is opened* (size, author association, ...) plus point-in-time
          history aggregates (author / repo track record strictly before the
          open time). Nothing on a PR node depends on what happens after it is
          opened, so a PR node can be shown to any seed that is later in time.
    user  [1, is_bot]   (time-invariant)
    repo  [1]           (time-invariant)

Edges: every edge carries ``time`` (unix seconds) = when the interaction
happened. Each forward relation has a ``rev_*`` twin with the same times.

    user  -opens->              pr
    pr    -in_repo->            repo   (first time the PR is seen)
    pr    -merged_into->        repo   (merge time)
    pr    -closed_unmerged_in-> repo
    user  -merges->             pr     (the merger)
    user  -closes->             pr     (the closer, PR closed without merge)
    user  -authored_merged->    pr     (the PR's author, at merge time)
    user  -authored_closed->    pr     (the PR's author, at unmerged-close time)
    user  -approves->           pr
    user  -reviews->            pr     (commented / changes requested / dismissed)
    user  -comments_on->        pr
    user  -pushes_to->          repo
    user  -stars->              repo
    user  -forks->              repo
    user  -issue_activity->     repo
    user  -releases->           repo

The authored_* relations put an author's past outcomes two hops from a new
PR of that author (seed <- author <- earlier PR), which a 2-layer model can
reach; through opens/merges alone it would take three hops.

Leakage control
---------------
* Neighbour sampling is temporal: for a seed PR opened at t, only edges with
  ``time <= t`` are sampled (PyG/pyg-lib edge-level temporal sampling; the
  cuGraph-PyG backend uses its own temporal sampler, see training/backends).
  The PR's own merge/close/review edges are therefore invisible to it.
* Node features are either time-invariant or computed as of the PR open time.
* Splits are by open time with an embargo: a split's labels must be fully
  resolved (open + horizon) before the next split starts, so no model is fit
  on outcomes that happen after the PRs it is evaluated on were opened.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

from core.dataset.gharchive_ingest import is_bot_expr

log = logging.getLogger(__name__)

ASSOCIATIONS = [
    "OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR",
    "FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER", "NONE",
]

# Features known when the PR is opened (from the PR payload itself).
INTRINSIC_FEATURES = [
    "has_open_event", "log_additions", "log_deletions", "log_changed_files", "log_commits",
    "draft", "log_title_len", "log_body_len", "to_default_branch", "log_n_labels",
    "log_n_req_reviewers", "is_revert", "author_is_bot", "hour_sin", "hour_cos", "weekend",
    *[f"assoc_{a.lower()}" for a in ASSOCIATIONS],
]
# Point-in-time aggregates of the author's / repo's history before the open time.
HISTORY_FEATURES = [
    "log_author_prior_opened", "log_author_prior_merged", "log_author_prior_closed",
    "author_merge_rate", "log_author_prior_reviews", "log_author_repo_prior_merged",
    "log_repo_prior_opened", "log_repo_prior_merged", "log_repo_prior_closed", "repo_merge_rate",
]
PR_FEATURES = INTRINSIC_FEATURES + HISTORY_FEATURES

NODE_TYPES = ["pr", "user", "repo"]
FORWARD_RELATIONS = [
    ("user", "opens", "pr"),
    ("pr", "in_repo", "repo"),
    ("pr", "merged_into", "repo"),
    ("pr", "closed_unmerged_in", "repo"),
    ("user", "merges", "pr"),
    ("user", "closes", "pr"),
    ("user", "authored_merged", "pr"),
    ("user", "authored_closed", "pr"),
    ("user", "approves", "pr"),
    ("user", "reviews", "pr"),
    ("user", "comments_on", "pr"),
    ("user", "pushes_to", "repo"),
    ("user", "stars", "repo"),
    ("user", "forks", "repo"),
    ("user", "issue_activity", "repo"),
    ("user", "releases", "repo"),
]
EDGE_TYPES = FORWARD_RELATIONS + [(d, f"rev_{r}", s) for s, r, d in FORWARD_RELATIONS]
METADATA = (NODE_TYPES, EDGE_TYPES)


def _to_ts(s: str) -> int:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


@dataclass(frozen=True)
class PRMergeTaskConfig:
    """Time ranges are ISO dates/datetimes (UTC), [start, end)."""

    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    horizon_hours: int = 48
    max_train_seeds: int | None = None
    max_eval_seeds: int | None = None
    sample_seed: int = 0
    exclude_bot_authors: bool = True

    def bounds(self) -> dict[str, tuple[int, int]]:
        return {
            "train": (_to_ts(self.train_start), _to_ts(self.train_end)),
            "val": (_to_ts(self.val_start), _to_ts(self.val_end)),
            "test": (_to_ts(self.test_start), _to_ts(self.test_end)),
        }

    @property
    def horizon_s(self) -> int:
        return self.horizon_hours * 3600

    @property
    def data_end(self) -> int:
        return self.bounds()["test"][1] + self.horizon_s

    def validate(self) -> None:
        b = self.bounds()
        for name, (s, e) in b.items():
            if s >= e:
                raise ValueError(f"{name}: start must be before end")
        if self.horizon_hours <= 0:
            raise ValueError("horizon_hours must be > 0")
        # Embargo: labels of a split are resolved before the next split starts.
        if b["train"][1] + self.horizon_s > b["val"][0]:
            raise ValueError("train_end + horizon must be <= val_start (label embargo)")
        if b["val"][1] + self.horizon_s > b["test"][0]:
            raise ValueError("val_end + horizon must be <= test_start (label embargo)")


@dataclass
class SeedSet:
    node: torch.Tensor   # (N,) pr node indices
    time: torch.Tensor   # (N,) open time, unix seconds
    y: torch.Tensor      # (N,) float {0, 1}

    def __len__(self) -> int:
        return int(self.node.numel())


@dataclass
class TemporalTask:
    """A temporal HeteroData graph plus labelled seed nodes for each split."""

    data: "HeteroData"  # noqa: F821
    splits: dict[str, SeedSet]
    feature_names: dict[str, list[str]]
    meta: dict = field(default_factory=dict)

    @property
    def seed_type(self) -> str:
        return self.meta.get("seed_type", "pr")

    @property
    def metadata(self) -> tuple[list[str], list[tuple[str, str, str]]]:
        """Canonical (node_types, edge_types) order used to build models (see HGT note)."""
        m = self.meta.get("metadata")
        if m is None:
            return METADATA
        return list(m[0]), [tuple(e) for e in m[1]]

    @property
    def history_features(self) -> list[str]:
        """Engineered relational / point-in-time aggregates (droppable with use_history=false)."""
        return list(self.meta.get("history_features", HISTORY_FEATURES))

    @property
    def history_feature_idx(self) -> list[int]:
        names = self.feature_names[self.seed_type]
        return [names.index(f) for f in self.history_features if f in names]

    def tabular(self, split: str, with_history: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Seed node features as a plain matrix, for non-graph baselines."""
        s = self.splits[split]
        x = self.data[self.seed_type].x[s.node]
        if not with_history:
            keep = [i for i in range(x.size(1)) if i not in set(self.history_feature_idx)]
            x = x[:, keep]
        return x.numpy(), s.y.numpy()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "data": self.data.to_dict(),
                "splits": {k: asdict(v) for k, v in self.splits.items()},
                "feature_names": self.feature_names,
                "meta": self.meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TemporalTask":
        from torch_geometric.data import HeteroData

        blob = torch.load(path, weights_only=False)
        return cls(
            data=HeteroData.from_dict(blob["data"]),
            splits={k: SeedSet(**v) for k, v in blob["splits"].items()},
            feature_names=blob["feature_names"],
            meta=blob["meta"],
        )


# --------------------------------------------------------------------------- #
# Event loading                                                               #
# --------------------------------------------------------------------------- #

def load_events(paths: list[str | Path], until_ts: int | None = None) -> pl.DataFrame:
    """Read ingested hourly Parquet files (dirs or files) into one sorted frame."""
    files: list[str] = []
    for p in paths:
        p = Path(p)
        files.extend(sorted(str(f) for f in p.glob("*.parquet")) if p.is_dir() else [str(p)])
    if not files:
        raise FileNotFoundError(f"no parquet files under {paths}")
    lf = pl.scan_parquet(files)
    if until_ts is not None:
        lf = lf.filter(pl.col("ts") < until_ts)
    return lf.unique().sort("ts").collect()


# --------------------------------------------------------------------------- #
# Point-in-time history features                                              #
# --------------------------------------------------------------------------- #

def _asof_count(queries: pl.DataFrame, events: pl.DataFrame, keys: list[str], name: str) -> pl.DataFrame:
    """
    For each query row (keys..., ts) count events with the same keys and
    event ts strictly lower than the query ts. Returns queries + column `name`.
    """
    if events.height == 0:
        return queries.with_columns(pl.lit(0).alias(name))
    cum = (
        events.group_by([*keys, "ts"]).len()
        .sort([*keys, "ts"])
        .with_columns(pl.col("len").cum_sum().over(keys).alias(name))
        .drop("len")
        .sort("ts")
    )
    out = queries.sort("ts").join_asof(
        cum, on="ts", by=keys, strategy="backward", allow_exact_matches=False,
        check_sortedness=False,  # both sides are sorted by ts globally above
    )
    return out.with_columns(pl.col(name).fill_null(0))


def _history_features(opens: pl.DataFrame, ev: pl.DataFrame) -> pl.DataFrame:
    """opens: one row per PR open with columns pr_id, ts, author, repo."""
    closes = ev.filter(pl.col("etype") == "pr_close").with_columns(
        pl.col("pr_author").alias("author")
    )
    merged = closes.filter(pl.col("merged"))
    unmerged = closes.filter(~pl.col("merged"))
    all_opens = ev.filter(pl.col("etype") == "pr_open").with_columns(
        pl.coalesce("pr_author", "actor").alias("author")
    )
    reviews = ev.filter(pl.col("etype") == "pr_review").rename({"actor": "author"})

    q = opens.select("pr_id", "ts", "author", "repo")
    q = _asof_count(q, all_opens.select("author", "ts"), ["author"], "author_prior_opened")
    q = _asof_count(q, merged.select("author", "ts"), ["author"], "author_prior_merged")
    q = _asof_count(q, unmerged.select("author", "ts"), ["author"], "author_prior_closed")
    q = _asof_count(q, reviews.select("author", "ts"), ["author"], "author_prior_reviews")
    q = _asof_count(q, merged.select("author", "repo", "ts"), ["author", "repo"], "author_repo_prior_merged")
    q = _asof_count(q, all_opens.select("repo", "ts"), ["repo"], "repo_prior_opened")
    q = _asof_count(q, merged.select("repo", "ts"), ["repo"], "repo_prior_merged")
    q = _asof_count(q, unmerged.select("repo", "ts"), ["repo"], "repo_prior_closed")

    def log1p(c: str) -> pl.Expr:
        return pl.col(c).cast(pl.Float64).log1p().alias(f"log_{c}")

    return q.select(
        "pr_id",
        log1p("author_prior_opened"),
        log1p("author_prior_merged"),
        log1p("author_prior_closed"),
        ((pl.col("author_prior_merged") + 1) / (pl.col("author_prior_merged") + pl.col("author_prior_closed") + 2))
        .alias("author_merge_rate"),
        log1p("author_prior_reviews"),
        log1p("author_repo_prior_merged"),
        log1p("repo_prior_opened"),
        log1p("repo_prior_merged"),
        log1p("repo_prior_closed"),
        ((pl.col("repo_prior_merged") + 1) / (pl.col("repo_prior_merged") + pl.col("repo_prior_closed") + 2))
        .alias("repo_merge_rate"),
    )


# --------------------------------------------------------------------------- #
# Graph construction                                                          #
# --------------------------------------------------------------------------- #

def _index(values: pl.Series, name: str) -> pl.DataFrame:
    """Deterministic id mapping: sorted unique values -> 0..n-1."""
    u = values.drop_nulls().unique().sort()
    return pl.DataFrame({name: u, f"{name}_idx": pl.arange(0, u.len(), eager=True, dtype=pl.Int64)})


def build_pr_merge_task(events: pl.DataFrame, cfg: PRMergeTaskConfig) -> TemporalTask:
    """Build the temporal heterogeneous graph and the labelled PR seeds."""
    from torch_geometric.data import HeteroData

    cfg.validate()
    ev = events.filter(pl.col("ts") < cfg.data_end)
    if ev.height == 0:
        raise ValueError("no events before data_end")

    pr_events = ev.filter(pl.col("number").is_not_null() & pl.col("etype").str.starts_with("pr_"))

    # ---- node ids ---------------------------------------------------------
    users = _index(pl.concat([ev["actor"], ev["pr_author"], ev["merged_by"]]), "user")
    repos = _index(ev["repo"], "repo")
    pr_first = (
        pr_events.group_by("repo", "number").agg(pl.col("ts").min().alias("first_ts"))
        .sort("repo", "number")
        .with_row_index("pr_id")
        .with_columns(pl.col("pr_id").cast(pl.Int64))
    )
    n_pr, n_user, n_repo = pr_first.height, users.height, repos.height

    def with_ids(df: pl.DataFrame, user_col: str | None = None) -> pl.DataFrame:
        out = df.join(repos, on="repo", how="left")
        if "number" in df.columns:
            out = out.join(pr_first.select("repo", "number", "pr_id"), on=["repo", "number"], how="left")
        if user_col is not None:
            out = out.join(users.rename({"user": user_col}), on=user_col, how="left")
        return out

    # ---- PR node features --------------------------------------------------
    opens = (
        ev.filter(pl.col("etype") == "pr_open")
        .sort("ts")
        .unique(subset=["repo", "number"], keep="first", maintain_order=True)
        .with_columns(pl.coalesce("pr_author", "actor").alias("author"))
    )
    opens = opens.join(pr_first.select("repo", "number", "pr_id"), on=["repo", "number"], how="inner")
    hour = (pl.col("ts") % 86400) / 3600.0
    weekday = ((pl.col("ts") // 86400) + 4) % 7  # 1970-01-01 was a Thursday -> Monday=0
    intrinsic = opens.select(
        "pr_id",
        pl.lit(1.0).alias("has_open_event"),
        pl.col("additions").fill_null(0).cast(pl.Float64).log1p().alias("log_additions"),
        pl.col("deletions").fill_null(0).cast(pl.Float64).log1p().alias("log_deletions"),
        pl.col("changed_files").fill_null(0).cast(pl.Float64).log1p().alias("log_changed_files"),
        pl.col("commits").fill_null(0).cast(pl.Float64).log1p().alias("log_commits"),
        pl.col("draft").fill_null(False).cast(pl.Float64).alias("draft"),
        pl.col("title_len").fill_null(0).cast(pl.Float64).log1p().alias("log_title_len"),
        pl.col("body_len").fill_null(0).cast(pl.Float64).log1p().alias("log_body_len"),
        pl.col("to_default_branch").fill_null(False).cast(pl.Float64).alias("to_default_branch"),
        pl.col("n_labels").fill_null(0).cast(pl.Float64).log1p().alias("log_n_labels"),
        pl.col("n_req_reviewers").fill_null(0).cast(pl.Float64).log1p().alias("log_n_req_reviewers"),
        pl.col("is_revert").fill_null(False).cast(pl.Float64).alias("is_revert"),
        is_bot_expr("author").cast(pl.Float64).alias("author_is_bot"),
        (2 * math.pi * hour / 24).sin().alias("hour_sin"),
        (2 * math.pi * hour / 24).cos().alias("hour_cos"),
        (weekday >= 5).cast(pl.Float64).alias("weekend"),
        *[(pl.col("author_assoc") == a).fill_null(False).cast(pl.Float64).alias(f"assoc_{a.lower()}")
          for a in ASSOCIATIONS],
    )
    hist = _history_features(opens, ev)
    feats = (
        pr_first.select("pr_id")
        .join(intrinsic, on="pr_id", how="left")
        .join(hist, on="pr_id", how="left")
        .sort("pr_id")
        .select(PR_FEATURES)
        .fill_null(0.0)
    )
    x_pr = torch.from_numpy(feats.to_numpy().astype(np.float32))

    x_user = torch.ones(n_user, 2)
    x_user[:, 1] = torch.from_numpy(
        users.select(is_bot_expr("user"))["user"].to_numpy().astype(np.float32)
    )
    x_repo = torch.ones(n_repo, 1)

    open_ts = (
        pr_first.select("pr_id").join(opens.select("pr_id", "ts"), on="pr_id", how="left")
        .sort("pr_id")["ts"].fill_null(-1).to_numpy().astype(np.int64)
    )

    data = HeteroData()
    data["pr"].x = x_pr
    data["pr"].open_ts = torch.from_numpy(open_ts)  # -1 when the open event was not observed
    data["user"].x = x_user
    data["repo"].x = x_repo

    # ---- edges -------------------------------------------------------------
    def put(rel: tuple[str, str, str], df: pl.DataFrame, src: str, dst: str) -> None:
        df = df.select(pl.col(src).alias("s"), pl.col(dst).alias("d"), "ts").drop_nulls().unique().sort("ts")
        ei = torch.from_numpy(np.stack([df["s"].to_numpy(), df["d"].to_numpy()]).astype(np.int64))
        t = torch.from_numpy(df["ts"].to_numpy().astype(np.int64))
        s, r, d = rel
        data[s, r, d].edge_index = ei
        data[s, r, d].time = t
        data[d, f"rev_{r}", s].edge_index = ei.flip(0).contiguous()
        data[d, f"rev_{r}", s].time = t

    by = {k: g for (k,), g in ev.partition_by("etype", as_dict=True).items()}
    empty = ev.head(0)

    def et(name: str) -> pl.DataFrame:
        return by.get(name, empty)

    put(("user", "opens", "pr"), with_ids(opens.select("author", "repo", "number", "ts"), "author"), "user_idx", "pr_id")
    put(("pr", "in_repo", "repo"), pr_first.join(repos, on="repo").rename({"first_ts": "ts"}), "pr_id", "repo_idx")
    closes = with_ids(et("pr_close"), "actor")
    merged = closes.filter(pl.col("merged"))
    put(("pr", "merged_into", "repo"), merged, "pr_id", "repo_idx")
    put(("pr", "closed_unmerged_in", "repo"), closes.filter(~pl.col("merged")), "pr_id", "repo_idx")
    merged_by = with_ids(
        et("pr_close").filter(pl.col("merged")).with_columns(pl.coalesce("merged_by", "actor").alias("merger")),
        "merger",
    )
    put(("user", "merges", "pr"), merged_by, "user_idx", "pr_id")
    put(("user", "closes", "pr"), closes.filter(~pl.col("merged")), "user_idx", "pr_id")
    authored = with_ids(et("pr_close").drop("actor").rename({"pr_author": "actor"}), "actor")
    put(("user", "authored_merged", "pr"), authored.filter(pl.col("merged")), "user_idx", "pr_id")
    put(("user", "authored_closed", "pr"), authored.filter(~pl.col("merged")), "user_idx", "pr_id")
    reviews = with_ids(et("pr_review"), "actor")
    put(("user", "approves", "pr"), reviews.filter(pl.col("review_state") == "approved"), "user_idx", "pr_id")
    put(("user", "reviews", "pr"), reviews.filter(pl.col("review_state") != "approved"), "user_idx", "pr_id")
    comments = with_ids(pl.concat([et("pr_comment"), et("pr_review_comment")]), "actor")
    put(("user", "comments_on", "pr"), comments, "user_idx", "pr_id")
    for rel, names in [
        ("pushes_to", ["push"]), ("stars", ["watch"]), ("forks", ["fork"]),
        ("issue_activity", ["issue_open", "issue_close", "issue_comment"]), ("releases", ["release"]),
    ]:
        df = pl.concat([et(n) for n in names]).drop("number")
        put(("user", rel, "repo"), with_ids(df, "actor"), "user_idx", "repo_idx")

    # ---- seeds and labels --------------------------------------------------
    merge_time = (
        ev.filter((pl.col("etype") == "pr_close") & pl.col("merged"))
        .group_by("repo", "number").agg(pl.col("ts").min().alias("merge_ts"))
    )
    seeds = opens.join(merge_time, on=["repo", "number"], how="left").with_columns(
        (pl.col("merge_ts").is_not_null()
         & (pl.col("merge_ts") > pl.col("ts"))
         & (pl.col("merge_ts") <= pl.col("ts") + cfg.horizon_s)).cast(pl.Float32).alias("y")
    )
    if cfg.exclude_bot_authors:
        seeds = seeds.filter(~is_bot_expr("author"))

    rng = np.random.default_rng(cfg.sample_seed)
    splits: dict[str, SeedSet] = {}
    eligible: dict[str, dict[str, float]] = {}
    for name, (start, end) in cfg.bounds().items():
        s = seeds.filter((pl.col("ts") >= start) & (pl.col("ts") < end)).sort("pr_id")
        eligible[name] = {"n": s.height, "pos_rate": float(s["y"].mean()) if s.height else float("nan")}
        cap = cfg.max_train_seeds if name == "train" else cfg.max_eval_seeds
        if cap is not None and s.height > cap:
            keep = np.sort(rng.choice(s.height, size=cap, replace=False))
            s = s[keep]
        splits[name] = SeedSet(
            node=torch.from_numpy(s["pr_id"].to_numpy().astype(np.int64)),
            time=torch.from_numpy(s["ts"].to_numpy().astype(np.int64)),
            y=torch.from_numpy(s["y"].to_numpy().astype(np.float32)),
        )

    meta = {
        "config": asdict(cfg),
        "num_nodes": {"pr": n_pr, "user": n_user, "repo": n_repo},
        "num_edges": {"/".join(k): int(data[k].edge_index.size(1)) for k in data.edge_types},
        "num_events": ev.height,
        "event_ts_range": [int(ev["ts"].min()), int(ev["ts"].max())],
        "eligible_seeds_before_subsampling": eligible,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_pos_rate": {k: float(v.y.mean()) if len(v) else float("nan") for k, v in splits.items()},
    }
    log.info("Built graph: %s nodes, %d edges", meta["num_nodes"], sum(meta["num_edges"].values()))
    return TemporalTask(data=data, splits=splits, feature_names={"pr": PR_FEATURES}, meta=meta)
