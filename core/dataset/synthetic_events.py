"""
Synthetic software-delivery event stream with a *known* mechanism.

SYNTHETIC DATA — not GitHub data. Used for
  * integration tests: the temporal graph + GNN pipeline must recover a signal
    that is only visible through graph history,
  * scale benchmarks: sampling / training throughput at controlled sizes.

The output has exactly the schema of ``core/dataset/gharchive_ingest.SCHEMA``,
so it goes through the same graph builder as real data.

Generative mechanism
--------------------
* Each user k has a latent skill  u_k ~ N(0, 1)   (never observed directly).
* Each repo r has a latent strictness s_r ~ N(0, 1).
* Each repo has 2-3 maintainers; each user contributes to 1-3 repos.
* PR opened by k in r at time t is merged with probability
      sigmoid(skill_w * u_k - strict_w * s_r + 0.4 * is_maintainer - 0.15 * log1p(size) + 0.5)
  with merge delay ~ Exp(mean 14h) (so some merges miss a 48h horizon).
  Unmerged PRs are closed after Exp(mean 60h) with probability 0.7.
* Maintainers review / approve / comment; users push, star and fork.

u_k and s_r are only observable through the outcomes of the author's and the
repo's *earlier* PRs, i.e. through the temporal graph (or hand-built history
aggregates). Intrinsic PR features carry only a weak size signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from core.dataset.gharchive_ingest import SCHEMA

HOUR = 3600
DAY = 86400


@dataclass(frozen=True)
class SyntheticConfig:
    n_users: int = 2000
    n_repos: int = 200
    days: int = 21
    start_ts: int = 1_756_684_800          # 2025-09-01T00:00:00Z
    prs_per_user_per_day: float = 0.25
    skill_weight: float = 1.6
    strict_weight: float = 1.0
    bot_fraction: float = 0.02
    noise_events_per_user_per_day: float = 1.0
    seed: int = 0


def generate_events(cfg: SyntheticConfig) -> pl.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    users = np.array([f"user{i}" for i in range(cfg.n_users)], dtype=object)
    n_bots = int(cfg.n_users * cfg.bot_fraction)
    users[:n_bots] = [f"auto{i}[bot]" for i in range(n_bots)]
    repos = np.array([f"org{i % 50}/repo{i}" for i in range(cfg.n_repos)], dtype=object)

    skill = rng.normal(size=cfg.n_users)
    strict = rng.normal(size=cfg.n_repos)

    # repo maintainers and user home repos
    maintainers = [rng.choice(cfg.n_users, size=rng.integers(2, 4), replace=False) for _ in range(cfg.n_repos)]
    is_maint = {(int(m), r) for r, ms in enumerate(maintainers) for m in ms}
    home = [rng.choice(cfg.n_repos, size=rng.integers(1, 4), replace=False) for _ in range(cfg.n_users)]
    for r, ms in enumerate(maintainers):
        for m in ms:
            if r not in home[m]:
                home[m] = np.append(home[m], r)

    horizon_end = cfg.start_ts + cfg.days * DAY
    rows: dict[str, list] = {k: [] for k in SCHEMA}

    def add(**kw) -> None:
        for k in SCHEMA:
            rows[k].append(kw.get(k))

    pr_counter = np.zeros(cfg.n_repos, dtype=np.int64)
    n_prs = rng.poisson(cfg.prs_per_user_per_day * cfg.days, size=cfg.n_users)
    for k in range(cfg.n_users):
        if n_prs[k] == 0:
            continue
        opens = np.sort(rng.uniform(cfg.start_ts, horizon_end, size=n_prs[k])).astype(np.int64)
        for t in opens:
            r = int(rng.choice(home[k]))
            pr_counter[r] += 1
            number = int(pr_counter[r])
            size = float(rng.lognormal(4.0, 1.2))
            maint = (k, r) in is_maint
            assoc = "MEMBER" if maint else rng.choice(["CONTRIBUTOR", "NONE", "FIRST_TIME_CONTRIBUTOR"], p=[0.6, 0.3, 0.1])
            add(ts=int(t), etype="pr_open", actor=users[k], repo=repos[r], number=number,
                additions=int(size * 0.7), deletions=int(size * 0.3), changed_files=int(1 + size / 50),
                commits=int(rng.integers(1, 6)), draft=bool(rng.random() < 0.08),
                title_len=int(rng.integers(10, 80)), body_len=int(rng.integers(0, 2000)),
                author_assoc=assoc, to_default_branch=True, n_labels=int(rng.integers(0, 3)),
                n_req_reviewers=int(rng.integers(0, 3)), is_revert=False, pr_author=users[k])

            logit = (cfg.skill_weight * skill[k] - cfg.strict_weight * strict[r]
                     + 0.4 * maint - 0.15 * np.log1p(size) + 0.5)
            merged = rng.random() < 1.0 / (1.0 + np.exp(-logit))
            reviewer = int(rng.choice(maintainers[r]))
            if merged:
                t_end = int(t + rng.exponential(14 * HOUR)) + 60
            else:
                t_end = int(t + rng.exponential(60 * HOUR)) + 60
            # review activity before the outcome
            for _ in range(int(rng.integers(0, 3))):
                tr = int(rng.uniform(t + 30, t_end))
                state = "approved" if (merged and rng.random() < 0.6) else rng.choice(["commented", "changes_requested"])
                add(ts=tr, etype="pr_review", actor=users[reviewer], repo=repos[r], number=number, review_state=state)
            for _ in range(int(rng.integers(0, 3))):
                tc = int(rng.uniform(t + 30, t_end))
                who = users[reviewer] if rng.random() < 0.5 else users[k]
                add(ts=tc, etype="pr_comment", actor=who, repo=repos[r], number=number)
            if merged:
                add(ts=t_end, etype="pr_close", actor=users[reviewer], repo=repos[r], number=number,
                    merged=True, pr_created_ts=int(t), pr_author=users[k], merged_by=users[reviewer])
            elif rng.random() < 0.7:
                closer = users[k] if rng.random() < 0.5 else users[reviewer]
                add(ts=t_end, etype="pr_close", actor=closer, repo=repos[r], number=number,
                    merged=False, pr_created_ts=int(t), pr_author=users[k])

    # background activity (pushes, stars, forks, issues) — no signal by design
    n_noise = rng.poisson(cfg.noise_events_per_user_per_day * cfg.days, size=cfg.n_users)
    kinds = np.array(["push", "watch", "fork", "issue_comment"])
    for k in range(cfg.n_users):
        for _ in range(n_noise[k]):
            kind = kinds[rng.choice(4, p=[0.6, 0.2, 0.05, 0.15])]
            r = int(rng.choice(home[k])) if kind in ("push", "issue_comment") else int(rng.integers(cfg.n_repos))
            t = int(rng.uniform(cfg.start_ts, horizon_end))
            add(ts=t // HOUR * HOUR if kind == "push" else t, etype=kind, actor=users[k], repo=repos[r],
                n_commits=int(rng.integers(1, 5)) if kind == "push" else None)

    df = pl.DataFrame(rows, schema=SCHEMA, strict=False)
    return df.filter(pl.col("ts") < horizon_end).sort("ts")
