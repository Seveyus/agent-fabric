"""
Stream GH Archive hourly dumps into compact, typed Parquet event tables.

Why a second GH Archive module (``gh_archive.py`` already exists)?
``gh_archive.py`` downloads raw ``.json.gz`` files to disk and keeps full event
dicts in memory. One hour of 2025 GitHub activity is ~120 MB compressed /
~1 GB of JSON, which makes multi-week collection impractical on a laptop.
This module streams each hour over HTTP, decompresses on the fly, keeps only
the fields needed by the temporal graph builder (``core/dataset/temporal_graph.py``)
and writes one small Parquet file per hour. Raw JSON never touches the disk.

Data caveat (measured, see docs/GRAPH_LEARNING.md):
    GitHub trimmed public Events API payloads between 2025-10-06 and
    2025-10-13. After that date ``PullRequestEvent.payload.pull_request`` only
    contains ``{base, head, id, number, url}`` (no title, author, merged flag,
    size) and ``PushEvent`` has no commit list. Use data up to 2025-10-05 for
    any PR-level task.

Usage (two phases, to keep the on-disk footprint small):
    # 1. full ingest of a warm-up period
    python -m core.dataset.gharchive_ingest ingest \
        --start 2025-08-25 --end 2025-08-31 --out data/gharchive/warmup_raw
    # 2. keep "collaborative" repos (>= 2 distinct human PR authors in warm-up)
    python -m core.dataset.gharchive_ingest select-repos \
        --events data/gharchive/warmup_raw --out data/gharchive/repos.txt
    python -m core.dataset.gharchive_ingest filter \
        --events data/gharchive/warmup_raw --repos data/gharchive/repos.txt \
        --out data/gharchive/events
    # 3. ingest the rest of the period, filtered on the fly
    python -m core.dataset.gharchive_ingest ingest \
        --start 2025-09-01 --end 2025-09-13 --out data/gharchive/events \
        --repo-filter data/gharchive/repos.txt
"""

from __future__ import annotations

import argparse
import logging
import re
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

import httpx
import orjson
import polars as pl

log = logging.getLogger(__name__)

GH_ARCHIVE_BASE = "https://data.gharchive.org"

# GitHub's "Revert" button produces a PR body "Reverts owner/repo#123".
_REVERTS_RE = re.compile(rb"Reverts\s+([\w.-]+/[\w.-]+)#(\d+)")

_BOT_RE = re.compile(r"(\[bot\]$)|([-_]bot$)|(^bot[-_])", re.IGNORECASE)
_KNOWN_BOTS = frozenset({
    "github-actions", "dependabot", "renovate", "dependabot-preview", "greenkeeper",
    "codecov-io", "pre-commit-ci", "allcontributors", "snyk-bot", "imgbot", "mergify",
    "copilot", "copilot-swe-agent",
})


def is_bot(login: str | None) -> bool:
    """Heuristic bot detector on GitHub logins (``[bot]`` suffix, ``-bot`` names, known apps)."""
    if not login:
        return False
    return bool(_BOT_RE.search(login)) or login.lower() in _KNOWN_BOTS


def is_bot_expr(col: str) -> pl.Expr:
    """Vectorised polars version of :func:`is_bot`."""
    c = pl.col(col)
    return (
        c.str.contains(r"(?i)(\[bot\]$)|([-_]bot$)|(^bot[-_])")
        | c.str.to_lowercase().is_in(sorted(_KNOWN_BOTS))
    ).fill_null(False)

# Columns of the output table. Nullable columns are None for event types that
# do not carry the field.
SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Int64,                # event time, unix seconds (UTC)
    "etype": pl.Utf8,              # canonical event type, see _EXTRACTORS
    "actor": pl.Utf8,              # login of the user who triggered the event
    "repo": pl.Utf8,               # "owner/name"
    "number": pl.Int32,            # PR / issue number
    # PR fields known at the moment the event is emitted
    "additions": pl.Int32,
    "deletions": pl.Int32,
    "changed_files": pl.Int32,
    "commits": pl.Int32,
    "draft": pl.Boolean,
    "title_len": pl.Int32,
    "body_len": pl.Int32,
    "author_assoc": pl.Utf8,       # OWNER / MEMBER / COLLABORATOR / CONTRIBUTOR / FIRST_TIME_CONTRIBUTOR / NONE
    "to_default_branch": pl.Boolean,
    "n_labels": pl.Int16,
    "n_req_reviewers": pl.Int16,
    "is_revert": pl.Boolean,       # title starts with 'Revert "'
    "reverts_number": pl.Int32,    # PR number referenced by "Reverts owner/repo#N" (same repo only)
    # PR close fields
    "merged": pl.Boolean,
    "pr_created_ts": pl.Int64,
    "pr_author": pl.Utf8,
    "merged_by": pl.Utf8,
    # review fields
    "review_state": pl.Utf8,
    # push fields (aggregated per actor/repo/hour)
    "n_commits": pl.Int32,
}

# Cheap pre-filter on the raw line: the "type" key appears in the first bytes
# of every GH Archive record, so we can skip ~40% of lines without parsing.
_WANTED_TYPES = (
    b'"PullRequestEvent"',
    b'"PullRequestReviewEvent"',
    b'"PullRequestReviewCommentEvent"',
    b'"IssueCommentEvent"',
    b'"IssuesEvent"',
    b'"PushEvent"',
    b'"WatchEvent"',
    b'"ForkEvent"',
    b'"ReleaseEvent"',
)


def _ts(s: str | None) -> int | None:
    if not s:
        return None
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def _base_row(ev: dict, etype: str) -> dict:
    return {
        "ts": _ts(ev.get("created_at")),
        "etype": etype,
        "actor": (ev.get("actor") or {}).get("login"),
        "repo": (ev.get("repo") or {}).get("name"),
    }


def _pull_request(ev: dict) -> dict | None:
    p = ev.get("payload") or {}
    pr = p.get("pull_request") or {}
    action = p.get("action")
    if not pr or "title" not in pr:  # trimmed payload (post 2025-10) -> unusable
        return None
    row = _base_row(ev, "")
    row["number"] = pr.get("number")
    if action == "opened":
        title = pr.get("title") or ""
        body = pr.get("body") or ""
        base = pr.get("base") or {}
        default_branch = (base.get("repo") or {}).get("default_branch")
        reverts_number = None
        m = _REVERTS_RE.search(body.encode("utf-8", "ignore"))
        if m and m.group(1).decode() == row["repo"]:
            reverts_number = int(m.group(2))
        row.update(
            etype="pr_open",
            additions=pr.get("additions"),
            deletions=pr.get("deletions"),
            changed_files=pr.get("changed_files"),
            commits=pr.get("commits"),
            draft=bool(pr.get("draft", False)),
            title_len=len(title),
            body_len=len(body),
            author_assoc=pr.get("author_association"),
            to_default_branch=(base.get("ref") == default_branch) if default_branch else None,
            n_labels=len(pr.get("labels") or []),
            n_req_reviewers=len(pr.get("requested_reviewers") or []),
            is_revert=title.startswith('Revert "'),
            reverts_number=reverts_number,
            pr_author=(pr.get("user") or {}).get("login"),
        )
        return row
    if action == "closed":
        row.update(
            etype="pr_close",
            merged=bool(pr.get("merged", False)),
            pr_created_ts=_ts(pr.get("created_at")),
            pr_author=(pr.get("user") or {}).get("login"),
            merged_by=(pr.get("merged_by") or {}).get("login"),
        )
        return row
    if action == "reopened":
        row.update(etype="pr_reopen", pr_author=(pr.get("user") or {}).get("login"))
        return row
    return None


def _review(ev: dict) -> dict | None:
    p = ev.get("payload") or {}
    pr = p.get("pull_request") or {}
    row = _base_row(ev, "pr_review")
    row["number"] = pr.get("number")
    row["review_state"] = ((p.get("review") or {}).get("state") or "").lower() or None
    return row


def _review_comment(ev: dict) -> dict | None:
    p = ev.get("payload") or {}
    row = _base_row(ev, "pr_review_comment")
    row["number"] = (p.get("pull_request") or {}).get("number")
    return row


def _issue_comment(ev: dict) -> dict | None:
    p = ev.get("payload") or {}
    issue = p.get("issue") or {}
    on_pr = "pull_request" in issue
    row = _base_row(ev, "pr_comment" if on_pr else "issue_comment")
    row["number"] = issue.get("number")
    return row


def _issue(ev: dict) -> dict | None:
    p = ev.get("payload") or {}
    action = p.get("action")
    if action not in ("opened", "closed"):
        return None
    row = _base_row(ev, "issue_open" if action == "opened" else "issue_close")
    row["number"] = (p.get("issue") or {}).get("number")
    return row


def _push(ev: dict) -> dict | None:
    p = ev.get("payload") or {}
    row = _base_row(ev, "push")
    # 'size' disappeared with the 2025-10 trimming; 'commits' too.
    row["n_commits"] = p.get("size", len(p.get("commits") or []))
    return row


def _simple(etype: str):
    def f(ev: dict) -> dict | None:
        return _base_row(ev, etype)
    return f


def _release(ev: dict) -> dict | None:
    if (ev.get("payload") or {}).get("action") not in ("published", "released"):
        return None
    return _base_row(ev, "release")


_EXTRACTORS = {
    "PullRequestEvent": _pull_request,
    "PullRequestReviewEvent": _review,
    "PullRequestReviewCommentEvent": _review_comment,
    "IssueCommentEvent": _issue_comment,
    "IssuesEvent": _issue,
    "PushEvent": _push,
    "WatchEvent": _simple("watch"),
    "ForkEvent": _simple("fork"),
    "ReleaseEvent": _release,
}


def extract(ev: dict) -> dict | None:
    """Map one raw GH Archive event to a row of SCHEMA (or None to drop it)."""
    fn = _EXTRACTORS.get(ev.get("type", ""))
    if fn is None:
        return None
    row = fn(ev)
    if row is None or row["ts"] is None or not row["actor"] or not row["repo"]:
        return None
    return row


def iter_gzip_lines(chunks: Iterator[bytes]) -> Iterator[bytes]:
    """Decompress a (possibly multi-member) gzip byte stream into lines."""
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    buf = b""
    for chunk in chunks:
        while chunk:
            out = d.decompress(chunk)
            buf += out
            if d.eof:  # start of a new gzip member
                chunk = d.unused_data
                d = zlib.decompressobj(16 + zlib.MAX_WBITS)
            else:
                chunk = b""
        *lines, buf = buf.split(b"\n")
        yield from lines
    tail = buf + d.flush()
    if tail:
        yield from tail.split(b"\n")


def rows_from_lines(lines: Iterator[bytes]) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    stats = {"lines": 0, "parsed": 0, "kept": 0, "bad_json": 0}
    for line in lines:
        if not line:
            continue
        stats["lines"] += 1
        head = line[:160]
        if not any(t in head for t in _WANTED_TYPES):
            continue
        try:
            ev = orjson.loads(line)
        except orjson.JSONDecodeError:
            stats["bad_json"] += 1
            continue
        stats["parsed"] += 1
        row = extract(ev)
        if row is not None:
            rows.append(row)
    stats["kept"] = len(rows)
    return rows, stats


def rows_to_frame(rows: list[dict]) -> pl.DataFrame:
    df = pl.DataFrame(rows, schema=SCHEMA, strict=False)
    # Aggregate pushes per (actor, repo, hour): bots push hundreds of times per
    # hour and the graph only needs "who pushed where, when, how much".
    pushes = df.filter(pl.col("etype") == "push")
    if pushes.height:
        agg = (
            pushes.with_columns((pl.col("ts") // 3600 * 3600).alias("ts"))
            .group_by(["ts", "actor", "repo"])
            .agg(pl.col("n_commits").sum())
            .with_columns(pl.lit("push").alias("etype"))
        )
        df = pl.concat(
            [df.filter(pl.col("etype") != "push"), agg], how="diagonal_relaxed"
        ).select(list(SCHEMA)).cast(SCHEMA)
    return df.sort("ts")


@dataclass
class HourResult:
    hour: str
    ok: bool
    rows: int = 0
    seconds: float = 0.0
    stats: dict = field(default_factory=dict)
    error: str | None = None


# Optional repo allow-list, loaded once per worker process (see _init_worker).
_REPO_FILTER: frozenset[str] | None = None


def _init_worker(repo_filter_path: str | None) -> None:
    global _REPO_FILTER
    _REPO_FILTER = load_repo_list(repo_filter_path) if repo_filter_path else None


def load_repo_list(path: str | Path) -> frozenset[str]:
    return frozenset(line.strip() for line in Path(path).read_text().splitlines() if line.strip())


def ingest_hour(hour: str, out_dir: str, retries: int = 3, timeout: float = 180.0) -> HourResult:
    """Download, parse and write one GH Archive hour (``YYYY-MM-DD-H``)."""
    out_path = Path(out_dir) / f"{hour}.parquet"
    if out_path.exists():
        return HourResult(hour, ok=True, rows=-1)
    url = f"{GH_ARCHIVE_BASE}/{hour}.json.gz"
    t0 = time.perf_counter()
    last_err = None
    for attempt in range(retries):
        try:
            with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as r:
                if r.status_code == 404:
                    return HourResult(hour, ok=False, error="404")
                r.raise_for_status()
                rows, stats = rows_from_lines(iter_gzip_lines(r.iter_raw(1 << 20)))
            if _REPO_FILTER is not None:
                rows = [row for row in rows if row["repo"] in _REPO_FILTER]
            df = rows_to_frame(rows)
            tmp = out_path.with_suffix(".tmp")
            df.write_parquet(tmp, compression="zstd", compression_level=6)
            tmp.rename(out_path)
            return HourResult(hour, True, df.height, time.perf_counter() - t0, stats)
        except (httpx.HTTPError, zlib.error, OSError) as exc:  # retry on transient errors
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
    return HourResult(hour, ok=False, error=last_err, seconds=time.perf_counter() - t0)


def hour_keys(start: str, end: str, hours: list[int] | None = None) -> list[str]:
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    hs = hours if hours is not None else list(range(24))
    keys = []
    d = d0
    while d <= d1:
        keys.extend(f"{d.isoformat()}-{h}" for h in hs)
        d += timedelta(days=1)
    return keys


def select_collaborative_repos(events_dir: str | Path, min_authors: int = 2) -> list[str]:
    """Repos with >= ``min_authors`` distinct non-bot PR authors in ``events_dir``."""
    opens = (
        pl.scan_parquet(str(Path(events_dir) / "*.parquet"))
        .filter(pl.col("etype") == "pr_open")
        .select("repo", "actor")
        .unique()
        .collect()
    )
    human = opens.filter(~is_bot_expr("actor"))
    counts = human.group_by("repo").agg(pl.col("actor").n_unique().alias("n"))
    return sorted(counts.filter(pl.col("n") >= min_authors)["repo"].to_list())


def filter_event_files(events_dir: str | Path, repos: frozenset[str], out_dir: str | Path) -> int:
    """Copy hourly event files keeping only rows of ``repos``. Returns rows kept."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    repo_list = sorted(repos)
    kept = 0
    for f in sorted(Path(events_dir).glob("*.parquet")):
        df = pl.read_parquet(f).filter(pl.col("repo").is_in(repo_list))
        df.write_parquet(out / f.name, compression="zstd", compression_level=6)
        kept += df.height
    return kept


def _cmd_ingest(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = hour_keys(args.start, args.end, args.hours)
    if date.fromisoformat(args.end) >= date(2025, 10, 6):
        log.warning("Payloads after 2025-10-06 are trimmed by GitHub; PR fields will be missing.")
    log.info("Ingesting %d hours into %s with %d workers", len(keys), out, args.workers)

    failed: list[HourResult] = []
    done = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(args.repo_filter,)
    ) as ex:
        futs = {ex.submit(ingest_hour, k, str(out)): k for k in keys}
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            if not res.ok:
                failed.append(res)
                log.warning("%s failed: %s", res.hour, res.error)
            elif res.rows >= 0:
                log.info(
                    "[%d/%d] %s rows=%d lines=%d %.1fs",
                    done, len(keys), res.hour, res.rows, res.stats.get("lines", 0), res.seconds,
                )
    log.info("Done in %.1f min. %d failed.", (time.perf_counter() - t0) / 60, len(failed))
    if failed:
        log.warning("Failed hours: %s", ", ".join(r.hour for r in failed))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="stream hourly files into parquet")
    pi.add_argument("--start", required=True, help="first day (inclusive), YYYY-MM-DD")
    pi.add_argument("--end", required=True, help="last day (inclusive), YYYY-MM-DD")
    pi.add_argument("--out", default="data/gharchive/events")
    pi.add_argument("--workers", type=int, default=4)
    pi.add_argument("--hours", type=int, nargs="*", default=None, help="subset of hours (default: all 24)")
    pi.add_argument("--repo-filter", default=None, help="text file with one owner/name per line")

    ps = sub.add_parser("select-repos", help="list collaborative repos from ingested events")
    ps.add_argument("--events", required=True)
    ps.add_argument("--out", required=True)
    ps.add_argument("--min-authors", type=int, default=2)

    pf = sub.add_parser("filter", help="filter ingested hourly files to a repo list")
    pf.add_argument("--events", required=True)
    pf.add_argument("--repos", required=True)
    pf.add_argument("--out", required=True)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.cmd == "ingest":
        _cmd_ingest(args)
    elif args.cmd == "select-repos":
        repos = select_collaborative_repos(args.events, args.min_authors)
        Path(args.out).write_text("\n".join(repos) + "\n")
        log.info("%d repos with >= %d distinct human PR authors -> %s", len(repos), args.min_authors, args.out)
    elif args.cmd == "filter":
        kept = filter_event_files(args.events, load_repo_list(args.repos), args.out)
        log.info("kept %d rows -> %s", kept, args.out)


if __name__ == "__main__":
    main()
