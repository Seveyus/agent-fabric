"""Offline tests for the streaming GH Archive ingester (no network)."""

from __future__ import annotations

import gzip

import orjson
import pytest

pl = pytest.importorskip("polars")

from core.dataset.gharchive_ingest import (  # noqa: E402
    extract,
    iter_gzip_lines,
    rows_from_lines,
    rows_to_frame,
)


def ev(type_, payload, actor="alice", repo="o/r", ts="2025-09-01T10:00:00Z"):
    return {"type": type_, "actor": {"login": actor}, "repo": {"name": repo},
            "created_at": ts, "payload": payload}


PR_OPEN = ev("PullRequestEvent", {"action": "opened", "pull_request": {
    "number": 7, "title": 'Revert "Add cache"', "body": "Reverts o/r#5", "additions": 3,
    "deletions": 40, "changed_files": 2, "commits": 1, "draft": False,
    "author_association": "MEMBER", "user": {"login": "alice"}, "labels": [{}],
    "requested_reviewers": [{}, {}], "base": {"ref": "main", "repo": {"default_branch": "main"}}}})
PR_CLOSE = ev("PullRequestEvent", {"action": "closed", "pull_request": {
    "number": 7, "title": "x", "merged": True, "created_at": "2025-09-01T09:00:00Z",
    "user": {"login": "alice"}, "merged_by": {"login": "bob"}}}, actor="bob")
TRIMMED = ev("PullRequestEvent", {"action": "opened", "pull_request": {
    "base": {}, "head": {}, "id": 1, "number": 8, "url": "u"}})


def test_pr_open_fields():
    row = extract(PR_OPEN)
    assert row["etype"] == "pr_open" and row["number"] == 7
    assert row["is_revert"] and row["reverts_number"] == 5
    assert row["to_default_branch"] is True
    assert row["n_req_reviewers"] == 2 and row["author_assoc"] == "MEMBER"
    assert row["ts"] == 1756720800


def test_pr_close_fields():
    row = extract(PR_CLOSE)
    assert row["etype"] == "pr_close" and row["merged"] is True
    assert row["merged_by"] == "bob" and row["pr_author"] == "alice"
    assert row["pr_created_ts"] == 1756717200


def test_trimmed_payload_is_dropped():
    assert extract(TRIMMED) is None


def test_reverts_reference_to_other_repo_is_ignored():
    e = orjson.loads(orjson.dumps(PR_OPEN))
    e["payload"]["pull_request"]["body"] = "Reverts other/repo#5"
    assert extract(e)["reverts_number"] is None


def test_comment_on_pr_vs_issue():
    on_pr = extract(ev("IssueCommentEvent", {"issue": {"number": 3, "pull_request": {}}}))
    on_issue = extract(ev("IssueCommentEvent", {"issue": {"number": 4}}))
    assert on_pr["etype"] == "pr_comment" and on_issue["etype"] == "issue_comment"


def test_multi_member_gzip_and_push_aggregation():
    lines = [orjson.dumps(e) for e in (
        PR_OPEN, PR_CLOSE, TRIMMED,
        ev("PushEvent", {"size": 2}, ts="2025-09-01T10:05:00Z"),
        ev("PushEvent", {"size": 3}, ts="2025-09-01T10:45:00Z"),
        ev("CreateEvent", {}),  # not wanted -> skipped by the pre-filter
    )]
    # two concatenated gzip members, split into odd-sized chunks
    blob = gzip.compress(b"\n".join(lines[:3]) + b"\n") + gzip.compress(b"\n".join(lines[3:]) + b"\n")
    chunks = [blob[i:i + 37] for i in range(0, len(blob), 37)]
    rows, stats = rows_from_lines(iter_gzip_lines(iter(chunks)))
    assert stats["lines"] == 6
    df = rows_to_frame(rows)
    assert df.filter(pl.col("etype") == "push").height == 1
    assert df.filter(pl.col("etype") == "push")["n_commits"][0] == 5
    assert set(df["etype"]) == {"pr_open", "pr_close", "push"}
    assert df["ts"].is_sorted()


def test_is_bot():
    from core.dataset.gharchive_ingest import is_bot
    assert is_bot("dependabot[bot]") and is_bot("my-ci-bot") and is_bot("github-actions")
    assert not is_bot("abbott") and not is_bot("robotics-dev") and not is_bot(None)


def test_select_and_filter(tmp_path):
    from core.dataset.gharchive_ingest import filter_event_files, select_collaborative_repos
    rows = [extract(ev("PullRequestEvent", PR_OPEN["payload"], actor=a, repo=r))
            for a, r in [("alice", "o/a"), ("bob", "o/a"), ("alice", "o/b"),
                         ("dependabot[bot]", "o/b"), ("carol", "o/c")]]
    src = tmp_path / "src"
    src.mkdir()
    rows_to_frame(rows).write_parquet(src / "h.parquet")
    repos = select_collaborative_repos(src, min_authors=2)
    assert repos == ["o/a"]
    kept = filter_event_files(src, frozenset(repos), tmp_path / "dst")
    assert kept == 2


def test_is_bot_expr_matches_python_version():
    from core.dataset.gharchive_ingest import is_bot, is_bot_expr
    logins = ["dependabot[bot]", "my-ci-bot", "github-actions", "abbott", "robotics-dev", "Renovate", "bot_x"]
    got = pl.DataFrame({"a": logins}).select(is_bot_expr("a"))["a"].to_list()
    assert got == [is_bot(x) for x in logins]
