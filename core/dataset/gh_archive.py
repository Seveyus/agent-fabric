"""
GH Archive downloader and event processor.

GH Archive (https://www.gharchive.org/) records all public GitHub events
since 2011. Each hourly file is ~20-80 MB gzipped JSON — one event per line.

Usage:
    from core.dataset.gh_archive import GHArchiveDownloader

    dl = GHArchiveDownloader(output_dir="data/raw")
    dl.download_range("2024-01-01", "2024-03-31")   # ~3 months

    events = dl.iter_events("data/raw/2024-01-01-0.json.gz")
    for ev in events:
        print(ev["type"], ev["repo"]["name"])
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import httpx
from tqdm import tqdm

log = logging.getLogger(__name__)

GH_ARCHIVE_BASE = "https://data.gharchive.org"

# Events we care about for the world model
RELEVANT_EVENT_TYPES = {
    "PushEvent",
    "PullRequestEvent",
    "PullRequestReviewEvent",
    "IssuesEvent",
    "IssueCommentEvent",
    "CreateEvent",
    "DeleteEvent",
    "ReleaseEvent",
    "ForkEvent",
    "WatchEvent",
    "CommitCommentEvent",
}


class GHArchiveDownloader:
    def __init__(self, output_dir: str = "data/raw/gh_archive") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_range(
        self,
        start: str,
        end: str,
        hours: list[int] | None = None,
        skip_existing: bool = True,
    ) -> list[Path]:
        """
        Download all hourly files between start and end (inclusive).

        Args:
            start: ISO date string, e.g. "2024-01-01"
            end:   ISO date string, e.g. "2024-03-31"
            hours: specific hours to download (0-23). Defaults to all 24.
            skip_existing: skip files already on disk.

        Returns:
            List of local paths downloaded.
        """
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
        hours = hours or list(range(24))

        urls = []
        current = start_date
        while current <= end_date:
            for h in hours:
                fname = f"{current}-{h}.json.gz"
                url = f"{GH_ARCHIVE_BASE}/{fname}"
                urls.append((url, self.output_dir / fname))
            current += timedelta(days=1)

        downloaded: list[Path] = []
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            for url, dest in tqdm(urls, desc="Downloading GH Archive"):
                if skip_existing and dest.exists():
                    downloaded.append(dest)
                    continue
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    dest.write_bytes(resp.content)
                    downloaded.append(dest)
                except httpx.HTTPStatusError as exc:
                    log.warning("Skipping %s — HTTP %s", url, exc.response.status_code)
        return downloaded

    def iter_events(
        self, path: str | Path, event_types: set[str] | None = None
    ):
        """
        Yield parsed events from a single GH Archive .json.gz file.

        Args:
            path: local path to the .json.gz file
            event_types: filter to these event types (None = all)
        """
        types = event_types or RELEVANT_EVENT_TYPES
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") in types:
                    yield ev

    def iter_all_events(
        self,
        raw_dir: str | Path | None = None,
        event_types: set[str] | None = None,
    ):
        """Yield events from ALL .json.gz files in raw_dir."""
        raw_dir = Path(raw_dir or self.output_dir)
        files = sorted(raw_dir.glob("*.json.gz"))
        log.info("Processing %d GH Archive files in %s", len(files), raw_dir)
        for path in tqdm(files, desc="Parsing GH Archive"):
            yield from self.iter_events(path, event_types=event_types)


class EventNormalizer:
    """
    Flatten raw GH Archive events into a consistent schema for the graph builder.
    Each normalized event has:
        repo      str      — "owner/repo"
        type      str      — canonical event type (pr_opened, pr_merged, etc.)
        actor     str      — GitHub login
        timestamp str      — ISO datetime
        payload   dict     — event-specific fields
    """

    def normalize(self, raw_event: dict) -> dict | None:
        ev_type = raw_event.get("type", "")
        repo = raw_event.get("repo", {}).get("name", "")
        actor = raw_event.get("actor", {}).get("login", "unknown")
        created_at = raw_event.get("created_at", "")
        payload = raw_event.get("payload", {})

        handler = {
            "PullRequestEvent": self._pr,
            "PullRequestReviewEvent": self._pr_review,
            "IssuesEvent": self._issue,
            "PushEvent": self._push,
            "ReleaseEvent": self._release,
            "IssueCommentEvent": self._comment,
        }.get(ev_type)

        if handler is None:
            return None

        result = handler(payload)
        if result is None:
            return None

        return {
            "repo": repo,
            "type": result["type"],
            "actor": actor,
            "timestamp": created_at,
            "payload": result["payload"],
        }

    def _pr(self, payload: dict) -> dict | None:
        action = payload.get("action", "")
        pr = payload.get("pull_request", {})
        if not pr:
            return None
        type_map = {
            "opened": "pr_opened",
            "closed": "pr_merged" if pr.get("merged") else "pr_closed",
            "reopened": "pr_reopened",
            "review_requested": "pr_review_requested",
        }
        canonical = type_map.get(action)
        if not canonical:
            return None
        return {
            "type": canonical,
            "payload": {
                "pr_number": pr.get("number"),
                "title": pr.get("title", "")[:200],
                "additions": pr.get("additions", 0),
                "deletions": pr.get("deletions", 0),
                "changed_files": pr.get("changed_files", 0),
                "merged": pr.get("merged", False),
                "merged_at": pr.get("merged_at"),
                "base_branch": pr.get("base", {}).get("ref", ""),
                "draft": pr.get("draft", False),
                "body_length": len(pr.get("body") or ""),
            },
        }

    def _pr_review(self, payload: dict) -> dict | None:
        review = payload.get("review", {})
        state = review.get("state", "").lower()
        if state not in {"approved", "changes_requested", "dismissed"}:
            return None
        return {
            "type": f"pr_review_{state}",
            "payload": {
                "pr_number": payload.get("pull_request", {}).get("number"),
                "state": state,
            },
        }

    def _issue(self, payload: dict) -> dict | None:
        action = payload.get("action", "")
        issue = payload.get("issue", {})
        type_map = {"opened": "issue_opened", "closed": "issue_closed", "reopened": "issue_reopened"}
        canonical = type_map.get(action)
        if not canonical:
            return None
        return {
            "type": canonical,
            "payload": {
                "issue_number": issue.get("number"),
                "title": issue.get("title", "")[:200],
                "labels": [lb.get("name", "") for lb in issue.get("labels", [])],
                "state": issue.get("state", ""),
                "body_length": len(issue.get("body") or ""),
            },
        }

    def _push(self, payload: dict) -> dict | None:
        commits = payload.get("commits", [])
        return {
            "type": "push",
            "payload": {
                "commit_count": len(commits),
                "ref": payload.get("ref", ""),
                "distinct_count": payload.get("distinct_size", 0),
            },
        }

    def _release(self, payload: dict) -> dict | None:
        action = payload.get("action", "")
        release = payload.get("release", {})
        if action not in {"published", "released"}:
            return None
        return {
            "type": "release_published",
            "payload": {
                "tag": release.get("tag_name", ""),
                "prerelease": release.get("prerelease", False),
                "draft": release.get("draft", False),
            },
        }

    def _comment(self, payload: dict) -> dict | None:
        return {
            "type": "issue_comment",
            "payload": {
                "issue_number": payload.get("issue", {}).get("number"),
                "body_length": len((payload.get("comment") or {}).get("body") or ""),
            },
        }
