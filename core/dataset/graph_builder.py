"""
Build PyTorch Geometric heterogeneous graphs from GitHub event streams.

A project "snapshot" at time T is a HeteroData graph with node types:
  - pr          : open pull requests
  - file_zone   : high-level file areas (auth, api, db, infra, test, other)
  - contributor : active contributors
  - milestone   : inferred release window

Each graph is a 7-day window of activity.  The JEPA training task predicts the
representation at T+14 from the representation at T, conditioned on events
that occurred between T and T+14.

Node feature dimensions:
  pr          : PR_FEAT_DIM  = 12
  file_zone   : FILE_FEAT_DIM = 8
  contributor : CONTRIB_FEAT_DIM = 6
  milestone   : MILE_FEAT_DIM = 6
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

log = logging.getLogger(__name__)

PR_FEAT_DIM = 12
FILE_FEAT_DIM = 8
CONTRIB_FEAT_DIM = 6
MILE_FEAT_DIM = 6
EVENT_FEAT_DIM = 16  # inter-snapshot event vector

FILE_ZONES = ["auth", "api", "db", "infra", "test", "config", "frontend", "other"]

# Labels that signal a revert / regression
REVERT_KEYWORDS = {"revert", "rollback", "undo", "hotfix", "fix regression"}


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _zone_for_path(path: str) -> str:
    p = path.lower()
    if any(k in p for k in ("auth", "oauth", "token", "login", "session", "permission")):
        return "auth"
    if any(k in p for k in ("api", "router", "endpoint", "handler", "view", "controller")):
        return "api"
    if any(k in p for k in ("db", "database", "migration", "model", "schema", "sql")):
        return "db"
    if any(k in p for k in ("infra", "deploy", "docker", "k8s", "ci", "cd", ".github")):
        return "infra"
    if any(k in p for k in ("test", "spec", "__test__", "jest", "pytest")):
        return "test"
    if any(k in p for k in ("config", "settings", "env", ".yaml", ".toml", ".ini")):
        return "config"
    if any(k in p for k in ("frontend", "ui", "component", "page", "style", "css")):
        return "frontend"
    return "other"


class RepoTimeline:
    """
    Accumulates normalized events for a single repo and produces snapshots.
    """

    def __init__(self, repo: str) -> None:
        self.repo = repo
        self.events: list[dict] = []

    def add_event(self, ev: dict) -> None:
        self.events.append(ev)

    def get_snapshots(
        self,
        window_days: int = 7,
        step_days: int = 7,
        min_events: int = 5,
    ) -> list["Snapshot"]:
        """
        Slide a window over the event timeline, yielding Snapshot objects.
        Each snapshot covers [t, t+window_days).
        """
        if not self.events:
            return []

        sorted_events = sorted(self.events, key=lambda e: e["timestamp"])
        first_ts = _parse_ts(sorted_events[0]["timestamp"])
        last_ts = _parse_ts(sorted_events[-1]["timestamp"])

        snapshots = []
        t = first_ts
        while t + timedelta(days=window_days) <= last_ts:
            window_end = t + timedelta(days=window_days)
            window_events = [
                e for e in sorted_events
                if t <= _parse_ts(e["timestamp"]) < window_end
            ]
            if len(window_events) >= min_events:
                snap = Snapshot(
                    repo=self.repo,
                    t_start=t,
                    t_end=window_end,
                    events=window_events,
                )
                snapshots.append(snap)
            t += timedelta(days=step_days)
        return snapshots


class Snapshot:
    """
    A time-windowed view of a repo's activity.
    Produces feature tensors and labels.
    """

    def __init__(
        self,
        repo: str,
        t_start: datetime,
        t_end: datetime,
        events: list[dict],
    ) -> None:
        self.repo = repo
        self.t_start = t_start
        self.t_end = t_end
        self.events = events

    @property
    def label_is_regression(self) -> int:
        """1 if any event in this window looks like a revert/regression."""
        for ev in self.events:
            if ev["type"] in {"pr_merged", "pr_closed"}:
                title = ev["payload"].get("title", "").lower()
                if any(kw in title for kw in REVERT_KEYWORDS):
                    return 1
        return 0

    def to_node_features(self) -> dict[str, np.ndarray]:
        """
        Compute node feature matrices for this snapshot.
        Returns dict with keys matching node types.
        """
        prs = self._pr_features()
        file_zones = self._file_zone_features()
        contribs = self._contributor_features()
        milestone = self._milestone_features()
        return {
            "pr": prs,
            "file_zone": file_zones,
            "contributor": contribs,
            "milestone": milestone,
        }

    def to_event_vector(self) -> np.ndarray:
        """Compact representation of all events in this window (EVENT_FEAT_DIM)."""
        counts = defaultdict(int)
        for ev in self.events:
            counts[ev["type"]] += 1

        vec = np.zeros(EVENT_FEAT_DIM, dtype=np.float32)
        vec[0] = _safe_log(counts.get("pr_opened", 0))
        vec[1] = _safe_log(counts.get("pr_merged", 0))
        vec[2] = _safe_log(counts.get("pr_closed", 0))
        vec[3] = _safe_log(counts.get("pr_review_approved", 0))
        vec[4] = _safe_log(counts.get("pr_review_changes_requested", 0))
        vec[5] = _safe_log(counts.get("issue_opened", 0))
        vec[6] = _safe_log(counts.get("issue_closed", 0))
        vec[7] = _safe_log(counts.get("push", 0))
        vec[8] = _safe_log(counts.get("release_published", 0))
        vec[9] = _safe_log(counts.get("issue_comment", 0))
        vec[10] = len(set(ev["actor"] for ev in self.events)) / 20.0  # unique actors
        vec[11] = len(self.events) / 100.0  # total event density
        # revert signal
        revert_count = sum(
            1 for ev in self.events
            if ev["type"] in {"pr_merged", "pr_closed"}
            and any(kw in ev["payload"].get("title", "").lower() for kw in REVERT_KEYWORDS)
        )
        vec[12] = min(1.0, revert_count / 3.0)
        # PR size signals
        pr_events = [ev for ev in self.events if ev["type"] in {"pr_opened", "pr_merged"}]
        if pr_events:
            avg_additions = np.mean([ev["payload"].get("additions", 0) for ev in pr_events])
            vec[13] = _safe_log(avg_additions)
        # review lag (approx: ratio of reviews to PRs)
        review_count = counts.get("pr_review_approved", 0) + counts.get("pr_review_changes_requested", 0)
        pr_total = max(1, counts.get("pr_opened", 0))
        vec[14] = min(1.0, review_count / pr_total)
        # weekend activity (rough signal for crunch)
        weekend_events = sum(1 for ev in self.events if _parse_ts(ev["timestamp"]).weekday() >= 5)
        vec[15] = weekend_events / max(1, len(self.events))
        return vec

    # ------------------------------------------------------------------ #
    # Private feature builders                                             #
    # ------------------------------------------------------------------ #

    def _pr_features(self) -> np.ndarray:
        """One row per PR touched in this window. Shape: (n_prs, PR_FEAT_DIM)."""
        prs: dict[int, dict] = {}
        for ev in self.events:
            if ev["type"] not in {"pr_opened", "pr_merged", "pr_closed", "pr_reopened"}:
                continue
            num = ev["payload"].get("pr_number")
            if num is None:
                continue
            if num not in prs:
                prs[num] = {"events": [], "payload": ev["payload"]}
            prs[num]["events"].append(ev["type"])

        if not prs:
            return np.zeros((1, PR_FEAT_DIM), dtype=np.float32)

        rows = []
        for pr_data in prs.values():
            p = pr_data["payload"]
            is_merged = "pr_merged" in pr_data["events"]
            is_revert = any(
                kw in p.get("title", "").lower() for kw in REVERT_KEYWORDS
            )
            row = np.array([
                _safe_log(p.get("additions", 0)),
                _safe_log(p.get("deletions", 0)),
                min(1.0, p.get("changed_files", 0) / 20.0),
                float(p.get("draft", False)),
                float(is_merged),
                float(is_revert),
                float(p.get("base_branch", "") in {"main", "master"}),
                min(1.0, p.get("body_length", 0) / 1000.0),
                0.0, 0.0, 0.0, 0.0,  # reserved
            ], dtype=np.float32)
            rows.append(row)
        return np.stack(rows)

    def _file_zone_features(self) -> np.ndarray:
        """One row per file zone. Shape: (8, FILE_FEAT_DIM)."""
        zone_pr_count = defaultdict(int)
        zone_revert_count = defaultdict(int)
        zone_push_count = defaultdict(int)

        for ev in self.events:
            if ev["type"] == "push":
                # We don't have file paths in GH Archive push events directly
                # — use ref as a proxy signal
                ref = ev["payload"].get("ref", "")
                zone = "infra" if "deploy" in ref or "release" in ref else "other"
                zone_push_count[zone] += ev["payload"].get("commit_count", 0)
            elif ev["type"] in {"pr_merged", "pr_closed"}:
                title = ev["payload"].get("title", "").lower()
                zone = "auth" if any(k in title for k in ("auth", "token", "login")) else \
                       "db" if any(k in title for k in ("db", "migration", "schema")) else \
                       "infra" if any(k in title for k in ("deploy", "ci", "infra")) else \
                       "test" if "test" in title else "other"
                zone_pr_count[zone] += 1
                if any(kw in title for kw in REVERT_KEYWORDS):
                    zone_revert_count[zone] += 1

        rows = []
        for zone in FILE_ZONES:
            pr_c = zone_pr_count.get(zone, 0)
            rev_c = zone_revert_count.get(zone, 0)
            push_c = zone_push_count.get(zone, 0)
            rows.append(np.array([
                _safe_log(pr_c),
                _safe_log(push_c),
                min(1.0, rev_c / max(1, pr_c)),  # revert rate in this zone
                float(zone == "auth"),
                float(zone == "infra"),
                float(zone == "test"),
                float(pr_c > 0),  # was this zone active?
                float(rev_c > 0),  # did this zone have a revert?
            ], dtype=np.float32))
        return np.stack(rows)

    def _contributor_features(self) -> np.ndarray:
        """One row per unique contributor. Shape: (n_contribs, CONTRIB_FEAT_DIM)."""
        contrib_events: dict[str, list[str]] = defaultdict(list)
        for ev in self.events:
            contrib_events[ev["actor"]].append(ev["type"])

        if not contrib_events:
            return np.zeros((1, CONTRIB_FEAT_DIM), dtype=np.float32)

        rows = []
        for actor, ev_types in contrib_events.items():
            counts = defaultdict(int)
            for t in ev_types:
                counts[t] += 1
            revert_prs = sum(
                1 for ev in self.events
                if ev["actor"] == actor
                and ev["type"] in {"pr_merged", "pr_closed"}
                and any(kw in ev["payload"].get("title", "").lower() for kw in REVERT_KEYWORDS)
            )
            rows.append(np.array([
                _safe_log(counts.get("pr_opened", 0) + counts.get("pr_merged", 0)),
                _safe_log(counts.get("push", 0)),
                _safe_log(counts.get("pr_review_approved", 0)),
                min(1.0, revert_prs / 3.0),
                float(counts.get("pr_review_changes_requested", 0) > 0),
                min(1.0, len(ev_types) / 20.0),
            ], dtype=np.float32))
        return np.stack(rows[:20])  # cap at 20 contributors

    def _milestone_features(self) -> np.ndarray:
        """Single-row milestone / release window features. Shape: (1, MILE_FEAT_DIM)."""
        releases = [ev for ev in self.events if ev["type"] == "release_published"]
        pr_merged = [ev for ev in self.events if ev["type"] == "pr_merged"]
        issues_closed = [ev for ev in self.events if ev["type"] == "issue_closed"]
        issues_opened = [ev for ev in self.events if ev["type"] == "issue_opened"]
        pushes = [ev for ev in self.events if ev["type"] == "push"]

        total_commits = sum(ev["payload"].get("commit_count", 0) for ev in pushes)
        completion_proxy = len(issues_closed) / max(1, len(issues_opened) + len(issues_closed))

        return np.array([[
            float(len(releases) > 0),          # had a release this window
            min(1.0, len(releases) / 2.0),      # release count
            completion_proxy,                   # issue close rate
            _safe_log(len(pr_merged)),          # merge velocity
            _safe_log(total_commits),           # commit volume
            float(len(releases) > 0 and any(r["payload"].get("prerelease") for r in releases)),
        ]], dtype=np.float32)


def _safe_log(value: float) -> float:
    return math.log1p(max(0.0, value)) / 10.0
