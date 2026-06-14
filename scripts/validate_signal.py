"""
Signal Validation Script
========================

Réponse à UNE question : est-ce qu'un signal observable dans GitHub
permet de prédire des "delivery failures" mieux que le hasard ?

Ce script est intentionnellement simple. Pas de JEPA, pas de HGT, pas de Qdrant.
Juste : données → features tabulaires → baseline → AUC.

Si AUC > 0.65 → le signal existe → on continue avec JEPA.
Si AUC ≈ 0.50 → revoir les labels ou les features, pas l'architecture.

Usage :
    # Étape 1 : télécharger les données (1-2h selon débit)
    python scripts/validate_signal.py download --days 30

    # Étape 2 : construire les features et labels (10-30 min)
    python scripts/validate_signal.py build

    # Étape 3 : entraîner les baselines et voir l'AUC (2-5 min)
    python scripts/validate_signal.py train

    # Tout d'un coup (long)
    python scripts/validate_signal.py all

Dépendances minimales (pas besoin du projet complet) :
    pip install httpx polars scikit-learn tqdm orjson
    pip install lightgbm  # optionnel mais recommandé
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# On ajoute le root du projet pour les imports si lancé depuis scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import polars as pl
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_RESULTS = Path("data/results")

GH_ARCHIVE_BASE = "https://data.gharchive.org"

# Repos cibles — actifs, publics, avec des failures connues dans leur historique.
# On filtre GH Archive sur ces repos pour réduire le bruit.
TARGET_REPOS = {
    # Meta / React ecosystem
    "facebook/react",
    "facebook/react-native",
    "vercel/next.js",
    # Microsoft
    "microsoft/vscode",
    "microsoft/TypeScript",
    # Python
    "django/django",
    "pallets/flask",
    "psf/requests",
    "encode/httpx",
    # Ruby
    "rails/rails",
    # Go
    "golang/go",
    "kubernetes/kubernetes",
    # Rust
    "rust-lang/rust",
    # DB / infra
    "redis/redis",
    "postgres/postgres",
    "hashicorp/terraform",
    # JS tooling
    "webpack/webpack",
    "vitejs/vite",
    "prettier/prettier",
    # Python ML
    "huggingface/transformers",
    "pytorch/pytorch",
    "scikit-learn/scikit-learn",
}

# Mots-clés qui définissent un "failure" dans un titre de PR, commit, ou issue
FAILURE_KEYWORDS = {
    "revert", "rollback", "hotfix", "hot-fix", "hot fix",
    "fix regression", "regression", "broken", "breaking",
    "emergency", "urgent fix", "critical fix", "patch",
    "undo", "restore previous",
}

WINDOW_DAYS = 7        # features construites sur 7 jours
HORIZON_DAYS = 7       # label : failure dans les HORIZON_DAYS jours suivants

# Heures téléchargées depuis GH Archive (pour réduire le volume)
# 0,6,12,18 = 4 fichiers par jour au lieu de 24
DOWNLOAD_HOURS = [0, 6, 12, 18]


# ─────────────────────────────────────────────
# ÉTAPE 0 : DOWNLOAD
# ─────────────────────────────────────────────

def cmd_download(days: int = 30, start_date: str | None = None) -> None:
    """
    Télécharge N jours de GH Archive.
    Seulement 4 fichiers/jour pour économiser le disque.
    30 jours × 4 fichiers × ~25MB = ~3 GB.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    end = date.today() - timedelta(days=1)  # hier (le jour courant peut être incomplet)
    start = date.fromisoformat(start_date) if start_date else end - timedelta(days=days)

    urls = []
    d = start
    while d <= end:
        for h in DOWNLOAD_HOURS:
            fname = f"{d}-{h}.json.gz"
            url = f"{GH_ARCHIVE_BASE}/{fname}"
            dest = DATA_RAW / fname
            urls.append((url, dest))
        d += timedelta(days=1)

    log.info("Téléchargement de %d fichiers (%d jours × %d heures)", len(urls), days, len(DOWNLOAD_HOURS))
    log.info("Destination : %s", DATA_RAW.absolute())
    log.info("Volume estimé : ~%.0f MB", len(urls) * 25)

    downloaded = skipped = errors = 0
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for url, dest in tqdm(urls, desc="GH Archive"):
            if dest.exists():
                skipped += 1
                continue
            try:
                r = client.get(url)
                r.raise_for_status()
                dest.write_bytes(r.content)
                downloaded += 1
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    pass  # Heure pas encore disponible
                else:
                    log.warning("Erreur HTTP %s pour %s", exc.response.status_code, url)
                errors += 1

    log.info("✓ Téléchargé %d | Déjà présents : %d | Erreurs : %d", downloaded, skipped, errors)


# ─────────────────────────────────────────────
# ÉTAPE 1 : PARSING + NORMALISATION
# ─────────────────────────────────────────────

def _is_failure_event(ev_type: str, payload: dict) -> bool:
    """
    Retourne True si cet événement ressemble à un failure signal.
    Utilisé pour construire les LABELS (côté futur T+7).
    """
    title = ""
    if ev_type == "PullRequestEvent":
        pr = payload.get("pull_request", {})
        title = (pr.get("title") or "").lower()
        body = (pr.get("body") or "").lower()
        # Aussi : PR réouverte après merge = régression probable
        if payload.get("action") == "reopened":
            return True
        if any(kw in title or kw in body for kw in FAILURE_KEYWORDS):
            return True
    elif ev_type == "PushEvent":
        commits = payload.get("commits", [])
        for c in commits:
            msg = (c.get("message") or "").lower()
            if any(kw in msg for kw in FAILURE_KEYWORDS):
                return True
    elif ev_type == "IssuesEvent":
        issue = payload.get("issue", {})
        title = (issue.get("title") or "").lower()
        # Issue réouverte = signal fort
        if payload.get("action") == "reopened":
            return True
        if any(kw in title for kw in {"regression", "broken", "hotfix", "critical"}):
            return True
    return False


def parse_archive_files(files: list[Path]) -> list[dict]:
    """
    Parse les fichiers GH Archive et retourne une liste d'événements normalisés.
    Ne garde que les repos TARGET_REPOS.
    """
    events = []
    for path in tqdm(files, desc="Parsing"):
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    repo = raw.get("repo", {}).get("name", "")
                    if repo not in TARGET_REPOS:
                        continue

                    ev_type = raw.get("type", "")
                    if ev_type not in {
                        "PushEvent", "PullRequestEvent", "IssuesEvent",
                        "PullRequestReviewEvent", "ReleaseEvent", "IssueCommentEvent",
                    }:
                        continue

                    payload = raw.get("payload", {})
                    ts = raw.get("created_at", "")

                    events.append({
                        "repo": repo,
                        "type": ev_type,
                        "actor": raw.get("actor", {}).get("login", ""),
                        "ts": ts,
                        "is_failure": _is_failure_event(ev_type, payload),
                        # Features brutes
                        "pr_additions": payload.get("pull_request", {}).get("additions", 0),
                        "pr_deletions": payload.get("pull_request", {}).get("deletions", 0),
                        "pr_changed_files": payload.get("pull_request", {}).get("changed_files", 0),
                        "pr_merged": int(
                            ev_type == "PullRequestEvent"
                            and payload.get("pull_request", {}).get("merged", False)
                        ),
                        "pr_opened": int(
                            ev_type == "PullRequestEvent"
                            and payload.get("action") == "opened"
                        ),
                        "issue_opened": int(
                            ev_type == "IssuesEvent"
                            and payload.get("action") == "opened"
                        ),
                        "issue_reopened": int(
                            ev_type == "IssuesEvent"
                            and payload.get("action") == "reopened"
                        ),
                        "commit_count": len(payload.get("commits", [])) if ev_type == "PushEvent" else 0,
                        "is_release": int(ev_type == "ReleaseEvent"),
                    })
        except Exception as exc:
            log.warning("Erreur lecture %s : %s", path.name, exc)

    log.info("✓ %d événements parsés pour %d repos cibles", len(events), len(TARGET_REPOS))
    return events


# ─────────────────────────────────────────────
# ÉTAPE 2 : FEATURES + LABELS
# ─────────────────────────────────────────────

def build_windows(events: list[dict]) -> list[dict]:
    """
    Pour chaque repo, construit des fenêtres glissantes de WINDOW_DAYS jours.

    Feature window : [T - WINDOW_DAYS, T)
    Label window   : [T, T + HORIZON_DAYS)   ← PAS de leakage

    Returns liste de dicts {features..., label}.
    """
    # Grouper par repo
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        try:
            ts = datetime.fromisoformat(ev["ts"].replace("Z", "+00:00"))
            by_repo[ev["repo"]].append({**ev, "dt": ts})
        except ValueError:
            continue

    windows = []
    for repo, repo_events in by_repo.items():
        repo_events.sort(key=lambda e: e["dt"])
        if not repo_events:
            continue

        first = repo_events[0]["dt"].date()
        last = repo_events[-1]["dt"].date()

        t = first + timedelta(days=WINDOW_DAYS)
        while t + timedelta(days=HORIZON_DAYS) <= last:
            feat_start = datetime(t.year, t.month, t.day, tzinfo=timezone.utc) - timedelta(days=WINDOW_DAYS)
            feat_end = datetime(t.year, t.month, t.day, tzinfo=timezone.utc)
            label_end = feat_end + timedelta(days=HORIZON_DAYS)

            feat_evs = [e for e in repo_events if feat_start <= e["dt"] < feat_end]
            label_evs = [e for e in repo_events if feat_end <= e["dt"] < label_end]

            if len(feat_evs) < 3:  # fenêtre trop creuse
                t += timedelta(days=WINDOW_DAYS)
                continue

            # ── FEATURES (observées avant T) ──────────────────────────────
            actors = {e["actor"] for e in feat_evs}
            pr_sizes = [
                e["pr_additions"] + e["pr_deletions"]
                for e in feat_evs
                if e["pr_merged"]
            ]
            days = max(1, WINDOW_DAYS)

            feat = {
                "repo": repo,
                "window_start": feat_start.isoformat(),
                # Vélocité
                "n_events": len(feat_evs),
                "n_commits": sum(e["commit_count"] for e in feat_evs),
                "n_pr_merged": sum(e["pr_merged"] for e in feat_evs),
                "n_pr_opened": sum(e["pr_opened"] for e in feat_evs),
                "n_issue_opened": sum(e["issue_opened"] for e in feat_evs),
                "n_issue_reopened": sum(e["issue_reopened"] for e in feat_evs),
                "n_releases": sum(e["is_release"] for e in feat_evs),
                # Auteurs
                "n_unique_authors": len(actors),
                # Taille des PRs
                "max_pr_size": max(pr_sizes) if pr_sizes else 0,
                "mean_pr_size": sum(pr_sizes) / len(pr_sizes) if pr_sizes else 0,
                "n_large_prs": sum(1 for s in pr_sizes if s > 500),
                # Entropie d'activité (signal de rush)
                "commits_per_day": sum(e["commit_count"] for e in feat_evs) / days,
                "prs_per_day": sum(e["pr_merged"] for e in feat_evs) / days,
                # Signal d'alerte passé (30 derniers jours dans la fenêtre)
                "n_failures_in_window": sum(1 for e in feat_evs if e["is_failure"]),
                # Ratio issues réouvertes (signal fort de régressions passées)
                "reopen_ratio": (
                    sum(e["issue_reopened"] for e in feat_evs)
                    / max(1, sum(e["issue_opened"] for e in feat_evs))
                ),
            }

            # ── LABEL (ce qui arrive dans les HORIZON_DAYS suivants) ──────
            label = int(any(e["is_failure"] for e in label_evs))
            feat["label"] = label
            windows.append(feat)

            t += timedelta(days=WINDOW_DAYS)

    log.info("✓ %d fenêtres construites", len(windows))
    pos = sum(1 for w in windows if w["label"] == 1)
    neg = len(windows) - pos
    log.info("  Positifs (failure) : %d  |  Négatifs : %d  |  Taux : %.1f%%",
              pos, neg, 100 * pos / max(1, len(windows)))
    return windows


# ─────────────────────────────────────────────
# ÉTAPE 3 : BUILD (download → parse → windows → parquet)
# ─────────────────────────────────────────────

def cmd_build() -> None:
    files = sorted(DATA_RAW.glob("*.json.gz"))
    if not files:
        log.error("Aucun fichier GH Archive dans %s — lance d'abord 'download'", DATA_RAW)
        sys.exit(1)

    log.info("Parsing de %d fichiers...", len(files))
    events = parse_archive_files(files)

    if not events:
        log.error("Aucun événement trouvé. Vérifie que les repos TARGET_REPOS ont de l'activité.")
        sys.exit(1)

    log.info("Construction des fenêtres temporelles...")
    windows = build_windows(events)

    if not windows:
        log.error("Aucune fenêtre construite. Données insuffisantes.")
        sys.exit(1)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(windows)
    out = DATA_PROCESSED / "windows.parquet"
    df.write_parquet(out)
    log.info("✓ Sauvegardé : %s  (%d lignes, %d colonnes)", out, len(df), len(df.columns))

    # Aperçu rapide
    print("\n── Aperçu des features ──")
    print(df.describe())
    print(f"\nDistribution labels: {df['label'].value_counts()}")


# ─────────────────────────────────────────────
# ÉTAPE 4 : TRAIN + EVALUATE
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "n_events", "n_commits", "n_pr_merged", "n_pr_opened",
    "n_issue_opened", "n_issue_reopened", "n_releases",
    "n_unique_authors", "max_pr_size", "mean_pr_size", "n_large_prs",
    "commits_per_day", "prs_per_day",
    "n_failures_in_window", "reopen_ratio",
]


def cmd_train() -> None:
    import numpy as np  # noqa: F811
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_score
    from sklearn.preprocessing import StandardScaler

    out_path = DATA_PROCESSED / "windows.parquet"
    if not out_path.exists():
        log.error("windows.parquet introuvable — lance d'abord 'build'")
        sys.exit(1)

    df = pl.read_parquet(out_path)
    log.info("Chargé %d fenêtres", len(df))

    # ── Split temporel strict ─────────────────────────────────────────
    df = df.sort("window_start")
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_df = df[:n_train]
    val_df = df[n_train:n_train + n_val]
    test_df = df[n_train + n_val:]

    log.info("Split — Train: %d | Val: %d | Test: %d", len(train_df), len(val_df), len(test_df))

    # Si le test set n'a aucun positif (trop petit), évaluer sur tout le dataset.
    # C'est une borne optimiste (légère fuite), mais utile pour savoir si le signal existe.
    import numpy as np
    test_labels = test_df["label"].to_numpy().astype(int)
    full_eval_mode = int(test_labels.sum()) == 0
    if full_eval_mode:
        log.warning(
            "Test set sans positifs (%d samples, 0 failure) — "
            "évaluation sur le dataset complet (résultat optimiste, indicatif). "
            "Ajouter plus de données pour une mesure fiable.",
            len(test_df),
        )

    def to_xy(df_: pl.DataFrame):
        X = df_.select(FEATURE_COLS).to_numpy().astype(np.float32)
        y = df_["label"].to_numpy().astype(int)
        return X, y

    X_train, y_train = to_xy(train_df)
    X_val, y_val = to_xy(val_df)
    X_test, y_test = to_xy(test_df)

    # En mode full_eval on réentraîne sur tout et évalue sur tout (indicatif)
    if full_eval_mode:
        X_all, y_all = to_xy(df)
    else:
        X_all, y_all = X_test, y_test

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    if full_eval_mode:
        X_all_s = scaler.fit_transform(X_all)

    results = {}

    # ── Baseline 1 : Régression logistique ───────────────────────────
    log.info("Entraînement Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, C=1.0)
    if full_eval_mode:
        lr.fit(X_all_s, y_all)
        lr_proba = lr.predict_proba(X_all_s)[:, 1]
        results["LogisticRegression"] = _eval(y_all, lr_proba, "LogisticRegression")
    else:
        lr.fit(X_train_s, y_train)
        lr_proba = lr.predict_proba(X_test_s)[:, 1]
        results["LogisticRegression"] = _eval(y_test, lr_proba, "LogisticRegression")

    # ── Baseline 2 : LightGBM (si dispo) ─────────────────────────────
    try:
        import lightgbm as lgb
        log.info("Entraînement LightGBM...")
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 10,
            "verbose": -1,
        }
        model = lgb.train(
            params, dtrain,
            num_boost_round=200,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(50)],
        )
        if full_eval_mode:
            lgb_proba = model.predict(X_all, num_iteration=model.best_iteration)
            results["LightGBM"] = _eval(y_all, lgb_proba, "LightGBM")
        else:
            lgb_proba = model.predict(X_test, num_iteration=model.best_iteration)
            results["LightGBM"] = _eval(y_test, lgb_proba, "LightGBM")

        # Feature importance
        print("\n── Top features (LightGBM) ──")
        fi = sorted(
            zip(FEATURE_COLS, model.feature_importance(importance_type="gain")),
            key=lambda x: -x[1],
        )
        for name, score in fi[:10]:
            bar = "█" * int(score / max(f[1] for f in fi) * 30)
            print(f"  {name:<30} {bar} ({score:.0f})")

    except ImportError:
        log.info("LightGBM non installé (pip install lightgbm) — skip")

    # ── Résultats finaux ──────────────────────────────────────────────
    print("\n" + "═" * 60)
    label = "DATASET COMPLET (indicatif — données insuffisantes)" if full_eval_mode else "TEST SET"
    print(f"RÉSULTATS — {label}")
    print("═" * 60)
    for name, metrics in results.items():
        print(f"\n  {name}")
        for k, v in metrics.items():
            print(f"    {k:<25} {v:.4f}")

    print("\n── VERDICT ──")
    best_auc = max(m["auc_roc"] for m in results.values())
    if best_auc >= 0.70:
        print(f"  ✓ AUC = {best_auc:.3f}  →  Signal fort. Le JEPA a du sens.")
        print("    Prochaine étape : passer à models/world_model.py")
    elif best_auc >= 0.62:
        print(f"  ◑ AUC = {best_auc:.3f}  →  Signal présent mais faible.")
        print("    Ajouter plus de données (plus de repos, plus de jours).")
        print("    Ou affiner les labels (cf. FAILURE_KEYWORDS en haut du script).")
    else:
        print(f"  ✗ AUC = {best_auc:.3f}  →  Signal absent ou trop faible.")
        print("    Ne pas faire de JEPA. Revoir les labels ou les features.")
        print("    Diagnostic : regarde les histogrammes des features (col n_failures_in_window).")

    print("═" * 60)

    # Sauvegarde
    DATA_RESULTS.mkdir(parents=True, exist_ok=True)
    summary = pl.DataFrame([
        {"model": name, **metrics}
        for name, metrics in results.items()
    ])
    summary.write_parquet(DATA_RESULTS / "baseline_results.parquet")
    log.info("Résultats sauvegardés dans %s", DATA_RESULTS / "baseline_results.parquet")


def _eval(y_true, y_proba, name: str) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score

    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    # Precision@5% : parmi les 5% de cas les plus risqués prédits, combien sont vrais ?
    threshold_idx = int(len(y_proba) * 0.95)
    sorted_indices = np.argsort(y_proba)
    top5_indices = sorted_indices[threshold_idx:]
    p5 = y_true[top5_indices].mean() if len(top5_indices) > 0 else 0.0

    base_rate = y_true.mean()
    lift = p5 / base_rate if base_rate > 0 else 0.0

    log.info(
        "[%s] AUC=%.3f  AP=%.3f  P@5%%=%.3f  Lift=%.1fx",
        name, auc, ap, p5, lift,
    )
    return {
        "auc_roc": auc,
        "avg_precision": ap,
        "precision_at_5pct": p5,
        "lift_at_5pct": lift,
        "base_rate": base_rate,
    }


# ─────────────────────────────────────────────
# DIAGNOSTIC : analyse rapide sans training
# ─────────────────────────────────────────────

def cmd_diagnose() -> None:
    """
    Analyse rapide des données brutes sans entraîner de modèle.
    Utile pour vérifier que les labels ont du sens.
    """
    out_path = DATA_PROCESSED / "windows.parquet"
    if not out_path.exists():
        log.error("Lance d'abord 'build'")
        sys.exit(1)

    df = pl.read_parquet(out_path)

    print("\n══════════════════════════════════════════")
    print("DIAGNOSTIC — Analyse des fenêtres")
    print("══════════════════════════════════════════")
    print(f"  Total fenêtres  : {len(df)}")
    print(f"  Repos couverts  : {df['repo'].n_unique()}")
    print(f"  Période         : {df['window_start'].min()} → {df['window_start'].max()}")

    pos = df.filter(pl.col("label") == 1)
    neg = df.filter(pl.col("label") == 0)
    rate = len(pos) / max(1, len(df))
    print(f"\n  Failure rate    : {rate:.1%}  ({len(pos)} positifs / {len(neg)} négatifs)")

    if rate < 0.03:
        print("  ⚠️  Taux très bas — labels peut-être trop stricts.")
        print("      Vérifie FAILURE_KEYWORDS ou élargis les critères.")
    elif rate > 0.50:
        print("  ⚠️  Taux très élevé — labels peut-être trop larges.")
        print("      Réduis les FAILURE_KEYWORDS aux signaux forts uniquement.")
    else:
        print("  ✓ Taux raisonnable pour un problème déséquilibré.")

    print("\n── Features moyennes : positifs vs négatifs ──")
    compare_cols = [
        "n_pr_merged", "n_large_prs", "n_issue_reopened",
        "n_failures_in_window", "reopen_ratio", "commits_per_day",
    ]
    for col in compare_cols:
        m_pos = pos[col].mean() or 0
        m_neg = neg[col].mean() or 0
        diff = (m_pos - m_neg) / max(0.001, abs(m_neg))
        signal = "↑" if diff > 0.1 else ("↓" if diff < -0.1 else "≈")
        print(f"  {col:<30}  positifs={m_pos:.2f}  négatifs={m_neg:.2f}  {signal} {diff:+.0%}")

    print("\n── Repos avec le plus de failures ──")
    by_repo = (
        df.group_by("repo")
        .agg([
            pl.col("label").sum().alias("n_failures"),
            pl.col("label").count().alias("n_windows"),
        ])
        .with_columns((pl.col("n_failures") / pl.col("n_windows")).alias("failure_rate"))
        .sort("n_failures", descending=True)
        .head(10)
    )
    print(by_repo)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Validation du signal de livraison depuis GH Archive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Quick check — 7 jours seulement, vite
  python scripts/validate_signal.py download --days 7
  python scripts/validate_signal.py build
  python scripts/validate_signal.py diagnose
  python scripts/validate_signal.py train

  # Full run — 30 jours
  python scripts/validate_signal.py all --days 30

  # Reprendre à partir du parsing si téléchargement déjà fait
  python scripts/validate_signal.py build
  python scripts/validate_signal.py train
        """,
    )
    sub = p.add_subparsers(dest="cmd")

    dl = sub.add_parser("download", help="Télécharge les données GH Archive")
    dl.add_argument("--days", type=int, default=30, help="Nombre de jours (défaut: 30)")
    dl.add_argument("--start", default=None, help="Date de début ISO (YYYY-MM-DD)")

    sub.add_parser("build", help="Parse les données et construit les features")
    sub.add_parser("train", help="Entraîne les baselines et affiche l'AUC")
    sub.add_parser("diagnose", help="Analyse rapide des données sans entraînement")

    all_p = sub.add_parser("all", help="Exécute toutes les étapes")
    all_p.add_argument("--days", type=int, default=30)
    all_p.add_argument("--start", default=None)

    args = p.parse_args()

    if args.cmd == "download":
        cmd_download(days=args.days, start_date=args.start)
    elif args.cmd == "build":
        cmd_build()
    elif args.cmd == "train":
        cmd_train()
    elif args.cmd == "diagnose":
        cmd_diagnose()
    elif args.cmd == "all":
        cmd_download(days=args.days, start_date=args.start)
        cmd_build()
        cmd_diagnose()
        cmd_train()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
