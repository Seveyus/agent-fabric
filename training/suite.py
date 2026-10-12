"""
Run a named comparison suite over several seeds and aggregate the results.

    python -m training.suite --suite main --config configs/synthetic_pr_merge.toml --seeds 0 1 2
    python -m training.suite --suite jepa --config configs/synthetic_pr_merge.toml --seeds 0 1 2

Writes experiments/results/<config-name>__<suite>.json (every run) and .md
(mean ± std table + paired comparisons). All variants share the cached task
(same graph, same seeds, same splits); the seed changes model initialisation,
mini-batch order and neighbour sampling. Each (variant, seed) runs in its own
process and is saved under runs/suite_runs/, so an interrupted suite resumes
where it stopped (--force re-runs).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from training.config import load_config

log = logging.getLogger(__name__)

# (label, kind, overrides). kind: "gnn" | "logreg" | "lgbm" | "jepa:<mode>"
SUITES: dict[str, list[tuple[str, str, list[str]]]] = {
    "main": [
        ("LogReg (intrinsic)", "logreg", ["features.use_history=false"]),
        ("LightGBM (intrinsic)", "lgbm", ["features.use_history=false"]),
        ("MLP (intrinsic)", "gnn", ['model.encoder="mlp"', "features.use_history=false"]),
        ("HeteroSAGE (intrinsic)", "gnn", ['model.encoder="sage"', "features.use_history=false"]),
        ("HGT (intrinsic)", "gnn", ['model.encoder="hgt"', "features.use_history=false"]),
        ("LogReg (+history)", "logreg", ["features.use_history=true"]),
        ("LightGBM (+history)", "lgbm", ["features.use_history=true"]),
        ("MLP (+history)", "gnn", ['model.encoder="mlp"', "features.use_history=true"]),
        ("HeteroSAGE (+history)", "gnn", ['model.encoder="sage"', "features.use_history=true"]),
        ("HGT (+history)", "gnn", ['model.encoder="hgt"', "features.use_history=true"]),
    ],
    # Self-supervised evaluation, intrinsic features only (the setting where the
    # graph has to supply the history signal).
    "jepa": [
        ("HGT scratch, 100% labels", "gnn", ['model.encoder="hgt"', "features.use_history=false"]),
        ("HGT scratch, 10% labels", "gnn", ['model.encoder="hgt"', "features.use_history=false", "train.label_fraction=0.1"]),
        ("Linear probe, random frozen HGT", "gnn", ['model.encoder="hgt"', "features.use_history=false", "pretrain.freeze_random_encoder=true"]),
        ("Linear probe, JEPA HGT", "jepa:probe", ['model.encoder="hgt"', "features.use_history=false"]),
        ("JEPA HGT fine-tuned, 100% labels", "jepa:finetune", ['model.encoder="hgt"', "features.use_history=false"]),
        ("JEPA HGT fine-tuned, 10% labels", "jepa:finetune", ['model.encoder="hgt"', "features.use_history=false", "train.label_fraction=0.1"]),
    ],
}


# Paired comparisons (A, B) reported as AUC(A) - AUC(B) with a paired bootstrap CI.
PAIRS: dict[str, list[tuple[str, str]]] = {
    "main": [
        ("HeteroSAGE (intrinsic)", "MLP (intrinsic)"),
        ("HGT (intrinsic)", "MLP (intrinsic)"),
        ("HGT (intrinsic)", "HeteroSAGE (intrinsic)"),
        ("LightGBM (+history)", "LightGBM (intrinsic)"),
        ("HeteroSAGE (+history)", "LightGBM (+history)"),
        ("HGT (+history)", "LightGBM (+history)"),
        ("HGT (+history)", "MLP (+history)"),
        ("HGT (+history)", "HeteroSAGE (+history)"),
    ],
    "jepa": [
        ("Linear probe, JEPA HGT", "Linear probe, random frozen HGT"),
        ("JEPA HGT fine-tuned, 100% labels", "HGT scratch, 100% labels"),
        ("JEPA HGT fine-tuned, 10% labels", "HGT scratch, 10% labels"),
    ],
}


def _comparisons(suite: str, pred_dir: Path, labels: set[str]) -> list[dict]:
    from training.compare import compare

    out = []
    for a, b in PAIRS.get(suite, []):
        if a in labels and b in labels:
            try:
                out.append(compare(pred_dir, a, b))
            except (FileNotFoundError, ValueError) as exc:
                log.warning("comparison %s vs %s skipped: %s", a, b, exc)
    return out


def _comparison_markdown(comps: list[dict]) -> str:
    if not comps:
        return ""
    lines = ["", "## Paired comparisons (same test seeds)", "",
             "| A | B | AUC(A) − AUC(B) | 95% paired bootstrap CI | P(diff ≤ 0) |", "|---|---|---|---|---|"]
    for c in comps:
        lines.append(f"| {c['a']} | {c['b']} | {c['auc_diff']:+.4f} | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | "
                     f"{c['p_le_0']:.3f} |")
    lines.append("")
    lines.append("Mean over model seeds of the per-seed AUC difference; 1,000 bootstrap resamples of the "
                 "test seeds shared by both models.")
    return "\n".join(lines) + "\n"


def _run_one(config_path: str, kind: str, overrides: list[str], seed: int) -> dict:
    from training.node_task import prepare_task, run_experiment, select_features, standardise

    overrides = [*overrides, f"experiment.seed={seed}"]
    if kind in ("logreg", "lgbm"):
        from training.tabular_baselines import run_tabular

        cfg = load_config(config_path, overrides)
        task = prepare_task(cfg)
        use_history = cfg.features.use_history
        select_features(task, use_history)
        standardise(task)
        return run_tabular(task, kind, use_history, seed=seed, label_fraction=cfg.train.label_fraction)
    if kind.startswith("jepa:"):
        from training.pretrain_jepa import pretrain

        base = load_config(config_path, overrides)
        hist = "hist" if base.features.use_history else "nohist"
        ckpt = Path(base.experiment.output_dir) / base.experiment.name / f"jepa_{base.model.encoder}_{hist}_seed{seed}.pt"
        if not ckpt.exists():  # one pretraining per (encoder, features, seed), shared by probe / fine-tune runs
            pretrain(load_config(config_path, [*overrides, f'jepa.output="{ckpt}"']))
        pre = json.loads(ckpt.with_suffix(".json").read_text())
        extra = [f'pretrain.checkpoint="{ckpt}"']
        if kind == "jepa:probe":
            extra.append("pretrain.freeze_encoder=true")
        result = run_experiment(load_config(config_path, [*overrides, *extra]))
        result["jepa_pretraining"] = {"seconds": pre["seconds"], "anchors": pre["anchors"],
                                      "final": pre["history"][-1]}
        return result
    return run_experiment(load_config(config_path, overrides))


def _run_isolated(config_path: str, kind: str, overrides: list[str], seed: int, out_json: Path,
                  out_npz: Path) -> dict:
    """Run one experiment in a fresh interpreter (``python -m training.suite --child-run``):
    memory is returned between runs (each run loads the task and forks sampling
    workers) and no library state (e.g. PyG's HeteroLinear timing cache) leaks
    from one run to the next."""
    spec = json.dumps({"config": config_path, "kind": kind, "overrides": overrides, "seed": seed,
                       "out_json": str(out_json), "out_npz": str(out_npz)})
    subprocess.run([sys.executable, "-m", "training.suite", "--child-run", spec], check=True)
    return json.loads(out_json.read_text())


def _child_run(spec_json: str) -> None:
    spec = json.loads(spec_json)
    r = _run_one(spec["config"], spec["kind"], spec["overrides"], spec["seed"])
    y, prob = r.pop("_test_pred")
    np.savez_compressed(spec["out_npz"], y=y, p=prob)
    tmp = Path(spec["out_json"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=2, default=float))
    tmp.rename(spec["out_json"])


def _summary(runs: list[dict]) -> dict:
    def stat(key: str, sub: str = "test") -> tuple[float, float]:
        v = np.array([r[sub][key] for r in runs], dtype=float)
        return float(np.nanmean(v)), float(np.nanstd(v))

    t = np.array([r["timing"]["train_seconds_total"] for r in runs], dtype=float)
    thr = np.array([r["timing"].get("test_seeds_per_second", np.nan) for r in runs], dtype=float)
    return {
        "n_seeds": len(runs),
        "test_roc_auc": stat("roc_auc"),
        "test_avg_precision": stat("avg_precision"),
        "test_f1": stat("f1"),
        "val_roc_auc": stat("roc_auc", "val"),
        "train_seconds": (float(t.mean()), float(t.std())),
        "test_seeds_per_second": float(np.nanmean(thr)),
        "trainable_parameters": runs[0].get("trainable_parameters"),
    }


def _markdown(title: str, meta: dict, rows: list[tuple[str, dict]]) -> str:
    lines = [f"# {title}", "", f"- config: `{meta['config']}`", f"- seeds: {meta['seeds']}",
             f"- data: {meta['data_source']}; seeds per split {meta['split_sizes']}, "
             f"positive rate {meta['split_pos_rate']}",
             f"- environment: {meta['environment']}", "",
             "| Model | Test ROC-AUC | Test AP | Test F1 | Val ROC-AUC | Train s | Test seeds/s | Params |",
             "|---|---|---|---|---|---|---|---|"]
    for label, s in rows:
        f = lambda m: f"{m[0]:.3f} ± {m[1]:.3f}"  # noqa: E731
        lines.append(f"| {label} | {f(s['test_roc_auc'])} | {f(s['test_avg_precision'])} | {f(s['test_f1'])} | "
                     f"{f(s['val_roc_auc'])} | {s['train_seconds'][0]:.1f} | {s['test_seeds_per_second']:.0f} | "
                     f"{s['trainable_parameters']} |")
    lines += ["", "Mean ± std over seeds (model initialisation, batch order, neighbour sampling); "
              "the graph, the seeds and the splits are fixed. Thresholds for F1 are chosen on validation. "
              "Params: trainable parameters (LightGBM: number of trees)."]
    return "\n".join(lines) + "\n"


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label).strip("_").lower()


def run_suite(suite: str, config_path: str, seeds: list[int], out_dir: str = "experiments/results",
              only: list[str] | None = None, pred_root: str = "runs/predictions",
              run_root: str = "runs/suite_runs", force: bool = False) -> dict:
    """Run missing (variant, seed) pairs, each in its own process, then (re)build
    the report from every saved run of this suite."""
    stem = f"{Path(config_path).stem}__{suite}"
    pred_dir, run_dir = Path(pred_root) / stem, Path(run_root) / stem
    pred_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    variants = [v for v in SUITES[suite] if not only or v[0] in only]
    t0 = time.perf_counter()
    for label, kind, overrides in variants:
        for seed in seeds:
            name = f"{_slug(label)}__seed{seed}"
            if (run_dir / f"{name}.json").exists() and (pred_dir / f"{name}.npz").exists() and not force:
                log.info("=== %s | seed %d (done, skipped)", label, seed)
                continue
            log.info("=== %s | seed %d", label, seed)
            _run_isolated(config_path, kind, overrides, seed, run_dir / f"{name}.json", pred_dir / f"{name}.npz")
    return build_report(suite, config_path, seeds, out_dir, pred_root, run_root,
                        wall_seconds=time.perf_counter() - t0)


def build_report(suite: str, config_path: str, seeds: list[int], out_dir: str = "experiments/results",
                 pred_root: str = "runs/predictions", run_root: str = "runs/suite_runs",
                 wall_seconds: float | None = None) -> dict:
    stem = f"{Path(config_path).stem}__{suite}"
    runs: dict[str, list[dict]] = {}
    for label, _, _ in SUITES[suite]:
        for seed in seeds:
            f = Path(run_root) / stem / f"{_slug(label)}__seed{seed}.json"
            if f.exists():
                runs.setdefault(label, []).append(json.loads(f.read_text()))
    if not runs:
        raise FileNotFoundError(f"no saved runs for {stem}")
    gnn_run = next((r[0] for r in runs.values() if "environment" in r[0]), None)
    env = gnn_run["environment"] if gnn_run else {}
    task_meta = gnn_run["task"] if gnn_run else {}
    meta = {
        "suite": suite, "config": config_path, "seeds": seeds,
        "data_source": gnn_run.get("data_source") if gnn_run else None,
        "split_sizes": task_meta.get("split_sizes"),
        "split_pos_rate": {k: round(v, 3) for k, v in (task_meta.get("split_pos_rate") or {}).items()},
        "environment": {k: env.get(k) for k in ("backend", "torch", "torch_geometric", "pyg_lib", "cpu",
                                                "cpu_threads", "git_commit", "git_dirty")},
        "isolation": "one process per run",
        "wall_seconds_last_invocation": wall_seconds,
    }
    summary = {label: _summary(rs) for label, rs in runs.items()}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comps = _comparisons(suite, Path(pred_root) / stem, set(runs))
    (out / f"{stem}.json").write_text(json.dumps(
        {"meta": meta, "summary": summary, "comparisons": comps, "runs": runs}, indent=2, default=float))
    (out / f"{stem}.md").write_text(_markdown(f"{stem}", meta, list(summary.items())) + _comparison_markdown(comps))
    log.info("wrote %s", out / f"{stem}.md")
    return {"meta": meta, "summary": summary, "comparisons": comps}


def main(argv: list[str] | None = None) -> None:
    if argv is None and len(sys.argv) > 2 and sys.argv[1] == "--child-run":   # internal: one isolated run
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        _child_run(sys.argv[2])
        return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True, choices=sorted(SUITES))
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default="experiments/results")
    p.add_argument("--only", nargs="*", default=None, help="subset of variant labels")
    p.add_argument("--report-only", action="store_true",
                   help="rebuild the report (tables + paired comparisons) from saved runs")
    p.add_argument("--force", action="store_true", help="re-run (variant, seed) pairs that already have results")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.report_only:
        build_report(args.suite, args.config, args.seeds, args.out)
        return
    run_suite(args.suite, args.config, args.seeds, args.out, args.only, force=args.force)


if __name__ == "__main__":
    main()
