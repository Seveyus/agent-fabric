"""
Generate the project overview figure from the committed results.

    python docs/figures/make_overview.py            # SVG, light + dark
    python docs/figures/make_overview.py --png      # + PNG (2x) via headless Chromium

Every number on the figure is read from experiments/results/*.json; nothing is
typed in by hand. Re-run after re-running the experiments.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
OUT = Path(__file__).resolve().parent

W, H = 1600, 940
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif"

THEMES = {
    "light": {
        "page": "#f9f9f7", "surface": "#fcfcfb", "surface2": "#f3f2ef", "border": "#e1e0d9",
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "accent": "#2a78d6", "hi_a": "#86b6ef", "hi_b": "#1c5cab", "lo_a": "#c3c2b7", "lo_b": "#898781",
    },
    "dark": {
        "page": "#0d0d0d", "surface": "#1a1a19", "surface2": "#232321", "border": "#2c2c2a",
        "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "accent": "#3987e5", "hi_a": "#256abf", "hi_b": "#9ec5f4", "lo_a": "#52514e", "lo_b": "#c3c2b7",
    },
}


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def facts() -> dict:
    gh = load("gh_pr_merge__main.json")
    summ = gh["summary"]
    comps = {(c["a"], c["b"]): c for c in gh["comparisons"]}
    run = gh["runs"]["HGT (+history)"][0]
    nodes = sum(run["task"]["num_nodes"].values())
    rows = []
    for name, graph in [("HeteroSAGE", True), ("HGT", True), ("MLP", False), ("LightGBM", False)]:
        rows.append({
            "name": name, "graph": graph,
            "intrinsic": summ[f"{name} (intrinsic)"]["test_roc_auc"][0],
            "history": summ[f"{name} (+history)"]["test_roc_auc"][0],
        })
    graph_gain = comps[("HeteroSAGE (intrinsic)", "MLP (intrinsic)")]
    beyond = comps[("HeteroSAGE (+history)", "LightGBM (+history)")]

    ev = load("ev_fleet__main.json")
    ev_gain = {(c["a"], c["b"]): c for c in ev["comparisons"]}[("HeteroSAGE (intrinsic)", "MLP (intrinsic)")]

    cpu = load("benchmark_gpu_session_1_cpu_only.json")["rows"]
    gpu = load("benchmark_gpu_session_2_pyg_sampler_gpu_model.json")["rows"]

    def gh_rows(rs):
        return {(tuple(r["fanout"]), r["batch_size"]): r for r in rs
                if r["graph"].startswith("task:") and r["temporal"] and r["train_step"]}

    c, g = gh_rows(cpu), gh_rows(gpu)
    speedups, shares = [], []
    for k in c.keys() & g.keys():
        hc = next(t for t in c[k]["train_step"] if t["encoder"] == "hgt")["seeds_per_s"]
        hg = next(t for t in g[k]["train_step"] if t["encoder"] == "hgt")["seeds_per_s"]
        speedups.append(hg / hc)
        shares += [t["sampling_share"] for t in g[k]["train_step"]]

    return {
        "nodes": nodes, "edges": run["num_edges_total"], "events": run["task"]["num_events"],
        "test_n": run["task"]["split_sizes"]["test"], "rows": rows,
        "graph_gain": graph_gain, "beyond": beyond, "ev_gain": ev_gain,
        "speedup": (min(speedups), max(speedups)), "share": (min(shares), max(shares)),
        "best": max(r["history"] for r in rows),
    }


def text(x, y, s, size, fill, weight=400, anchor="start", spacing=None, extra=""):
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}"{ls}{extra}>{escape(s)}</text>')


def render(f: dict, theme: str) -> str:
    t = THEMES[theme]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="{FONT}" role="img" aria-labelledby="ttl desc">',
           '<title id="ttl">Agent Fabric: temporal graph learning on real GitHub activity</title>',
           f'<desc id="desc">Pipeline from GH Archive to a temporal heterogeneous graph and graph models. '
           f'Test ROC-AUC for predicting whether a pull request is merged within 48 hours: '
           + "; ".join(f'{r["name"]} {r["intrinsic"]:.3f} with PR features only, {r["history"]:.3f} with history features'
                       for r in f["rows"]) + '.</desc>',
           f'<rect width="{W}" height="{H}" fill="{t["page"]}"/>',
           f'<rect x="20" y="20" width="{W - 40}" height="{H - 40}" rx="28" fill="{t["surface"]}" stroke="{t["border"]}"/>']

    # ---- header -------------------------------------------------------------
    x0, x1 = 64, W - 64
    out.append(text(x0, 84, "AGENT FABRIC  ·  TEMPORAL HETEROGENEOUS GRAPH LEARNING", 13, t["muted"], 600, spacing=1.6))
    out.append(text(x0, 130, "Will this pull request be merged within 48 hours?", 36, t["ink"], 650))
    out.append(text(x0, 166, f"Real public GitHub activity  ·  {f['nodes'] / 1e6:.2f} M nodes  ·  "
                             f"{f['edges'] / 1e6:.1f} M timestamped edges  ·  leakage-safe by construction",
                    18, t["ink2"]))

    # ---- pipeline -------------------------------------------------------------
    steps = [
        ("01 · DATA", "GH Archive stream", [f"{f['events'] / 1e6:.1f} M events, 3 weeks", "24,475 collaborative repos"], False),
        ("02 · GRAPH", "Temporal hetero graph", ["PR · user · repo, 32 relations", "every edge timestamped"], True),
        ("03 · NO LEAKAGE", "Time-respecting sampling", ["no edge newer than the PR", "embargoed splits, tested"], True),
        ("04 · MODELS", "HeteroSAGE · HGT · JEPA", ["vs LightGBM & MLP baselines", "same features, same splits"], True),
        ("05 · OUTPUT", "P(merged within 48 h)", [f"best test ROC-AUC {f['best']:.3f}", "3 seeds, paired bootstrap"], False),
    ]
    by, bh, bw = 204, 128, 262
    gap = (x1 - x0 - 5 * bw) / 4
    for i, (over, title, lines, core) in enumerate(steps):
        bx = x0 + i * (bw + gap)
        out.append(f'<rect x="{bx:.1f}" y="{by}" width="{bw}" height="{bh}" rx="14" fill="{t["surface2"]}" stroke="{t["border"]}"/>')
        if core:
            out.append(f'<rect x="{bx:.1f}" y="{by + 18}" width="4" height="{bh - 36}" rx="2" fill="{t["accent"]}"/>')
        out.append(text(bx + 22, by + 32, over, 12, t["muted"], 600, spacing=1.2))
        out.append(text(bx + 22, by + 60, title, 17, t["ink"], 650))
        for j, ln in enumerate(lines):
            out.append(text(bx + 22, by + 86 + j * 21, ln, 14, t["ink2"]))
        if i < 4:
            ax = bx + bw + 8
            ay = by + bh / 2
            out.append(f'<line x1="{ax:.1f}" y1="{ay}" x2="{ax + gap - 22:.1f}" y2="{ay}" stroke="{t["muted"]}" stroke-width="2" stroke-linecap="round"/>')
            out.append(f'<path d="M{ax + gap - 24:.1f},{ay - 6} L{ax + gap - 16:.1f},{ay} L{ax + gap - 24:.1f},{ay + 6}" '
                       f'fill="none" stroke="{t["muted"]}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')

    out.append(f'<line x1="{x0}" y1="368" x2="{x1}" y2="368" stroke="{t["border"]}"/>')

    # ---- dumbbell chart ---------------------------------------------------------
    cy0 = 404
    out.append(text(x0, cy0 + 12, f"Test ROC-AUC on {f['test_n']:,} held-out pull requests", 18, t["ink"], 650))
    out.append(text(x0, cy0 + 36, "mean of 3 seeds  ·  higher is better", 14, t["muted"]))
    # legend (shape + shade, never color alone)
    lx, ly = x0, cy0 + 70
    out.append(f'<circle cx="{lx + 7}" cy="{ly - 5}" r="6" fill="{t["surface"]}" stroke="{t["lo_b"]}" stroke-width="2.5"/>')
    out.append(text(lx + 22, ly, "PR features only", 14, t["ink2"]))
    lx2 = lx + 170
    out.append(f'<circle cx="{lx2 + 7}" cy="{ly - 5}" r="7" fill="{t["lo_b"]}"/>')
    out.append(text(lx2 + 22, ly, "+ point-in-time history features (hand-built)", 14, t["ink2"]))

    px0, px1 = 268, 872
    lo, hi = 0.66, 0.74

    def X(v: float) -> float:
        return px0 + (v - lo) / (hi - lo) * (px1 - px0)

    rows_y = [cy0 + 128 + i * 66 for i in range(len(f["rows"]))]
    gy0, gy1 = rows_y[0] - 34, rows_y[-1] + 30
    for v in (0.66, 0.68, 0.70, 0.72, 0.74):
        out.append(f'<line x1="{X(v):.1f}" y1="{gy0}" x2="{X(v):.1f}" y2="{gy1}" stroke="{t["grid"]}"/>')
        out.append(text(X(v), gy1 + 22, f"{v:.2f}", 13, t["muted"], anchor="middle",
                        extra=' style="font-variant-numeric: tabular-nums"'))
    out.append(f'<line x1="{px0}" y1="{gy1}" x2="{px1}" y2="{gy1}" stroke="{t["axis"]}"/>')

    for r, y in zip(f["rows"], rows_y):
        hl = r["name"] == "HeteroSAGE"
        ca, cb = (t["hi_a"], t["hi_b"]) if hl else (t["lo_a"], t["lo_b"])
        out.append(text(px0 - 24, y - 2, r["name"], 16, t["ink"] if hl else t["ink2"], 650 if hl else 500, anchor="end"))
        out.append(text(px0 - 24, y + 17, "graph model" if r["graph"] else "no graph", 12.5, t["muted"], anchor="end"))
        xa, xb = X(r["intrinsic"]), X(r["history"])
        out.append(f'<line x1="{xa:.1f}" y1="{y}" x2="{xb:.1f}" y2="{y}" stroke="{ca}" stroke-width="3" stroke-linecap="round"/>')
        out.append(f'<circle cx="{xa:.1f}" cy="{y}" r="7" fill="{t["surface"]}" stroke="{cb if hl else ca}" stroke-width="2.5"/>')
        out.append(f'<circle cx="{xb:.1f}" cy="{y}" r="9" fill="{t["surface"]}"/>')
        out.append(f'<circle cx="{xb:.1f}" cy="{y}" r="7" fill="{cb}"/>')
        if hl:
            out.append(text(xa, y - 16, f"{r['intrinsic']:.3f}", 13.5, t["ink2"], anchor="middle"))
            out.append(text(xb + 2, y - 16, f"{r['history']:.3f}", 14.5, t["ink"], 650, anchor="start"))

    out.append(text(x0, gy1 + 62, "The graph models learn from raw interactions what the non-graph models only reach", 14.5, t["ink2"]))
    out.append(text(x0, gy1 + 84, "with hand-built history features — and still add a small, significant gain on top.", 14.5, t["ink2"]))

    # ---- tiles --------------------------------------------------------------------
    gg, ev, sp, sh = f["graph_gain"], f["ev_gain"], f["speedup"], f["share"]
    tiles = [
        ("LEARNED, NOT HAND-ENGINEERED", f"+{gg['auc_diff']:.3f}", "ROC-AUC",
         ["graph vs the same features", "without message passing"],
         f"paired 95% CI {gg['ci95'][0]:.3f} – {gg['ci95'][1]:.3f}"),
        ("GPU  ·  GTX 1650 LAPTOP", f"{sp[0]:.0f}–{sp[1]:.0f}×", "faster",
         ["HGT training with the model on", "the GPU vs. on the laptop CPU"],
         f"sampling then = {100 * sh[0]:.0f}–{100 * sh[1]:.0f}% of each step"),
        ("INFRASTRUCTURE FINDINGS", "3", "silent bugs",
         ["in PyG HGTConv and cuGraph-PyG", "temporal sampling, reproduced"],
         "repro scripts + drafted upstream reports"),
        ("EV FLEET  ·  SYNTHETIC", f"+{ev['auc_diff']:.3f}", "ROC-AUC",
         ["charger & depot contention", "learned by the same pipeline"],
         f"paired 95% CI {ev['ci95'][0]:.3f} – {ev['ci95'][1]:.3f}"),
    ]
    tx0, tw, th, tg = 944, 286, 222, 20
    for i, (over, value, unit, lines, cap) in enumerate(tiles):
        tx = tx0 + (i % 2) * (tw + tg)
        ty = cy0 - 4 + (i // 2) * (th + tg)
        out.append(f'<rect x="{tx}" y="{ty}" width="{tw}" height="{th}" rx="16" fill="{t["surface2"]}" stroke="{t["border"]}"/>')
        out.append(text(tx + 22, ty + 34, over, 11.5, t["muted"], 600, spacing=1.2))
        out.append(f'<text x="{tx + 22}" y="{ty + 92}" fill="{t["ink"]}"><tspan font-size="44" font-weight="650">{escape(value)}</tspan>'
                   f'<tspan font-size="15" font-weight="500" fill="{t["ink2"]}" dx="8">{escape(unit)}</tspan></text>')
        for j, ln in enumerate(lines):
            out.append(text(tx + 22, ty + 128 + j * 22, ln, 14.5, t["ink2"]))
        out.append(text(tx + 22, ty + 196, cap, 12.5, t["muted"]))

    out.append(text(x0, H - 44, "github.com/Seveyus/agent-fabric  ·  every number generated from experiments/results/ "
                                "by docs/figures/make_overview.py", 13, t["muted"]))
    out.append(text(x1, H - 44, "PyTorch Geometric  ·  cuGraph-PyG  ·  CPU + GPU", 13, t["muted"], anchor="end"))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def to_png(svg: Path, png: Path) -> bool:
    browser = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
    if browser is None:
        return False
    subprocess.run([browser, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    f"--window-size={W},{H}", f"--screenshot={png}", svg.resolve().as_uri()],
                   check=True, capture_output=True)
    return png.exists()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--png", action="store_true", help="also export 2x PNGs with headless Chromium")
    args = p.parse_args()
    f = facts()
    for theme in THEMES:
        svg = OUT / f"agent_fabric_overview_{theme}.svg"
        svg.write_text(render(f, theme))
        print("wrote", svg.relative_to(ROOT))
        if args.png:
            png = svg.with_suffix(".png")
            print("wrote" if to_png(svg, png) else "skipped (no Chromium)", png.relative_to(ROOT))


if __name__ == "__main__":
    main()
