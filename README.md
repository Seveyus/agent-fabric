# Delivery World Model

> A JEPA-inspired world model for software delivery — trained on millions of public GitHub project timelines, predicts delivery failures before humans can see them.

---

## What this is

Most AI coding agents (Cursor, Claude Code, Copilot) are contextually blind. They see the code, the tickets, the docs — but not the **causal physics** that produced them. They don't know that the weird retry logic in `payment_service.py` exists because of a silent timeout failure in prod (March 2024), and that three attempts to "simplify" it all caused regressions.

This system builds and maintains a **temporal causal knowledge graph** of software projects, and exposes it via a pre-flight API that agents query before acting.

**Core question it answers:** "Before you modify X — here is what will probably break, why, and how confident we are."

---

## Architecture

```
GH Archive (public GitHub data)
    ↓
core/dataset/          Graph snapshots — 7-day windows of project state
    ↓
models/                JEPA world model
    encoder.py         HGT (Heterogeneous Graph Transformer) → 128-dim repr
    predictor.py       Predicts future repr from current repr + events
    world_model.py     Full training loop with EMA target encoder
    ↓
training/              Training + evaluation scripts
    train.py           python -m training.train --pairs data/pairs.pkl
    evaluate.py        AUC, prediction MSE, representation health
    ↓
core/services/
    knowledge_graph_service.py   Temporal causal graph (append-only edges)
    knowledge_search_service.py  Hybrid search: Qdrant vectors + graph traversal
    ↓
apps/api/routers/
    preflight.py       POST /preflight — the core product endpoint
```

---

## The JEPA training loop

```
graph_t  ──[online encoder]──→  s_x (128-dim)
                                    │
                                [predictor + events]
                                    │
                                  ŝ_y (predicted)

graph_t_k ──[target encoder (EMA)]──→  s_y (128-dim, stop gradient)

Loss = ||ŝ_y - s_y||²   ← prediction in latent space, not raw space
```

The target encoder is **never trained by backprop** — it is updated via EMA (τ=0.99) after each optimizer step. This prevents representational collapse, the key insight from I-JEPA (Assran et al., 2023).

---

## Pre-flight API

```bash
curl -X POST http://localhost:8000/preflight \
  -H "Content-Type: application/json" \
  -d '{
    "action": "modify",
    "target_ref": "auth_middleware.py",
    "project_ref": "myorg/backend"
  }'
```

Response:
```json
{
  "risk_level": "high",
  "risk_score": 0.78,
  "consequences": [
    {
      "entity_ref": "payment_service.py",
      "probability": 0.87,
      "relation_type": "constrained_by",
      "reason": "payment_service.py is constrained by this — changing it may violate the constraint"
    }
  ],
  "decision_history": [
    {
      "decision": "Retry logic hardened to 3 attempts",
      "rationale": "Silent timeout failures from payment processor (incident 2024-03)",
      "causal_type": "triggered_by",
      "confidence": 0.9
    }
  ],
  "recommendation": "Require human review — high-probability impact on payment_service.py"
}
```

---

## Getting started

```bash
# Install
pip install -e .
# For GPU training:
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install -e ".[train]"

# Run the API
uvicorn apps.api.main:app --reload

# Collect data (start small — 1 week of GH Archive)
python -c "
from core.dataset.gh_archive import GHArchiveDownloader
dl = GHArchiveDownloader('data/raw')
dl.download_range('2024-01-01', '2024-01-07', hours=[9,12,15,18])
"

# Build training pairs
python -c "
from core.dataset.gh_archive import GHArchiveDownloader, EventNormalizer
from core.dataset.graph_builder import RepoTimeline
from training.dataset import build_pairs_from_timelines
import pickle, collections

dl = GHArchiveDownloader('data/raw')
normalizer = EventNormalizer()
timelines = collections.defaultdict(lambda: RepoTimeline(''))

for raw_ev in dl.iter_all_events():
    ev = normalizer.normalize(raw_ev)
    if ev:
        if timelines[ev['repo']].repo == '':
            from core.dataset.graph_builder import RepoTimeline as RT
            timelines[ev['repo']] = RT(ev['repo'])
        timelines[ev['repo']].add_event(ev)

build_pairs_from_timelines(list(timelines.values()), 'data/pairs_14d.pkl')
"

# Train (CPU, ~4h for 50K pairs)
python -m training.train --pairs data/pairs_14d.pkl --epochs 50 --device cpu

# Or GPU (your 20GB VM)
python -m training.train --pairs data/pairs_14d.pkl --epochs 100 --device cuda

# Evaluate
python -m training.evaluate --checkpoint checkpoints/best.pt --pairs data/pairs_14d.pkl
```

---

## Validation signal

Before building more, run this first:

```python
# Does the signal exist in public data?
# Target: AUC > 0.65 on regression prediction
python -m training.evaluate --checkpoint checkpoints/best.pt --pairs data/pairs_14d.pkl
# If AUC > 0.65 → signal is real → continue
# If AUC ≈ 0.50 → no signal → revisit features
```

---

## Papers

- **I-JEPA** — Assran et al., 2023 — https://arxiv.org/abs/2301.08243
- **A Path Towards Autonomous Machine Intelligence** — LeCun, 2022 — https://openreview.net/pdf?id=BZ5a1r-kVsf
- **Heterogeneous Graph Transformer (HGT)** — Hu et al., 2020 — https://arxiv.org/abs/2003.01332
- **Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs** — Trivedi et al., 2017 — https://arxiv.org/abs/1705.05742

---

## Compute requirements

| Task | Hardware | Time |
|---|---|---|
| Data collection (50K repos) | CPU + internet | 2–3 weeks |
| Feature engineering | CPU | Hours |
| Training (50K pairs, 100 epochs) | 4 GB VRAM | ~8h |
| Training (500K pairs, 100 epochs) | 20 GB VRAM | ~24h |
| Inference (pre-flight API) | CPU | <10ms |

No massive compute required. The model is ~700K parameters.
