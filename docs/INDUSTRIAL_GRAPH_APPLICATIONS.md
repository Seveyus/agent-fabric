# Industrial graph applications: energy systems and EV fleets

This document separates three things clearly:

1. **Implemented**: a small synthetic EV-fleet example that runs through the
   same pipeline as the software-delivery task (§1–§3).
2. **Measured**: what that example shows about graph models vs. simpler
   methods (§3).
3. **Proposed**: how the architecture could extend to real energy and fleet
   operations, and what would have to be true first (§4–§6). Nothing in §4–§6
   is implemented or validated.

The EV data is produced by a simulator in `core/dataset/ev_fleet.py`. It is
not derived from any real fleet, operator, depot or grid.

## 1. Implemented example: departure-energy risk in depot charging

**Operational question.** At 15:00, with tonight's charging plan known (which
vehicles return, when, with what state of charge, which charger each is
planned on, when each must leave and with how much energy), which vehicles
will leave short of their departure energy target?

This is a real class of problem for electric bus and delivery fleets: depots
are usually connected to the grid with less capacity than the sum of their
charger ratings, and several depots can sit behind the same feeder. Whether a
given vehicle gets its energy depends on everyone else plugged in that night.

**Simulator (synthetic).** 6 feeders × 2–3 depots; 6–14 chargers per depot
(11/22 kW AC, 50/150 kW DC) and 1.0–1.6× as many vehicles; depot caps at
35–60 % of installed charger power, feeder caps at 60–85 % of the sum of their
depots' caps; FIFO queues per charger; 15-minute power allocation scaled down
proportionally when a depot or feeder exceeds its cap; latent per-charger
fault probabilities (a few chargers are bad). 60 nights; nights 0–39 train,
40–49 validation, 50–59 test. About 6 % of sessions miss their target.

**Graph.**

```
                 feeder ──────────────── feeder_night
                   │ fed_by                  │ part_of / in_feeder
   charger ── located_at ── depot ── of ── depot_night
      │                       │ assigned_to       │ in
      └──── uses ──── session ┘── of_vehicle ── vehicle
                         (seed)
```

`depot_night` and `feeder_night` are per-night context nodes: the sessions
competing tonight are two hops away from each other, sessions from other
nights are not mixed in, and every edge carries the planning time of its
night so temporal sampling never reaches tomorrow's plan.

**Features.** Session-local: arrival SoC, required SoC, needed energy, dwell
time, arrival/departure hour, vehicle capacity / AC and DC limits / state of
health, planned charger power and type. Optional engineered aggregates
(`features.use_history = true`): planned depot-night and feeder-night load
relative to the cap, number of sessions queued on the same charger, queue
position, and the charger's fault count over the previous 30 nights
(point-in-time).

## 2. How to run it

```bash
python -m training.node_task --config configs/ev_fleet.toml                               # HGT, local features
python -m training.node_task --config configs/ev_fleet.toml --set features.use_history=true
python -m training.suite --suite main --config configs/ev_fleet.toml --seeds 0 1 2        # full comparison
```

## 3. What the example shows

Source: [`experiments/results/ev_fleet__main.md`](../experiments/results/ev_fleet__main.md).
Synthetic data; 7,384 / 1,843 / 1,829 train / validation / test sessions,
5.6 % positives in test; mean ± std over 3 seeds; CPU.

| Model | Test ROC-AUC, local features | + engineered aggregates | Test AP, + aggregates |
|---|---|---|---|
| Logistic regression | 0.788 ± 0.000 | 0.899 ± 0.000 | 0.562 |
| LightGBM | 0.835 ± 0.014 | **0.931 ± 0.003** | **0.746** |
| MLP | 0.791 ± 0.008 | 0.917 ± 0.004 | 0.684 |
| HeteroSAGE | **0.869 ± 0.008** | 0.926 ± 0.003 | 0.715 |
| HGT | 0.863 ± 0.016 | 0.919 ± 0.005 | 0.725 |

Paired bootstrap on the same test sessions:

| Comparison | Δ ROC-AUC | 95 % CI |
|---|---|---|
| HeteroSAGE vs MLP, local features | +0.078 | [+0.049, +0.113] |
| LightGBM + aggregates vs LightGBM local | +0.096 | [+0.061, +0.135] |
| HeteroSAGE + aggregates vs LightGBM + aggregates | −0.004 | [−0.021, +0.012] |
| HGT + aggregates vs LightGBM + aggregates | −0.012 | [−0.029, +0.006] |

What it shows, and what it does not:

* **The graph models learn the contention effect from structure.** With only
  per-session information, HeteroSAGE beats the same-loop MLP by 0.08 AUC and
  beats LightGBM on the same features (0.869 vs 0.835): the shared depot,
  feeder and charger context that decides whether a vehicle gets its energy
  is two hops away and the GNN uses it.
* **An expert who knows the mechanism does as well or better with a GBDT.**
  Five engineered aggregates (depot/feeder load ratios, charger queue,
  recent faults) lift LightGBM to 0.931; the GNNs with the same aggregates are
  not better (paired differences include zero or are negative).
* The same pattern as on real software-delivery data (`docs/GRAPH_LEARNING.md`
  §10.1): graph learning recovers relational signal that hand-built features
  can also capture; its value is in not having to build and maintain those
  features, not in beating them.
* It does not show anything about real fleets: the simulator encodes one
  simple allocation policy, and the contention mechanism was designed into it.

## 4. Proposed: where graph learning may help in energy and fleet operations

Not implemented. Each item lists the relational structure and the reason a
per-asset model may fall short.

| Problem | Graph | Why relational |
|---|---|---|
| Depot charging risk with real telemetry (extension of §1) | sessions, vehicles, chargers, depots, feeders, tariffs/constraints | contention for shared power and chargers; fault propagation through shared hardware |
| Grid-constrained charging schedules (forecasting input to an optimiser) | depots ↔ feeders ↔ substations, with time-stamped load | upstream constraints couple sites that do not share equipment |
| Fleet assignment risk (which vehicle on which route) | vehicles ↔ routes ↔ depots ↔ drivers, with history | the same vehicle/route/driver combinations recur; outcomes are history-dependent |
| Industrial asset failure propagation | assets ↔ assets (process flow), maintenance events, sensors | failures propagate along physical dependencies |
| Change risk in operational software (this repository's original domain) | code owners, components, deployments, incidents | identical structure: actors, artefacts, outcomes over time |

The pieces that transfer directly from this repository: the temporal
heterogeneous graph with timestamped edges, temporal neighbour sampling as the
leakage guard, point-in-time features, embargoed splits, the controlled
non-graph baseline in the same training loop, and the sampling backend
abstraction (PyG on CPU, cuGraph-PyG on GPU for large graphs).

## 5. Where simpler methods are likely sufficient

* When the relational effect can be summarised by a few well-understood
  aggregates (depot load ratio, queue length, recent fault count), a
  gradient-boosted model on those aggregates is cheaper, easier to explain
  and — in the synthetic example — at least as accurate (§3). A domain expert
  who already knows the mechanism should build those features first.
* When the decision is an optimisation (schedule charging under caps), the
  physics and constraints belong in the optimiser; a learned model is at most
  a forecasting input (arrival times, energy needs, fault risk).
* When data is small (one depot, a few months), graph models have little to
  learn from and overfit easily.

Graph learning is more plausible when: the topology changes often (new
depots, re-wired feeders, new routes) so hand-built aggregates go stale;
effects travel more than one hop; or there are many interacting entity types
and nobody knows which aggregates matter. Those are hypotheses to test with
real data, not results.

## 6. What would have to be true before production use

* Real data with timestamps for every interaction (charger telemetry, plans,
  departures, faults), and a definition of the decision time — the leakage
  rules of `docs/GRAPH_LEARNING.md` §5 apply unchanged.
* A baseline that the operations team already trusts (rules or gradient
  boosting on engineered aggregates), evaluated on the same temporal splits.
* Calibration of predicted probabilities (not done in this repository) and an
  explanation per alert (which shared resource drives the risk) — operators
  act on reasons, not scores.
* Monitoring of drift: depot layouts and fleet composition change.
* For large networks, the sampling cost study of `docs/GRAPH_LEARNING.md` §11
  repeated on GPU before choosing infrastructure.
