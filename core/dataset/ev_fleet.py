"""
EV fleet depot charging as a temporal heterogeneous graph — SYNTHETIC example.

SYNTHETIC DATA: produced by a simple depot charging simulator in this file.
It is not derived from any real fleet, operator or grid. It exists to show
that the graph pipeline used for software delivery (temporal heterogeneous
graph, temporal neighbour sampling, MLP / SAGE / HGT, tabular baselines)
applies unchanged to an industrial operations problem.

Task
----
At planning time for night n (15:00), for every planned charging session:
will the vehicle leave with less energy than required for its next shift?
(label "missed", resolved at departure).

Why it is relational
--------------------
Whether a session gets its energy depends on the *other* sessions planned the
same night: they share the charger (FIFO queue), the depot grid connection
(power cap) and the feeder that supplies several depots (second cap).
A per-vehicle model sees only its own needs; the contention is a property of
the night's graph.

Simulator (15-minute steps, per night)
--------------------------------------
* feeders supply 2-3 depots; depot caps and feeder caps are below the sum of
  charger ratings (oversubscribed, as is common to save grid-connection cost);
* each depot has 6-14 chargers (11/22 kW AC, 50/150 kW DC) and 1.0-1.6x as
  many vehicles; vehicles queue FIFO on their planned charger;
* requested power = min(charger rating, vehicle AC/DC limit, remaining/dt);
  if a depot exceeds its cap all its sessions are scaled down
  proportionally, then likewise for the feeder;
* each charger has a latent fault probability (a few chargers are bad); a
  faulted charger delivers nothing that night;
* missed = remaining energy > 0.5 kWh at departure.

Graph (all edges carry ``time`` = night start, static ones 0)
-------------------------------------------------------------
    session -of_vehicle-> vehicle        session -uses-> charger
    session -in-> depot_night            session -in_feeder-> feeder_night
    depot_night -of-> depot              feeder_night -of-> feeder
    depot_night -part_of-> feeder_night  charger -located_at-> depot
    vehicle -assigned_to-> depot         depot -fed_by-> feeder
(+ rev_* for each). ``depot_night`` / ``feeder_night`` are per-night context
nodes, so the sessions competing tonight are two hops away and sessions of
other nights are not mixed in.

Engineered aggregates (the "history" features of this task, droppable with
features.use_history = false): planned depot-night and feeder-night load
relative to cap, sessions queued on the same charger, and the charger's fault
count over the previous 30 nights (point-in-time).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from core.dataset.temporal_graph import SeedSet, TemporalTask

DAY = 86400
T0 = 1_756_684_800  # 2025-09-01, used only to give nights a timestamp

LOCAL_FEATURES = [
    "arrival_soc", "required_soc", "log_needed_kwh", "dwell_hours", "arrival_hour", "departure_hour",
    "weekend", "vehicle_capacity_kwh", "vehicle_ac_kw", "vehicle_dc_kw", "vehicle_soh",
    "charger_kw", "charger_is_dc", "needed_kwh_over_charger_kwh_dwell",
]
AGGREGATE_FEATURES = [
    "depot_night_load_ratio", "feeder_night_load_ratio", "sessions_on_same_charger",
    "queue_position", "charger_faults_prev_30_nights",
]
SESSION_FEATURES = LOCAL_FEATURES + AGGREGATE_FEATURES

FORWARD_RELATIONS = [
    ("session", "of_vehicle", "vehicle"),
    ("session", "uses", "charger"),
    ("session", "in", "depot_night"),
    ("session", "in_feeder", "feeder_night"),
    ("depot_night", "of", "depot"),
    ("feeder_night", "of", "feeder"),
    ("depot_night", "part_of", "feeder_night"),
    ("charger", "located_at", "depot"),
    ("vehicle", "assigned_to", "depot"),
    ("depot", "fed_by", "feeder"),
]
NODE_TYPES = ["session", "vehicle", "charger", "depot", "feeder", "depot_night", "feeder_night"]
EDGE_TYPES = FORWARD_RELATIONS + [(d, f"rev_{r}", s) for s, r, d in FORWARD_RELATIONS]
METADATA = (NODE_TYPES, EDGE_TYPES)


@dataclass(frozen=True)
class EVFleetConfig:
    n_feeders: int = 6
    nights: int = 60
    train_nights: int = 40        # nights [0, train) train, [train, train+val) val, rest test
    val_nights: int = 10
    history_nights: int = 30      # window of the charger fault count
    step_minutes: int = 15
    seed: int = 0


def simulate(cfg: EVFleetConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    # ---- static infrastructure ---------------------------------------------
    depot_feeder, depot_cap, depot_nchg = [], [], []
    chg_depot, chg_kw, chg_fault_p = [], [], []
    for f in range(cfg.n_feeders):
        for _ in range(rng.integers(2, 4)):
            d = len(depot_feeder)
            n = int(rng.integers(6, 15))
            kws = rng.choice([11.0, 22.0, 50.0, 150.0], size=n, p=[0.3, 0.35, 0.25, 0.1])
            depot_feeder.append(f)
            depot_nchg.append(n)
            depot_cap.append(float(kws.sum() * rng.uniform(0.35, 0.6)))
            for kw in kws:
                chg_depot.append(d)
                chg_kw.append(kw)
                chg_fault_p.append(rng.uniform(0.15, 0.35) if rng.random() < 0.08 else rng.beta(1, 60))
    depot_feeder, depot_cap = np.array(depot_feeder), np.array(depot_cap)
    chg_depot, chg_kw, chg_fault_p = np.array(chg_depot), np.array(chg_kw), np.array(chg_fault_p)
    feeder_cap = np.array([depot_cap[depot_feeder == f].sum() * rng.uniform(0.6, 0.85)
                           for f in range(cfg.n_feeders)])

    veh_depot, veh_cap, veh_ac, veh_dc, veh_soh, veh_req = [], [], [], [], [], []
    for d, n in enumerate(depot_nchg):
        for _ in range(int(round(n * rng.uniform(1.0, 1.6)))):
            veh_depot.append(d)
            veh_cap.append(rng.choice([60.0, 80.0, 100.0, 250.0], p=[0.3, 0.35, 0.25, 0.1]))
            veh_ac.append(rng.choice([7.4, 11.0, 22.0], p=[0.3, 0.5, 0.2]))
            veh_dc.append(rng.choice([50.0, 100.0, 150.0], p=[0.3, 0.4, 0.3]))
            veh_soh.append(rng.uniform(0.8, 1.0))
            veh_req.append(rng.uniform(0.75, 0.95))
    veh_depot = np.array(veh_depot)
    veh_cap, veh_ac, veh_dc = np.array(veh_cap), np.array(veh_ac), np.array(veh_dc)
    veh_soh, veh_req = np.array(veh_soh), np.array(veh_req)
    depot_chargers = [np.nonzero(chg_depot == d)[0] for d in range(len(depot_cap))]

    # ---- nights ------------------------------------------------------------
    dt_h = cfg.step_minutes / 60.0
    sessions: dict[str, list] = {k: [] for k in (
        "night", "vehicle", "charger", "arrival_h", "departure_h", "arrival_soc", "required_soc",
        "needed_kwh", "queue_position", "missed")}
    fault_log = np.zeros((cfg.nights, len(chg_kw)), dtype=bool)
    for night in range(cfg.nights):
        faulted = rng.random(len(chg_kw)) < chg_fault_p
        fault_log[night] = faulted
        present = np.nonzero(rng.random(len(veh_depot)) < 0.9)[0]
        arr = np.clip(rng.normal(18.5, 1.5, len(present)), 15.0, 23.5)
        dep = np.clip(rng.normal(30.0, 1.0, len(present)), 28.0, 33.0)   # next morning, hours since 00:00
        soc = rng.beta(2, 3, len(present))
        need = np.maximum(0.0, veh_req[present] - soc) * veh_cap[present] * veh_soh[present]

        # planned charger: FIFO by arrival on the depot charger with the shortest queue
        charger = np.empty(len(present), dtype=int)
        qpos = np.zeros(len(present), dtype=int)
        for d in range(len(depot_cap)):
            idx = np.nonzero(veh_depot[present] == d)[0]
            idx = idx[np.argsort(arr[idx])]
            queue_len = {int(c): 0 for c in depot_chargers[d]}
            for i in idx:
                c = min(queue_len, key=lambda k: (queue_len[k], -chg_kw[k]))
                charger[i] = c
                qpos[i] = queue_len[c]
                queue_len[c] += 1

        remaining = need.copy()
        done = remaining <= 0
        t = 15.0
        while t < 33.0:
            # the head of each charger's queue that has arrived and not departed charges
            active = np.zeros(len(present), dtype=bool)
            for c in np.unique(charger):
                q = np.nonzero((charger == c) & ~done & (arr <= t) & (dep > t))[0]
                if len(q):
                    active[q[np.argmin(qpos[q])]] = True
            rate = np.where(chg_kw[charger] >= 50.0,
                            np.minimum(chg_kw[charger], veh_dc[present]),
                            np.minimum(chg_kw[charger], veh_ac[present]))
            p = np.where(active & ~faulted[charger], np.minimum(rate, remaining / dt_h), 0.0)
            dep_of = chg_depot[charger]
            for d in np.unique(dep_of[p > 0]):
                m = dep_of == d
                s = p[m].sum()
                if s > depot_cap[d]:
                    p[m] *= depot_cap[d] / s
            fdr = depot_feeder[dep_of]
            for f in np.unique(fdr[p > 0]):
                m = fdr == f
                s = p[m].sum()
                if s > feeder_cap[f]:
                    p[m] *= feeder_cap[f] / s
            remaining = np.maximum(0.0, remaining - p * dt_h)
            done = done | (remaining <= 1e-6) | (dep <= t)
            t += dt_h
        for i, v in enumerate(present):
            sessions["night"].append(night)
            sessions["vehicle"].append(int(v))
            sessions["charger"].append(int(charger[i]))
            sessions["arrival_h"].append(float(arr[i]))
            sessions["departure_h"].append(float(dep[i]))
            sessions["arrival_soc"].append(float(soc[i]))
            sessions["required_soc"].append(float(veh_req[v]))
            sessions["needed_kwh"].append(float(need[i]))
            sessions["queue_position"].append(int(qpos[i]))
            sessions["missed"].append(float(remaining[i] > 0.5))
    return {
        "sessions": {k: np.array(v) for k, v in sessions.items()},
        "depot_feeder": depot_feeder, "depot_cap": depot_cap, "feeder_cap": feeder_cap,
        "chg_depot": chg_depot, "chg_kw": chg_kw, "fault_log": fault_log,
        "veh_depot": veh_depot, "veh_cap": veh_cap, "veh_ac": veh_ac, "veh_dc": veh_dc, "veh_soh": veh_soh,
    }


def build_ev_task(cfg: EVFleetConfig) -> TemporalTask:
    from torch_geometric.data import HeteroData

    sim = simulate(cfg)
    s = sim["sessions"]
    n_s, n_dep, n_fdr = len(s["night"]), len(sim["depot_cap"]), len(sim["feeder_cap"])
    nights = cfg.nights
    night_t = T0 + s["night"] * DAY + 15 * 3600            # planning time: 15:00
    dep_of = sim["chg_depot"][s["charger"]]
    fdr_of = sim["depot_feeder"][dep_of]
    dn = s["night"] * n_dep + dep_of                       # depot_night id
    fn = s["night"] * n_fdr + fdr_of                       # feeder_night id
    dwell = s["departure_h"] - s["arrival_h"]

    # engineered aggregates (all from tonight's plan or strictly earlier nights)
    dn_need = np.bincount(dn, weights=s["needed_kwh"], minlength=nights * n_dep)
    fn_need = np.bincount(fn, weights=s["needed_kwh"], minlength=nights * n_fdr)
    depot_ratio = dn_need[dn] / (sim["depot_cap"][dep_of] * 10.0)    # ~10 h overnight window
    feeder_ratio = fn_need[fn] / (sim["feeder_cap"][fdr_of] * 10.0)
    key = s["night"] * len(sim["chg_kw"]) + s["charger"]
    same_charger = np.bincount(key, minlength=nights * len(sim["chg_kw"]))[key]
    cum_faults = np.vstack([np.zeros((1, len(sim["chg_kw"]))), np.cumsum(sim["fault_log"], axis=0)])
    lo = np.maximum(0, s["night"] - cfg.history_nights)
    prev_faults = cum_faults[s["night"], s["charger"]] - cum_faults[lo, s["charger"]]

    v = s["vehicle"]
    kw = sim["chg_kw"][s["charger"]]
    feats = np.column_stack([
        s["arrival_soc"], s["required_soc"], np.log1p(s["needed_kwh"]), dwell, s["arrival_h"],
        s["departure_h"] - 24, ((s["night"] + 0) % 7 >= 5).astype(float),
        sim["veh_cap"][v], sim["veh_ac"][v], sim["veh_dc"][v], sim["veh_soh"][v],
        kw, (kw >= 50).astype(float), s["needed_kwh"] / (kw * np.maximum(dwell, 0.5)),
        depot_ratio, feeder_ratio, same_charger, s["queue_position"], prev_faults,
    ]).astype(np.float32)

    data = HeteroData()
    data["session"].x = torch.from_numpy(feats)
    data["vehicle"].x = torch.from_numpy(np.column_stack([
        sim["veh_cap"], sim["veh_ac"], sim["veh_dc"], sim["veh_soh"]]).astype(np.float32))
    data["charger"].x = torch.from_numpy(np.column_stack([
        sim["chg_kw"], (sim["chg_kw"] >= 50).astype(float)]).astype(np.float32))
    data["depot"].x = torch.from_numpy(np.column_stack([
        sim["depot_cap"], np.bincount(sim["chg_depot"], minlength=n_dep),
        np.bincount(sim["veh_depot"], minlength=n_dep)]).astype(np.float32))
    data["feeder"].x = torch.from_numpy(sim["feeder_cap"][:, None].astype(np.float32))
    data["depot_night"].x = torch.ones(nights * n_dep, 1)
    data["feeder_night"].x = torch.ones(nights * n_fdr, 1)

    nn_dep = np.arange(nights * n_dep)
    nn_fdr = np.arange(nights * n_fdr)
    t_dn = T0 + (nn_dep // n_dep) * DAY + 15 * 3600
    t_fn = T0 + (nn_fdr // n_fdr) * DAY + 15 * 3600
    sid = np.arange(n_s)
    edges = {
        ("session", "of_vehicle", "vehicle"): (sid, v, night_t),
        ("session", "uses", "charger"): (sid, s["charger"], night_t),
        ("session", "in", "depot_night"): (sid, dn, night_t),
        ("session", "in_feeder", "feeder_night"): (sid, fn, night_t),
        ("depot_night", "of", "depot"): (nn_dep, nn_dep % n_dep, t_dn),
        ("feeder_night", "of", "feeder"): (nn_fdr, nn_fdr % n_fdr, t_fn),
        ("depot_night", "part_of", "feeder_night"): (
            nn_dep, (nn_dep // n_dep) * n_fdr + sim["depot_feeder"][nn_dep % n_dep], t_dn),
        ("charger", "located_at", "depot"): (np.arange(len(sim["chg_kw"])), sim["chg_depot"], None),
        ("vehicle", "assigned_to", "depot"): (np.arange(len(sim["veh_depot"])), sim["veh_depot"], None),
        ("depot", "fed_by", "feeder"): (np.arange(n_dep), sim["depot_feeder"], None),
    }
    for (src, rel, dst), (a, b, t) in edges.items():
        ei = torch.from_numpy(np.stack([a, b]).astype(np.int64))
        tt = torch.from_numpy((np.zeros(len(a)) if t is None else t).astype(np.int64))
        data[src, rel, dst].edge_index = ei
        data[src, rel, dst].time = tt
        data[dst, f"rev_{rel}", src].edge_index = ei.flip(0).contiguous()
        data[dst, f"rev_{rel}", src].time = tt

    y = torch.from_numpy(s["missed"].astype(np.float32))
    t_seed = torch.from_numpy(night_t.astype(np.int64))
    ranges = {"train": (0, cfg.train_nights), "val": (cfg.train_nights, cfg.train_nights + cfg.val_nights),
              "test": (cfg.train_nights + cfg.val_nights, nights)}
    splits = {}
    for name, (a, b) in ranges.items():
        idx = torch.from_numpy(np.nonzero((s["night"] >= a) & (s["night"] < b))[0].astype(np.int64))
        splits[name] = SeedSet(node=idx, time=t_seed[idx], y=y[idx])
    meta = {
        "config": asdict(cfg), "data_source": "ev_fleet_synthetic", "seed_type": "session",
        "metadata": [NODE_TYPES, [list(e) for e in EDGE_TYPES]],
        "history_features": AGGREGATE_FEATURES,
        "num_nodes": {nt: int(data[nt].num_nodes) for nt in NODE_TYPES},
        "num_edges": {"/".join(k): int(data[k].edge_index.size(1)) for k in data.edge_types},
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_pos_rate": {k: float(v.y.mean()) for k, v in splits.items()},
    }
    return TemporalTask(data=data, splits=splits, feature_names={"session": SESSION_FEATURES}, meta=meta)
