"""Greedy conditional marginal verification value — the utility complement to n_eff.

n_eff (dependence) asks "how statistically redundant are the decisions?"; this asks
"how much does a new source change the DECISION the current set S would make?". Two signed
components, both conditioned on what S already does and both using the same cascade
union-semantics (an item is flagged iff some member rejects it):

    Delta_catch_i(S) = #{error x : S misses x, i catches x} / #{error x : S misses x}   (good)
    Delta_fa_i(S)    = #{clean x : S passes x, i flags x}  / #{clean x : S passes x}     (bad)
    V_i(S)           = w_catch * Delta_catch_i - w_fa * Delta_fa_i - lambda * C_i

C_i = median latency (cheapest verifier = 1). A source usefully diversifies only if it
carries different information AND creates net decision value, so the controller weighs
catch, false alarm and compute together. Reading dependence alone (pick least-correlated /
highest n_eff) can select a near-constant verifier that catches nothing — this module
exposes that, and phase_diagram() maps where the policy SCALEs, DIVERSIFYs, or STOPs as the
false-alarm/miss cost ratio and the compute price lambda vary.

Greedy (one trace per failure mode), item-level catch/flag = majority-reject over a model's
pooled gates. Read-only over the raw gate records; no model calls.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd


def _per_model_reject(dfm, thr=0.5, is_error=True):
    """{model:{item: 1 if the model MAJORITY-REJECTS it}} over items of the given class.
    On an error item a reject is a catch (good); on a clean item a reject is a false alarm."""
    mk = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # model->item->[accepts, decided]
    for _, r in dfm.iterrows():
        if bool(r["ground_truth_is_error"]) != is_error:
            continue
        for g in r["gates"]:
            acc = g.get("accepted")
            if acc not in (True, False):
                continue
            cell = mk[g.get("verifier_id")][r["item_id"]]
            cell[1] += 1
            if acc is True:
                cell[0] += 1
    return {v: {it: (1 if c[0] / c[1] < thr else 0) for it, c in items.items() if c[1] > 0}
            for v, items in mk.items()}


def _self_split_reject(dfm, model, thr=0.5, is_error=True):
    """An independent 2nd draw of `model` via a per-item gate half-split -> {item:0/1 reject}."""
    h = defaultdict(lambda: [[0, 0], [0, 0]])
    for _, r in dfm.iterrows():
        if bool(r["ground_truth_is_error"]) != is_error:
            continue
        gs = [g for g in r["gates"]
              if g.get("verifier_id") == model and g.get("accepted") in (True, False)]
        for j, g in enumerate(gs):
            hi = 0 if j < len(gs) / 2 else 1
            h[r["item_id"]][hi][1] += 1
            if g["accepted"] is True:
                h[r["item_id"]][hi][0] += 1
    return {it: (1 if v[1][0] / v[1][1] < thr else 0) for it, v in h.items() if v[1][1] > 0}


def _median_latency(dfm, model):
    xs = [g.get("latency_seconds") for _, r in dfm.iterrows() for g in r["gates"]
          if g.get("verifier_id") == model and g.get("latency_seconds")]
    return float(np.median(xs)) if xs else float("nan")


def _wilson(k, n, z=1.959963984540054):
    """Wilson score CI for a binomial proportion; honest at the k=0 boundary ([0, upper>0])."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _matrix(reject_dict, items, idx):
    v = np.zeros(len(items), dtype=bool)
    for it, val in reject_dict.items():
        if val and it in idx:
            v[idx[it]] = True
    return v


_OUT_OF_POOL_PROBES = frozenset({"mistral_small_24b", "gemma2_9b"})
# Out-of-pool probe verifiers (the 24B escalate target, the gemma diversity probe) whose
# verdicts live in the same raw dirs but are NOT part of the frozen confirmatory pool.
# Excluding them by default keeps the frozen _prep-based analyses reproducible even after
# those probes are collected. Pass exclude=() to keep every model, or a custom set.


def _prep(dfm, thr, exclude=_OUT_OF_POOL_PROBES):
    """Per-mode structures shared by the greedy trace and the phase sweep."""
    catch = _per_model_reject(dfm, thr, True)
    fa = _per_model_reject(dfm, thr, False)
    models = [m for m in sorted(catch) if m not in exclude]
    err_items = sorted({it for v in models for it in catch.get(v, {})})
    cln_items = sorted({it for v in models for it in fa.get(v, {})})
    eidx = {it: i for i, it in enumerate(err_items)}
    cidx = {it: i for i, it in enumerate(cln_items)}
    Ce = {v: _matrix(catch.get(v, {}), err_items, eidx) for v in models}
    Cf = {v: _matrix(fa.get(v, {}), cln_items, cidx) for v in models}
    Se = {v: _matrix(_self_split_reject(dfm, v, thr, True), err_items, eidx) for v in models}
    Sf = {v: _matrix(_self_split_reject(dfm, v, thr, False), cln_items, cidx) for v in models}
    lat = {v: _median_latency(dfm, v) for v in models}
    finite = [x for x in lat.values() if x == x]
    cmin = min(finite) if finite else float("nan")
    cost = {v: (lat[v] / cmin if cmin == cmin and cmin else float("nan")) for v in models}
    return dict(models=models, ne=len(err_items), nc=len(cln_items),
                Ce=Ce, Cf=Cf, Se=Se, Sf=Sf, cost=cost)


def _greedy_run(P, w_catch, w_fa, lam, record=False):
    """One operating point. record=False -> region summary; record=True -> per-candidate rows."""
    models, ne, nc = P["models"], P["ne"], P["nc"]
    Ce, Cf, Se, Sf, cost = P["Ce"], P["Cf"], P["Se"], P["Sf"], P["cost"]
    if not models:
        return [] if record else {"s0": None, "first_action": "NA", "set_size": 0,
                                   "n_diversify": 0, "n_scale": 0}

    def solo_V(v):
        dc = Ce[v].sum() / ne if ne else 0.0
        df = Cf[v].sum() / nc if nc else 0.0
        return w_catch * dc - w_fa * df - lam * cost[v]

    s0 = max(models, key=solo_V)
    S, Ecov, Fcov = [s0], Ce[s0].copy(), Cf[s0].copy()
    rows, actions = [], []
    if record:
        dl, dh = _wilson(int(Ce[s0].sum()), ne)
        rows.append({"step": 0, "S_before": "", "cand_type": "INIT", "candidate": s0,
                     "delta": round(Ce[s0].sum() / ne if ne else 0, 4),
                     "delta_ci_low": round(dl, 4), "delta_ci_high": round(dh, 4),
                     "unique_catches": int(Ce[s0].sum()), "denom": ne,
                     "delta_fa": round(Cf[s0].sum() / nc if nc else 0, 4),
                     "rel_cost": round(cost[s0], 3), "V": round(solo_V(s0), 4),
                     "selected": True, "action": "START"})

    for step in range(1, len(models) + 2):
        emiss, cpass = ~Ecov, ~Fcov
        eden, cden = int(emiss.sum()), int(cpass.sum())
        if eden == 0:
            actions.append("EXHAUST")
            break
        cands = []
        for v in models:
            if v not in S:
                cands.append(("NEW", v, Ce[v], Cf[v]))
        for v in S:
            cands.append(("SCALE", v, Se[v], Sf[v]))
        scored = []
        for typ, v, ev, fv in cands:
            dc = (ev & emiss).sum() / eden
            dfa = (fv & cpass).sum() / cden if cden else 0.0
            V = w_catch * dc - w_fa * dfa - lam * cost[v]
            lo, hi = _wilson(int((ev & emiss).sum()), eden)
            scored.append((typ, v, ev, fv, dc, dfa, lo, hi, V))
        scored.sort(key=lambda c: -c[8])
        best = scored[0]
        proceed = (best[8] > 0) and (best[6] > 0)   # V>0 AND Delta_catch significant (Wilson low>0)
        act = "STOP" if not proceed else ("SCALE" if best[0] == "SCALE" else "DIVERSIFY")
        actions.append(act)
        if record:
            for rank, (typ, v, ev, fv, dc, dfa, lo, hi, V) in enumerate(scored):
                rows.append({"step": step, "S_before": "+".join(S), "cand_type": typ,
                             "candidate": v, "delta": round(dc, 4),
                             "delta_ci_low": round(lo, 4), "delta_ci_high": round(hi, 4),
                             "unique_catches": int((ev & emiss).sum()), "denom": eden,
                             "delta_fa": round(dfa, 4), "rel_cost": round(cost[v], 3),
                             "V": round(V, 4), "selected": rank == 0,
                             "action": act if rank == 0 else ""})
        if not proceed:
            break
        if best[0] == "NEW":
            S.append(best[1])
        Ecov, Fcov = Ecov | best[2], Fcov | best[3]

    if record:
        return rows
    first = next((a for a in actions if a != "EXHAUST"), "STOP")
    return {"s0": s0, "first_action": first, "set_size": len(S),
            "n_diversify": actions.count("DIVERSIFY"), "n_scale": actions.count("SCALE")}


def greedy_marginal_value(df, lam=0.1, thr=0.5, w_catch=1.0, w_fa=0.0):
    """One row per (failure_mode, step, candidate): Delta_catch (+Wilson CI), Delta_fa, cost, V."""
    out = []
    for mode, dfm in df.groupby("failure_mode_id"):
        for r in _greedy_run(_prep(dfm, thr), w_catch, w_fa, lam, record=True):
            out.append({"failure_mode_id": mode, **r})
    return pd.DataFrame(out)


def phase_diagram(df, cost_ratios=None, lambdas=None, thr=0.5, w_catch=1.0):
    """Sweep (false-alarm/miss cost ratio, compute price lambda); label each cell's policy.

    cost_ratio r = w_fa / w_catch (how costly a false alarm is relative to a caught miss).
    Returns one row per (failure_mode, r, lambda) with the first post-S0 action
    (DIVERSIFY / SCALE / STOP) and the resulting verifier-set size.
    """
    cost_ratios = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0] if cost_ratios is None else cost_ratios
    lambdas = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8] if lambdas is None else lambdas
    out = []
    for mode, dfm in df.groupby("failure_mode_id"):
        P = _prep(dfm, thr)
        for r in cost_ratios:
            for lam in lambdas:
                s = _greedy_run(P, w_catch, w_catch * r, lam, record=False)
                out.append({"failure_mode_id": mode, "cost_ratio_fa_over_miss": r,
                            "lambda_cost": lam, "s0": s["s0"],
                            "first_action": s["first_action"], "set_size": s["set_size"],
                            "n_diversify": s["n_diversify"], "n_scale": s["n_scale"]})
    return pd.DataFrame(out)
