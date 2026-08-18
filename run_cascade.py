from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cascade.alpha_recorder import AlphaRecorder
from cascade.gate_runner import GateRunner
from cascade.paths import raw_run_dir, table_path
from cascade.mast_loader import (
    FULL_FILENAME,
    HUMAN_FILENAME,
    MastDownloadError,
    MastSchemaError,
    analyse_label_noise,
    describe,
    load_mast_items,
    resolve_path,
    system_prompt_for,
)
from cascade.ollama_backend import (
    BackendError,
    OllamaVerifierBackend,
    load_backend_settings,
    load_pools,
    load_verifier_specs,
)
from cascade.backend_factory import build_backend
from cascade.mcqa_loader import load_mcqa_items, mcqa_system_prompt
from cascade.gsm8k_loader import load_gsm8k_items, gsm8k_system_prompt
from cascade.mbpp_loader import load_mbpp_items, mbpp_system_prompt
from cascade.scenario_loader import default_testbed_root, load_items, summarise
from cascade.verifier_chain import VerifierChain
from theory.beta_fit import fit_beta_mle, rho_v_from_beta
from theory.ceiling_test import run_ceiling_test
from theory.concavity_test import run_concavity_test
from theory.net_utility import net_utility_curve
from theory.odds_law import odds_law_log_odds
from theory.trichotomy_test import find_k_dagger

RAW_ROOT = os.path.join("data", "raw_cascade_runs")
PROCESSED_DIR = os.path.join("data", "processed")
TABLES_DIR = os.path.join("results", "tables")


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def experiment_dir(source: str, target: str, context_mode: str = "truncate") -> str:
    """Raw-run dir, isolated by context condition (see cascade.paths.raw_run_dir)."""
    return raw_run_dir(target, context_mode, source)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_cascade")
    parser.add_argument("--pool", default="same_model")
    parser.add_argument("--source", default="mast",
                        choices=["mast", "testbed", "mcqa", "gsm8k", "mbpp"])
    parser.add_argument("--benchmark", default=None,
                        help="mcqa benchmark: arc | mmlu | csqa | truthfulqa (for --source mcqa)")
    parser.add_argument("--mast-target", default="any",
                        help="any, fc3, or a single mode id such as 3.2")
    parser.add_argument("--mast-path", default=None,
                        help="directory holding the MAST json files, if already downloaded")
    parser.add_argument("--mast-mas", nargs="*", default=None)
    parser.add_argument("--mast-benchmark", nargs="*", default=None)
    parser.add_argument("--mast-max-chars", type=int, default=None,
                        help="default: derived from the smallest num_ctx in the pool")
    parser.add_argument("--ctx-reserve-tokens", type=int, default=600)
    parser.add_argument("--budget", choices=["fast", "balanced", "full"], default="balanced",
                        help="prompt size preset; fast truncates hard, full uses the whole "
                             "context window and costs several times more compute")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--label-noise", action="store_true")
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--n-errors", type=int, default=None)
    parser.add_argument("--n-non-errors", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--testbed", default=None)
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--thresholds-config", default="configs/cascade_thresholds.yaml")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--analyse-only", action="store_true")
    parser.add_argument("--skip-analyse", action="store_true",
                        help="data-collection only; skip the end-of-run analyse so a single-pool "
                             "run never writes the shared-name analysis tables")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--false-alarm-cost", type=float, default=1.0,
                        help="weight on false alarms when locating k dagger")
    parser.add_argument("--k-gates", type=int, default=None,
                        help="analyse only records at this cascade depth (n_gates); use to "
                             "separate a k=50 sweep from an earlier k=20 pilot sharing a pool")
    parser.add_argument("--lambda-cost", type=float, default=0.1,
                        help="cost weight in the greedy marginal-value policy V=Delta-lambda*C")
    parser.add_argument("--context-mode", choices=["truncate", "evidence"], default="truncate",
                        help="how to fit long transcripts: 'truncate' cuts the middle; "
                             "'evidence' keeps a large full tail + head + a structural skeleton "
                             "of the middle (deterministic, no summariser). Use the two as a "
                             "context factor to decompose context-induced vs intrinsic blindness.")
    return parser


def _first_event_gates(sub_df: "pd.DataFrame", is_error: bool) -> "list[int]":
    """1-indexed first flag gate per item (k_max+1 if never flagged), for the bootstrap.

    For an error item the flag is a correct reject (``caught_at``); for a clean item it is
    the first false-alarm reject drawn from its gate list.
    """
    out = []
    for _, row in sub_df.iterrows():
        gates = row.get("gates")
        k_dec = int(row.get("n_decided") or row.get("n_gates") or (len(gates) if isinstance(gates, list) else 0))
        if is_error:
            fe = row.get("caught_at")
        else:
            fe = None
            if isinstance(gates, list):
                for g in gates:
                    if g.get("accepted") is False:
                        fe = g.get("gate_index")
                        break
        missing = fe is None or (isinstance(fe, float) and fe != fe)
        out.append(k_dec + 1 if missing else int(fe))
    return out


def _attach_curve_ci(by_k: "pd.DataFrame", df: "pd.DataFrame", is_error: bool,
                     seed: int) -> "pd.DataFrame":
    """Add cluster-bootstrap (item-resampled) CI columns to a by-k curve table."""
    import numpy as np
    import pandas as pd
    from theory.uncertainty import bootstrap_curve_ci

    subset = df[df["ground_truth_is_error"] == is_error]
    pieces = []
    for keys, g in by_k.groupby(["pairing_condition", "failure_mode_id"]):
        items = subset[(subset["pairing_condition"] == keys[0])
                       & (subset["failure_mode_id"] == keys[1])]
        k_max = int(g["k"].max())
        fe = np.asarray(_first_event_gates(items, is_error))
        _, lo, hi = bootstrap_curve_ci(fe, k_max, n_boot=2000, seed=seed)
        gg = g.sort_values("k").copy()
        gg["ci_low"] = [round(float(lo[k - 1]), 4) for k in gg["k"]]
        gg["ci_high"] = [round(float(hi[k - 1]), 4) for k in gg["k"]]
        pieces.append(gg)
    return pd.concat(pieces, ignore_index=True) if pieces else by_k


def analyse(recorder: AlphaRecorder, thresholds: dict,
            false_alarm_cost: float = 1.0, k_gates: int | None = None,
            lambda_cost: float = 0.1, context_mode: str = "truncate") -> int:
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace

    from theory.betabinom_mixture import fit_mixture
    from theory.net_utility import net_utility_cost_frontier
    from theory.marginal_value import greedy_marginal_value, phase_diagram
    from mast_link.mast_fc3_reference import load_mast_fc3_reference
    from mast_link.ceiling_vs_fc3 import compare_ceiling_to_fc3

    df = recorder.load_raw()
    if df.empty:
        print("no cascade runs recorded yet", file=sys.stderr)
        return 1
    if k_gates is not None and "n_gates" in df.columns:
        before = len(df)
        df = df[df["n_gates"] == k_gates].copy()
        print(f"--k-gates {k_gates}: kept {len(df)} of {before} records", file=sys.stderr)
        if df.empty:
            print(f"no records at n_gates={k_gates}", file=sys.stderr)
            return 1

    os.makedirs(TABLES_DIR, exist_ok=True)
    seed = int(thresholds.get("seed_base", 20260701))
    min_gates = int(thresholds.get("min_gates_for_ceiling_fit", 10))

    # ---- descriptive curves (with cluster-bootstrap CIs) ----
    inst = recorder.per_instance_alpha(df)
    rel = recorder.reliability_by_k_and_condition(df)
    fa = recorder.false_alarm_rate(df)
    fa_k = recorder.false_alarm_by_k_and_condition(df)
    counts = recorder.item_unwanted_counts(df)

    if not rel.empty:
        rel = _attach_curve_ci(rel, df, is_error=True, seed=seed)
    if not fa_k.empty:
        fa_k = _attach_curve_ci(fa_k, df, is_error=False, seed=seed + 1)

    indet = recorder.indeterminate_report(df)
    if not indet.empty:
        indet.to_csv(table_path("indeterminate_gates.csv", context_mode), index=False)
        worst = indet["indeterminate_rate"].max()
        if worst > 0.02:
            print(f"WARNING: up to {worst:.1%} of gates were indeterminate and excluded. "
                  "See results/tables/indeterminate_gates.csv", file=sys.stderr)

    recorder.save_processed(inst, "per_instance_alpha")
    recorder.save_processed(rel, "reliability_by_k")
    rel.to_csv(table_path("reliability_by_k.csv", context_mode), index=False)
    if not fa.empty:
        fa.to_csv(table_path("false_alarm_rate.csv", context_mode), index=False)
    if not fa_k.empty:
        fa_k.to_csv(table_path("false_alarm_by_k.csv", context_mode), index=False)

    try:
        fc3 = load_mast_fc3_reference()
    except FileNotFoundError:
        fc3 = None

    cost_grid = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
    report = []
    frontier_rows = []

    for keys, err_c in counts[counts["ground_truth_is_error"]].groupby(
            ["pairing_condition", "failure_mode_id"]):
        cond, mode = keys
        row = {"pairing_condition": cond, "failure_mode_id": mode,
               "k_max": int(err_c["K"].max()), "n_items": int(len(err_c))}

        # ---- H1 / H3 / H4 : item-level Beta-Binomial mixture on the miss counts ----
        fit = fit_mixture(err_c["m_unwanted"].to_numpy(), err_c["K"].to_numpy(),
                          min_gates_for_fit=min_gates, compute_ci=True)
        row["mu_hat_miss_rate"] = round(fit.mu_hat, 4)
        row["preferred_decay"] = fit.preferred_decay
        row["delta_aic"] = round(fit.delta_aic, 3)
        row["p_heterogeneity"] = float(f"{fit.p_heterogeneity:.3g}")
        row["rho_v"] = round(fit.rho_v, 4)
        row["rho_v_ci_low"] = round(fit.rho_v_ci[0], 4)
        row["rho_v_ci_high"] = round(fit.rho_v_ci[1], 4)
        row["beta_a"] = round(fit.a_hat, 4)
        row["beta_b"] = round(fit.b_hat, 4)
        row["design_effect_at_k_max"] = round(fit.design_effect_at_k_max, 3)
        row["n_eff_at_k_max"] = round(fit.n_eff_at_k_max, 3)
        row["blind_spot_mass"] = round(fit.blind_spot_mass, 4)
        row["blind_spot_ci_low"] = round(fit.blind_spot_ci[0], 4)
        row["blind_spot_ci_high"] = round(fit.blind_spot_ci[1], 4)
        row["ceiling_p_value"] = float(f"{fit.p_ceiling:.3g}")
        row["ceiling_detected"] = fit.ceiling_detected
        row["odds_law_is_upper_bound"] = fit.odds_law_is_upper_bound
        row["below_odds_law_fraction"] = round(fit.below_odds_law_fraction, 4)
        row["fit_warnings"] = ";".join(fit.warnings)

        # ---- symmetric false-alarm heterogeneity on the clean items ----
        clean_c = counts[(counts["pairing_condition"] == cond)
                         & (counts["failure_mode_id"] == mode)
                         & (~counts["ground_truth_is_error"])]
        if len(clean_c) >= 4:
            fa_fit = fit_mixture(clean_c["m_unwanted"].to_numpy(), clean_c["K"].to_numpy(),
                                 min_gates_for_fit=min_gates, compute_ci=False)
            row["fa_rate_mean"] = round(fa_fit.mu_hat, 4)
            row["fa_rho_v"] = round(fa_fit.rho_v, 4)
            row["fa_always_flag_mass"] = round(fa_fit.blind_spot_mass, 4)
            row["n_clean_items"] = int(len(clean_c))
        else:
            row["fa_rate_mean"] = row["fa_rho_v"] = row["fa_always_flag_mass"] = None
            row["n_clean_items"] = int(len(clean_c))

        # ---- trichotomy / net-utility cost frontier + MAST FC3 linkage ----
        g_rel = rel[(rel["pairing_condition"] == cond) & (rel["failure_mode_id"] == mode)]
        g_fa = fa_k[(fa_k["pairing_condition"] == cond) & (fa_k["failure_mode_id"] == mode)]
        merged = g_rel.merge(g_fa[["k", "false_alarm_rate"]], on="k", how="inner")
        if len(merged) >= 3 and not merged["false_alarm_rate"].isna().all():
            ks = merged["k"].tolist()
            cr = merged["reliability"].tolist()
            fr = merged["false_alarm_rate"].tolist()
            eq = net_utility_cost_frontier(ks, cr, fr, [false_alarm_cost], prevalence=None)[0]
            row["k_best_net_equal_weight"] = eq["k_best"]
            row["k_dagger_net"] = eq["k_dagger"]
            prev = fc3.fc3_rate if fc3 is not None else None
            for c in cost_grid:
                fr_eq = net_utility_cost_frontier(ks, cr, fr, [c], prevalence=None)[0]
                fr_mast = (net_utility_cost_frontier(ks, cr, fr, [c], prevalence=prev)[0]
                           if prev is not None else None)
                frontier_rows.append({
                    "pairing_condition": cond, "failure_mode_id": mode,
                    "false_alarm_cost": c,
                    "k_best_equal_weight": fr_eq["k_best"],
                    "gates_help_equal_weight": fr_eq["gates_help"],
                    "k_best_mast_prevalence": fr_mast["k_best"] if fr_mast else None,
                    "gates_help_mast_prevalence": fr_mast["gates_help"] if fr_mast else None,
                    "mast_prevalence": prev,
                })
            # operating depth a MAST-base-rate operator would pick at unit cost
            op = (net_utility_cost_frontier(ks, cr, fr, [1.0], prevalence=prev)[0]
                  if prev is not None else eq)
            row["k_best_mast_operating"] = op["k_best"]
        else:
            row["k_best_net_equal_weight"] = row["k_dagger_net"] = None
            row["k_best_mast_operating"] = None

        # ---- FC3 comparison (reuse the audited mast_link comparator via a shim) ----
        if fc3 is not None:
            shim = SimpleNamespace(
                blind_spot_mass=fit.blind_spot_mass,
                blind_spot_ci=fit.blind_spot_ci,
                p_value=fit.p_ceiling,
                is_ceiling_detected=fit.ceiling_detected,
            )
            cmp = compare_ceiling_to_fc3(shim, fc3)
            row["fc3_rate"] = fc3.fc3_rate
            row["blind_spot_vs_fc3"] = cmp.interpretation

        report.append(row)

    table = pd.DataFrame(report)
    out = table_path("cascade_theory_report.csv", context_mode)
    table.to_csv(out, index=False)
    if frontier_rows:
        pd.DataFrame(frontier_rows).to_csv(
            table_path("net_utility_by_cost.csv", context_mode), index=False)

    cols = ["pairing_condition", "failure_mode_id", "k_max", "n_items", "preferred_decay",
            "delta_aic", "rho_v", "rho_v_ci_low", "rho_v_ci_high", "n_eff_at_k_max",
            "blind_spot_mass", "blind_spot_ci_high", "ceiling_detected",
            "k_best_net_equal_weight", "k_best_mast_operating"]
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(table[[c for c in cols if c in table.columns]].to_string(index=False))
    print(f"\nwrote {out}")

    # ---- utility layer: greedy conditional marginal value (complement to n_eff) ----
    mv = greedy_marginal_value(df, lam=lambda_cost)
    if not mv.empty:
        mv_out = table_path("marginal_value_greedy.csv", context_mode)
        mv.to_csv(mv_out, index=False)
        sel = mv[mv["selected"]]
        show = ["failure_mode_id", "step", "cand_type", "candidate", "delta",
                "delta_ci_low", "delta_ci_high", "delta_fa", "unique_catches", "denom",
                "rel_cost", "V", "action"]
        with pd.option_context("display.width", 200, "display.max_columns", 40):
            print("\ngreedy marginal-value trace (selected step winners; "
                  f"lambda={lambda_cost}, w_fa=0):")
            print(sel[show].to_string(index=False))
        print(f"wrote {mv_out}")

        # ---- Stage 2: SCALE / DIVERSIFY / STOP phase diagram over (fa/miss cost, lambda) ----
        ph = phase_diagram(df)
        ph_out = table_path("marginal_value_phase.csv", context_mode)
        ph.to_csv(ph_out, index=False)
        code = {"DIVERSIFY": "D", "SCALE": "S", "STOP": "T", "NA": "-"}
        for mode, g in ph.groupby("failure_mode_id"):
            grid = (g.assign(a=g["first_action"].map(lambda x: code.get(x, "?")))
                     .pivot(index="cost_ratio_fa_over_miss", columns="lambda_cost", values="a"))
            print(f"\nphase diagram {mode} (rows=fa/miss cost, cols=lambda; "
                  "D=diversify S=scale T=stop):")
            print(grid.to_string())
        print(f"wrote {ph_out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = load_yaml(args.thresholds_config)

    if args.source == "mast":
        target_label = args.mast_target
    elif args.source == "mcqa":
        target_label = args.benchmark or "unknown"
    elif args.source == "gsm8k":
        target_label = "gsm8k"
    elif args.source == "mbpp":
        target_label = "mbpp"
    else:
        target_label = "testbed"
    scope = experiment_dir(args.source, target_label, args.context_mode)
    recorder = AlphaRecorder(scope, PROCESSED_DIR)

    if args.analyse_only:
        return analyse(recorder, thresholds, args.false_alarm_cost, args.k_gates,
                       args.lambda_cost, args.context_mode)

    if args.label_noise:
        try:
            path = resolve_path(HUMAN_FILENAME, args.mast_path)
            report = analyse_label_noise(path)
        except (MastDownloadError, MastSchemaError) as exc:
            print(str(exc), file=sys.stderr)
            return 3
        os.makedirs(TABLES_DIR, exist_ok=True)
        out = os.path.join(TABLES_DIR, "label_noise.json")
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
        print(f"traces {report.n_traces}  annotations {report.n_annotations}  "
              f"rounds {report.rounds}")
        print(f"unanimity over all rows          : {report.overall_full_agreement:.3f}")
        print(f"unanimity over informative rows  : "
              f"{report.overall_full_agreement_informative:.3f}  "
              f"(n={report.n_informative})")
        if report.ids_outside_final_taxonomy:
            print(f"\nmode ids not in the final 14-mode taxonomy: "
                  f"{report.ids_outside_final_taxonomy}")
        if report.ids_with_multiple_titles:
            print("mode ids carrying more than one definition across rounds:")
            for mode, names in sorted(report.ids_with_multiple_titles.items()):
                print(f"  {mode}: {len(names)} distinct titles")
        print(f"\n{'mode':<8} {'n':>4} {'unanim':>8} {'n_info':>7} {'info_unanim':>12} "
              f"{'any_pos':>8}")
        for mode, row in sorted(report.per_mode.items()):
            info = row["informative_unanimous_rate"]
            info_s = "         n/a" if info != info else f"{info:>12.3f}"
            print(f"{mode:<8} {int(row['n']):>4} {row['unanimous_rate']:>8.3f} "
                  f"{int(row['n_informative']):>7} {info_s} {row['any_positive_rate']:>8.3f}")
        print(f"\n{report.note}")
        print(f"\nwrote {out}")
        return 0

    if args.describe:
        if args.source != "mast":
            print("--describe applies to the mast source", file=sys.stderr)
            return 2
        try:
            path = resolve_path(FULL_FILENAME, args.mast_path)
            print(json.dumps(describe(path, args.mast_target), indent=2))
        except (MastDownloadError, MastSchemaError) as exc:
            print(str(exc), file=sys.stderr)
            return 3
        return 0

    k = args.k or int(thresholds["k_pilot"] if args.pilot else thresholds["k_max"])
    n_err = args.n_errors if args.n_errors is not None else (
        10 if args.pilot else int(thresholds["n_items_per_failure_mode"]))
    n_clean = args.n_non_errors if args.n_non_errors is not None else max(n_err // 2, 5)

    pools = load_pools(args.models_config)
    specs = load_verifier_specs(args.models_config)
    if args.pool not in pools:
        print(f"unknown pool {args.pool!r}; available: {sorted(pools)}", file=sys.stderr)
        return 2
    min_ctx = min(specs[v].num_ctx for v in pools[args.pool])
    budget_fraction = {"fast": 0.15, "balanced": 0.40, "full": 1.0}[args.budget]
    max_chars = args.mast_max_chars
    if max_chars is None:
        ceiling_chars = max(4000, (min_ctx - args.ctx_reserve_tokens) * 4)
        max_chars = max(4000, int(ceiling_chars * budget_fraction))

    if args.source == "mast":
        try:
            full_path = resolve_path(FULL_FILENAME, args.mast_path)
            items = load_mast_items(
                full_path, target=args.mast_target,
                n_errors=n_err, n_non_errors=n_clean,
                mas_names=args.mast_mas, benchmarks=args.mast_benchmark,
                max_chars=max_chars, context_mode=args.context_mode,
            )
        except (MastDownloadError, MastSchemaError) as exc:
            print(str(exc), file=sys.stderr)
            return 3
        origin = f"MAST-Data ({full_path})"
    elif args.source == "mcqa":
        if not args.benchmark:
            print("--source mcqa requires --benchmark", file=sys.stderr)
            return 2
        items = load_mcqa_items(args.benchmark, n_errors=n_err, n_non_errors=n_clean,
                                seed=int(thresholds.get("seed_base", 20260701)))
        origin = f"mcqa ({args.benchmark})"
    elif args.source == "gsm8k":
        items = load_gsm8k_items(n_errors=n_err, n_non_errors=n_clean,
                                 seed=int(thresholds.get("seed_base", 20260701)))
        origin = "gsm8k (gemma2:9b solutions)"
    elif args.source == "mbpp":
        items = load_mbpp_items(n_errors=n_err, n_non_errors=n_clean,
                                seed=int(thresholds.get("seed_base", 20260701)))
        origin = "mbpp (gemma2:9b solutions)"
    else:
        testbed = args.testbed or default_testbed_root()
        if not testbed:
            print(
                "no testbed traces found.\n"
                "expected one of:\n"
                "  shared_testbed/data/raw_runs/*.jsonl\n"
                "  ../minimal-mas-failure-modes/data/raw_runs/*.jsonl\n"
                "run the sibling testbed first, or pass --testbed <path>",
                file=sys.stderr,
            )
            return 3
        items = load_items(testbed, args.scenarios, n_errors=n_err, n_non_errors=n_clean)
        origin = f"testbed ({testbed})"

    info = summarise(items)
    n_trunc = sum(1 for i in items if i.source_config.get("was_truncated"))
    n_compressed = sum(1 for i in items if i.source_config.get("was_compressed"))
    info["context_mode"] = args.context_mode
    info["n_truncated"] = n_trunc
    info["truncated_fraction"] = round(n_trunc / len(items), 4) if items else 0.0
    info["n_evidence_compressed"] = n_compressed
    print(json.dumps({"source": args.source, "origin": origin, "k": k, **info},
                     indent=2), file=sys.stderr)
    if n_trunc:  # only fires in truncate mode: the middle is destructively removed
        print(
            f"WARNING: {n_trunc} of {len(items)} transcripts were middle-TRUNCATED at "
            f"{max_chars} chars. Evidence of the failure may have been removed, which biases "
            "the false-accept rate upward. Report this fraction, and check sensitivity by "
            "rerunning with --context-mode evidence.",
            file=sys.stderr,
        )
    if n_compressed:  # evidence mode: full trace read, middle skeletonised, tail+head intact
        print(
            f"note: {n_compressed} of {len(items)} transcripts exceeded the context and were "
            "evidence-COMPRESSED (deterministic skeleton of the middle over the FULL trace; head "
            "and a large tail kept intact). This is not information-destructive middle truncation.",
            file=sys.stderr,
        )
    if not items:
        print("no usable items after filtering", file=sys.stderr)
        return 3
    if info["n_errors"] == 0:
        print("no ground-truth error items; alpha is undefined", file=sys.stderr)
        return 3

    settings = load_backend_settings(args.models_config)
    if args.source == "mast":
        try:
            verifier_prompt = system_prompt_for(args.mast_target)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    elif args.source == "mcqa":
        verifier_prompt = mcqa_system_prompt()
    elif args.source == "gsm8k":
        verifier_prompt = gsm8k_system_prompt()
    elif args.source == "mbpp":
        verifier_prompt = mbpp_system_prompt()
    else:
        verifier_prompt = system_prompt_for("any")

    backends = [
        build_backend(specs[verifier_key], settings, thresholds, verifier_prompt)
        for verifier_key in pools[args.pool]
    ]

    if args.dry_run:
        print(json.dumps({
            "pool": args.pool,
            "target": args.mast_target if args.source == "mast" else "any",
            "gates_cycle": [b.spec.checkpoint for b in backends],
            "calls_planned": len(items) * k * args.repeats,
            "num_ctx": [b.spec.num_ctx for b in backends],
            "budget": args.budget,
            "max_chars_used": max_chars,
            "est_prompt_tokens_max": max(len(i.payload["prompt"]) for i in items) // 4,
            "prompt_chars_max": max(len(i.payload["prompt"]) for i in items),
            "prompt_chars_median": sorted(
                len(i.payload["prompt"]) for i in items)[len(items) // 2],
            "verifier_question": verifier_prompt.split("Decide one thing only: ")[1]
                                                .split(".\n\n")[0],
        }, indent=2))
        return 0

    if not backends[0].health():
        print(f"backend unreachable at {backends[0].host}", file=sys.stderr)
        return 2
    try:
        for backend in backends:
            backend.ensure_available()
    except BackendError as exc:
        print(str(exc), file=sys.stderr)
        return 4

    by_mode: dict[str, list] = {}
    for item in items:
        by_mode.setdefault(item.failure_mode_id, []).append(item)

    total_records = 0
    for failure_mode_id, mode_items in sorted(by_mode.items()):
        chain = VerifierChain(
            backends, verifier_id=args.pool, k_max=k,
            retry_on_backend_error=int(thresholds.get("retry_on_backend_error", 3)),
            timeout_seconds_per_gate=float(thresholds.get("timeout_seconds_per_gate", 60)),
        )
        runner = GateRunner(
            scenario_source=[{
                "item_id": i.item_id,
                "ground_truth_is_error": i.ground_truth_is_error,
                "payload": i.payload,
            } for i in mode_items],
            verifier_chain=chain,
            output_dir=scope,
            pairing_condition=args.pool,
            failure_mode_id=failure_mode_id,
            seed_base=int(thresholds.get("seed_base", 20260701)),
        )
        records = runner.run_all(k=k, n_repeats=args.repeats)
        total_records += len(records)
        print(f"{failure_mode_id}: {len(records)} records, "
              f"{len(mode_items)} items, k={k}", file=sys.stderr)

    unparsed = sum(b.unparsed_count for b in backends)
    calls = sum(b.call_count for b in backends)
    overflow = sum(b.context_overflow_count for b in backends)
    max_chars_seen = max((b.max_prompt_chars_seen for b in backends), default=0)
    print(f"total records {total_records}, backend calls {calls}, "
          f"unparsed verdicts {unparsed}", file=sys.stderr)
    print(f"raw runs written to {scope}", file=sys.stderr)
    print(f"longest prompt seen {max_chars_seen} chars "
          f"(~{max_chars_seen // 4} tokens), context overflows {overflow}", file=sys.stderr)
    if unparsed:
        rate = unparsed / max(calls + unparsed, 1)
        print(f"WARNING: {unparsed} replies carried no verdict ({rate:.1%} of attempts). "
              "Those gates are recorded as indeterminate and excluded from alpha, not "
              "counted as accepts. If the rate is high the verifier is not following the "
              "output format and alpha rests on a thinned sample.", file=sys.stderr)
    if overflow:
        print(f"WARNING: {overflow} prompts exceeded num_ctx and were silently truncated "
              "by the server. Lower --mast-max-chars or raise num_ctx in configs/models.yaml.",
              file=sys.stderr)

    if args.skip_analyse:
        print("skipped end-of-run analyse (--skip-analyse); analyse offline via a pilot script",
              file=sys.stderr)
        return 0
    return analyse(recorder, thresholds, args.false_alarm_cost, args.k_gates,
                   args.lambda_cost, args.context_mode)


if __name__ == "__main__":
    raise SystemExit(main())
