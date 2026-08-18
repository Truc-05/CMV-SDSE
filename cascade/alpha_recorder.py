from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


class AlphaRecorder:
    def __init__(self, raw_runs_dir: str, processed_dir: str):
        self.raw_runs_dir = Path(raw_runs_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_raw(self, pattern: str = "*.jsonl") -> pd.DataFrame:
        records = []
        for path in self.raw_runs_dir.glob(pattern):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
        if not records:
            return pd.DataFrame(
                columns=[
                    "item_id", "repeat_index", "pairing_condition", "failure_mode_id",
                    "verifier_id", "ground_truth_is_error", "n_gates", "n_decided",
                    "n_indeterminate", "false_accepts", "survived_all_gates", "caught_at",
                ]
            )
        return pd.DataFrame(records)

    @staticmethod
    def _denominator(row) -> float:
        decided = row.get("n_decided")
        if decided is None or (isinstance(decided, float) and decided != decided):
            decided = row.get("n_gates", 0)
        return float(decided or 0)

    def per_instance_alpha(self, df: pd.DataFrame, min_decided: int = 1) -> pd.DataFrame:
        error_df = df[df["ground_truth_is_error"]].copy()
        error_df["n_decided_eff"] = error_df.apply(self._denominator, axis=1)
        error_df["alpha_empirical"] = error_df.apply(
            lambda row: (row["false_accepts"] / row["n_decided_eff"]
                         if row["n_decided_eff"] >= min_decided else float("nan")),
            axis=1,
        )
        return error_df

    def alpha_by_config(self, df: pd.DataFrame) -> pd.DataFrame:
        inst = self.per_instance_alpha(df)
        group_cols = ["pairing_condition", "failure_mode_id", "verifier_id"]
        return (
            inst.groupby(group_cols)
            .agg(
                n_instances=("item_id", "count"),
                alpha_mean=("alpha_empirical", "mean"),
                alpha_var=("alpha_empirical", "var"),
                blind_spot_rate=("survived_all_gates", "mean"),
            )
            .reset_index()
        )

    def reliability_by_k(self, df: pd.DataFrame) -> pd.DataFrame:
        error_df = df[df["ground_truth_is_error"]].copy()
        max_k = int(error_df["n_gates"].max()) if len(error_df) else 0
        rows = []
        for k in range(1, max_k + 1):
            eligible = error_df[error_df.apply(self._denominator, axis=1) >= k]
            if len(eligible) == 0:
                continue
            caught_by_k = eligible["caught_at"].apply(lambda c: c is not None and c <= k)
            rows.append({"k": k, "reliability": float(caught_by_k.mean()),
                         "n": int(len(eligible))})
        return pd.DataFrame(rows)

    def reliability_by_k_and_condition(self, df: pd.DataFrame) -> pd.DataFrame:
        error_df = df[df["ground_truth_is_error"]].copy()
        rows = []
        for keys, group in error_df.groupby(["pairing_condition", "failure_mode_id"]):
            max_k = int(group["n_gates"].max()) if len(group) else 0
            for k in range(1, max_k + 1):
                eligible = group[group.apply(self._denominator, axis=1) >= k]
                if len(eligible) == 0:
                    continue
                caught = eligible["caught_at"].apply(lambda c: c is not None and c <= k)
                rows.append({
                    "pairing_condition": keys[0],
                    "failure_mode_id": keys[1],
                    "k": k,
                    "reliability": float(caught.mean()),
                    "n": int(len(eligible)),
                })
        return pd.DataFrame(rows)

    def false_alarm_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        clean = df[~df["ground_truth_is_error"]].copy()
        if clean.empty:
            return pd.DataFrame(columns=["pairing_condition", "failure_mode_id",
                                         "n_clean_items", "false_alarm_rate"])
        clean["any_reject"] = clean["gates"].apply(
            lambda gates: any(g["accepted"] is False for g in gates)
        )
        return (
            clean.groupby(["pairing_condition", "failure_mode_id"])
            .agg(n_clean_items=("item_id", "count"),
                 false_alarm_rate=("any_reject", "mean"))
            .reset_index()
        )

    def false_alarm_by_k_and_condition(self, df: pd.DataFrame) -> pd.DataFrame:
        clean = df[~df["ground_truth_is_error"]].copy()
        if clean.empty:
            return pd.DataFrame(columns=["pairing_condition", "failure_mode_id", "k",
                                         "false_alarm_rate", "n"])

        def first_reject(gates) -> int | None:
            for g in gates:
                if g["accepted"] is False:
                    return g["gate_index"]
            return None

        clean["first_reject"] = clean["gates"].apply(first_reject)
        rows = []
        for keys, group in clean.groupby(["pairing_condition", "failure_mode_id"]):
            max_k = int(group["n_gates"].max()) if len(group) else 0
            for k in range(1, max_k + 1):
                eligible = group[group.apply(self._denominator, axis=1) >= k]
                if len(eligible) == 0:
                    continue
                flagged = eligible["first_reject"].apply(
                    lambda c: c is not None and c <= k)
                rows.append({
                    "pairing_condition": keys[0],
                    "failure_mode_id": keys[1],
                    "k": k,
                    "false_alarm_rate": float(flagged.mean()),
                    "n": int(len(eligible)),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _gate_counts(row) -> tuple[int, int, int]:
        """Return (n_decided, n_accept, n_reject) from a record's gate list.

        Falls back to the record-level fields when the gate list is absent (older logs).
        """
        gates = row.get("gates")
        if isinstance(gates, list) and gates:
            accepts = sum(1 for g in gates if g.get("accepted") is True)
            rejects = sum(1 for g in gates if g.get("accepted") is False)
            return accepts + rejects, accepts, rejects
        decided = row.get("n_decided")
        if decided is None or (isinstance(decided, float) and decided != decided):
            decided = row.get("n_gates", 0) or 0
        accepts = int(row.get("false_accepts") or 0)
        return int(decided), accepts, int(decided) - accepts

    def item_unwanted_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-item count of the *unwanted* verdict, the sufficient statistic for the
        item-level Beta-Binomial estimator.

        For a ground-truth error item the unwanted verdict is a false accept (a miss);
        for a clean item it is a false alarm (a reject). ``m`` is that count and ``K`` is
        the number of decided gates. Items with no decided gate are dropped.
        """
        rows = []
        for _, row in df.iterrows():
            n_dec, n_acc, n_rej = self._gate_counts(row)
            if n_dec <= 0:
                continue
            is_error = bool(row["ground_truth_is_error"])
            rows.append({
                "pairing_condition": row["pairing_condition"],
                "failure_mode_id": row["failure_mode_id"],
                "item_id": row["item_id"],
                "repeat_index": row.get("repeat_index", 0),
                "ground_truth_is_error": is_error,
                "K": int(n_dec),
                # unwanted verdict: miss for errors, false alarm for clean items
                "m_unwanted": int(n_acc if is_error else n_rej),
            })
        return pd.DataFrame(rows)

    def indeterminate_report(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        work = df.copy()
        if "n_indeterminate" not in work.columns:
            work["n_indeterminate"] = 0
        return (
            work.groupby(["pairing_condition", "failure_mode_id"])
            .agg(n_items=("item_id", "count"),
                 gates_total=("n_gates", "sum"),
                 gates_indeterminate=("n_indeterminate", "sum"))
            .assign(indeterminate_rate=lambda d: d["gates_indeterminate"] / d["gates_total"])
            .reset_index()
        )

    def save_processed(self, df: pd.DataFrame, name: str) -> Path:
        out_path = self.processed_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)
        return out_path
