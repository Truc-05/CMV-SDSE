import pandas as pd

t = pd.read_csv("results/tables/cascade_theory_report.csv")
m = pd.read_csv("results/tables/main_results_3mode.csv")
tn = int(t.loc[t.pairing_condition == "same_model", "n_items"].iloc[0])
mn = int(m.loc[(m["mode"] == "FM-3.3") & (m["pool"] == "same_model"), "n"].iloc[0])
tag = "MATCH" if tn == mn else "STILL DIVERGENT"
print(f"[reconcile] theory same_model n_items={tn}  main n={mn}  -> {tag}")
