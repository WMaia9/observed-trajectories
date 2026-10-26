"""Summarize semisynthetic_matched.py results over seeds (P2 review weakness 4).

Groups by world (mode, tau, gamma), model size and observation parameters; reports the mean, standard deviation and
range over seeds of real, erased and count AUCs and rho for the plain, matched and wash-out analyses. Synthetic only.
Usage: python experiments/semisynthetic_summary.py results/semisynthetic_matched.json [--min_seeds 3]
"""
import argparse, json
import numpy as np
import pandas as pd

KEYS = ["mode", "tau", "gamma", "layers", "embd", "visit_rate", "p_detect", "iters"]


def flatten(results):
    rows = []
    for r in results:
        base = {k: r.get(k, d) for k, d in zip(KEYS, ["frailty", None, None, 4, 64, 0.7, 0.2, None])}
        base["seed"] = r.get("seed", 0)
        for an in ["plain", "matched", "washout"]:
            for c in ["real", "erased", "count_only", "rho"]:
                v = r[an].get(c)
                base[f"{an}_{c}"] = np.nan if v is None else v
        rows.append(base)
    return pd.DataFrame(rows)


def summarize(df, min_seeds=1):
    val = [c for c in df.columns if c not in KEYS + ["seed"]]
    g = df.groupby(KEYS, dropna=False)
    out = g[val].agg(["mean", "std", "min", "max"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g["seed"].nunique()
    return out[out["n_seeds"] >= min_seeds].reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--min_seeds", type=int, default=1)
    a = ap.parse_args()
    s = summarize(flatten(json.load(open(a.path))), a.min_seeds)
    cols = KEYS[:7] + ["n_seeds"] + [f"{an}_{c}_{st}" for an in ["plain", "matched"] for c in ["real", "erased", "rho"]
                                     for st in ["mean", "std"]]
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        print(s[cols].to_string(index=False))
    print("SUMMARY_JSON", s.to_json(orient="records"))
