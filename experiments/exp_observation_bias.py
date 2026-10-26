#!/usr/bin/env python
"""Paper experiment 1: does the likelihood objective matter when EHR observation is partial?
Trains the faithful Delphi objective and the exact point process objective on SIMULATED records with
known truth, and scores both against (a) the recorded labels and (b) the true onsets.
Usage: python exp_observation_bias.py [p_burst] [seed]"""
import json, sys
import numpy as np
from delphi_aou.eval import metrics as eval_auc
from delphi_aou.baselines import classical as s3
from delphi_aou.train import delphi as s4
from delphi_aou.sim import observation as so
from delphi_aou import config as common

CFG = common.load_config()


def calib(pred, Y, E):
    """Per disease observed / expected among eligible rows, summarised as mean |log(O/E)|."""
    out = []
    for j in range(Y.shape[1]):
        e = E[:, j]
        if e.sum() < 200 or Y[e, j].sum() < 25:
            continue
        out.append(np.log(Y[e, j].mean() / max(pred[e, j].mean(), 1e-9)))
    return float(np.mean(np.abs(out))) if out else float("nan"), [round(float(x), 3) for x in out]


def main(p_burst=0.8, seed=0, n=150000, iters=1500, H=5):
    cohort, fo, truth = so.simulate(n=n, seed=seed, p_burst=p_burst)
    an = s3.build_anchors(cohort)
    te = an[an["split"] == "test"].reset_index(drop=True); te["anchor_id"] = np.arange(len(te))
    names = sorted(t for t in fo["token"].unique() if t != "DEATH") + ["DEATH"]
    Yr, Pr, Er, _ = s3.attach(te, fo, names[:-1], names[:-1], H, 1)
    Yt, Et = so.true_labels(te, truth, names, H)
    Yt[:, -1], Et[:, -1] = Yr[:, -1], Er[:, -1]
    bands = [tuple(b) for b in CFG["evaluation"]["age_bands"]]
    res = {"p_burst": p_burst, "seed": seed}
    for label, mode, pp in [("delphi_faithful", "delphi", False), ("observed_delphiloss", "observed", False),
                            ("observed_pp", "observed", True)]:
        model, r, sc, tab = s4.run(cohort, fo, mode=mode, censor=pp, horizon_years=H, n_diseases=len(names) - 1,
                                   iters=iters, batch=128, n_layer=2, n_embd=48, n_head=2, lr=2e-3, seed=seed,
                                   log=lambda *a: None)
        sc = s4.score(model, s4.Data(cohort, fo), te, names, mode, H, 1)
        a_rec = eval_auc.summarize(eval_auc.evaluate_auc(sc, Yr, Er, te["age_years"].to_numpy(), te["sex"].to_numpy(), names, bands, 25, 25))
        a_tru = eval_auc.summarize(eval_auc.evaluate_auc(sc[:, :-1], Yt[:, :-1], Et[:, :-1], te["age_years"].to_numpy(), te["sex"].to_numpy(), names[:-1], bands, 25, 25))
        c_tru, per = calib(sc[:, :-1], Yt[:, :-1], Et[:, :-1])
        c_rec, _ = calib(sc[:, :-1], Yr[:, :-1], Er[:, :-1])
        res[label] = {"auc_recorded": round(a_rec["macro_auc"], 4), "auc_true": round(a_tru["macro_auc"], 4),
                      "mean_abs_log_OE_true": round(c_tru, 3), "mean_abs_log_OE_recorded": round(c_rec, 3),
                      "log_OE_true_per_disease": per}
        print(label, res[label], flush=True)
    return res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        out = [main(pb, sd) for pb in (0.0, 0.5, 1.0) for sd in (0, 1)]
        common.write_results("results/sim_sweep.json", out)
        print("SWEEP DONE")
    else:
        pb = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
        sd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        print(json.dumps(main(pb, sd), indent=1))
