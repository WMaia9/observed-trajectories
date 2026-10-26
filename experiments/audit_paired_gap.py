"""Paired intervals for the full minus identity-free comparison, and AUC by disjoint time-to-target bins (P2 round 9).

The main checkpoint (tag T) and the identity-free model (tag T + _idfree, or another suffix) were scored on the same
batch with the same selection seed, so their saved points coincide. We check that they do, join them into one set of
rows with the variants real, erased (main checkpoint) and idfree (the identity-free model on its own input), and report
  (1) a joint person-level Poisson bootstrap (B resamples) of the mean AUCs, of real minus idfree and of
      rho_idfree, over all points and without the controls whose target is a no-event token (padding controls);
  (2) the same for the matched AUC (count quintile x recency band);
  (3) mean AUC and rho over disjoint bins of time from prediction to target (0-1, 1-2, 2-5, >5 years), with and
      without padding controls (bins on the prediction-to-target gap zk - z);
  (4) post-record points only (at least one diagnosis before the point: a zero-length wash-out), and the one-year
      wash-out restricted to the diseases evaluable on all points;
  (5) per ICD-10 chapter: mean AUC, rho and bootstrap intervals (B_chapter resamples).
Only aggregates are written, to ~/results/audit_paired_gap<suffix>.json.
Usage: python -u experiments/audit_paired_gap.py [--tag off0.1_horopen] [--free _idfree] [--B 500] [--Bm 200]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from delphi_aou.replication.evaluate import OFFICIAL
from delphi_aou.replication import audit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_matching_checks import matched, summary
from audit_corrected_conditional import target_noevent, subset

BINS = [(0, 365.25), (365.25, 730.5), (730.5, 1826.25), (1826.25, np.inf)]


def join(rows_main, rows_free, names_main, names_free, free_col="erased"):
    """Rows with scores [real, erased, idfree]; raises if the two point sets differ."""
    fm = {(r["sex"], r["k"]): r for r in rows_free}
    out = []
    i_r, i_e, i_f = names_main.index("real"), names_main.index("erased"), names_free.index(free_col)
    for r in rows_main:
        f = fm[(r["sex"], r["k"])]
        for key in ("person", "case", "zk", "z", "bracket"):
            if not np.array_equal(r[key], f[key]):
                raise ValueError(f"points differ for {r['sex']} {r['k']} ({key})")
        s = dict(r)
        s["scores"] = np.stack([r["scores"][:, i_r], r["scores"][:, i_e], f["scores"][:, i_f]], 1)
        out.append(s)
    if len(out) != len(rows_free):
        raise ValueError("different unit files")
    return out


def boot_matched(rows, names, B, seed=0):
    persons = np.unique(np.concatenate([r["person"] for r in rows]))
    W = np.random.default_rng(seed).poisson(1.0, (B, len(persons))).astype(np.float32)
    A = matched(rows, names, False, W, persons)
    mean = {n: np.nanmean(A[n], 1) for n in names}
    ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
    out = {"B": B, **{f"auc_{n}": ci(mean[n]) for n in names}}
    for n in names:
        if n != "real":
            out[f"diff_real_minus_{n}"] = ci(mean["real"] - mean[n])
            out[f"rho_{n}"] = ci((mean[n] - 0.5) / (mean["real"] - 0.5))
    return out


def by_chapter(rows, names, chapter_of, B, seed=0):
    out = {}
    for ch in sorted(set(chapter_of.values())):
        rs = [r for r in rows if chapter_of.get(r["k"]) == ch]
        if len({r["k"] for r in rs}) < 5:
            continue
        bs = audit.bootstrap(rs, names, B=B, seed=seed)
        out[ch] = {"n_diseases": bs["n_diseases"],
                   **{f"auc_{n}": [round(bs[f"mean_auc_{n}"]["point"], 4)] + [round(x, 4) for x in bs[f"mean_auc_{n}"]["ci"]] for n in names},
                   **{f"rho_{n}": [round(bs[f"rho_{n}"]["point"], 4)] + [round(x, 4) for x in bs[f"rho_{n}"]["ci"]] for n in names if n != "real"},
                   "diff_real_minus_idfree": [round(bs["diff_real_minus_idfree"]["point"], 4)] + [round(x, 4) for x in bs["diff_real_minus_idfree"]["ci"]]}
    return out


def by_bins(rows, names):
    res = {}
    for lo, hi in BINS:
        f = lambda r, lo=lo, hi=hi: ((r["zk"] - r["z"]) >= lo) & ((r["zk"] - r["z"]) < hi)
        df = audit.point_aucs(rows, names, filt=f)
        res[f"{lo / 365.25:g}-{hi / 365.25:g}y"] = summary(df[[c for c in df if c != "recency_only"]])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "delphi_rep_data"))
    ap.add_argument("--tag", default="off0.1_horopen")
    ap.add_argument("--free", default="_idfree", help="suffix of the identity-free tag (e.g. _idfree, _idrec)")
    ap.add_argument("--main_suffix", default="", help="suffix of the main tag (e.g. _s1)")
    ap.add_argument("--B", type=int, default=500); ap.add_argument("--Bm", type=int, default=200)
    ap.add_argument("--Bc", type=int, default=200, help="bootstrap resamples per chapter")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(OFFICIAL))
    from utils import get_p2i, get_batch
    val = np.fromfile(f"{a.data}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    p2i = get_p2i(val)
    np.random.seed(1337); torch.manual_seed(1337)
    d = get_batch(range(len(p2i)), val, p2i, select="left", block_size=80, device="cpu", padding="random",
                  no_event_token_rate=5)
    _, _, y, b = [t.numpy() for t in d]
    base = Path.home() / "runs" / "audit"
    mm, rm = audit.load(base / (a.tag + a.main_suffix))
    mf, rf = audit.load(base / (a.tag + a.free))
    rows = join(rm, rf, mm["variants"], mf["variants"])
    del rm, rf
    names = ["real", "erased", "idfree"]
    target_noevent(rows, y, b)
    out = {"tag": a.tag, "main": a.tag + a.main_suffix, "free": a.tag + a.free}
    import pandas as pd
    lab = pd.read_csv(OFFICIAL / "delphi_labels_chapters_colours_icd.csv")
    chapter_of = dict(zip(lab["index"].astype(int), lab["ICD-10 Chapter (short)"].astype(str)))
    plain_set = set(int(k) for k in audit.point_aucs(rows, names, extra_scores=False).dropna().index)
    for label, rs in (("all", rows), ("padding_removed", subset(rows, lambda r: r["case"] | ~r["target_noevent"]))):
        out[label] = {"plain_bootstrap": audit.bootstrap(rs, names, B=a.B),
                      "matched_point": summary(matched(rs, names, False)[0]),
                      "matched_bootstrap": boot_matched(rs, names, a.Bm),
                      "time_to_target_bins": by_bins(rs, names)}
        post = subset(rs, lambda r: r["nreal"] >= 1)
        dfp = audit.point_aucs(post, names)
        out[label]["post_record"] = summary(dfp[[c for c in dfp if c != "recency_only"]])
        dfw = audit.point_aucs([r for r in rs if r["k"] in plain_set], names, filt=audit.washout(365.25))
        out[label]["washout_1y_plain_disease_set"] = summary(dfw[[c for c in dfw if c != "recency_only"]])
        if label == "all":
            out[label]["by_chapter"] = by_chapter(rs, names, chapter_of, a.Bc)
        print("RESULT", label, json.dumps(out[label]["matched_bootstrap"]), flush=True)
        fn = a.out or str(Path.home() / "results" / f"audit_paired_gap{a.main_suffix}{a.free}.json")
        Path(fn).write_text(json.dumps(out, indent=1, default=float))
    print("PAIRED GAP COMPLETE", flush=True)
