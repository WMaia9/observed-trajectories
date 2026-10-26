"""Conditional analyses of the P2 paired audit under the corrected control definition (reviewer rounds 3 to 7, B item).

The matched and wash-out AUCs of the paper use every point of the official protocol, including the controls whose
target is a randomly placed no-event token. Here we recompute them (a) without those controls and (b) additionally
restricted to points scored after the person's first recorded diagnosis (nreal >= 1, "inside the record"), for the
main checkpoint and for the identity-free model (tags with suffix _idfree; its own score is the "erased" variant).

The target token of each saved point is recovered from the same batch as run_audit.py (same seeds), exactly as in
audit_protocol_checks.py. Only aggregates are written, to ~/results/audit_corrected_conditional.json.
Usage: python -u experiments/audit_corrected_conditional.py [--tags off0.1_horopen,...] [--B 100]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from delphi_aou.replication.evaluate import OFFICIAL
from delphi_aou.replication import audit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_matching_checks import matched, summary


def target_noevent(rows, y, b):
    """Flag points whose next target is the no-event token (id 1 in the batch), as in audit_protocol_checks."""
    rec = []
    for r in rows:
        pr, zk = r["person"].astype(np.int64), r["zk"]
        hit = np.isclose(b[pr], zk[:, None], atol=0.5)
        r["target_noevent"] = (hit & (y[pr] == 1)).any(1) & ~(hit & (y[pr] != 1)).any(1)
        rec.append(hit.any(1).mean())
    return float(np.mean(rec))


def subset(rows, fn):
    """Rows restricted to the points where fn(r) is True (all per-point arrays sliced)."""
    out = []
    for r in rows:
        m = fn(r)
        if m.sum() == 0:
            continue
        s = {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in r.items()}
        out.append(s)
    return out


def analyse(rows, names, B, seed=0):
    res = {}
    plain = audit.point_aucs(rows, names)
    res["plain"] = summary(plain[[c for c in plain if c != "recency_only"]])
    for fine in (False, True):
        df, kept = matched(rows, names, fine)
        res["matched_fine" if fine else "matched_default"] = {**summary(df), "share_cases_kept": round(kept, 4)}
    df = audit.point_aucs(rows, names, filt=audit.washout(365.25))
    res["washout_1y"] = summary(df[[c for c in df if c != "recency_only"]])
    ctrl = np.concatenate([~r["case"] for r in rows]); nz = np.concatenate([r["nreal"] == 0 for r in rows])
    res["share_controls_no_prior_diagnosis"] = round(float(nz[ctrl].mean()), 4)
    res["share_cases_no_prior_diagnosis"] = round(float(nz[~ctrl].mean()), 4)
    if B:
        persons = np.unique(np.concatenate([r["person"] for r in rows]))
        W = np.random.default_rng(seed).poisson(1.0, (B, len(persons))).astype(np.float32)
        A = matched(rows, names, False, W, persons)
        ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
        mean = {n: np.nanmean(A[n], 1) for n in names}
        res["matched_default_bootstrap"] = {"B": B, **{f"auc_{n}": ci(mean[n]) for n in names}}
        if "real" in names:
            res["matched_default_bootstrap"].update(
                {f"rho_{n}": ci((mean[n] - 0.5) / (mean["real"] - 0.5)) for n in names if n != "real"})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "delphi_rep_data"))
    ap.add_argument("--tags", default="off0.1_horopen,off0.1_horopen_idfree,off0.1_hor365.25,off0.1_hor365.25_idfree,"
                                      "off365.25_horopen,off365.25_horopen_idfree")
    ap.add_argument("--B", type=int, default=100)
    ap.add_argument("--out", default=str(Path.home() / "results" / "audit_corrected_conditional.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(OFFICIAL))
    from utils import get_p2i, get_batch
    val = np.fromfile(f"{a.data}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    p2i = get_p2i(val)
    np.random.seed(1337); torch.manual_seed(1337)
    d = get_batch(range(len(p2i)), val, p2i, select="left", block_size=80, device="cpu", padding="random",
                  no_event_token_rate=5)
    _, _, y, b = [t.numpy() for t in d]
    out = {}
    for tag in a.tags.split(","):
        meta, rows = audit.load(Path.home() / "runs" / "audit" / tag)
        names = meta["variants"]
        rec = target_noevent(rows, y, b)
        res = {"share_points_target_recovered": round(rec, 4)}
        corrected = subset(rows, lambda r: r["case"] | ~r["target_noevent"])
        res["no_event_controls_removed"] = analyse(corrected, names, a.B)
        print(tag, "removed", json.dumps(res["no_event_controls_removed"]), flush=True)
        inside = subset(corrected, lambda r: r["nreal"] >= 1)
        res["removed_and_after_first_diagnosis"] = analyse(inside, names, a.B)
        print(tag, "inside", json.dumps(res["removed_and_after_first_diagnosis"]), flush=True)
        out[tag] = res
        Path(a.out).write_text(json.dumps(out, indent=1))
    print("CORRECTED CONDITIONAL COMPLETE", flush=True)
