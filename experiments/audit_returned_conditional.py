"""Matched and wash-out AUCs for the people the official window cut drops (P2, reviewer round 8, item 4).

audit_keep_sex.py rebuilt the official batch with the sex token kept and saved the paired points of the real input, the
erased input and the identity-free model to ~/runs/audit/off0.1_horopen_keepsex. Here we run the conditional analyses
of audit_corrected_conditional.py (matched cells, fine matching, one-year wash-out, person bootstrap of the matched
AUC) separately for the people the official batch keeps and for the returned people, with and without the controls
whose target is a no-event token, and once more on the diseases evaluable in every group (common disease set).
Only aggregates are written, to ~/results/audit_returned_conditional.json.
Usage: python -u experiments/audit_returned_conditional.py [--B 100]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from delphi_aou.replication.evaluate import OFFICIAL
from delphi_aou.replication import audit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_keep_sex import make_get_batch_keep_sex
from audit_corrected_conditional import target_noevent, subset, analyse


def evaluable(rows, names):
    """Diseases with at least one evaluable unit (2 cases and 2 controls) in these rows."""
    df = audit.point_aucs(rows, names, extra_scores=False)
    return set(int(k) for k in df.index[df[names[0]].notna()])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "delphi_rep_data"))
    ap.add_argument("--B", type=int, default=100)
    ap.add_argument("--out", default=str(Path.home() / "results" / "audit_returned_conditional.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(OFFICIAL))
    from utils import get_p2i, get_batch
    val = np.fromfile(f"{a.data}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    p2i = get_p2i(val)
    kw = dict(select="left", block_size=80, device="cpu", padding="random", no_event_token_rate=5)
    np.random.seed(1337); torch.manual_seed(1337)
    x0 = get_batch(range(len(p2i)), val, p2i, **kw)[0].numpy()
    np.random.seed(1337); torch.manual_seed(1337)
    d = make_get_batch_keep_sex()(range(len(p2i)), val, p2i, **kw)
    kept_off = ((x0 == 2) | (x0 == 3)).any(1)
    y, b = d[2].numpy(), d[3].numpy()
    meta, rows = audit.load(Path.home() / "runs" / "audit" / "off0.1_horopen_keepsex")
    names = meta["variants"]
    rec = target_noevent(rows, y, b)
    for r in rows:
        r["returned"] = ~kept_off[r["person"].astype(np.int64)]
    out = {"share_points_target_recovered": round(rec, 4), "variants": names}
    groups = {"official_kept": lambda r: ~r["returned"], "returned_only": lambda r: r["returned"]}
    sets = {}
    for corr in (False, True):
        for g, f in groups.items():
            fn = (lambda r, f=f: f(r) & (r["case"] | ~r["target_noevent"])) if corr else f
            sub = subset(rows, fn)
            key = g + ("_noevent_controls_removed" if corr else "")
            sets[key] = sub
            out[key] = analyse(sub, names, a.B)
            print("RESULT", key, json.dumps(out[key]), flush=True)
            Path(a.out).write_text(json.dumps(out, indent=1))
    common = evaluable(sets["official_kept"], names) & evaluable(sets["returned_only"], names)
    out["n_common_diseases"] = len(common)
    for key in ("official_kept", "returned_only"):
        sub = [r for r in sets[key] if r["k"] in common]
        out[key + "_common"] = analyse(sub, names, 0)
        print("RESULT", key + "_common", json.dumps(out[key + "_common"]), flush=True)
    out["all_people_common"] = analyse([r for r in rows if r["k"] in common], names, 0)
    print("RESULT all_people_common", json.dumps(out["all_people_common"]), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("RETURNED CONDITIONAL COMPLETE", flush=True)
