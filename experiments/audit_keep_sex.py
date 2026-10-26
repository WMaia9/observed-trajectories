"""How much does the silent exclusion of the official evaluation change the AUC? (P2, reviewer B item)

The official batch (utils.get_batch, select='left', block_size=80, random padding) cuts every row from the left to
81 tokens after inserting no-event tokens; for full windows this removes the sex token, and evaluate_auc then drops
those people (7.4% of the validation set). Here we rebuild the batch with one change: when the cut would remove the
sex token, the sex token is kept and the earliest token after it is removed instead. Everything else (window,
padding draws, seeds, selection rule, aggregation) is the official code. People kept by the official batch get
exactly the same rows, so the only difference is that the excluded people return.

Variants: the main checkpoint (real, erased) and the identity-free model (its own input). We report the mean AUC
over all people, over the people the official batch keeps, and over the returned people alone, with and without the
controls whose target is a no-event token. Points go to ~/runs/audit/off0.1_horopen_keepsex (Workbench only);
aggregates to ~/results/audit_keep_sex.json.
Usage: python -u experiments/audit_keep_sex.py [--chunk 50]
"""
import argparse, inspect, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from delphi_aou.replication.evaluate import OFFICIAL, EraseIdentity
from delphi_aou.replication import audit

OLD_CUT = """        cut_margin = tokens.shape[1] - block_size - 1
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]
"""
NEW_CUT = """        cut_margin = tokens.shape[1] - block_size - 1
        _sx = (tokens == 2) | (tokens == 3)
        _has = _sx[:, :cut_margin].any(1)
        _t = tokens[:, cut_margin:].clone(); _a = ages[:, cut_margin:].clone()
        if _has.any():
            _r = _has.nonzero().squeeze(1)
            _sp = _sx.float().argmax(1)[_r]
            _t[_r, 1:] = tokens[_r, cut_margin + 1:]; _a[_r, 1:] = ages[_r, cut_margin + 1:]
            _t[_r, 0] = tokens[_r, _sp]; _a[_r, 0] = ages[_r, _sp]
        tokens, ages = _t, _a
"""


def make_get_batch_keep_sex():
    """The official get_batch with the left cut changed to keep the sex token (source patched, nothing else)."""
    sys.path.insert(0, str(OFFICIAL))
    import utils
    src = inspect.getsource(utils.get_batch)
    assert src.count(OLD_CUT) == 1, "official cut not found"
    ns = dict(utils.__dict__)
    exec(src.replace(OLD_CUT, NEW_CUT).replace("def get_batch(", "def get_batch_keep_sex(", 1), ns)
    return ns["get_batch_keep_sex"]


def load_model(path, device):
    sys.path.insert(0, str(OFFICIAL))
    from model import Delphi, DelphiConfig
    ck = torch.load(path, map_location=device)
    m = Delphi(DelphiConfig(**ck["model_args"])); m.load_state_dict(ck["model"]); return m.eval().to(device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "delphi_rep_data"))
    ap.add_argument("--ckpt", default=str(Path.home() / "delphi_rep_runs/full/ckpt/ckpt.pt"))
    ap.add_argument("--ckpt_idfree", default=str(Path.home() / "delphi_rep_runs/idfree/ckpt/ckpt.pt"))
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    sys.path.insert(0, str(OFFICIAL))
    from utils import get_p2i, get_batch
    gbk = make_get_batch_keep_sex()
    val = np.fromfile(f"{a.data}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    p2i = get_p2i(val)
    kw = dict(select="left", block_size=80, device="cpu", padding="random", no_event_token_rate=5)
    np.random.seed(1337); torch.manual_seed(1337)
    d_off = get_batch(range(len(p2i)), val, p2i, **kw)
    np.random.seed(1337); torch.manual_seed(1337)
    d = gbk(range(len(p2i)), val, p2i, **kw)
    x0, x1 = d_off[0].numpy(), d[0].numpy()
    kept_off = ((x0 == 2) | (x0 == 3)).any(1)
    kept_new = ((x1 == 2) | (x1 == 3)).any(1)
    same = (x0[kept_off] == x1[kept_off]).all() and np.allclose(d_off[1].numpy()[kept_off], d[1].numpy()[kept_off])
    res = {"share_kept_official": round(float(kept_off.mean()), 4), "share_kept_new": round(float(kept_new.mean()), 4),
           "rows_of_kept_people_identical": bool(same)}
    print("BATCH", json.dumps(res), flush=True)
    assert same, "keep-sex batch must equal the official batch for kept people"
    m = load_model(a.ckpt, a.device); mf = load_model(a.ckpt_idfree, a.device)
    models = {"real": m, "erased": EraseIdentity(m).eval(), "idfree": EraseIdentity(mf).eval()}
    labels = pd.read_csv(OFFICIAL / "delphi_labels_chapters_colours_icd.csv")
    out = Path.home() / "runs" / "audit" / "off0.1_horopen_keepsex"
    if not (out / "meta.json").exists():
        audit.collect(models, [t.to(a.device) for t in d], labels, out, offset=0.1, horizon=None, device=a.device,
                      disease_chunk_size=a.chunk)
    meta, rows = audit.load(out)
    names = meta["variants"]
    y, b = d[2].numpy(), d[3].numpy()
    for r in rows:                                                       # person = batch row index (all kept now)
        pr, zk = r["person"].astype(np.int64), r["zk"]
        hit = np.isclose(b[pr], zk[:, None], atol=0.5)
        r["target_noevent"] = (hit & (y[pr] == 1)).any(1) & ~(hit & (y[pr] != 1)).any(1)
        r["returned"] = ~kept_off[pr]
    groups = {"all": None, "official_kept": lambda r: ~r["returned"], "returned_only": lambda r: r["returned"]}
    for corr in (False, True):
        for g, f in groups.items():
            filt = (lambda r, f=f: (r["case"] | ~r["target_noevent"]) & (np.ones(len(r["case"]), bool) if f is None else f(r))) \
                if corr else f
            df = audit.point_aucs(rows, names, filt=filt)
            s = {c: round(float(df[c].mean()), 4) for c in names + ["count_only"]}
            for c in names + ["count_only"]:
                if c != "real":
                    s[f"rho_{c}"] = round((s[c] - 0.5) / (s["real"] - 0.5), 4)
            s["n_diseases"] = int(df["real"].notna().sum())
            key = f"{g}{'_noevent_controls_removed' if corr else ''}"
            res[key] = s
            print("RESULT", key, json.dumps(s), flush=True)
    pts = np.concatenate([r["returned"] for r in rows]); cs = np.concatenate([r["case"] for r in rows])
    res["share_points_from_returned_people"] = {"cases": round(float(pts[cs].mean()), 4),
                                                "controls": round(float(pts[~cs].mean()), 4)}
    (Path.home() / "results" / "audit_keep_sex.json").write_text(json.dumps(res, indent=1))
    print("KEEP SEX COMPLETE", flush=True)
