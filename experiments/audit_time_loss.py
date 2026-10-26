"""Does the identity-free model learn the record calendar as well as the full model? (P2, reviewer round 11)

The matched gap compares two separately trained models; it is biased upward if the identity-free model fits the
calendar worse. Here every checkpoint is scored on the same validation batch as the audit (official get_batch,
select='left', block 80, random no-event padding, seed 1337) with the official loss: the time-to-next-token term
loss_dt (exponential log-likelihood of the gap to the next token, from the total rate) and, for reference, the
cross-entropy loss_ce. Each model sees its own training input (full: real tokens; identity-free: diseases and death
replaced by no event; record-token: replaced by token 1270). loss_dt is reported over all target positions and over
positions whose target is a disease or death (the calendar of records), with a person bootstrap (B resamples) of the
difference full minus identity-free. Only aggregates are written: ~/results/audit_time_loss.json
Usage: python -u experiments/audit_time_loss.py [--max_people N] [--B 200]
"""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delphi_aou.replication.evaluate import OFFICIAL

FIRST_DISEASE, VOCAB = 13, 1270

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "delphi_rep_data"))
    ap.add_argument("--runs", default=str(Path.home() / "delphi_rep_runs"))
    ap.add_argument("--max_people", type=int, default=-1)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--out", default=str(Path.home() / "results" / "audit_time_loss.json"))
    a = ap.parse_args()
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, str(OFFICIAL))
    from model import Delphi, DelphiConfig
    from utils import get_p2i, get_batch
    val = np.fromfile(f"{a.data}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    p2i = get_p2i(val)
    n = len(p2i) if a.max_people < 0 else min(a.max_people, len(p2i))
    np.random.seed(1337); torch.manual_seed(1337)
    d = get_batch(range(n), val, p2i, select="left", block_size=80, device="cpu", padding="random", no_event_token_rate=5)
    x, ag, y, b = d
    print("BATCH", tuple(x.shape), f"{time.time() - t0:.0f}s", flush=True)
    ck = {Path(p).parts[-3]: p for p in sorted(glob.glob(f"{a.runs}/*/ckpt/ckpt.pt"))}
    print("CHECKPOINTS", list(ck), flush=True)
    variants = [("full", "full", None), ("idfree", "idfree", 1), ("idrec", "idrec", VOCAB)]
    per = {}
    for name, run, tok in variants:
        if run not in ck:
            print("SKIP", name, "no checkpoint", flush=True); continue
        c = torch.load(ck[run], map_location=dev)
        m = Delphi(DelphiConfig(**c["model_args"])); m.load_state_dict(c["model"]); m.eval().to(dev)
        ign = list(m.config.ignore_tokens)
        dt_all = np.zeros(n); n_all = np.zeros(n); dt_dis = np.zeros(n); n_dis = np.zeros(n); ce = np.zeros(n)
        with torch.no_grad():
            for s in range(0, n, a.bs):
                xb, ab, yb, bb = [t[s:s + a.bs].to(dev) for t in (x, ag, y, b)]
                xin = xb if tok is None else torch.where((xb >= FIRST_DISEASE) & (xb < VOCAB), torch.full_like(xb, tok), xb)
                # per-position losses, recomputed as in the official forward (model.py, loss_dt and loss_ce)
                logits, _, _ = m(xin, ab)
                lse = torch.logsumexp(logits, -1)
                lse = -torch.log(torch.exp(-lse) + m.config.t_min)
                dt = torch.clamp(bb - ab, min=1.0)
                ldt = -torch.log(dt + m.config.t_min)
                ldt_pos = -(lse - torch.exp(lse - ldt))
                lce = torch.nn.functional.cross_entropy(logits.transpose(1, 2), yb.clamp(min=0), reduction="none")
                pas = yb != -1
                for k in ign:
                    pas &= yb != k
                dis = pas & (yb >= FIRST_DISEASE)
                i = slice(s, s + len(xb))
                dt_all[i] = (ldt_pos * pas).sum(1).cpu().numpy(); n_all[i] = pas.sum(1).cpu().numpy()
                dt_dis[i] = (ldt_pos * dis).sum(1).cpu().numpy(); n_dis[i] = dis.sum(1).cpu().numpy()
                ce[i] = (lce * pas).sum(1).cpu().numpy()
        per[name] = dict(dt_all=dt_all, n_all=n_all, dt_dis=dt_dis, n_dis=n_dis, ce=ce)
        print("MODEL", name, "loss_dt", round(dt_all.sum() / n_all.sum(), 4), "loss_dt disease targets",
              round(dt_dis.sum() / n_dis.sum(), 4), "loss_ce", round(ce.sum() / n_all.sum(), 4), f"{time.time() - t0:.0f}s", flush=True)
        del m; torch.cuda.empty_cache()
    W = np.random.default_rng(0).poisson(1.0, (a.B, n)).astype(np.float32)
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    out = {"n_people": n, "models": {}, "diff": {}}
    for name, p in per.items():
        out["models"][name] = {k: round(float(p[k].sum() / p[nk].sum()), 4) for k, nk in
                               (("dt_all", "n_all"), ("dt_dis", "n_dis"), ("ce", "n_all"))}
    for other in [v for v in per if v != "full"]:
        for k, nk in (("dt_all", "n_all"), ("dt_dis", "n_dis")):
            f, o = per["full"], per[other]
            bf = (W @ f[k]) / (W @ f[nk]); bo = (W @ o[k]) / (W @ o[nk])
            out["diff"][f"{other}_minus_full_{k}"] = {"point": round(float(o[k].sum() / o[nk].sum() - f[k].sum() / f[nk].sum()), 4),
                                                      "ci": ci(bo - bf)}
    print("RESULT", json.dumps(out), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("TIME LOSS COMPLETE", f"{time.time() - t0:.0f}s", flush=True)
