"""Score the paired-audit points of Delphi-2M (AoU) with CEHR-GPT and evaluate them the same way (P2, second family).

The points (person row of the official batch, prediction age z, disease k, case/control) are those saved by
run_audit.py for --tag; CEHR-GPT sees the same batch row (the official 80-token window) up to z, converted to a
CEHR-GPT sequence. Three variants: cg_full (full model, full input), cg_erased (full model, codes in the input
replaced by [OOV]) and cg_idfree (identity-free model on the erased input). Read-out: delphi_aou.cehrgpt.model.
Per-point scores are written to ~/runs/audit/<tag>_cehrgpt/ (participant level, never exported); aggregates to
~/results/audit_cehrgpt.json: mean AUC over all points and without padding controls, matched cells (default and
fine), one-year wash-out, person bootstrap intervals (plain and matched) for the shares rho = (AUC_x - 0.5) /
(AUC_cg_full - 0.5), and the Delphi variants on the same points for reference.
Usage: python -u experiments/cehrgpt_score.py --full ~/cg_runs/full/best --idfree ~/cg_runs/idfree/best
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delphi_aou.replication.evaluate import OFFICIAL
from delphi_aou.replication import audit
from delphi_aou.cehrgpt import model as M
from audit_matching_checks import matched, summary
from audit_corrected_conditional import target_noevent, subset
from audit_paired_gap import boot_matched

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(Path.home() / "delphi_rep_data"))
ap.add_argument("--tag", default="off0.1_horopen")
ap.add_argument("--full", required=True); ap.add_argument("--idfree", required=True)
ap.add_argument("--tokenizer", default=str(Path.home() / "cg_runs" / "tokenizer"))
ap.add_argument("--people_per_batch", type=int, default=128)
ap.add_argument("--max_people", type=int, default=-1, help="smoke test: score only the first N batch rows")
ap.add_argument("--B", type=int, default=200); ap.add_argument("--Bm", type=int, default=100)
ap.add_argument("--out", default=str(Path.home() / "results" / "audit_cehrgpt.json"))
ap.add_argument("--eval_only", action="store_true", help="reuse the saved per-point CEHR-GPT scores (no GPU)")
ap.add_argument("--label", default="all,padding_removed", help="point sets to evaluate (one per process saves RAM)")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
sys.path.insert(0, str(OFFICIAL))
from utils import get_p2i, get_batch
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel

val = np.fromfile(f"{a.data}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
p2i = get_p2i(val)
np.random.seed(1337); torch.manual_seed(1337)
d = get_batch(range(len(p2i)), val, p2i, select="left", block_size=80, device="cpu", padding="random",
              no_event_token_rate=5)
x, ag, y, b = [t.numpy() for t in d]
meta, rows = audit.load(Path.home() / "runs" / "audit" / a.tag)
if a.max_people > 0:
    rows = subset(rows, lambda r: r["person"] < a.max_people)
cg = ["cg_full", "cg_erased", "cg_idfree"]
odir = Path.home() / "runs" / "audit" / (a.tag + "_cehrgpt")
keep_delphi = [c for c in ("real", "erased") if c in meta["variants"]]
names = keep_delphi + cg
if not a.eval_only:
    tok = CehrGptTokenizer.from_pretrained(a.tokenizer); vocab = M.Vocab(tok)
    models = {"cg_full": (CEHRGPT2LMHeadModel.from_pretrained(a.full), False),
              "cg_erased": (None, True), "cg_idfree": (CEHRGPT2LMHeadModel.from_pretrained(a.idfree), True)}
    models["cg_erased"] = (models["cg_full"][0], True)
    for mm, _ in models.values():
        mm.eval().to(dev)

    # index the points by person row
    P = np.concatenate([r["person"].astype(np.int64) for r in rows])
    Z = np.concatenate([r["z"] for r in rows]).astype(np.float64)
    K = np.concatenate([np.full(len(r["person"]), r["k"]) for r in rows])
    order = np.argsort(P, kind="stable"); persons, starts = np.unique(P[order], return_index=True)
    ends = np.r_[starts[1:], len(order)]
    S = {n: np.zeros(len(P), np.float32) for n in models}
    for b0 in range(0, len(persons), a.people_per_batch):
        grp = range(b0, min(b0 + a.people_per_batch, len(persons)))
        people = [(int(persons[i]), ag[persons[i]].astype(np.float64), x[persons[i]], Z[order[starts[i]:ends[i]]],
                   K[order[starts[i]:ends[i]]]) for i in grp]
        for n, (mm, er) in models.items():
            res = M.score_people(mm, vocab, people, dev, erase=er)
            for i, s in zip(grp, res):
                S[n][order[starts[i]:ends[i]]] = s
        if (b0 // a.people_per_batch) % 20 == 0:
            print(f"scored {grp.stop}/{len(persons)} people {time.time() - t0:.0f}s", flush=True)
    i = 0
    odir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        n = len(r["person"])
        np.save(odir / f"cg_{r['sex']}_{r['k']}.npy", np.stack([S[m][i:i + n] for m in models], 1)); i += n
    del S, P, Z, K, order

for r in rows:                       # keep only the columns needed (175M points do not fit twice in 26 GB)
    extra = np.load(odir / f"cg_{r['sex']}_{r['k']}.npy")
    r["scores"] = np.concatenate([r["scores"][:, [meta["variants"].index(c) for c in keep_delphi]].astype(np.float32),
                                  extra.astype(np.float32)], 1)
target_noevent(rows, y, b)
out = {"tag": a.tag, "variants": names, "max_people": a.max_people,
       "readout": "first code of the next visit, mixture over 8 time-token bins after z"}
prev = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
out = {**prev, **out}
for label in a.label.split(","):
    rs = rows if label == "all" else subset(rows, lambda r: r["case"] | ~r["target_noevent"])
    df = audit.point_aucs(rs, names); df = df[[c for c in df if c != "recency_only"]]
    res = {"plain_delphi_ref": summary(df), "plain_cg_ref": summary(df, ref="cg_full")}
    mt = matched(rs, names, False)[0]
    res["matched_default"] = {**summary(mt), **{f"cgref_{k}": v for k, v in summary(mt, ref="cg_full").items()}}
    res["matched_fine"] = summary(matched(rs, names, True)[0], ref="cg_full")
    dw = audit.point_aucs(rs, names, filt=audit.washout(365.25)); dw = dw[[c for c in dw if c != "recency_only"]]
    res["washout_1y"] = summary(dw, ref="cg_full")
    # paired intervals with cg_full as the reference: rows whose scores are [cg_full, cg_erased, cg_idfree]
    idx = [names.index(c) for c in cg]
    rc = [{**r, "scores": r["scores"][:, idx]} for r in rs]
    bs = audit.bootstrap(rc, ["real", "erased", "idfree"], B=a.B)
    res["plain_bootstrap_cg"] = {k: v for k, v in bs.items() if k.startswith(("mean_auc", "rho", "diff"))}
    res["matched_bootstrap_cg"] = boot_matched(rc, ["real", "erased", "idfree"], a.Bm)
    out[label] = res
    print("RESULT", label, json.dumps({k: res[k] for k in ("plain_cg_ref", "matched_default")}), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
print("CEHRGPT SCORE COMPLETE", f"{time.time() - t0:.0f}s", flush=True)
