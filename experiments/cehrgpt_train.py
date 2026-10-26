"""Train CEHR-GPT (cehrgpt 0.1.9) on the Delphi-2M (AoU) training people (P2, second model family).

Sequences come from the same train.bin as Delphi-2M (delphi_aou.cehrgpt.sequences); 5% of the training people
(crc32 of the person id) are held out for early stopping, the validation people of Delphi are never seen.
--erase trains the identity-free model (disease and death codes in the input become [OOV], labels unchanged).
The checkpoint with the lowest held-out loss is kept (as in the official Delphi train.py).
Writes <out>/best/ (HF model), <out>/log.json; the tokenizer is shared through --tokenizer.
Usage: python -u experiments/cehrgpt_train.py --out ~/cg_runs/full [--erase] [--steps 30000] [--smoke]
"""
import argparse, json, math, os, sys, time, zlib
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delphi_aou.cehrgpt.sequences import records
from delphi_aou.cehrgpt import model as M

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(Path.home() / "delphi_rep_data"))
ap.add_argument("--out", required=True)
ap.add_argument("--tokenizer", default=str(Path.home() / "cg_runs" / "tokenizer"))
ap.add_argument("--erase", action="store_true")
ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--steps", type=int, default=40000)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--warmup", type=int, default=1000)
ap.add_argument("--eval_every", type=int, default=1000)
ap.add_argument("--eval_batches", type=int, default=100)
ap.add_argument("--n_embd", type=int, default=128); ap.add_argument("--n_layer", type=int, default=8)
ap.add_argument("--n_head", type=int, default=8); ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--smoke", action="store_true", help="2000 people, 300 steps")
a = ap.parse_args()
if a.smoke:
    a.steps, a.eval_every, a.eval_batches, a.warmup = 300, 150, 5, 50
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
import datasets

t0 = time.time()
arr = np.fromfile(f"{a.data}/train.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
if a.smoke:
    pids = np.unique(arr[:, 0])[:2000]; arr = arr[np.isin(arr[:, 0], pids)]
pid_all = np.unique(arr[:, 0])
hold_ids = set(int(p) for p in pid_all if zlib.crc32(str(int(p)).encode()) % 100 < 5)


def gen(holdout, chunk=20000):
    """Records written straight to an Arrow cache (memory-mapped): a list of 480k dicts does not fit in 26 GB."""
    for s0 in range(0, len(pid_all), chunk):
        ids = pid_all[s0:s0 + chunk]
        sub = arr[(arr[:, 0] >= ids[0]) & (arr[:, 0] <= ids[-1])]
        for r in records(sub):
            if (int(r["person_id"]) in hold_ids) == holdout:
                yield r


cache = str(out / "hf_cache")
ds = datasets.DatasetDict({"train": datasets.Dataset.from_generator(gen, gen_kwargs={"holdout": False}, cache_dir=cache),
                           "validation": datasets.Dataset.from_generator(gen, gen_kwargs={"holdout": True}, cache_dir=cache)})
del arr
lens = np.concatenate([np.asarray(ds[s]["num_of_concepts"]) for s in ("train", "validation")])
stats = {"n_train_people": len(ds["train"]), "n_holdout_people": len(ds["validation"]),
         "seq_len_quantiles": {q: float(np.quantile(lens, q)) for q in (0.5, 0.9, 0.99)},
         "share_longer_than_max_len": float((lens > a.max_len).mean())}
print("DATA", json.dumps(stats), f"{time.time() - t0:.0f}s", flush=True)
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
if Path(a.tokenizer).exists():
    tok = CehrGptTokenizer.from_pretrained(a.tokenizer)
else:
    tok = M.train_tokenizer(ds); tok.save_pretrained(a.tokenizer)
vocab = M.Vocab(tok)
pds = M.tokenize(ds, tok, num_proc=a.workers)
print("TOKENIZED vocab", tok.vocab_size, f"{time.time() - t0:.0f}s", flush=True)
from cehrgpt.data.hf_cehrgpt_dataset_collator import CehrGptDataCollator
col = CehrGptDataCollator(tok, max_length=a.max_len, pretraining=True)
cols = ["input_ids", "ages", "epoch_times", "value_indicators", "values", "person_id", "concept_ids"]


class LengthGrouped(torch.utils.data.Sampler):
    """Batches of similar length (as HF group_by_length): shuffle, sort within mega-batches of 50 batches, shuffle the
    batches. Cuts the padding of the eager attention (T4 has no flash attention), not the data seen per epoch."""

    def __init__(self, lengths, bs, seed):
        self.l, self.bs, self.rng = np.minimum(np.asarray(lengths), a.max_len), bs, np.random.default_rng(seed)

    def __iter__(self):
        idx = self.rng.permutation(len(self.l)); mb = self.bs * 50; batches = []
        for s in range(0, len(idx), mb):
            c = idx[s:s + mb]; c = c[np.argsort(-self.l[c], kind="stable")]
            batches += [c[i:i + self.bs].tolist() for i in range(0, len(c) - self.bs + 1, self.bs)]
        for j in self.rng.permutation(len(batches)):
            yield batches[j]

    def __len__(self):
        return len(self.l) // self.bs


dl_tr = torch.utils.data.DataLoader(pds["train"], batch_sampler=LengthGrouped(pds["train"]["num_of_concepts"], a.bs, a.seed),
                                    collate_fn=col, num_workers=a.workers, persistent_workers=True)
dl_va = torch.utils.data.DataLoader(pds["validation"], batch_size=a.bs, shuffle=False, collate_fn=col,
                                    num_workers=a.workers, persistent_workers=True)
m = M.make_model(tok, a.n_embd, a.n_layer, a.n_head, a.max_len, a.dropout).to(dev)
n_par = sum(p.numel() for p in m.parameters())
print("MODEL params", n_par, "erase", a.erase, flush=True)
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.1)
sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
keep = ("input_ids", "attention_mask", "ages", "labels")


def evaluate():
    m.eval(); tot, n = 0.0, 0
    with torch.no_grad():
        for i, b in enumerate(dl_va):
            if i >= a.eval_batches:
                break
            b = {k: b[k].to(dev) for k in keep}
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                tot += M.lm_loss(m, b, vocab, a.erase).item(); n += 1
    m.train(); return tot / max(n, 1)


log, best, step, t1 = [], float("inf"), 0, time.time()
m.train()
while step < a.steps:
    for b in dl_tr:
        if step >= a.steps:
            break
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        b = {k: b[k].to(dev, non_blocking=True) for k in keep}
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            loss = M.lm_loss(m, b, vocab, a.erase)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt); scaler.update(); step += 1
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} {(time.time() - t1) / step:.3f}s/step", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            vl = evaluate()
            log.append({"step": step, "train_loss": float(loss.item()), "val_loss": vl})
            if vl < best:
                best = vl; m.save_pretrained(out / "best")
            print("EVAL", json.dumps(log[-1]), "best", round(best, 4), flush=True)
            (out / "log.json").write_text(json.dumps({"args": vars(a), "stats": stats, "params": n_par,
                                                      "best_val_loss": best, "log": log}, indent=1))
print("TRAIN COMPLETE", f"{time.time() - t0:.0f}s", flush=True)
