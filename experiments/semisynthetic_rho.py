"""Does the timing share rho measure what it should? Semi-synthetic worlds with known structure (P2 review item R10).

Each person has an activity level A (lognormal). Disease k is recorded with hazard
    lambda_k(t) = r_k * age_factor(t) * A^tau * exp(gamma * has_parent_k(t)),
so tau sets how much the record calendar (through A) predicts every disease, and gamma how much disease identity
(the specific parent disease) adds. The official Delphi-2M code is trained on each world (through our converter, exactly
as on All of Us), and rho = (AUC_erased - 0.5) / (AUC_real - 0.5) is computed with the same controls and fast evaluation
as in the paper. Expected: rho near 1 when gamma = 0 (identity carries nothing), lower as gamma grows at fixed tau, and
the erased AUC near the age-sex floor when tau = 0 and gamma = 0.

Usage: python experiments/semisynthetic_rho.py --grid "1:0,1:1,1:2,0.5:2,0:2" --n 20000 --iters 3000 --device cuda
Writes ~/results/semisynthetic_rho.json (synthetic data only, nothing from All of Us).
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd, torch
from delphi_aou.replication import to_delphi_bin as tdb
from delphi_aou.replication.evaluate import OFFICIAL, EraseIdentity

DAY = 365.25


def world(n, tau, gamma, K=40, seed=0, step=0.25):
    r = np.random.default_rng(seed)
    sys.path.insert(0, str(OFFICIAL))
    import evaluate_auc as ev
    labels = pd.read_csv(OFFICIAL / "delphi_labels_chapters_colours_icd.csv")
    dis = ev.get_common_diseases(labels, 100)
    inv = {v: k for k, v in tdb.label_dict().items()}
    codes = [inv[int(i) - 1] for i in r.choice(dis, K, replace=False)]
    rate = r.uniform(0.004, 0.02, K)
    parent = np.array([r.choice([j for j in range(K) if j != k]) for k in range(K)])
    pid = np.arange(1000, 1000 + n)
    birth = pd.Timestamp("1935-01-01") + pd.to_timedelta(r.integers(0, 365 * 25, n), unit="D")
    entry = r.uniform(40, 55, n); exit_ = entry + r.uniform(10, 25, n)
    A = np.exp(r.normal(0, 0.8, n))
    has = np.zeros((n, K), bool); rows = []
    t = entry.copy()
    while (t < exit_).any():
        alive = t < exit_
        lam = rate[None, :] * (1.03 ** (t[:, None] - 50)) * (A[:, None] ** tau) * np.exp(gamma * has[:, parent])
        p = 1 - np.exp(-lam * step)
        new = (r.random((n, K)) < p) & ~has & alive[:, None]
        for i, k in zip(*np.nonzero(new)):
            rows.append((pid[i], codes[k], birth[i] + pd.Timedelta(days=float((t[i] + r.uniform(0, step)) * DAY))))
        has |= new
        t = t + step
    icd = pd.DataFrame(rows, columns=["person_id", "code3", "date"])
    persons = pd.DataFrame({"person_id": pid, "birth_date": birth, "sex": r.choice(["Female", "Male"], n)})
    death = pd.DataFrame({"person_id": [], "date": []})
    at = birth + pd.to_timedelta(50 * 365, unit="D")
    bmi = pd.DataFrame({"person_id": pid, "date": at, "value": r.normal(27, 4, n)})
    smoking = pd.DataFrame({"person_id": pid, "date": at, "answer": r.choice(["Every Day", "Some Days", "Not At All"], n)})
    alcohol = pd.DataFrame({"person_id": pid, "date": at, "answer": r.choice(["Never", "2 to 3 Times a Week"], n)})
    return persons, icd, death, bmi, smoking, alcohol


def world_detect(n, tau, gamma, K=40, seed=0, step=0.25, visit_rate=0.7, p_detect=0.2):
    """Separate onset and recording. True onset of disease k has hazard r_k * age factor * exp(gamma * [parent onset]),
    independent of the activity level A. A disease is recorded only at a visit: visits during the EHR window follow a
    Poisson process with rate visit_rate * A^tau per year, and each visit detects each undiagnosed onset with probability
    p_detect. Onsets start at age 30, before EHR entry, so prevalent conditions are recorded in a burst after entry (left
    truncation). tau is now a pure observation-process effect: it changes when diseases are recorded, never whether or
    when they begin."""
    r = np.random.default_rng(seed)
    sys.path.insert(0, str(OFFICIAL))
    import evaluate_auc as ev
    labels = pd.read_csv(OFFICIAL / "delphi_labels_chapters_colours_icd.csv")
    dis = ev.get_common_diseases(labels, 100)
    inv = {v: k for k, v in tdb.label_dict().items()}
    codes = [inv[int(i) - 1] for i in r.choice(dis, K, replace=False)]
    rate = r.uniform(0.004, 0.02, K)
    parent = np.array([r.choice([j for j in range(K) if j != k]) for k in range(K)])
    pid = np.arange(1000, 1000 + n)
    birth = pd.Timestamp("1935-01-01") + pd.to_timedelta(r.integers(0, 365 * 25, n), unit="D")
    entry = r.uniform(40, 55, n); exit_ = entry + r.uniform(10, 25, n)
    A = np.exp(r.normal(0, 0.8, n))
    onset = np.zeros((n, K), bool); rec = np.zeros((n, K), bool); rows = []
    t = np.full(n, 30.0)
    while (t < exit_).any():
        alive = t < exit_
        lam = rate[None, :] * (1.03 ** (t[:, None] - 50)) * np.exp(gamma * onset[:, parent])
        new = (r.random((n, K)) < 1 - np.exp(-lam * step)) & ~onset & alive[:, None]
        onset |= new
        inwin = alive & (t >= entry)
        nv = r.poisson(visit_rate * A ** tau * step) * inwin
        pdet = 1 - (1 - p_detect) ** nv
        det = onset & ~rec & (r.random((n, K)) < pdet[:, None])
        for i, k in zip(*np.nonzero(det)):
            rows.append((pid[i], codes[k], birth[i] + pd.Timedelta(days=float((t[i] + r.uniform(0, step)) * DAY))))
        rec |= det
        t = t + step
    icd = pd.DataFrame(rows, columns=["person_id", "code3", "date"])
    persons = pd.DataFrame({"person_id": pid, "birth_date": birth, "sex": r.choice(["Female", "Male"], n)})
    death = pd.DataFrame({"person_id": [], "date": []})
    at = birth + pd.to_timedelta(50 * 365, unit="D")
    bmi = pd.DataFrame({"person_id": pid, "date": at, "value": r.normal(27, 4, n)})
    smoking = pd.DataFrame({"person_id": pid, "date": at, "answer": r.choice(["Every Day", "Some Days", "Not At All"], n)})
    alcohol = pd.DataFrame({"person_id": pid, "date": at, "answer": r.choice(["Never", "2 to 3 Times a Week"], n)})
    return persons, icd, death, bmi, smoking, alcohol


def train_world(tau, gamma, n, iters, device, work, seed=0, layers=4, embd=64, mode="frailty", visit_rate=0.7,
                p_detect=0.2, trainer="official"):
    """Build one world, train the official code on it, return (model, validation array, number of records).
    seed sets both the world and the official training seed (train.py --seed). trainer: "official" (train.py),
    "idfree" (identity-free, erased records share the no-event token) or "idrec" (identity-free with a record token)."""
    if mode == "frailty":
        persons = world(n, tau, gamma, seed=seed)
    else:
        persons = world_detect(n, tau, gamma, seed=seed, visit_rate=visit_rate, p_detect=p_detect)
    rows, _ = tdb.build_rows(*persons)
    tr, va = tdb.official_order_and_split(rows)
    tag = f"{mode}_{tau}_{gamma}_{seed}_{layers}x{embd}_{visit_rate}_{p_detect}"
    data = tdb.write_dataset(tr, va, work / f"w_{tag}")
    out = work / f"ck_{tag}" if trainer == "official" else work / f"ck_{tag}_{trainer}"
    entry = ["train.py"] if trainer == "official" else ["-m", "delphi_aou.replication.train_identity_free"]
    cmd = [sys.executable, *entry, "config/train_delphi.py", f"--dataset={data.resolve()}", f"--out_dir={out}",
           f"--device={device}", f"--max_iters={iters}", f"--lr_decay_iters={iters}", f"--eval_interval={max(iters // 2, 1)}",
           "--eval_iters=10", f"--warmup_iters={min(200, iters // 10)}", "--always_save_checkpoint=True", "--batch_size=128",
           f"--n_layer={layers}", f"--n_head=4", f"--n_embd={embd}", "--compile=False", f"--seed={42 + seed}"]
    if trainer == "idrec":
        cmd.append("--record_token=1270")
    r = subprocess.run(cmd, cwd=OFFICIAL, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    sys.path.insert(0, str(OFFICIAL))
    from model import Delphi, DelphiConfig
    ck = torch.load(out / "ckpt.pt", map_location=device)
    m = Delphi(DelphiConfig(**ck["model_args"])); m.load_state_dict(ck["model"]); m.eval().to(device)
    return m, va.astype(np.int64), len(rows)


def run_world(tau, gamma, n, iters, device, work, seed=0, layers=4, embd=64):
    m, val, nrec = train_world(tau, gamma, n, iters, device, work, seed, layers, embd)
    from utils import get_p2i, get_batch
    from delphi_aou.replication import fast_eval
    p2i = get_p2i(val)
    np.random.seed(1337); torch.manual_seed(1337)
    d = get_batch(range(len(p2i)), val, p2i, select="left", block_size=80, device=device, padding="random",
                  no_event_token_rate=5)
    labels = pd.read_csv(OFFICIAL / "delphi_labels_chapters_colours_icd.csv")
    res = {}
    for name, model in [("real", m), ("erased", EraseIdentity(m).eval())]:
        df = fast_eval.run(model, d, labels, [(0.1, None)], device=device, workers=4, batch_size=512)[0][1]
        res[name] = float(df["auc"].mean()); res[f"n_diseases_{name}"] = int(len(df))
    res["rho"] = (res["erased"] - 0.5) / (res["real"] - 0.5)
    res.update(tau=tau, gamma=gamma, n=n, iters=iters, records=int(nrec))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="1:0,1:1,1:2,0.5:2,0:2")
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(Path.home() / "results" / "semisynthetic_rho.json"))
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="semisyn_"))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    allr = json.loads(out.read_text()) if out.exists() else []
    for g in a.grid.split(","):
        tau, gamma = (float(x) for x in g.split(":"))
        res = run_world(tau, gamma, a.n, a.iters, a.device, work)
        allr.append(res); out.write_text(json.dumps(allr, indent=1))
        print("RESULT", json.dumps(res), flush=True)
    print("SEMISYNTHETIC COMPLETE", flush=True)
