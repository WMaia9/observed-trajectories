"""Protocol sweep for the official Delphi AUC evaluation (inference only, no retraining).

The official evaluate_auc.get_calibration_auc compares, per disease and 5 year age bracket, the model's
rate for that disease at the last input token at least `offset` days before the target event, between
cases (target = the disease) and controls (target = any other event of a person without the disease).
It has no horizon: the gap between the prediction token and the target may be decades. The paper's
longitudinal test (one year gap) and its horizon analysis are what bring the AUC from about 0.76 to 0.69.

This module runs the unchanged official pipeline for a grid of `offset` values (an argument the official
pipeline already exposes) and of horizons. A horizon needs the line the authors left commented out in
get_calibration_auc (`a *= (zk - z < 365.25)`): we install a copy of that function in which this line is
active with a configurable value. Everything else is the official code.

Writes, per configuration, df_both.parquet into OUT/<tag>/ and one aggregate JSON (means only, no counts)
to OUT/summary.json.
Usage: python -m delphi_aou.replication.sweep --input_path DATA --model_ckpt_path CKPT --out OUT
       [--erase_identity] [--grid offsets|horizons|both]
"""
import argparse, inspect, json, os, sys, textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from delphi_aou.replication.evaluate import OFFICIAL, EraseIdentity

OFFSETS = [0.1, 30.0, 90.0, 365.25]
HORIZONS = [365.25, 730.5, 1826.25, 3652.5]


def patched_calibration_auc(ev):
    """Return a copy of ev.get_calibration_auc whose commented horizon line is active, with the horizon
    taken from the module global HORIZON_DAYS (None = official behaviour, no horizon)."""
    src = textwrap.dedent(inspect.getsource(ev.get_calibration_auc))
    old = "        # a *= (zk - z < 365.25)"
    assert old in src, "official get_calibration_auc changed; horizon patch no longer applies"
    src = src.replace(old, "        if HORIZON_DAYS is not None:\n            a *= (zk - z < HORIZON_DAYS)")
    ns = dict(ev.__dict__); ns["HORIZON_DAYS"] = None
    exec(compile(src, "<patched get_calibration_auc>", "exec"), ns)
    return ns["get_calibration_auc"], ns


def summarize(df):
    w = df["n_diseased"].to_numpy(float)
    return {"n_diseases": int(df["token"].nunique()), "mean_auc": round(float(df["auc"].mean()), 4),
            "median_auc": round(float(df["auc"].median()), 4),
            "case_weighted_auc": round(float(np.average(df["auc"], weights=w)), 4)}


def grid(kind):
    g = []
    if kind in ("offsets", "both"):
        g += [(o, None) for o in OFFSETS]
    if kind in ("horizons", "both"):
        g += [(0.1, h) for h in HORIZONS] + [(365.25, 1826.25)]
    return g


def main():
    sys.path.insert(0, str(OFFICIAL))
    from model import Delphi, DelphiConfig
    from utils import get_p2i, get_batch
    import evaluate_auc as ev
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path"); ap.add_argument("--model_ckpt_path"); ap.add_argument("--out")
    ap.add_argument("--erase_identity", action="store_true", help="same as --control erased")
    ap.add_argument("--control", default="real", choices=["real", "erased", "shuffled", "agesex"])
    ap.add_argument("--settings", default="", help="subset, e.g. '0.1:open,365.25:open,0.1:365.25'")
    ap.add_argument("--grid", default="both", choices=["offsets", "horizons", "both"])
    ap.add_argument("--dataset_subset_size", type=int, default=-1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--official_loop", action="store_true", help="use the official pipeline once per setting (slow)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=512)
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    inp = str(Path(a.input_path).resolve())
    ck = torch.load(a.model_ckpt_path, map_location=a.device)
    model = Delphi(DelphiConfig(**ck["model_args"])); model.load_state_dict(ck["model"]); model.eval().to(a.device)
    if a.erase_identity:
        a.control = "erased"
    if a.control == "erased":
        model = EraseIdentity(model).eval()
    elif a.control == "shuffled":
        from delphi_aou.replication.controls import ShuffleIdentity
        model = ShuffleIdentity(model).eval().to(a.device)
    elif a.control == "agesex":
        from delphi_aou.replication.controls import AgeSexRates
        tr = np.fromfile(f"{inp}/train.bin", dtype=np.uint32).reshape(-1, 3)
        model = AgeSexRates(tr).eval().to(a.device)
    val = np.fromfile(f"{inp}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    p2i = get_p2i(val)
    n = len(p2i) if a.dataset_subset_size == -1 else a.dataset_subset_size
    d = get_batch(range(n), val, p2i, select="left", block_size=80, device=a.device, padding="random",
                  no_event_token_rate=5)
    labels = pd.read_csv(OFFICIAL / "delphi_labels_chapters_colours_icd.csv")
    fn, ns = patched_calibration_auc(ev)
    ev.get_calibration_auc = fn
    summ_path = out / "summary.json"
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    kind = a.control
    tag_of = lambda off, hor: f"{kind}_off{off:g}_hor{'open' if hor is None else f'{hor:g}'}"
    if not a.official_loop:
        from delphi_aou.replication import fast_eval
        g = grid(a.grid)
        if a.settings:
            g = [(float(o), None if h == "open" else float(h)) for o, h in (x.split(":") for x in a.settings.split(","))]
        todo = [c for c in g if tag_of(*c) not in summary]
        res = fast_eval.run(model, d, labels, todo, device=a.device, batch_size=a.batch_size,
                            workers=a.workers or None, horizon_fn=(fn, ns)) if todo else {}
        for ci, (off, hor) in enumerate(todo):
            unpooled, df_both = res[ci]
            tag = tag_of(off, hor); (out / tag).mkdir(exist_ok=True)
            df_both.to_parquet(out / tag / "df_both.parquet", index=False)
            unpooled.to_parquet(out / tag / "df_auc_unpooled.parquet", index=False)
            summary[tag] = {"offset_days": off, "horizon_days": hor, "fast": True, **summarize(df_both)}
            summ_path.write_text(json.dumps(summary, indent=1))
            print("RESULT", tag, json.dumps(summary[tag]), flush=True)
        print("SWEEP COMPLETE", flush=True)
        return
    cwd = os.getcwd(); os.chdir(OFFICIAL)
    try:
        for off, hor in grid(a.grid):
            tag = f"{'erased' if a.erase_identity else 'real'}_off{off:g}_hor{'open' if hor is None else f'{hor:g}'}"
            if tag in summary:
                print("skip", tag, flush=True); continue
            ns["HORIZON_DAYS"] = hor
            _, df_both = ev.evaluate_auc_pipeline(model, d, str(out / tag), labels, filter_min_total=100,
                                                     disease_chunk_size=200, device=a.device, seed=1337,
                                                     n_bootstrap=1, offset=off)
            summary[tag] = {"offset_days": off, "horizon_days": hor, **summarize(df_both)}
            summ_path.write_text(json.dumps(summary, indent=1))
            print("RESULT", tag, json.dumps(summary[tag]), flush=True)
    finally:
        os.chdir(cwd)
    print("SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
