"""Run the official evaluate_auc_pipeline on the All of Us validation file.

The upstream evaluate_auc.py main() (commit fb72166) passes health_token_replacement_prob to
utils.get_batch, whose signature does not accept it, so main() raises a TypeError. This wrapper does
exactly what main() does, without that argument (its default, 0.0, means no replacement anyway).
Nothing else differs: same block_size=80, select='left', padding='random', seed 1337, labels file,
filter_min_total=100 and disease chunking.
Usage: python -m delphi_aou.replication.evaluate --input_path DATA --output_path OUT --model_ckpt_path CKPT
"""
import argparse, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OFFICIAL = Path(__file__).resolve().parents[3] / "third_party" / "delphi_official"


class EraseIdentity(torch.nn.Module):
    """Control experiment: the model sees when records happened (ages) and sex, but every disease and death
    token in its input is replaced by the NO_EVENT token (id 1). Targets are untouched, so cases and
    controls are exactly the same as in the real evaluation. Measures how much AUC comes from timing alone."""

    def __init__(self, model, token=1):
        super().__init__(); self.m = model; self.token = token

    def forward(self, x, ages, *rest):
        return self.m(torch.where((x >= 13) & (x < 1270), torch.full_like(x, self.token), x), ages, *rest)


def main():
    sys.path.insert(0, str(OFFICIAL))
    from model import Delphi, DelphiConfig
    from utils import get_p2i, get_batch
    from evaluate_auc import evaluate_auc_pipeline
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path"); ap.add_argument("--output_path"); ap.add_argument("--model_ckpt_path")
    ap.add_argument("--no_event_token_rate", type=int, default=5)
    ap.add_argument("--dataset_subset_size", type=int, default=-1)
    ap.add_argument("--n_bootstrap", type=int, default=1)
    ap.add_argument("--filter_min_total", type=int, default=100)
    ap.add_argument("--disease_chunk_size", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--erase_identity", action="store_true", help="control: hide disease identities from the model input")
    a = ap.parse_args()
    a.output_path = str(Path(a.output_path).resolve()); a.input_path = str(Path(a.input_path).resolve())
    Path(a.output_path).mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.model_ckpt_path, map_location=a.device)
    model = Delphi(DelphiConfig(**ck["model_args"])); model.load_state_dict(ck["model"]); model.eval().to(a.device)
    if a.erase_identity:
        model = EraseIdentity(model).eval()
    val = np.fromfile(f"{a.input_path}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    p2i = get_p2i(val)
    n = len(p2i) if a.dataset_subset_size == -1 else a.dataset_subset_size
    d = get_batch(range(n), val, p2i, select="left", block_size=80, device=a.device, padding="random",
                  no_event_token_rate=a.no_event_token_rate)
    labels = pd.read_csv(OFFICIAL / "delphi_labels_chapters_colours_icd.csv")
    cwd = os.getcwd(); os.chdir(OFFICIAL)
    try:
        evaluate_auc_pipeline(model, d, a.output_path, labels, diseases_of_interest=None,
                              filter_min_total=a.filter_min_total, disease_chunk_size=a.disease_chunk_size,
                              device=a.device, seed=1337, n_bootstrap=a.n_bootstrap)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
