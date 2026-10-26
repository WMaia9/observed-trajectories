"""Identity-free Delphi-2M trained from scratch (P2 control; answers 'erasure is out of distribution').

Runs the authors' unchanged train.py (third_party/delphi_official), with one difference applied from outside the
vendored files: every input token >= 13 (diseases and death) is replaced by the no-event token (id 1) before the
model sees it, during training and validation. Targets are untouched, so the model is still trained to predict
the next disease identity and its time, but from ages, sex, lifestyle tokens and the calendar of records only.
This is the same input the evaluation-time erasure control (evaluate.EraseIdentity) gives a model trained on real
sequences; here the model is trained on that input distribution.

Usage (from anywhere): python -m delphi_aou.replication.train_identity_free config/train_delphi.py \
    --dataset=/path/to/data --out_dir=/path/to/ckpt --device=cuda [other train.py overrides]
Evaluate with: python -m delphi_aou.replication.evaluate ... --erase_identity (same erasure at test time).

Record-token variant (--record_token=1270): erased diseases become a dedicated input token instead of the no-event
token, so the model can tell a real (erased) record from the random no-event padding of the official code. The
vocabulary grows by one (vocab_size=1271) and the new token is added to ignore_tokens, so it is an input only, never a
target, exactly like the sex and lifestyle tokens of the official configuration."""
import os
import runpy
import sys
from pathlib import Path

import torch

OFFICIAL = Path(__file__).resolve().parents[3] / "third_party" / "delphi_official"
FIRST_DISEASE = 13
N_OFFICIAL_VOCAB = 1270
OFFICIAL_IGNORE = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
TOKEN = {"id": 1}


def erase(idx, token=None):
    """Diseases and death (ids >= 13) become no event (id 1) or the record token; padding, no event, sex, lifestyle kept.
    The record token itself (>= 1270) is never produced by the data, so it is left as is if it ever appears."""
    t = TOKEN["id"] if token is None else token
    return torch.where((idx >= FIRST_DISEASE) & (idx < N_OFFICIAL_VOCAB), torch.full_like(idx, t), idx)


def record_token_overrides(argv):
    """Strip --record_token=K from argv; return (argv for train.py, K or None) with vocab and ignore overrides added."""
    rest, tok = [], None
    for a in argv:
        if a.startswith("--record_token="):
            tok = int(a.split("=", 1)[1])
        else:
            rest.append(a)
    if tok is None:
        return rest, None
    assert tok == N_OFFICIAL_VOCAB, "the record token must be the first id after the official vocabulary"
    ign = ",".join(str(i) for i in OFFICIAL_IGNORE + [tok])
    return rest + [f"--vocab_size={tok + 1}", f"--ignore_tokens=[{ign}]"], tok


def patch_model():
    sys.path.insert(0, str(OFFICIAL))
    import model as official_model
    orig = official_model.Delphi.forward
    if getattr(orig, "_identity_free", False):
        return official_model
    seen = {"n": 0}

    def forward(self, idx, age, *rest, **kw):
        if seen["n"] == 0:
            print(f"IDENTITY_FREE: erased {int(((idx >= FIRST_DISEASE) & (idx < N_OFFICIAL_VOCAB)).sum())} disease tokens "
                  f"in the first batch to token {TOKEN['id']}", flush=True)
        seen["n"] += 1
        return orig(self, erase(idx), age, *rest, **kw)

    forward._identity_free = True
    official_model.Delphi.forward = forward
    return official_model


def main(argv):
    argv, tok = record_token_overrides(argv)
    if tok is not None:
        TOKEN["id"] = tok
    patch_model()
    os.chdir(OFFICIAL)
    sys.argv = ["train.py"] + list(argv)
    runpy.run_path(str(OFFICIAL / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
