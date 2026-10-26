"""Frozen configuration, deterministic person level split, and result writing.

Nothing in the evaluation code may contain typed in metric values. Every number
that reaches a report must be produced by code and written with write_results,
which stamps the config hash and the split hash.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def config_hash(cfg):
    blob = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def assign_split(person_ids, cfg=None):
    """Deterministic split from a hash of (salt, person_id).

    Does not depend on row order, on the cohort size, or on any random state, so
    notebooks 01, 02 and 03 cannot disagree. Returns an array of
    'train', 'val' or 'test'.
    """
    cfg = cfg or load_config()
    s = cfg["split"]
    t_train, t_val = s["train"], s["train"] + s["val"]
    out = np.empty(len(person_ids), dtype=object)
    for i, pid in enumerate(person_ids):
        h = hashlib.sha256(f"{s['salt']}:{int(pid)}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2**64
        out[i] = "train" if u < t_train else ("val" if u < t_val else "test")
    return out


def split_hash(person_ids, splits):
    """Fingerprint of the assignment. Assert this is identical across notebooks."""
    order = np.argsort(np.asarray(person_ids, dtype=np.int64))
    pid = np.asarray(person_ids, dtype=np.int64)[order]
    sp = np.asarray(splits)[order]
    h = hashlib.sha256()
    h.update(pid.tobytes())
    code = {"train": "0", "val": "1", "test": "2"}
    h.update("".join(code[x] for x in sp).encode())
    return h.hexdigest()[:12]


def write_results(path, results, cfg=None, splits_fingerprint=None):
    """Write results as JSON stamped with the config hash and split hash."""
    cfg = cfg or load_config()
    payload = {
        "config_name": cfg["name"],
        "config_hash": config_hash(cfg),
        "split_hash": splits_fingerprint,
        "results": results,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    return payload
