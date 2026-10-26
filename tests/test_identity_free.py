"""Identity-free Delphi-2M (P2 control): erasure rule and a short training run of the official code."""
import subprocess
import sys
import pytest
from test_replication import tdb, synthetic

def test_identity_free_erase_rule():
    import torch
    from delphi_aou.replication.train_identity_free import erase
    x = torch.tensor([[0, 1, 2, 12, 13, 500, 1269]])
    assert erase(x).tolist() == [[0, 1, 2, 12, 1, 1, 1]]
    assert erase(x, token=1270).tolist() == [[0, 1, 2, 12, 1270, 1270, 1270]]


def test_record_token_overrides():
    from delphi_aou.replication.train_identity_free import record_token_overrides
    argv, tok = record_token_overrides(["config/train_delphi.py", "--record_token=1270", "--seed=42"])
    assert tok == 1270 and "--vocab_size=1271" in argv and "--seed=42" in argv
    assert "--ignore_tokens=[0,2,3,4,5,6,7,8,9,10,11,12,1270]" in argv
    assert record_token_overrides(["a", "--seed=1"]) == (["a", "--seed=1"], None)


def test_erase_wrappers_record_token():
    import torch
    from delphi_aou.replication.evaluate import EraseIdentity
    from delphi_aou.replication.audit import EraseIdentityAndLifestyle
    seen = {}
    class M(torch.nn.Module):
        def forward(self, x, a):
            seen["x"] = x.tolist(); return x
    x = torch.tensor([[0, 1, 2, 5, 13, 1269]])
    EraseIdentity(M())(x, None); assert seen["x"] == [[0, 1, 2, 5, 1, 1]]
    EraseIdentity(M(), 1270)(x, None); assert seen["x"] == [[0, 1, 2, 5, 1270, 1270]]
    EraseIdentityAndLifestyle(M())(x, None); assert seen["x"] == [[0, 1, 2, 1, 1, 1]]
    EraseIdentityAndLifestyle(M(), 1270)(x, None); assert seen["x"] == [[0, 1, 2, 1, 1270, 1270]]


@pytest.mark.slow
def test_identity_free_training_runs(tmp_path):
    rows, _ = tdb.build_rows(*synthetic())
    tr, va = tdb.official_order_and_split(rows)
    data = tdb.write_dataset(tr, va, tmp_path / "aou_test")
    out = tmp_path / "ckpt"
    cmd = [sys.executable, "-m", "delphi_aou.replication.train_identity_free", "config/train_delphi.py",
           f"--dataset={data.resolve()}", f"--out_dir={out}", "--device=cpu", "--max_iters=40", "--lr_decay_iters=40",
           "--eval_interval=20", "--eval_iters=2", "--warmup_iters=5", "--always_save_checkpoint=True", "--batch_size=32"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "IDENTITY_FREE: erased" in r.stdout and (out / "ckpt.pt").exists()
    n = int(r.stdout.split("IDENTITY_FREE: erased ")[1].split()[0])
    assert n > 0


@pytest.mark.slow
def test_record_token_training_runs(tmp_path):
    import torch
    rows, _ = tdb.build_rows(*synthetic())
    tr, va = tdb.official_order_and_split(rows)
    data = tdb.write_dataset(tr, va, tmp_path / "aou_test")
    out = tmp_path / "ckpt"
    cmd = [sys.executable, "-m", "delphi_aou.replication.train_identity_free", "config/train_delphi.py",
           f"--dataset={data.resolve()}", f"--out_dir={out}", "--device=cpu", "--max_iters=40", "--lr_decay_iters=40",
           "--eval_interval=20", "--eval_iters=2", "--warmup_iters=5", "--always_save_checkpoint=True", "--batch_size=32",
           "--record_token=1270"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "to token 1270" in r.stdout
    ck = torch.load(out / "ckpt.pt", map_location="cpu")
    assert ck["model_args"]["vocab_size"] == 1271 and 1270 in ck["model_args"]["ignore_tokens"]
