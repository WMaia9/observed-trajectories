"""CEHR-GPT (cehrgpt 0.1.9) training and read-out on Delphi-2M (AoU) data (P2, second model family).

The package's tokenizer, tokenization mapping, collator and CEHRGPT2LMHeadModel (GPT-2 backbone, package defaults:
no positional encoding, no value or time-to-visit heads) are used unchanged; we only write the training loop, the
identity-free input and the read-out.

Identity-free model: the disease and death codes (Delphi tokens 13..1269) in the INPUT become the tokenizer's [OOV]
token, which never occurs otherwise and is never a target; the labels keep the original codes. Lifestyle codes,
the demographic prompt, [VS]/[VE], the visit type and the artificial time tokens are kept, as in the Delphi
identity-free model.

Read-out at a prediction point z (the score of disease k): the prefix is the record up to and including the last
visit on or before z. At its [VE] the model predicts the time token to the next visit; we group the time tokens into
8 bins, drop the part of each bin that lies before z (the next visit must come after z), and for one representative
token per bin feed [ATT, VS, 9202] and read the distribution of the first code of that visit (renormalized over code
tokens). The score is the bin-weighted mixture. Without a visit before z the prefix is the demographic prompt with
the age at z, followed by [VS, 9202]. The model has no positional encoding, so continuing from cached keys and
values with the later cached positions masked is exactly the same as running the truncated prefix (tested).
"""
from typing import Dict, List, Optional, Sequence, Tuple
import math
import numpy as np
import torch

from .sequences import to_record, last_visit_day, VISIT_TYPE

ERASE_LO, ERASE_HI = 13, 1269
BIN_EDGES = [1, 7, 30, 90, 180, 365, 730, 1081]              # days; last bin [730, 1081) then LT (> 1080 days)
BIN_REP = ["D3", "D14", "D60", "D120", "D270", "D540", "D900", "LT"]


def train_tokenizer(ds_dict):
    import datasets
    from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
    from cehrgpt.models.pretrained_embeddings import PretrainedEmbeddings
    from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
    da = DataTrainingArguments(data_folder=".", dataset_prepared_path=".", preprocessing_num_workers=1,
                               preprocessing_batch_size=256)
    both = datasets.concatenate_datasets([ds_dict["train"], ds_dict["validation"]])
    return CehrGptTokenizer.train_tokenizer(both, {}, da, PretrainedEmbeddings(None), min_prevalence=0.0)


def tokenize(ds_dict, tok, num_proc=1):
    from cehrgpt.data.hf_cehrgpt_dataset import create_cehrgpt_pretraining_dataset
    from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
    da = DataTrainingArguments(data_folder=".", dataset_prepared_path=".", preprocessing_num_workers=num_proc,
                               preprocessing_batch_size=256)
    return create_cehrgpt_pretraining_dataset(ds_dict, tok, da)


def make_model(tok, n_embd=128, n_layer=8, n_head=8, n_positions=512, dropout=0.1):
    from cehrgpt.models.config import CEHRGPTConfig
    from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
    cfg = CEHRGPTConfig(vocab_size=tok.vocab_size, value_vocab_size=tok.value_vocab_size,
                        time_token_vocab_size=tok.time_token_vocab_size, bos_token_id=tok.end_token_id,
                        eos_token_id=tok.end_token_id, lab_token_ids=tok.lab_token_ids,
                        token_to_time_token_mapping=tok.token_to_time_token_mapping, n_embd=n_embd, n_layer=n_layer,
                        n_head=n_head, n_positions=n_positions, resid_pdrop=dropout, embd_pdrop=dropout,
                        attn_pdrop=dropout)
    return CEHRGPT2LMHeadModel(cfg)


class Vocab:
    """Token-id helpers for one trained tokenizer."""

    def __init__(self, tok):
        v = tok.get_vocab(); self.tok = tok
        self.oov = tok._oov_token_id; self.pad = tok.pad_token_id
        self.code_of_k = np.full(1270, -1, np.int64)
        for name, i in v.items():
            if name.startswith("t") and name[1:].isdigit():
                self.code_of_k[int(name[1:])] = i
        self.code_ids = np.flatnonzero(np.isin(np.arange(tok.vocab_size), self.code_of_k[self.code_of_k >= 0]))
        lut = np.zeros(tok.vocab_size, bool)
        for k in range(ERASE_LO, ERASE_HI + 1):
            if self.code_of_k[k] >= 0:
                lut[self.code_of_k[k]] = True
        self.erase_lut = torch.as_tensor(lut)
        att = [(int(n[1:]), i) for n, i in v.items() if n.startswith("D") and n[1:].isdigit()]
        if "LT" in v:
            att.append((10 ** 6, v["LT"]))
        att.sort()
        self.att_days = np.array([d for d, _ in att], np.float64); self.att_ids = np.array([i for _, i in att])
        self.rep_ids = [v.get(r, v["LT"]) for r in BIN_REP]
        self.vs, self.vt = v["[VS]"], v[VISIT_TYPE]

    def erase(self, ids: torch.Tensor) -> torch.Tensor:
        lut = self.erase_lut.to(ids.device)
        return torch.where(lut[ids], torch.full_like(ids, self.oov), ids)

    def encode(self, rec) -> List[int]:
        return self.tok.encode(rec["concept_ids"])


def lm_loss(model, batch, vocab: Optional[Vocab] = None, erase=False):
    ids = batch["input_ids"]
    inp = vocab.erase(ids) if erase else ids
    out = model(input_ids=inp, attention_mask=batch["attention_mask"], ages=batch["ages"])
    logits = out.logits[:, :-1].float(); lab = batch["labels"][:, 1:]
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1), ignore_index=-100)


# ------------------------------------------------------------------------------------------------ read-out
def _pad(seqs, pad):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long); am = torch.zeros((len(seqs), L))
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.as_tensor(s); am[i, :len(s)] = 1
    return ids, am


def bin_weights(p_att: np.ndarray, att_days: np.ndarray, gmin: np.ndarray) -> np.ndarray:
    """p_att [n, A] probabilities of the time tokens (sorted by days), gmin [n] smallest admissible gap (days).
    Returns [n, 8] weights of the bins, after removing tokens with days < gmin, normalized to sum one."""
    edges = BIN_EDGES + [10 ** 7]
    W = np.zeros((len(p_att), len(edges) - 1))
    for j in range(len(edges) - 1):
        inb = (att_days >= edges[j]) & (att_days < edges[j + 1])
        ok = inb[None, :] & (att_days[None, :] >= gmin[:, None])
        W[:, j] = (p_att * ok).sum(1)
    s = W.sum(1, keepdims=True)
    W = np.where(s > 0, W / np.where(s > 0, s, 1), 0)
    W[s[:, 0] == 0, -1] = 1.0                                    # nothing admissible: all weight on LT
    return W


@torch.no_grad()
def code_dists_after(model, vocab: Vocab, seqs: List[List[int]], ends: List[List[int]], branches: List[List[int]],
                     device, erase=False, chunk=1024):
    """For each person sequence seqs[i] and each prefix end e in ends[i] (inclusive index), return
      p_next [sum ends, V]: next-token distribution at position e (from the full-sequence pass), and
      q [sum ends, len(branches), n_codes]: distribution over code tokens after appending each branch at e.
    Uses the cached keys/values of seqs with the positions after e masked (exact: no positional encoding)."""
    ids, am = _pad(seqs, vocab.pad)
    ids, am = ids.to(device), am.to(device)
    inp = vocab.erase(ids) if erase else ids
    out = model(input_ids=inp, attention_mask=am, ages=torch.zeros_like(ids), use_cache=True)
    pres = out.past_key_values
    rows = [(i, e) for i in range(len(seqs)) for e in ends[i]]
    pi = torch.as_tensor([r[0] for r in rows], device=device); pe = torch.as_tensor([r[1] for r in rows], device=device)
    p_next = torch.softmax(out.logits[pi, pe].float(), -1)
    code_ids = torch.as_tensor(vocab.code_ids, device=device)
    nb = len(branches)
    q = torch.zeros((len(rows), nb, len(code_ids)), device=device)
    pe_np = pe.cpu().numpy()
    step = max(1, chunk // nb)
    # group the prefixes by end position, slice the cache to that length (all positions <= e are real tokens, so no
    # attention mask is needed; the right padding of shorter sequences lies after e)
    for e in np.unique(pe_np):
        sel_all = np.flatnonzero(pe_np == e)
        for s0 in range(0, len(sel_all), step):
            sel = torch.as_tensor(sel_all[s0:s0 + step], device=device)
            bi = pi[sel].repeat_interleave(nb)
            br = torch.as_tensor(branches, device=device).repeat(len(sel), 1)
            past = tuple((k[bi, :, :e + 1], v[bi, :, :e + 1]) for k, v in pres)
            o = model(input_ids=br, past_key_values=past, ages=torch.zeros_like(br), use_cache=False)
            lg = o.logits[:, -1].float()[:, code_ids]
            q[sel] = torch.softmax(lg, -1).view(-1, nb, len(code_ids))
    return p_next.cpu().numpy(), q.cpu().numpy()


def bin_weights_fast(p_att: np.ndarray, att_days: np.ndarray, rows: np.ndarray, gmin: np.ndarray) -> np.ndarray:
    """Vectorized bin_weights for many points: p_att [R, A] per prefix, rows [n] prefix index of each point."""
    edges = np.array(BIN_EDGES + [10 ** 7], np.float64)
    tail = np.concatenate([np.cumsum(p_att[:, ::-1], 1)[:, ::-1], np.zeros((len(p_att), 1))], 1)
    lo = np.searchsorted(att_days, edges[:-1], "left"); hi = np.searchsorted(att_days, edges[1:], "left")
    s = np.searchsorted(att_days, gmin, "left")
    start = np.maximum(lo[None, :], s[:, None])
    W = np.clip(tail[rows[:, None], np.minimum(start, len(att_days))] - tail[rows[:, None], hi[None, :]], 0, None)
    W = np.where(start < hi[None, :], W, 0.0)
    tot = W.sum(1, keepdims=True)
    W = np.where(tot > 0, W / np.where(tot > 0, tot, 1), 0.0)
    W[tot[:, 0] == 0, -1] = 1.0
    return W


@torch.no_grad()
def score_people(model, vocab: Vocab, people: Sequence[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                 device, erase=False):
    """people: list of (pid, ages, tokens, z [n], k [n]) for the points of one batch row (ages/tokens of that row).
    Returns a list of score arrays [n]: the bin-weighted probability that k is the first code of the next visit."""
    kcol = np.full(1270, -1, np.int64)
    pos_of = {int(c): j for j, c in enumerate(vocab.code_ids)}
    for k in range(1270):
        if vocab.code_of_k[k] >= 0:
            kcol[k] = pos_of[int(vocab.code_of_k[k])]
    seqs, ends, per = [], [], []
    prompts = {}
    for pid, a, t, z, k in people:
        rec = to_record(pid, a, t)
        info = {"lv": np.full(len(z), -1), "vd": None, "seq": None, "pkey": None}
        if rec is not None:
            ve = np.array([i for i, c in enumerate(rec["concept_ids"]) if c == "[VE]"])
            vd = np.array([rec["epoch_times"][i] / 86400.0 for i in ve])
            lv = np.searchsorted(vd, np.asarray(z, np.float64), "right") - 1
            need = np.unique(lv[lv >= 0])
            if len(need):
                seqs.append(vocab.encode(rec)); ends.append(list(ve[need]))
                pos = np.full(len(ve), -1); pos[need] = np.arange(len(need))
                info.update(lv=lv, vd=vd, seq=len(seqs) - 1, pos=pos)
            else:
                info.update(lv=lv)
        pre = info["lv"] < 0
        if pre.any():
            ages_y = (np.asarray(z)[pre] // 365.25).astype(int)
            ua, inv = np.unique(ages_y, return_inverse=True)
            ukeys = []
            for ay in ua:
                pr = to_record(pid, a, t, upto=-1.0, prompt_age_day=float(ay) * 365.25 + 1)
                key = tuple(vocab.encode(pr)[:4])
                prompts.setdefault(key, len(prompts)); ukeys.append(prompts[key])
            info["pkey"] = np.array(ukeys)[inv]
        per.append(info)
    results = [np.zeros(len(p[3]), np.float32) for p in people]
    if seqs:
        branches = [[r, vocab.vs, vocab.vt] for r in vocab.rep_ids]
        p_next, q = code_dists_after(model, vocab, seqs, ends, branches, device, erase=erase)
        p_att = p_next[:, vocab.att_ids]
        p_att = p_att / np.maximum(p_att.sum(1, keepdims=True), 1e-12)
        off = np.cumsum([0] + [len(e) for e in ends])
        for (pid, a, t, z, k), info, res in zip(people, per, results):
            if info["seq"] is None:
                continue
            m = info["lv"] >= 0
            lv = info["lv"][m]; zz = np.asarray(z, np.float64)[m]
            r = off[info["seq"]] + info["pos"][lv]
            g = np.maximum(1.0, np.ceil(zz - info["vd"][lv]))
            W = bin_weights_fast(p_att, vocab.att_days, r, g)
            kc = kcol[np.asarray(k)[m].astype(int)]
            qk = q[r, :, np.maximum(kc, 0)]
            res[m] = np.where(kc >= 0, (W * qk).sum(1), 0.0)
    if prompts:
        useqs = [list(s) for s in prompts]
        _, q = code_dists_after(model, vocab, useqs, [[len(s) - 1] for s in useqs], [[vocab.vs, vocab.vt]],
                                device, erase=erase)
        for (pid, a, t, z, k), info, res in zip(people, per, results):
            if info["pkey"] is None:
                continue
            m = info["lv"] < 0
            kc = kcol[np.asarray(k)[m].astype(int)]
            res[m] = np.where(kc >= 0, q[info["pkey"], 0, np.maximum(kc, 0)], 0.0)
    return results
