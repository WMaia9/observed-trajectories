"""Delphi token sequences -> CEHR-GPT patient sequences (P2, second model family).

CEHR-GPT (Pang et al.) represents a patient as a demographic prompt [year, age, gender, race] followed by visits
[VS] visit-type codes... [VE] separated by artificial time tokens (ATT, 'D<days>' or 'LT' for more than 1080 days).
We build these sequences from the SAME tokens Delphi-2M sees, so both models get the same information:

  * a visit is the set of tokens that share one age in days (Delphi records have day resolution);
  * codes are the Delphi tokens 4..1269 (lifestyle, diseases, death) written as 't<id>'; the sex tokens 2/3 become
    the gender prompt token; padding (0) and no-event (1) tokens are dropped (CEHR-GPT has no such tokens);
  * no calendar dates are given to Delphi, so the year prompt is 'year:unknown' and race is 'Race/0' (unknown);
    epoch times are age in seconds from a common origin, so they carry no calendar information either;
  * every visit gets the outpatient visit type 9202 (the Delphi tokens carry no visit type).

The output dictionaries follow the format of cehrgpt.data.hf_cehrgpt_dataset_mapping.MedToCehrGPTDatasetMapping,
so the package's tokenizer, tokenization mapping and collator are used unchanged.
"""
from typing import Dict, Iterator, List, Optional, Tuple
import numpy as np

DAY_S = 86400.0
VISIT_TYPE = "9202"
GENDER = {2: "Gender/female", 3: "Gender/male"}
GENDER_UNKNOWN, RACE_UNKNOWN, YEAR_UNKNOWN = "Gender/0", "Race/0", "year:unknown"
FIRST_CODE, LAST_CODE = 4, 1269


def att_token(days: int) -> str:
    """CEHR-GPT 'day' artificial time token (cehrbert_data time_day_token): D<days> up to 1080 days, else LT."""
    return f"D{int(days)}" if days <= 1080 else "LT"


def code_name(tok: int) -> str:
    return f"t{int(tok)}"


def by_person(arr: np.ndarray) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """arr: [N, 3] (person, age_days, token) sorted by person (Delphi .bin layout)."""
    if len(arr) == 0:
        return
    cut = np.flatnonzero(np.diff(arr[:, 0])) + 1
    for s, e in zip(np.r_[0, cut], np.r_[cut, len(arr)]):
        yield int(arr[s, 0]), arr[s:e, 1].astype(np.float64), arr[s:e, 2].astype(np.int64)


def _blank(pid):
    return {"person_id": pid, "concept_ids": [], "ages": [], "concept_value_masks": [], "number_as_values": [],
            "concept_as_values": [], "units": [], "is_numeric_types": [], "epoch_times": []}


def _add(rec, code, day):
    rec["concept_ids"].append(code); rec["ages"].append(int(day // 365.25))
    rec["concept_value_masks"].append(0); rec["number_as_values"].append(0.0); rec["concept_as_values"].append("0")
    rec["units"].append("N/A"); rec["is_numeric_types"].append(0); rec["epoch_times"].append(float(day) * DAY_S)


def to_record(pid: int, ages: np.ndarray, toks: np.ndarray, upto: Optional[float] = None,
              prompt_age_day: Optional[float] = None) -> Optional[Dict]:
    """One CEHR-GPT record. upto: keep only tokens with age <= upto (prediction prefix). prompt_age_day: age used in
    the age prompt when the prefix has no visit (defaults to the first visit's age). Returns None if nothing is left
    and no prompt age is given."""
    o = np.argsort(ages, kind="stable"); ages, toks = ages[o], toks[o]
    gender = GENDER_UNKNOWN
    for t in toks:
        if int(t) in GENDER:
            gender = GENDER[int(t)]; break
    keep = (toks >= FIRST_CODE) & (toks <= LAST_CODE)
    if upto is not None:
        keep &= ages <= upto
    ages, toks = np.floor(ages[keep]), toks[keep]
    if len(toks) == 0 and prompt_age_day is None:
        return None
    rec = _blank(pid)
    first_day = float(ages[0]) if len(toks) else float(prompt_age_day)
    _add(rec, YEAR_UNKNOWN, first_day); _add(rec, f"age:{int(first_day // 365.25)}", first_day)
    _add(rec, gender, first_day); _add(rec, RACE_UNKNOWN, first_day)
    days = np.unique(ages)
    prev = None
    for d in days:
        if prev is not None:
            _add(rec, att_token(int(d - prev)), d)
        _add(rec, "[VS]", d); _add(rec, VISIT_TYPE, d)
        seen = set()
        for t in toks[ages == d]:
            if int(t) in seen:
                continue
            seen.add(int(t)); _add(rec, code_name(t), d)
        _add(rec, "[VE]", d)
        prev = d
    rec["orders"] = list(range(1, len(rec["concept_ids"]) + 1))
    rec["num_of_concepts"] = len(rec["concept_ids"]); rec["num_of_visits"] = int(len(days))
    return rec


def last_visit_day(ages: np.ndarray, toks: np.ndarray, upto: float) -> Optional[float]:
    m = (toks >= FIRST_CODE) & (toks <= LAST_CODE) & (ages <= upto)
    return float(np.floor(ages[m]).max()) if m.any() else None


def records(arr: np.ndarray, min_codes: int = 1) -> List[Dict]:
    out = []
    for pid, a, t in by_person(arr):
        r = to_record(pid, a, t)
        if r is not None and r["num_of_visits"] >= min_codes:
            out.append(r)
    return out
