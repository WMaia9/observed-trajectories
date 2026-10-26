"""Which first-record ICD-10 level 3 codes fall outside the Delphi vocabulary, and why.

Aggregates only, safe to export: shares by first letter (ICD-10 chapter family), and the most frequent
out-of-vocabulary codes with person counts; any count of 20 or fewer is suppressed.
Also lists A to Q codes (inside Delphi's chapter range) that are missing from the vocabulary, the
candidates for a small ICD-10-CM to WHO remap (for example E08, E09, C4A, M1A, O9A).
Usage: RUN_MODE=full python -m delphi_aou.replication.oov_report  (writes ~/results/delphi_rep_oov.json)
"""
import json, os
from pathlib import Path

import pandas as pd

from delphi_aou.data import cohort as s2
from delphi_aou.replication import to_delphi_bin as tdb

MIN_CELL = 21


def report(icd, vocab):
    c = icd["code3"].astype(str).str.upper().str[:3]
    oov = ~c.isin(vocab)
    df = pd.DataFrame({"pid": icd["person_id"].to_numpy(), "code": c.to_numpy(), "oov": oov.to_numpy()})
    df["letter"] = df["code"].str[0]
    by_letter = (df[df.oov].groupby("letter").size() / max(int(df.oov.sum()), 1)).round(4).sort_values(ascending=False)
    persons = df[df.oov].groupby("code")["pid"].nunique().sort_values(ascending=False)
    top = persons[persons >= MIN_CELL].head(40)
    in_range = df[df.oov & df["letter"].between("A", "Q")].groupby("code")["pid"].nunique().sort_values(ascending=False)
    return {
        "rows": int(len(df)), "oov_rows": int(df.oov.sum()), "oov_share": round(float(df.oov.mean()), 4),
        "oov_share_by_first_letter": by_letter.to_dict(),
        "top_oov_codes_persons": {k: int(v) for k, v in top.items()},
        "a_to_q_missing_from_vocab_persons": {k: int(v) for k, v in in_range[in_range >= MIN_CELL].head(60).items()},
        "a_to_q_missing_codes_total": int(len(in_range)),
        "note": f"person counts below {MIN_CELL} suppressed",
    }


def main():
    vocab = {k for k in tdb.label_dict() if len(k) == 3 and k[0].isalpha() and k[1].isdigit()}
    rep = report(s2._read_local_from_cache("rep_icd3"), vocab)
    out = Path(os.environ.get("OUT", str(Path.home() / "results" / "delphi_rep_oov.json")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1)); print("OOV REPORT COMPLETE", out)


if __name__ == "__main__":
    main()
