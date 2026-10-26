"""Compare the outputs of reproduce/run_public.sh with reproduce/expected_public.json and print PASS or FAIL per item.

Usage: python reproduce/check_expected.py --results OUT/home/results [--semisyn OUT/semisyn_quick.json]
Exit code 1 if any item fails.
"""
import argparse, json, sys
from pathlib import Path


def get(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--semisyn", default=None)
    ap.add_argument("--expected", default=str(Path(__file__).with_name("expected_public.json")))
    a = ap.parse_args()
    exp = json.loads(Path(a.expected).read_text())
    res = Path(a.results)
    checks = json.loads((res / "audit_protocol_checks_ukbsim.json").read_text())
    tag = next(k for k in checks if k != "window")
    audit = json.loads((res / f"audit_{tag}.json").read_text())
    audit["rho_erased"] = (audit["mean_auc"]["erased"] - 0.5) / (audit["mean_auc"]["real"] - 0.5)
    prot = dict(checks[tag]); prot["window"] = checks["window"]
    ok = True
    for section, src in [("protocol", prot), ("audit", audit)]:
        for key, spec in exp[section].items():
            got = float(get(src, key))
            good = abs(got - spec["value"]) <= spec["tol"]
            ok &= good
            print(f"{'PASS' if good else 'FAIL'}  {section:8s} {key:40s} got {got:.4f}  expected {spec['value']} +- {spec['tol']}")
    if a.semisyn:
        # qualitative claims of the paper that any seed should reproduce
        rows = {(r["mode"], r["tau"], r["gamma"]): r for r in json.loads(Path(a.semisyn).read_text())}
        claims = []
        r = rows.get(("frailty", 1.0, 0.0))
        if r:
            claims.append(("frailty tau=1 gamma=0: timing-only input keeps most of the AUC (rho > 0.5)",
                           r["plain"]["rho"] is not None and r["plain"]["rho"] > 0.5))
        r = rows.get(("detect", 2.0, 0.0))
        if r:
            claims.append(("detect tau=2 gamma=0: count beats timing-only model (count - erased > 0.03)",
                           r["plain"]["count_only"] - r["plain"]["erased"] > 0.03))
        for text, good in claims:
            ok &= good
            print(f"{'PASS' if good else 'FAIL'}  semisyn  {text}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
