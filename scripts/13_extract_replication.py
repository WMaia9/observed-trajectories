#!/usr/bin/env python
"""Stage 1d: extraction for the exact Delphi-2M replication (all ICD-10 level 3 codes, death, BMI,
smoking, alcohol). Workbench only. RUN_MODE=test first."""
import runpy

if __name__ == "__main__":
    runpy.run_module("delphi_aou.data.extract.replication", run_name="__main__", alter_sys=True)
