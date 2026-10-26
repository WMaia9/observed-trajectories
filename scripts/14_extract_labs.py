#!/usr/bin/env python
"""Stage 1d: lab-defined disease onset (silver standard for the Tier B latent onset). Workbench only (BigQuery).
Use RUN_MODE=test (5% sample) before RUN_MODE=full."""
import runpy

if __name__ == "__main__":
    runpy.run_module("delphi_aou.data.extract.labs", run_name="__main__", alter_sys=True)
