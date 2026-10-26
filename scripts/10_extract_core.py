#!/usr/bin/env python
"""Stage 1a: core extraction from the CDR (STOP_AFTER=N to stop after step N).
Runs inside the All of Us Workbench only (BigQuery). Use RUN_MODE=test (5% sample) before RUN_MODE=full."""
import runpy

if __name__ == "__main__":
    runpy.run_module("delphi_aou.data.extract.core", run_name="__main__", alter_sys=True)
