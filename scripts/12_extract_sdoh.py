#!/usr/bin/env python
"""Stage 1c: SDoH survey answers (housing, food insecurity, discrimination).
Runs inside the All of Us Workbench only (BigQuery). Use RUN_MODE=test (5% sample) before RUN_MODE=full."""
import runpy

if __name__ == "__main__":
    runpy.run_module("delphi_aou.data.extract.sdoh", run_name="__main__", alter_sys=True)
