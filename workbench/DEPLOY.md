# Running inside the All of Us Researcher Workbench

The Workbench apps sit inside a VPC Service Controls perimeter, so code is copied in and only
aggregate results come out.

1. Build a source archive locally: `git archive --format=tar.gz -o delphi-aou.tar.gz HEAD`
2. Upload it to the app home directory (JupyterLab upload), then in a terminal:
   `mkdir -p ~/delphi-aou && tar -xzf delphi-aou.tar.gz -C ~/delphi-aou && pip install -e ~/delphi-aou`
3. Always run the sample first (`RUN_MODE=test`), check the printed validation numbers, then `RUN_MODE=full`.
4. Use the CPU app (n1-highmem-4) for data preparation and the GPU app (T4) only for training.
   Stop apps as soon as a batch finishes.
5. Cache: `${WORKSPACE_BUCKET}/notebooks/cache_v8` (full) and `cache_v8_test` (5% sample).

## Layout of the Workbench home directory

Keep the home directory tidy; every runner follows this convention.

```
~/delphi-aou/            this repository (code only; never write outputs inside it)
~/logs/                  every log file (log_<what>.txt); runners create it
~/results/               small JSON summaries written by experiments (aggregates only)
~/runs/                  scratch directories of individual runs (TMPDIR), safe to delete after backup
~/delphi_rep_data[_test]/  official Delphi data set built by scripts/70 (participant level, never exported)
~/delphi_rep_runs/       official Delphi checkpoints and sweeps (backed up to the bucket)
~/archive/               old code and helper scripts kept for reference
```

Workbench-managed folders (`repos/`, `workspace/`, `load-env*`) are left alone.

## Updating the code on the VM

The VM copy is not a git checkout. Update it from a source archive (step 1 above) or by uploading the changed
files, and never while a runner that uses those files is still going: bash reads a script while it runs, so
editing a running `.sh` can break it.
