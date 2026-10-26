# delphi-aou

Auditing and modeling generative disease trajectory models under informative EHR observation, on the NIH
*All of Us* Research Program (Registered Tier, CDR v9, R2025Q4R6).

A diagnosis in an electronic health record is not a disease onset: it is an onset filtered through when a person is
seen and coded. This repository holds the code behind two papers:

- **P2, audit.** What does the headline AUC of a disease trajectory model measure? Delphi-2M (Shmatko et al.,
  *Nature* 2025) is retrained on All of Us with its unchanged official code and audited with input controls, a
  retrained identity-free model, matched cells on the record calendar, a second model family (CEHR-GPT), semi-synthetic
  worlds with known truth, calibration by utilization, and strata by access, race and ethnicity, and EHR site.
- **P1, method.** A visit-level marked point process that separates disease onset from its recording (in progress).

This repository contains code only: no All of Us data, no participant-level outputs, no model weights and no
results computed on All of Us data (see [`docs/DATA_POLICY.md`](docs/DATA_POLICY.md)). Workspace-specific
identifiers are read from the `WORKSPACE_CDR` and `WORKSPACE_BUCKET` environment variables.

## Repository layout

```
src/delphi_aou/              the library
  config.py                  frozen configuration, hashed person-level split, result stamping
  data/                      BigQuery extraction (extract/), cohort and first-occurrence tokens, visit table
  baselines/                 landmark anchors, incident labels, age-sex, logistic and gradient boosting baselines
  eval/                      AUC per disease, age band and sex with DeLong; calibration by visit quintile
  models/, train/            Delphi-style reimplementation and the visit-level point process (P1)
  replication/               official Delphi-2M on All of Us: data build, evaluation wrapper, input controls,
                             identity-free training, paired audit (P2)
  cehrgpt/                   CEHR-GPT sequences, training helpers and next-visit read-out (P2 second family)
  sim/                       simulator with known onsets observed through an EHR window
third_party/delphi_official/ official Delphi-2M code, vendored unchanged (commit fb72166)
scripts/                     numbered pipeline: 10-14 extraction, 20 cohort, 30 baselines, 40 Delphi-style training,
                             50-51 visit model, 60 calibration, 70 official Delphi data set
experiments/                 one script per analysis in the papers (index in experiments/README.md)
workbench/                   runners and job queues used inside the All of Us Workbench (DEPLOY.md)
reproduce/                   checks that run without All of Us access (public simulated data, semi-synthetic worlds)
tests/                       tests on synthetic data with known answers
results/                     small outputs of the synthetic checks (no All of Us data)
docs/DATA_POLICY.md          what may and may not leave the Workbench
```

## Install and test

```
pip install -e ".[dev]"            # core library and tests
pip install -e ".[aou]"            # BigQuery extraction inside the Workbench
pip install -e ".[cehrgpt]"        # CEHR-GPT second family (Python 3.12 recommended)
pytest -m "not slow"               # a few minutes, synthetic data only
pytest -m slow                     # trains small models
```

## Reproducing

Without All of Us access: `reproduce/run_public.sh` runs the official-protocol checks on the authors' public
simulated UK Biobank data and a short semi-synthetic run, and compares them with `reproduce/expected_public.json`.

With All of Us access (Registered Tier): follow [`workbench/DEPLOY.md`](workbench/DEPLOY.md). Run the 5% sample first
(`RUN_MODE=test`, the default), then `RUN_MODE=full`. The P2 analyses run in the order of the queues in
`workbench/queues/`; each experiment writes only aggregates to `~/results/`.

## Data use

All of Us data are subject to the All of Us Data Use Agreement and are not distributed here. Results computed on
All of Us data are published only in the papers, after review by the program.

## License and citation

MIT for the code in this repository; the vendored Delphi-2M code keeps its own license
(`third_party/delphi_official/LICENSE`). To cite this code, see `CITATION.cff`.
