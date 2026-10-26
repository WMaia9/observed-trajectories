# Experiments

One script per analysis. Every script states its question and usage in its docstring, writes only aggregates
(`~/results/*.json` inside the Workbench) and is run from the repository root, e.g.
`python -u experiments/run_audit.py --tag ...`. Scripts import each other by module name, so keep them in this folder.

## P2: audit of Delphi-2M on All of Us

| Paper section | Script | What it does |
|---|---|---|
| Who the protocol scores (Sec. 4) | `audit_protocol_checks.py` | Padding and pre-record controls in the official evaluation batch |
| | `audit_keep_sex.py` | People the official window cut drops, and the AUC with them returned |
| | `audit_returned_conditional.py` | Matched and wash-out AUCs for the returned people |
| | `protocol_on_public_sim.sh` | The same two checks on the authors' public simulated UK Biobank data |
| Without identities (Sec. 5) | `run_audit.py` | Paired audit: full, erased, shuffled and identity-free inputs on the same points |
| | `audit_selection_seeds.py` | Variance from the random choice of points |
| Conditioning on the calendar (Sec. 6) | `audit_matching_checks.py` | Matched cells (default and fine) and wash-out |
| | `audit_corrected_conditional.py` | Conditional analyses without padding controls, all settings |
| | `audit_paired_gap.py` | Person-bootstrap intervals for the matched gap, time-to-target bins, chapters |
| | `audit_reference_models.py` | Cross-fitted calendar regression and identities-without-timing reference scores |
| | `audit_time_loss.py` | Time-to-next-token loss of the full and identity-free models |
| | `audit_by_access.py` | Strata by income, insurance and education |
| | `audit_by_access_r12.py` | Strata by race and ethnicity, EHR site, and insurance at ages 65 to 80 |
| | `audit_site_density.py` | Record density of the EHR-site groups; post-record matched gaps |
| Second model family (Sec. 6) | `cehrgpt_train.py` | Train CEHR-GPT (full and identity-free) on the Delphi-2M training people |
| | `cehrgpt_score.py` | Score and evaluate CEHR-GPT on the audit points |
| Semi-synthetic worlds (Sec. 7) | `semisynthetic_rho.py` | Worlds with known structure; erased-input share |
| | `semisynthetic_matched.py`, `semisynthetic_summary.py` | Matched analysis in the worlds and its summary over seeds |
| | `semisynthetic_idfree.py` | Identity-free and record-token models in the worlds |
| Calibration by utilization (Sec. 8) | `official_anchored_calibration.py` | Anchored calibration of the official checkpoint by visit quintile |
| | `official_fidelity_check.py` | Fidelity of the anchored scoring path on official-protocol points |

## P1: visit-level point process (in progress)

| Script | What it does |
|---|---|
| `exp_observation_bias.py` | Does the likelihood objective matter when observation is partial? |
| `synthetic_detection.py` | Synthetic world with onsets independent of visits and 10-fold visit-rate variation |
| `synthetic_drift.py` | Does rollout drift come from illness-driven visits? |
| `rescore_visit_model.py` | Re-score a saved visit model with both risk estimators |
| `lab_onset.py` | Latent onset against lab-defined onset |
| `diag_followup.py`, `diag_readout.py`, `diag_visit_counts.py` | Diagnostics of follow-up definition, read-out and simulated visit counts |

Job queues that chain these scripts inside the Workbench are in `workbench/queues/`.
