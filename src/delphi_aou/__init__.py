"""delphi_aou: generative disease trajectory models under informative EHR observation processes.

Subpackages
  data        cohort construction, first occurrence tokens, visit table (All of Us Workbench)
  eval        Delphi equivalent AUC with DeLong, paired comparisons, calibration by access group
  baselines   age-sex, record density, visits, logistic and gradient boosting baselines
  models      Delphi style next token model and the visit level marked point process (Tier A)
  train       training, anchor scoring and trajectory sampling for both model families
  sim         simulator with known disease dynamics observed through an EHR window
"""
__version__ = "0.1.0"
