# All of Us data policy (summary for this repository)

This repository contains code only. It never contains participant level data, trained model weights,
or any file produced from participant level data, except aggregate statistics that satisfy the
All of Us Data and Statistics Dissemination Policy.

* Counts of 1 to 20 are never shown or derivable from published numbers. The code suppresses any cell
  below the `policy.min_cell_count_for_export` threshold (21) in exported summaries.
* Analyses, including model training, run only inside the Researcher Workbench.
* Participant data is never sent to external services, including hosted LLM APIs.
* Models trained on participant level data are not exported.
* Every publication or presentation is reported to the program at least two weeks in advance, with the
  required acknowledgement and data availability statement (CDR version: Registered Tier R2025Q4R6).

Check the current policy pages on support.researchallofus.org before each submission.
