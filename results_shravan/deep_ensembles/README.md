# MIDTERM ARCHIVE - superseded

Everything in this folder comes from the midterm Deep Ensembles run (5 members trained with `models/deep_ensembles/train_ensemble.py`,
since removed). It is kept because `models/deep-ensembles-model.ipynb` (midterm notebook) reads the tables here.

**Do not use these numbers for the final report.** Final Deep Ensembles numbers come from members trained through the shared
`pipeline/` (`train_with_noise(DeepEnsemble(5), noise_ratio=0, seed=s)`), see the Deep Ensembles section of the top-level README
and `results/de/`.

`archive_old_pipeline/` holds the logs, train times and dumped probabilities of the five midterm members (seeds 0-4). They are only used
for the equivalence cross-check `src/verify_midterm_equivalence.py`.
