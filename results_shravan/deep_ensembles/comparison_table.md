# Baseline vs Deep Ensemble

CIFAR-10 test n=12000, calibration n=12000, alpha=0.1. OOD = SVHN train chunks (test split not in repo) (subsampled to 12000).

| Metric | Edward (reported) | Shravan repro (reported) | Seed-0 ckpt (recomputed) | Deep Ensemble M=5 |
|---|---|---|---|---|
| Accuracy | 0.9028 | 0.9039 | 0.9028 | 0.9273 |
| NLL | 0.3026 | 0.2992 | 0.3026 | 0.2232 |
| ECE (15 bins) | 0.0312 | 0.0284 | 0.0312 | 0.0164 |
| Brier | n/a | n/a | 0.1433 | 0.1103 |
| Conformal coverage, shared code (split APS) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Avg set size, shared code | 6.31 | 6.32 | 6.31 | 6.27 |
| Conformal coverage, fixed (rand. APS, split) | n/a | n/a | 0.8959 | 0.8928 |
| Avg set size, fixed | n/a | n/a | 1.17 | 1.17 |
| Mondrian coverage, fixed | n/a | n/a | 0.8950 | 0.8928 |
| Mondrian avg set size, fixed | n/a | n/a | 1.17 | 1.17 |
| OOD AUROC (max-softmax) | n/a | n/a | 0.9139 | 0.9224 |
| OOD AUROC (entropy) | n/a | n/a | 0.9204 | 0.9331 |
| OOD AUROC (mutual info) | n/a | n/a | N/A | 0.8746 |
| OOD AUPR (max-softmax) | n/a | n/a | 0.8787 | 0.8913 |
| OOD AUPR (entropy) | n/a | n/a | 0.8834 | 0.8956 |
| OOD AUPR (mutual info) | n/a | n/a | N/A | 0.8045 |

"Shravan repro (reported)" values are those from the independent retrain quoted in the project notes (not recomputed here). The seed-0 column is the same checkpoint Edward reported, re-evaluated by this pipeline.
