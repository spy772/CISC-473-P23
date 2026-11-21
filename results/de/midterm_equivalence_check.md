# Midterm-checkpoint equivalence check (cross-check only, not final numbers)

main's `pipeline.evaluate_conformal` (seed 0, 100 re-splits) run on the midterm seed 0-4 checkpoints.

| model | Acc | NLL | ECE | Brier | APS-rand cov | APS-rand size | LAC cov | LAC size | resplit cov (ddof=0) | resplit size |
|---|---|---|---|---|---|---|---|---|---|---|
| single seed 0 | 0.9028 | 0.3026 | 0.0312 | 0.1433 | 0.8959 | 1.1675 | 0.9009 | 0.9931 | 0.8995 ± 0.0038 | 1.1881 ± 0.0089 |
| ensemble M=5 | 0.9273 | 0.2232 | 0.0164 | 0.1103 | 0.8928 | 1.1694 | 0.9016 | 0.9505 | 0.8997 ± 0.0039 | 1.1961 ± 0.0096 |

Midterm expectations: single seed 0 Acc/NLL/ECE (0.9028, 0.3026, 0.0312), APS-rand cov/size (0.8959, 1.1675); ensemble M=5 Acc/NLL/ECE (0.9273, 0.2232, 0.0164), APS-rand cov/size (0.8928, 1.1694)
(accuracy of the ensemble is 11127/12000 = 0.92725, printed as 0.9273 at the midterm; tolerance is half a unit of the 4th decimal.)

**Result: PASS**
