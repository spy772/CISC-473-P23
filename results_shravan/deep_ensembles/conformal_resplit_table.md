# Fixed conformal layer (randomized APS), averaged over 100 random cal/test re-splits

alpha = 0.1 (target coverage 0.90). Each row: mean +/- sd over 100 re-splits of the pooled 24,000 calibration+test points (12,000 / 12,000). Split = marginal split conformal; Mondrian = class-conditional.

| Configuration | Split coverage | Split avg set size | Mondrian coverage | Mondrian avg set size |
|---|---|---|---|---|
| baseline (seed 0) | 0.9000 ± 0.0042 | 1.188 | 0.9004 ± 0.0041 | 1.201 |
| member seed 1 | 0.9002 ± 0.0042 | 1.177 | 0.9007 ± 0.0042 | 1.190 |
| member seed 2 | 0.8998 ± 0.0039 | 1.183 | 0.9005 ± 0.0039 | 1.196 |
| member seed 3 | 0.9001 ± 0.0040 | 1.201 | 0.9005 ± 0.0041 | 1.214 |
| member seed 4 | 0.9005 ± 0.0042 | 1.194 | 0.9009 ± 0.0040 | 1.208 |
| ensemble M=2 | 0.9000 ± 0.0041 | 1.187 | 0.9003 ± 0.0042 | 1.195 |
| ensemble M=3 | 0.9001 ± 0.0041 | 1.190 | 0.9003 ± 0.0042 | 1.195 |
| ensemble M=4 | 0.8998 ± 0.0040 | 1.193 | 0.9002 ± 0.0040 | 1.197 |
| ensemble M=5 | 0.9000 ± 0.0041 | 1.196 | 0.9003 ± 0.0041 | 1.199 |

Per-class Mondrian coverage, ensemble M=5 (mean ± sd):

| airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|
| 0.901 ± 0.013 | 0.902 ± 0.011 | 0.898 ± 0.014 | 0.903 ± 0.012 | 0.900 ± 0.013 | 0.899 ± 0.013 | 0.900 ± 0.011 | 0.900 ± 0.011 | 0.900 ± 0.014 | 0.901 ± 0.012 |
