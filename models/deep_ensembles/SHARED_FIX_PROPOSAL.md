# Proposed fix for the shared conformal layer (nothing shared was edited)

Problem and evidence: [CONFORMAL_AUDIT.md](CONFORMAL_AUDIT.md). In short: the shared deterministic APS
gives coverage 1.0 and ~6.3-class sets, because the model is overconfident with accuracy about 1 - alpha,
and because the set rule does not match the calibration score.

## What to do (for whoever owns the shared notebook)
1. Add `conformal_fixed.py` (in `models/deep_ensembles/`) to shared code, or copy it to a top-level
   `conformal.py`. It only needs numpy.
2. In `baseline_resnet20_conformal.ipynb` (and the other methods' notebooks), replace the cell titled
   "Split conformal (APS)" (`aps_scores`, `aps_predict_sets`, the qhat and coverage code) with:

```python
import sys; sys.path.insert(0, "models/deep_ensembles")      # or wherever conformal_fixed.py lives
from conformal_fixed import split_conformal, mondrian_conformal

res  = split_conformal(cal_probs, cal_labels, test_probs, test_labels, alpha=ALPHA, score="aps_rand")
mond = mondrian_conformal(cal_probs, cal_labels, test_probs, test_labels, alpha=ALPHA, score="aps_rand")

conformal_coverage     = res["coverage"]       # ~0.90
conformal_avg_set_size = res["avg_set_size"]   # ~1.2
print(res)
print(mond["per_class_coverage"])
```
3. Rerun only the conformal cell (probabilities do not change) and update the results JSON. A ready-made
   corrected counterpart for the baseline is
   `results_shravan/deep_ensembles/baseline_conformal_fixed.json`.
4. Every method (MC Dropout, Laplace, SNGP, Deep Ensembles) must use the same conformal code, or the
   comparison is not fair.

## What changes
| | Shared (as-is) | Fixed |
|---|---|---|
| Score | deterministic APS (cumulative mass incl. true class) | randomized APS: mass above y + U * p_y |
| Set rule | up to and including the class that crosses qhat | include y iff score(x, y) <= qhat |
| Quantile | ceil((n+1)(1-alpha))/n upper quantile | unchanged |
| Baseline result | coverage 1.000, size 6.31 | coverage 0.900 (mean of 100 re-splits), size 1.19 |

Randomized APS can return empty sets (about 1-6% of points). That is normal; if non-empty sets are
required, always add the top-1 class (coverage can only go up). Use a fixed seed for reproducibility.
`score="lac"` (1 - p_true) is also available if the team prefers it.
