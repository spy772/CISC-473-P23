# Audit of the shared conformal layer

Scope: `baseline_resnet20_conformal.ipynb` (cell 14, "Split conformal (APS)"). Nothing shared was edited.
Checked on Edward's `baseline_resnet20_seed0.pt` (test acc 0.9028; I regenerated its cal/test softmax outputs).
Reproduced exactly: qhat = 0.9999951, coverage = 1.0, avg set size = 6.3088.

**Verdict: it is a real defect, not an intentional choice** (alpha is 0.1, the quantile is correct). The cause is the nonconformity score and set rule (items b and d below). Fixing them gives coverage of about 0.90 and sets of about 1.2 classes.

## Checklist (in order)

| | Question | Finding | Lines (ipynb) |
|---|---|---|---|
| a | alpha | `ALPHA = 0.1`, a module-level constant used by the conformal cell (not passed into a function). Correct, **not** the cause. | 88, 666 |
| b | score | **Deterministic (non-randomized) APS**: `cumsum` of sorted probs *including* the true class, taken at the true class's rank. Not flipped (higher = less conforming, which is right), but it has a known failure mode here (see "Why coverage = 1.0"). | 635-645 |
| c | quantile | `ceil((n+1)(1-alpha))/n` with `method="higher"`, on the calibration scores. Correct. | 666-668 |
| d | set rule vs score | **Mismatch.** Calibration uses `score = cumsum[true_rank]` (inclusive). Prediction (`k = searchsorted(cumsum, qhat) + 1`) includes every class up to and including the first rank where `cumsum >= qhat`, so it effectively uses `cumsum[rank-1] < qhat`. That is not "include y iff score(x,y) <= qhat". The extra class makes sets larger, always on the conservative side. | 656-658 |
| e | disjointness | Fine. `train/test/cal` come from one stratified permutation. I measured overlap = 0 for all pairs (`test∩cal`, `train∩cal`, `train∩test`). Cal is held out from training (36000/12000/12000). | cell 4 |

## Why coverage = 1.0 and set size is about 6.3

1. The ResNet-20 is very overconfident: the median max-prob is 0.998, and 28% of cal APS scores are >= 0.9999.
2. Top-1 accuracy (0.9037 on cal) is almost exactly `1 - alpha = 0.90`. So the 90% quantile of the deterministic scores falls in the cluster of *correctly classified, near-1.0* scores. It lands at qhat = 0.9999951, a hair under 1 (the 0.5 / 0.9 / 0.95 score quantiles are 0.9985 / 0.9999951 / 0.9999989).
3. Deterministic APS has no way to "partially" use the top class (no randomization). With qhat that close to 1, the cumulative mass reaches it only after the tail of near-zero probabilities is added, so a set grows to about 6 classes.
4. Plus the inclusive-crossing rule from (d). The result: coverage is 1.0 for both top-1-correct *and* top-1-wrong test points, with mean set size 5.97 on correct points and 9.45 on wrong ones. The sets carry almost no uncertainty information, so the comparison across UQ methods is distorted.

The guarantee is still valid (coverage >= 0.9 holds), but it is vacuously conservative. A better-calibrated method would not produce smaller sets under this layer, so the layer would hide the differences we want to measure.

## Proposed fix

Use **randomized APS** (Romano et al. 2020) with a consistent rule, as in [conformal_fixed.py](conformal_fixed.py):

* score(x, y) = (mass of classes ranked strictly above y) + U * p_y, with U ~ Uniform(0,1) drawn per point (seeded);
* qhat = `ceil((n+1)(1-alpha))/n` upper quantile of the calibration true-label scores (same as now);
* set(x) = { y : score(x, y) <= qhat } (the same function used in calibration);
* Mondrian version: one qhat per true class from that class's calibration scores.

LAC (`1 - p_true`) is also implemented as a reference. Both are in `conformal_fixed.py`.

## Results on the seed-0 baseline (alpha = 0.1, test n = 12000)

| method | coverage | avg set size | notes |
|---|---|---|---|
| shared (deterministic APS, as-is) | **1.000** | **6.31** | qhat = 0.9999951 |
| fixed: randomized APS, split | 0.896 | 1.17 | 5.9% empty sets; coverage across 5 U-seeds: 0.893-0.898 |
| fixed: randomized APS, Mondrian | 0.895 | 1.17 | per-class coverage 0.862-0.921 |
| ref: LAC, split | 0.901 | 0.99 | 1.4% empty sets |
| ref: LAC, Mondrian | 0.905 | 1.03 | per-class coverage 0.885-0.917 |

The fixed coverage is inside the 0.88-0.93 sanity window. Sets can be empty under randomized APS and LAC; this is standard and the guarantee still holds. If the team prefers non-empty sets, always add the top-1 class (this only increases coverage).

## Suggested change to the shared notebook (not applied)

Replace `aps_scores` / `aps_predict_sets` and the cell-14 glue with `split_conformal(..., score="aps_rand")` from `conformal_fixed.py`, or equivalent. The notebook also has no Mondrian code, so `mondrian_conformal` is new. Until the team decides, my results report both a "shared" and a "fixed" column.
