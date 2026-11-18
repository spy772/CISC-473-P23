# Implementation notes (Deep Ensembles)

## Data / split / members
- CIFAR-10 train+test pooled (60k), class-stratified 60/20/20, split seed 0 (`models/deep_ensembles/common.py`, same code as the baseline notebook) -> train 36000 / test 12000 / cal 12000. Indices in `results/raw/labels/split_indices.json`.
- Members differ only by seed (init, shuffle, aug). Seed 0 = baseline checkpoint (`models/baseline/baseline_resnet20_seed0.pt`), seeds 1-4 trained on the laptop CPU (`models/deep_ensembles/train_ensemble.py`), seeds 5-14 on GPU (`train_jobs/de_train.ipynb`, same script).
- Ensemble = arithmetic mean of member softmaxes. Ensembles: A = 0-4, B = 5-9, C = 10-14. M=10 ablation pool = seeds 0-9.
- OOD set: SVHN **official test split** (26,032 images, `test_32x32.mat` from ufldl.stanford.edu, git-ignored in `data_svhn_test/`), CIFAR-10 mean/std normalization, no augmentation. This replaces the midterm's 12,000-image subsample of SVHN *train* chunks, so OOD numbers are not comparable to the midterm table.
- Dumped probs are float32; the midterm probs for seeds 0-4 are bit-identical to the re-dumped ones (checked).

## Sanity check
- Seed 0: Acc 0.9028 / NLL 0.3026 / ECE 0.0312 reproduced. Ensemble A: NLL 0.2232 / ECE 0.0164 reproduced; accuracy is 11127/12000 = 0.92725 exactly, which the midterm printed as 0.9273 (round-half-up); Python `round` gives 0.9272. The check in `de_eval.py` therefore uses a +-0.00005 tolerance.

## Metrics
- Std everywhere = sample std (ddof=1) over members (single models) or over ensembles (DE). With one ensemble the DE column shows `(n=1)`, no std.
- ECE: 15 equal-width bins on max-prob, bins (lo, hi]. Adaptive ECE: 15 equal-mass bins (sorted by confidence, `array_split`).
- NLL clipped at 1e-12. Brier = sum over classes of squared error, averaged over samples (range 0-2).
- Mutual information = H(mean member prob) - mean member entropy (nats); undefined (n/a) for single models.
- Overconfidence gap = mean confidence - accuracy (negative = under-confident).
- Selective prediction: sort by confidence score (MSP; negative entropy; negative MI), stable sort; AURC = mean over all k of the risk (1 - acc) of the top-k most confident; accuracy at 80/90/95% coverage = accuracy of the top ceil(c*n).

## Conformal (`src/de_conformal.py`)
- alpha=0.1; qhat = ceil((n+1)(1-alpha))/n empirical quantile of the calibration true-class scores.
- Every method uses the same function for calibration and prediction: y in set iff score(x,y) <= qhat. Sets may be empty (fraction reported, no top-1 forced in). This is the corrected layer from `models/deep_ensembles/CONFORMAL_AUDIT.md`; the notebook's original inclusive-crossing APS (coverage 1.0, size 6.3) is NOT used.
- Non-randomized APS: score = inclusive cumulative mass up to y's rank. Randomized APS: mass strictly above y + U*p_y. RAPS: randomized APS + lam*max(0, rank - k_reg), k_reg=5, lam=0.01 (rank 1-indexed, no tuning of k_reg/lam). Mondrian randomized APS: one qhat per true class.
- Randomness: U ~ Uniform(0,1), one per image (shared across all classes of that image); seed = re-split index.
- 100 re-splits: pool cal+test (24k), `RandomState(r).permutation`, first 12k = cal, last 12k = test, r = 0..99. All numbers in the conformal table are averages over the 100 splits (per member/ensemble), then mean ± std over members/ensembles. The "std over splits" column is within-ensemble split variability.
- Per-class coverage range = min and max over the 10 classes, averaged over splits.
- "RAPS set size" used for OOD, ablation and significance = RAPS (k_reg=5, lam=0.01) with qhat from the original 12k cal split (seed 0 for cal randomization, a fixed second seed for test/SVHN). Set size as an OOD score has many ties (handled by sklearn / the weighted AUROC).
- The spec's "rAPS" is read as RAPS (regularized APS) for the OOD/ablation set sizes; "randomized APS" appears separately as `aps_rand`.

## OOD
- ID (CIFAR-10 test, 12000) vs OOD (SVHN test, 26032). Scores (higher = more OOD): -MSP, entropy, MI, RAPS set size.
- AUROC: OOD positive. AUPR-Out: OOD positive. AUPR-In: ID positive (score negated). FPR@95TPR: ID positive; the FPR is the fraction of SVHN accepted at the threshold keeping 95% of CIFAR test. AUPR is prevalence-dependent (OOD = 68% of the pooled set).

## Ensemble-size ablation
- Pool = seeds 0-9 (only the available ones). For each M, up to 10 distinct random size-M subsets via `RandomState(0).choice` over all C(N,M) combinations (all combinations if <=10). Error bars = sample std across subsets (0 for the single M=N subset). M=1 subsets are single members (MI AUROC n/a).

## DE vs single
- Delta = metric(ensemble) - mean over its own 5 members of the same metric. Paired bootstrap: 10,000 resamples of test images (the same resample for the ensemble and its members; `RandomState(0)`), percentile 95% CI. The OOD-AUROC delta also resamples the SVHN images (tie-aware weighted AUROC, only entropy-based OOD AUROC is bootstrapped).
- Paired t-test (`scipy.stats.ttest_1samp` of the per-ensemble deltas vs 0) is run only when >=3 ensembles exist.
- Bootstrap CIs account for test-set sampling noise only, not seed/training variance.

## Cost
- Params from `sum(p.numel())` of the hub ResNet-20. Inference time measured in `dump_member_probs.py` (batch 256, includes the numpy preprocessing loader, `num_workers=0`) over test+cal+SVHN; ensemble time = M x single (members run sequentially). Training times: wall-clock from `train_time_seed*.json`; seed 3's 342 min includes ~3.5 h laptop sleep (actual compute ~100 min); seed 0's time is not recorded.

## Failure analysis (ensemble A)
- Confusion matrix and top-5 off-diagonal pairs from the ensemble argmax. Confident wrong = wrong test predictions sorted by max-prob. Confident SVHN = SVHN images sorted by max-prob (with predicted class). Largest sets = RAPS set size, ties broken by lowest confidence. Highest MI = top MI test images. JSON indices are positions in the 12k test split (and global CIFAR index); figures in `figures/de/failure_*`.

## Other
- Seeds fixed and logged: split 0; training seed = member id; re-splits 0-99; ablation `RandomState(0)`; bootstrap `RandomState(0)`.
- The probability dumps are committed (~11 MB for seeds 0-4) so `python src/de_eval.py` works without checkpoints.
- Python 3.8 environment; `requirements-de.txt` is a separate pin file (the shared `requirements.txt` is UTF-16 with numpy 2.x pins from other environments), so the shared file was not edited.
