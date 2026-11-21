# Implementation notes (Deep Ensembles)

## What is main's and what is an add-on
- **Main's code, used unchanged:** `pipeline.data` (split + loaders), `pipeline.models.DeepEnsemble`, `pipeline.train.train_with_noise`, `pipeline.evaluate.evaluate_conformal` / `conformal_metrics` / `compare` / `summarize`, `common.compute_ece/nll/brier`, `conformal_fixed.py`.
- **Add-ons (DE-only):** `src/dump_member_probs.py` (per-member and SVHN-test probabilities), `src/de_conformal.py` (non-randomized APS, RAPS), `src/de_eval.py` (OOD, selective prediction, ablation, significance, failure analysis, MI, adaptive ECE, cost, tables/figures), `src/verify_midterm_equivalence.py`.

## Data, training, seeds
- CIFAR-10 pooled 60k, class-stratified 60/20/20, split seed 0 -> train 36000 / test 12000 / cal 12000 (`common.get_splits`, via `pipeline.data.get_cifar_splits`). Training uses `noise_ratio=0` (no SVHN noise images).
- Recipe: `pipeline.train._train_one` (SGD Nesterov, momentum 0.9, lr 0.1, wd 5e-4, cosine over 40 epochs, batch 128, random crop (pad 4, reflect) + flip, cross-entropy).
- Ensemble seed `s` trains members with seeds `s*100 + i`, i = 0..4 (`DeepEnsemble.member_seeds`). Expected: ensemble seeds 0, 1, 2 (members 0-4, 100-104, 200-204). Prediction = mean of member softmaxes (`pipeline.models._softmax_probs`).
- Checkpoints `models/noise_testing/deep_ensemble_noise0_seed{s}_m{i}.pt` (git-ignored); main-format results `results/noise_testing/deep_ensemble_noise0_seed{s}.json/.npz`; add-on probabilities `results/noise_testing/de_addon/deep_ensemble_noise0_seed{s}/{m0..m4.npz, labels.npz, meta.json}` (committed so the eval runs without checkpoints).
- The five midterm checkpoints (seeds 0-4, older training script) are NOT used for final numbers. Their init is bit-identical to main's ensemble seed 0 members; only the batch-shuffle / augmentation RNG plumbing differs. They are used only in the cross-check below.

## Conventions (follow main)
- Std: re-split std (over the 100 random cal/test splits) uses `np.std` ddof=0, as in `pipeline/evaluate.py`; across-seed std (members, ensemble seeds) uses ddof=1, as in `pipeline.evaluate.summarize`. Tables say which is which.
- Re-splits: pool cal+test, permutation `RandomState(seed*100000 + r)`, randomization draws seeded by `r` only (main's convention; different ensemble seeds therefore share the same draws). The add-on `resplit_eval` reproduces this exactly; `RESULTS.md` -> "Consistency checks" shows the max difference vs main's numbers (0).
- Single-split conformal: draws seeded with the run (ensemble) seed, cal draws first, then test draws from the same stream (`conformal_fixed.split_conformal`). Members are evaluated with their ensemble's seed.
- Conformal rule: class y in set iff score(x,y) <= qhat, qhat = ceil((n+1)(1-alpha))/n quantile, alpha 0.1; sets may be empty (fraction reported).
- "Single ResNet-20" rows = the individual members (15 when all ensembles exist); "DE" = mean ± std over ensemble seeds. With fewer than 3 ensemble seeds the t-test is TODO and the tables show `(n=1)` or n=2 stds.

## Add-on definitions
- Adaptive ECE: 15 equal-mass bins. Mutual information = H(mean member prob) - mean member entropy (n/a for single models). Overconfidence gap = mean confidence - accuracy.
- Non-randomized APS: inclusive cumulative mass at y's rank. RAPS: randomized APS + 0.01 * max(0, rank - 5) (k_reg=5, lam=0.01, untuned). "RAPS set size" in OOD/ablation/significance uses qhat from the cal split with draw seed 0.
- OOD: CIFAR-10 test vs SVHN **official test split** (26,032 images, `data_svhn_test/test_32x32.mat`, git-ignored, downloaded by the dumper), CIFAR-10 normalisation. Scores (higher = OOD): -MSP, entropy, MI, RAPS set size. AUROC/AUPR-Out: OOD positive; AUPR-In/FPR@95TPR: ID positive (FPR = fraction of SVHN accepted when 95% of CIFAR test is kept). AUPR depends on prevalence.
- Selective prediction: stable sort by confidence (MSP, -entropy, -MI); AURC = mean selective risk; accuracy at 80/90/95% coverage.
- Ablation: pool = all available pipeline members; per M, up to 10 distinct random subsets (`RandomState(0)`; all subsets if fewer; subsets may mix ensemble seeds); error bars = sample std over subsets.
- DE vs single: delta = ensemble metric - mean over its own 5 members. Paired bootstrap, 10,000 resamples of test images (same resample for ensemble and members; the OOD delta also resamples SVHN), percentile 95% CI, `RandomState(0)`. t-test (`ttest_1samp` of per-ensemble deltas) only with >=3 ensemble seeds. Bootstrap covers test-sampling noise only.
- Cost: params from the model; train minutes from main's checkpoint history; inference time measured by the dumper (batch 256, includes preprocessing); ensemble = M x single.
- Failure analysis: first available ensemble seed. Largest sets = RAPS size, ties by lowest confidence. Indices are test-split positions (+ pooled-60k indices).

## Checks that ship with the repo
- `RESULTS.md` -> Consistency checks: add-on randomized APS re-splits vs `conformal_metrics`, and (when present) the saved main `.json` vs numbers recomputed from the add-on probabilities.
- `results/de/midterm_equivalence_check.md`: main's `evaluate_conformal` on the midterm checkpoints reproduces Acc/NLL/ECE 0.9028/0.3026/0.0312 (seed 0) and 0.9273/0.2232/0.0164 (M=5) and the midterm randomized APS numbers. Ensemble accuracy is 11127/12000 = 0.92725 (printed 0.9273 at the midterm).
- `--smoke` outputs live only in `results/de_smoke/` (git-ignored) and carry a SMOKE banner.

## Notes for the team
- main's re-split code seeds the permutation with `seed*100000 + r` but the randomization with `r` alone (shared draws across seeds); followed as is.
- `models/deep-ensembles-model.ipynb` (midterm notebook, unchanged from main) has commented-out calls to removed midterm scripts and reads `results_shravan/deep_ensembles/`, which is kept as an archive.
