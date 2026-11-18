# Deep Ensembles (Lakshminarayanan et al., 2017)

Five ResNet-20 networks are trained on the same data with identical settings, differing only in random seed (initialization, batch order, augmentation). Member 0 is Edward's baseline checkpoint. At test time the ensemble prediction is the mean of the members' softmax probabilities. Uncertainty comes from averaging. Disagreement among members is also available as mutual information for OOD detection. No adversarial training (omitted on purpose for a fair comparison).

The conformal layer found a problem in the shared code, so read [CONFORMAL_AUDIT.md](CONFORMAL_AUDIT.md) before interpreting any "shared" conformal number. The proposed fix for the team is in [SHARED_FIX_PROPOSAL.md](SHARED_FIX_PROPOSAL.md); no shared file was edited.

## Commands (run from the repo root)
```
# 1-epoch smoke test (tiny subset, writes to a temp dir)
python models/deep_ensembles/train_ensemble.py --smoke
# full training: seeds 1-4, resumable (re-run the same command after any interruption)
python models/deep_ensembles/train_ensemble.py
# evaluation (members 0-4 -> metrics, tables, cached probabilities)
python models/deep_ensembles/evaluate_ensemble.py [--svhn-test-mat path/to/test_32x32.mat]
# conformal numbers averaged over 100 random cal/test re-splits (writes conformal_resplit_table.md,
# baseline_conformal_fixed.json)
python models/deep_ensembles/conformal_resplit.py
# plots
python models/deep_ensembles/make_plots.py
```
`models/deep-ensembles-model.ipynb` is a thin notebook that shows the same commands and results.
Outputs go to `results_shravan/deep_ensembles/`. Checkpoints (`resnet20_seed{k}.pt`) are git-ignored.

## Setup (identical to the baseline notebook, `baseline_resnet20_conformal.ipynb`)
- Model: `cifar10_resnet20` from `chenyaofo/pytorch-cifar-models`, `pretrained=False`.
- Data: CIFAR-10 train+test pooled and split 60/20/20 stratified with split seed 0, giving train 36000 / test 12000 / calibration 12000 (same split for every member).
- 40 epochs, batch 128, SGD lr 0.1, momentum 0.9 (Nesterov), weight decay 5e-4, `CosineAnnealingLR(T_max=40)`, 4px reflect-pad random crop + horizontal flip, cross-entropy.
- Seeds: 0 (Edward's `models/baseline/baseline_resnet20_seed0.pt`, loaded in place and not retrained), 1, 2, 3, 4.
- Device: CPU only (torch 2.4.1+cpu, `cuda.is_available() == False`).
- Training time: about 100 min per member on CPU (seed 1: 99.5 min, seed 2: 98.7 min, seed 4: 104.6 min). Seed 3's log says 342 min wall-clock because the machine slept for about 3.5 h mid-run; its actual compute was about 100 min. Total compute for four members is about 6.7 h.
- Per-epoch train/val loss and accuracy: `train_log_seed{k}.csv` ("val" = the calibration split, so test is never looked at during training; this does not affect the weights).

## Evaluation details
- ID metrics on the CIFAR-10 test split: accuracy, NLL, ECE (15 equal-width bins, the baseline's function), Brier (sum over classes of squared error, averaged over samples).
- Conformal, alpha = 0.1, calibrated on the calibration split: "shared" = the notebook's split APS code as-is; "fixed" = randomized APS in `conformal_fixed.py`. Mondrian = per-true-class qhat. The shared notebook has no Mondrian code, so the "shared Mondrian" numbers use my class-conditional version of the shared APS score and rule. LAC is also reported in `results.csv`.
- OOD: CIFAR-10 test (negatives) vs SVHN (positives). AUROC and AUPR using max-softmax, predictive entropy, and mutual information (entropy of the mean prediction minus mean member entropy; N/A for single models).

## Deviations and caveats
- **SVHN data:** the repo has no SVHN *test* split, only the train chunks (73,257 images). OOD uses a fixed random subsample of 12,000 of them (seed 0) to balance against the 12,000 CIFAR test images. Rerun with `--svhn-test-mat` to use the real SVHN test set.
- **Seed 0 is not a fresh training run** with this script (it is Edward's checkpoint). Training is single-threaded-CPU non-deterministic anyway, so exact bitwise reproduction of seed 0 would not be expected.
- The ensemble uses uniform averaging of probabilities, no adversarial training, and no hyperparameter tuning per member (all deviations from the paper's optional components).
- Members vary by about 1 point in accuracy (0.898-0.908), above the 0.5% sanity target; see `results_shravan/deep_ensembles/sanity_checks.md`.
- A single cal/test split gives fixed-conformal coverage within about +/-0.4% of 0.90 by chance (0.893 for M=5 on our split); averaged over 100 re-splits it is 0.900 for every configuration (`conformal_resplit_table.md`).
- Results are for one set of 5 seeds on one split; there are no error bars.
