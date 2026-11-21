# CISC-473-P23

## Setup & Installation

1. Clone the repo
2. Activate the virtual environment.

### Windows: (run in Command Prompt)

```
.venv\Scripts\activate.bat
```

### Mac: (run in Terminal)

```
source .venv\bin\activate
```

3. Install the requirements:

```
pip install
```

4. Run your program of choice.

## The Team

- Edward Tanurkov
- Elliott Vince
- Shravan Agnihotri

## Deep Ensembles (final numbers) - Shravan

Deep Ensembles (5 x ResNet-20, mean of softmaxes) trained and evaluated through the shared `pipeline/`, so the split, recipe, seed
convention, metrics and conformal code are identical to every other method. Ensemble seed `s` = members with seeds `s*100 + i`
(seed 0 -> 0-4, 1 -> 100-104, 2 -> 200-204); three ensemble seeds give mean ± std over 3 independent ensembles.

1. **Train + evaluate (GPU recommended):** `python src/de_train.py --seeds 0 1 2`. Per ensemble seed this calls
   `train_with_noise(DeepEnsemble(n=5), noise_ratio=0, seed=s)` (checkpoints `models/noise_testing/deep_ensemble_noise0_seed{s}_m{i}.pt`,
   resumed per member), `evaluate_conformal` (main-format `results/noise_testing/deep_ensemble_noise0_seed{s}.json/.npz`), and the add-on
   dumper `src/dump_member_probs.py` (per-member + SVHN-test probabilities in `results/noise_testing/de_addon/`).
   On Kaggle/Colab use `train_jobs/de_train.ipynb` (set `SEEDS` in the second cell, unzip the downloaded zip at the repo root).
2. **Evaluate:** `pip install -r requirements.txt` (Python >= 3.12), then `python src/de_eval.py` (about 10 min on CPU). It reads only the
   dumped probabilities. Tables (CSV + LaTeX): `results/de/tables/`; figures: `figures/de/`; `results/de/RESULTS.md`, `NOTES.md` (definitions,
   conventions), `MISSING.md` (what is still TODO); failure-analysis indices: `results/de/failure/`.
3. **Smoke test (minutes, CPU):** `python src/de_train.py --seeds 0 --smoke && python src/de_eval.py --smoke` (writes only under `results/de_smoke/`, git-ignored).
4. **Equivalence cross-check with the midterm run:** `python src/verify_midterm_equivalence.py` (needs the git-ignored midterm checkpoints).

Headline numbers use main's functions unchanged (`common.compute_*`, `pipeline.evaluate.conformal_metrics/compare/summarize`).
`src/de_conformal.py` adds non-randomized APS and RAPS on top. Midterm code/results are archived: `results_shravan/deep_ensembles/README.md`.
