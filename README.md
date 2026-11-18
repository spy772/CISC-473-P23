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

Everything is computed from dumped softmax probabilities in `results/raw/` (no model is run for any metric).

```
pip install -r requirements-de.txt
python src/dump_member_probs.py     # optional: re-dump probs from checkpoints (needs .pt files + internet for SVHN test)
python src/de_eval.py               # all tables, figures, failure data, RESULTS.md, MISSING.md (~5 min on CPU)
```
- Members: seeds 0-4 = ensemble A (seed 0 = baseline checkpoint), 5-9 = B, 10-14 = C. Seeds 5-14 are trained with
  `train_jobs/de_train.ipynb` on a GPU; unzip its output at the repo root and re-run `python src/de_eval.py`.
- Outputs: `results/de/tables/*.{csv,tex}`, `results/de/RESULTS.md`, `results/de/NOTES.md` (implementation choices),
  `results/de/MISSING.md` (what is still TODO), `results/de/failure/*.json`, `figures/de/*.{pdf,png}`.
- `src/de_conformal.py` is a reusable conformal module (takes probability arrays: APS, randomized APS, RAPS, Mondrian).
- Midterm exploratory DE code/results: `models/deep_ensembles/`, `results_shravan/deep_ensembles/`.
