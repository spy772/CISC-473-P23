"""Common conformal evaluation layer for any trained UQ adapter.

Uses the corrected conformal code in models/deep_ensembles/conformal_fixed.py
(randomized APS with a consistent set rule, LAC, Mondrian). See
models/deep_ensembles/CONFORMAL_AUDIT.md for why the original deterministic APS isn't used.
"""
import json
import math
from pathlib import Path

import numpy as np

import common
from conformal_fixed import (SCORES, _true_scores, conformal_quantile, mondrian_conformal,
                             split_conformal)

from . import REPO

DEFAULT_RESULTS_DIR = REPO / "results" / "noise_testing"
CONFORMAL_SCORES = ("aps_rand", "lac")


def _sets(cal_probs, cal_labels, test_probs, alpha, score, seed):
    """Boolean (n_test, K) prediction-set matrix. Draws U in the same order as
    conformal_fixed.split_conformal, so the sets match its reported numbers exactly."""
    f = SCORES[score]
    rng = np.random.RandomState(seed)
    u_cal = rng.rand(len(cal_labels))
    u_test = rng.rand(len(test_probs))
    qhat = conformal_quantile(_true_scores(f(cal_probs, u_cal), cal_labels), alpha)
    return f(test_probs, u_test) <= qhat


def conformal_metrics(cal_probs, cal_labels, test_probs, test_labels, alpha=common.ALPHA,
                      seed=0, n_resplits=0):
    """All conformal metrics from pre-computed probabilities (no model needed)."""
    K = test_probs.shape[1]
    out = {"alpha": alpha, "target_coverage": 1 - alpha}

    for score in CONFORMAL_SCORES:
        res = split_conformal(cal_probs, cal_labels, test_probs, test_labels,
                              alpha=alpha, score=score, seed=seed)
        sets = _sets(cal_probs, cal_labels, test_probs, alpha, score, seed)
        sizes = sets.sum(1)
        covered = sets[np.arange(len(test_labels)), test_labels]
        size_hist = np.bincount(sizes, minlength=K + 1)
        size_cov = {int(k): float(covered[sizes == k].mean()) for k in range(K + 1)
                    if (sizes == k).any()}
        out[score] = {**res,
                      "set_size_hist": size_hist.tolist(),          # count of sets of size 0..K
                      "coverage_by_set_size": size_cov,             # size-stratified coverage
                      "singleton_frac": float((sizes == 1).mean())}

    mond = mondrian_conformal(cal_probs, cal_labels, test_probs, test_labels,
                              alpha=alpha, score="aps_rand", seed=seed)
    mond["min_class_coverage"] = float(min(mond["per_class_coverage"]))
    out["mondrian_aps_rand"] = mond

    if n_resplits:
        # Pool cal+test and re-split many times: removes single-split luck from coverage.
        P = np.concatenate([cal_probs, test_probs])
        Y = np.concatenate([cal_labels, test_labels])
        n_cal = len(cal_labels)
        for score in CONFORMAL_SCORES:
            cov, size = [], []
            for r in range(n_resplits):
                perm = np.random.RandomState(seed * 100_000 + r).permutation(len(Y))
                c, t = perm[:n_cal], perm[n_cal:]
                res = split_conformal(P[c], Y[c], P[t], Y[t], alpha=alpha, score=score, seed=r)
                cov.append(res["coverage"])
                size.append(res["avg_set_size"])
            out[score]["resplit"] = {"n": n_resplits,
                                     "coverage_mean": float(np.mean(cov)),
                                     "coverage_std": float(np.std(cov)),
                                     "avg_set_size_mean": float(np.mean(size)),
                                     "avg_set_size_std": float(np.std(size))}
    return out


def evaluate_conformal(model, cal_loader, test_loader, alpha=common.ALPHA, seed=0, n_resplits=0,
                       tag=None, config=None, results_dir=DEFAULT_RESULTS_DIR, save=True):
    """Wrap a trained model in the common conformal layer and compute coverage metrics.

    model       : a trained adapter (e.g. train_with_noise(...)["model"]).
    cal_loader  : clean CIFAR calibration split; test_loader: clean CIFAR test split.
    seed        : seeds the randomized-APS draws and any MC sampling (MC Dropout / SNGP).
    n_resplits  : if > 0, also report mean/std coverage over that many random cal/test re-splits.
    tag/config  : run identifiers from train_with_noise (used in the result and file names).

    Returns a dict with accuracy, NLL, ECE, Brier and, per conformal score (aps_rand, lac):
    qhat, coverage, avg_set_size, frac_empty, set-size histogram, size-stratified coverage;
    plus Mondrian (per-class) coverage. Probabilities are saved to results_dir/{tag}.npz.
    """
    cal_probs, cal_labels = model.predict_probs(cal_loader, seed=seed)
    test_probs, test_labels = model.predict_probs(test_loader, seed=seed)

    ece, _ = common.compute_ece(test_probs, test_labels)
    result = {
        "tag": tag or model.name,
        "model": model.name,
        **({"noise_ratio": config["noise_ratio"], "n_svhn": config["n_svhn"],
            "seed": config["seed"], "noise_seed": config["noise_seed"],
            "epochs": config["epochs"], "smoke": config["smoke"]} if config else {}),
        "config": config or {"model_config": model.config()},
        "n_cal": int(len(cal_labels)), "n_test": int(len(test_labels)),
        "accuracy": float((test_probs.argmax(1) == test_labels).mean()),
        "nll": common.compute_nll(test_probs, test_labels),
        "ece": ece,
        "brier": common.compute_brier(test_probs, test_labels),
        "conformal": conformal_metrics(cal_probs, cal_labels, test_probs, test_labels,
                                       alpha=alpha, seed=seed, n_resplits=n_resplits),
    }

    if save:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(results_dir / f"{result['tag']}.npz", cal_probs=cal_probs,
                            cal_labels=cal_labels, test_probs=test_probs, test_labels=test_labels)
        with open(results_dir / f"{result['tag']}.json", "w") as f:
            json.dump(result, f, indent=2)
    return result


# ------------------------------ tables ----------------------------------------
ROW_KEYS = ["model", "noise_ratio", "n_svhn", "seed", "accuracy", "nll", "ece", "brier",
            "aps_coverage", "aps_set_size", "aps_frac_empty",
            "lac_coverage", "lac_set_size", "mondrian_min_class_cov"]


def _flat(r):
    c = r["conformal"]
    return {"model": r["model"], "noise_ratio": r.get("noise_ratio"), "n_svhn": r.get("n_svhn"),
            "seed": r.get("seed"), "accuracy": r["accuracy"], "nll": r["nll"], "ece": r["ece"],
            "brier": r["brier"],
            "aps_coverage": c["aps_rand"]["coverage"], "aps_set_size": c["aps_rand"]["avg_set_size"],
            "aps_frac_empty": c["aps_rand"]["frac_empty"],
            "lac_coverage": c["lac"]["coverage"], "lac_set_size": c["lac"]["avg_set_size"],
            "mondrian_min_class_cov": c["mondrian_aps_rand"]["min_class_coverage"]}


def compare(results, path=DEFAULT_RESULTS_DIR / "results.json"):
    """One flat row per run (model, noise_ratio, seed). Also writes all full results to `path`."""
    rows = [_flat(r) for r in results]
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
    return rows


def summarize(rows):
    """Group compare() rows by (model, noise_ratio): mean and std across seeds."""
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["noise_ratio"]), []).append(r)
    metrics = ROW_KEYS[4:]
    out = []

    # (model, noise_ratio)
    for (m, nr), g in groups.items():
        row = {"model": m, "noise_ratio": nr, "n_seeds": len(g)}
        for k in metrics:

            # Get the specified metric (e.g. NLL) for the given seed run (e.g. seed = 2)
            v = np.array([x[k] for x in g], dtype=float)
            row[k] = float(v.mean())  # combine all the metrics and get their mean (e.g. mean NLL)
            row[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0  # get the std dev
        out.append(row)
    return out


def to_markdown(rows, keys=None, digits=4, with_std=True):
    """Render compare()/summarize() rows as a Markdown table (no pandas needed)."""
    if not rows:
        return ""
    keys = keys or [k for k in rows[0] if not k.endswith("_std")]

    def fmt(r, k):
        v = r.get(k)
        if isinstance(v, float) and not math.isnan(v):
            s = f"{v:.{digits}f}"
            if with_std and f"{k}_std" in r and r.get("n_seeds", 1) > 1:
                s += f" ± {r[k + '_std']:.{digits}f}"
            return s
        return str(v)

    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r, k) for k in keys) + " |" for r in rows]
    return "\n".join(lines)
