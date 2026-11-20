"""Noise-testing pipeline: train ResNet-20 UQ variants on CIFAR-10 + SVHN noise,
evaluate on the clean CIFAR-10 test split with a common conformal layer.

Reuses the shared code in models/deep_ensembles/ (common.py, conformal_fixed.py).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_SHARED = REPO / "models" / "deep_ensembles"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from .data import build_train_aug, get_cifar_splits, load_svhn_train, make_loaders  # noqa: E402
from .models import SNGP, Baseline, DeepEnsemble, LaplaceLL, MCDropout  # noqa: E402
from .train import run_experiment, set_seed, train_with_noise  # noqa: E402
from .evaluate import compare, evaluate_conformal, summarize  # noqa: E402

__all__ = [
    "REPO", "build_train_aug", "get_cifar_splits", "load_svhn_train", "make_loaders",
    "Baseline", "MCDropout", "LaplaceLL", "SNGP", "DeepEnsemble",
    "train_with_noise", "run_experiment", "set_seed",
    "evaluate_conformal", "compare", "summarize",
]
