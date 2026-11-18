"""Report plots from results.json + cached probabilities (run evaluate_ensemble.py first).
  python models/deep_ensembles/make_plots.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import common as C  # noqa: E402

RES = C.RESULTS_DIR
PR = RES / "probs"
BASE_C, ENS_C = "#4C72B0", "#DD8452"   # colour-blind-safe pair
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})

data = json.load(open(RES / "results.json"))
rows = data["rows"]
Mmax = max(r["M"] for r in rows if r["kind"] == "ensemble") if any(r["kind"] == "ensemble" for r in rows) else 1
ens_name = f"Deep Ensemble (M={Mmax})"
y = np.load(PR / "cifar10_test_labels.npy")
base_test = np.load(PR / "member_seed0_test.npy")
base_svhn = np.load(PR / "member_seed0_svhn.npy")
sub = np.load(PR / "svhn_ood_subsample_idx.npy")
if Mmax > 1:
    ens_test = np.load(PR / f"ensemble_M{Mmax}_cifar10_test_probs.npy")
    ens_svhn = np.load(PR / f"ensemble_M{Mmax}_svhn_probs.npy")
else:
    ens_test, ens_svhn = base_test, base_svhn
pairs = [("Baseline (seed 0)", base_test, base_svhn, BASE_C), (ens_name, ens_test, ens_svhn, ENS_C)]


def entropy(p):
    return -np.sum(p * np.log(np.clip(p, 1e-12, 1)), axis=1)


# 1. reliability diagram
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True)
for ax, (name, pt, _, col) in zip(axes, pairs):
    ece, stats = C.compute_ece(pt, y)
    centers = [(lo + hi) / 2 for lo, hi, *_ in stats]
    ax.bar(centers, [s[3] for s in stats], width=1 / C.ECE_BINS, color=col, edgecolor="white",
           label="accuracy")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.set_title(f"{name}\nECE = {ece:.4f}")
    ax.set_xlabel("Confidence (max softmax probability)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
axes[0].set_ylabel("Accuracy within bin")
axes[0].legend(loc="upper left", frameon=False)
fig.suptitle("Reliability diagram, CIFAR-10 test (15 bins)")
fig.savefig(RES / "reliability_diagram.png"); plt.close(fig)

# 2. metrics vs ensemble size
ens_rows = [r for r in rows if r["M"] == 1 and r["kind"] == "baseline"] + \
           sorted([r for r in rows if r["kind"] == "ensemble"], key=lambda r: r["M"])
Ms = [r["M"] for r in ens_rows]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, (k, lab) in zip(axes, [("accuracy", "Accuracy"), ("nll", "NLL"), ("ece", "ECE (15 bins)")]):
    ax.plot(Ms, [r[k] for r in ens_rows], "o-", color=ENS_C, lw=2)
    ax.set_xlabel("Ensemble size M"); ax.set_ylabel(lab); ax.set_xticks(Ms)
    ax.set_title(f"{lab} vs ensemble size")
    ax.grid(alpha=.25)
fig.suptitle("Deep Ensembles, CIFAR-10 test (M=1 is the baseline)")
fig.tight_layout(); fig.savefig(RES / "metrics_vs_ensemble_size.png"); plt.close(fig)

# 3. predictive entropy histograms
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
bins = np.linspace(0, np.log(10), 40)
for ax, (name, pt, ps, col) in zip(axes, pairs):
    ax.hist(entropy(pt), bins=bins, alpha=.65, color="#4C72B0", label="CIFAR-10 test (in-dist.)", density=True)
    ax.hist(entropy(ps[sub]), bins=bins, alpha=.65, color="#C44E52", label="SVHN (OOD)", density=True)
    ax.set_title(name); ax.set_xlabel("Predictive entropy (nats)")
    ax.set_yscale("log")
axes[0].set_ylabel("Density (log scale)")
axes[0].legend(frameon=False, loc="upper right")
fig.suptitle("Predictive entropy: CIFAR-10 vs SVHN")
fig.tight_layout(); fig.savefig(RES / "entropy_histograms.png"); plt.close(fig)

# 4. per-class Mondrian coverage (fixed version)
names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
base_r = next(r for r in rows if r["kind"] == "baseline")
ens_r = next((r for r in rows if r["kind"] == "ensemble" and r["M"] == Mmax), base_r)
x = np.arange(10); w = .4
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.bar(x - w / 2, base_r["fixed_mondrian_per_class"], w, color=BASE_C, label="Baseline (seed 0)")
ax.bar(x + w / 2, ens_r["fixed_mondrian_per_class"], w, color=ENS_C, label=ens_name)
ax.axhline(0.90, color="k", ls="--", lw=1.2, label="target 0.90")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30)
ax.set_ylim(0.8, 1.0); ax.set_ylabel("Test coverage per true class")
ax.set_xlabel("True class")
ax.set_title("Mondrian (class-conditional) coverage, randomized APS, alpha = 0.1")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.28))
fig.savefig(RES / "mondrian_per_class_coverage.png"); plt.close(fig)
print("plots written to", RES)
