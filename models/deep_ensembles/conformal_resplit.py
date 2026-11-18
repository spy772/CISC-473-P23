"""Robust conformal numbers for the fixed (randomized APS) layer.

A single calibration/test split gives coverage that fluctuates by about +/-0.4%
(per class about +/-1.2%). To report stable numbers we pool the 12k calibration
and 12k test points of each configuration and re-split them at random 100 times,
recalibrating each time. Uses the cached probabilities from evaluate_ensemble.py.

Also writes baseline_conformal_fixed.json: the corrected counterpart of the
baseline's baseline_resnet20_results.json "conformal_metrics" block (the original
file is left untouched).

  python models/deep_ensembles/conformal_resplit.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import common as C  # noqa: E402
import conformal_fixed as cf  # noqa: E402

N_RESPLITS = 100
PR = C.RESULTS_DIR / "probs"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def run(P_all, Y_all, score="aps_rand"):
    cov, size, mcov, msize, pc = [], [], [], [], []
    n = len(Y_all) // 2
    for r in range(N_RESPLITS):
        perm = np.random.RandomState(1000 + r).permutation(len(Y_all))
        a, b = perm[:n], perm[n:]
        s = cf.split_conformal(P_all[a], Y_all[a], P_all[b], Y_all[b], C.ALPHA, score, seed=r)
        m = cf.mondrian_conformal(P_all[a], Y_all[a], P_all[b], Y_all[b], C.ALPHA, score, seed=r)
        cov.append(s["coverage"]); size.append(s["avg_set_size"])
        mcov.append(m["coverage"]); msize.append(m["avg_set_size"]); pc.append(m["per_class_coverage"])
    pc = np.array(pc)
    return dict(split_cov_mean=float(np.mean(cov)), split_cov_sd=float(np.std(cov)),
                split_size_mean=float(np.mean(size)),
                mondrian_cov_mean=float(np.mean(mcov)), mondrian_cov_sd=float(np.std(mcov)),
                mondrian_size_mean=float(np.mean(msize)),
                per_class_cov_mean=pc.mean(0).tolist(), per_class_cov_sd=pc.std(0).tolist())


def main():
    images, labels, (tr, te, ca) = C.get_splits()
    Y_all = np.r_[labels[ca], labels[te]]
    mem = {s: {n: np.load(PR / f"member_seed{s}_{n}.npy") for n in ("cal", "test")} for s in range(5)}
    configs = [("baseline (seed 0)", [0])] + [(f"member seed {s}", [s]) for s in range(1, 5)] + \
              [(f"ensemble M={M}", list(range(M))) for M in range(2, 6)]
    out = {}
    for name, ms in configs:
        P_all = np.concatenate([np.mean([mem[s][n] for s in ms], 0) for n in ("cal", "test")])
        out[name] = run(P_all, Y_all)
        print(name, round(out[name]["split_cov_mean"], 4), round(out[name]["split_size_mean"], 3), flush=True)

    json.dump(dict(n_resplits=N_RESPLITS, alpha=C.ALPHA, score="randomized APS", results=out),
              open(C.RESULTS_DIR / "conformal_resplit.json", "w"), indent=2)

    L = ["# Fixed conformal layer (randomized APS), averaged over 100 random cal/test re-splits", "",
         f"alpha = {C.ALPHA} (target coverage 0.90). Each row: mean +/- sd over {N_RESPLITS} re-splits of "
         "the pooled 24,000 calibration+test points (12,000 / 12,000). Split = marginal split conformal; "
         "Mondrian = class-conditional.", "",
         "| Configuration | Split coverage | Split avg set size | Mondrian coverage | Mondrian avg set size |",
         "|---|---|---|---|---|"]
    for name, r in out.items():
        L.append(f"| {name} | {r['split_cov_mean']:.4f} ± {r['split_cov_sd']:.4f} | {r['split_size_mean']:.3f} | "
                 f"{r['mondrian_cov_mean']:.4f} ± {r['mondrian_cov_sd']:.4f} | {r['mondrian_size_mean']:.3f} |")
    e = out["ensemble M=5"]
    L += ["", "Per-class Mondrian coverage, ensemble M=5 (mean ± sd):", "",
          "| " + " | ".join(CLASSES) + " |", "|" + "---|" * 10,
          "| " + " | ".join(f"{m:.3f} ± {s:.3f}" for m, s in zip(e["per_class_cov_mean"], e["per_class_cov_sd"])) + " |"]
    (C.RESULTS_DIR / "conformal_resplit_table.md").write_text("\n".join(L) + "\n", encoding="utf8")

    # corrected counterpart of Edward's baseline_resnet20_results.json conformal block
    b = json.load(open(C.REPO / "baseline_resnet20_results.json"))
    rows = json.load(open(C.RESULTS_DIR / "results.json"))["rows"]
    base = next(r for r in rows if r["kind"] == "baseline")
    fixed = dict(original_baseline_metrics=b["baseline_metrics"],
                 original_conformal_metrics_shared_code=b["conformal_metrics"],
                 corrected_conformal_metrics=dict(
                     method="split conformal (randomized APS, consistent set rule)", alpha=C.ALPHA,
                     target_coverage=1 - C.ALPHA,
                     single_split_coverage=base["fixed_split_cov"],
                     single_split_avg_set_size=base["fixed_split_size"],
                     mean_coverage_over_100_resplits=out["baseline (seed 0)"]["split_cov_mean"],
                     avg_set_size_over_100_resplits=out["baseline (seed 0)"]["split_size_mean"]),
                 note="Counterpart of baseline_resnet20_results.json; the original file is unchanged.")
    json.dump(fixed, open(C.RESULTS_DIR / "baseline_conformal_fixed.json", "w"), indent=2)
    print("wrote conformal_resplit.{json}, conformal_resplit_table.md, baseline_conformal_fixed.json")


if __name__ == "__main__":
    main()
