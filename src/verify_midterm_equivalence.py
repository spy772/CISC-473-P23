"""Cross-check (NOT final numbers): run main's pipeline evaluation on the five MIDTERM checkpoints (seeds 0-4, trained before
the shared pipeline existed) and confirm it reproduces the midterm numbers, i.e. that pipeline.evaluate_conformal and the
midterm evaluation are the same computation.

Needs the git-ignored midterm checkpoints: models/baseline/baseline_resnet20_seed0.pt, models/deep_ensembles/resnet20_seed{1..4}.pt.
Writes results/de/midterm_equivalence_check.md.   python src/verify_midterm_equivalence.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import pipeline  # noqa: E402,F401
import common as C  # noqa: E402
from pipeline import Baseline, DeepEnsemble, build_train_aug, evaluate_conformal, make_loaders  # noqa: E402

CK = {0: C.BASELINE_CKPT, **{s: C.ENS_DIR / ("resnet20_seed%d.pt" % s) for s in range(1, 5)}}
EXPECT = {"single seed 0": (0.9028, 0.3026, 0.0312), "ensemble M=5": (0.9273, 0.2232, 0.0164)}
EXPECT_APS = {"single seed 0": (0.8959, 1.1675), "ensemble M=5": (0.8928, 1.1694)}   # midterm randomized APS, seed 0, single split


def main():
    missing = [str(p) for p in CK.values() if not p.exists()]
    if missing:
        sys.exit("midterm checkpoints missing: %s" % missing)
    L = make_loaders(build_train_aug(noise_ratio=0, seed=0), seed=0)
    de = DeepEnsemble(5).build(0)
    for i, net in enumerate(de.nets):
        net.load_state_dict(torch.load(CK[i], map_location=C.DEVICE))
    one = Baseline().build(0)
    one.nets[0].load_state_dict(torch.load(CK[0], map_location=C.DEVICE))
    lines = ["# Midterm-checkpoint equivalence check (cross-check only, not final numbers)", "",
             "main's `pipeline.evaluate_conformal` (seed 0, 100 re-splits) run on the midterm seed 0-4 checkpoints.", "",
             "| model | Acc | NLL | ECE | Brier | APS-rand cov | APS-rand size | LAC cov | LAC size | resplit cov (ddof=0) | resplit size |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    ok = True
    for name, m in (("single seed 0", one), ("ensemble M=5", de)):
        r = evaluate_conformal(m, L["cal"], L["test"], seed=0, n_resplits=100, save=False)
        a, l = r["conformal"]["aps_rand"], r["conformal"]["lac"]
        got = (r["accuracy"], r["nll"], r["ece"])
        good = all(abs(g - e) <= 0.00005 + 1e-9 for g, e in zip(got, EXPECT[name])) and \
            all(abs(g - e) <= 0.00005 + 1e-9 for g, e in zip((a["coverage"], a["avg_set_size"]), EXPECT_APS[name]))
        ok &= good
        lines.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f ± %.4f | %.4f ± %.4f |" % (
            name, *got, r["brier"], a["coverage"], a["avg_set_size"], l["coverage"], l["avg_set_size"],
            a["resplit"]["coverage_mean"], a["resplit"]["coverage_std"], a["resplit"]["avg_set_size_mean"], a["resplit"]["avg_set_size_std"]))
    lines += ["", "Midterm expectations: " + "; ".join("%s Acc/NLL/ECE %s, APS-rand cov/size %s" % (k, EXPECT[k], EXPECT_APS[k]) for k in EXPECT),
              "(accuracy of the ensemble is 11127/12000 = 0.92725, printed as 0.9273 at the midterm; tolerance is half a unit of the 4th decimal.)",
              "", "**Result: %s**" % ("PASS" if ok else "FAIL")]
    out = REPO / "results" / "de" / "midterm_equivalence_check.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
