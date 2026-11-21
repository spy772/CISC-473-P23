"""Train + evaluate Deep Ensembles through main's pipeline (the only supported DE training path).

For each ensemble seed s: pipeline.train_with_noise(DeepEnsemble(n=5), noise_ratio=0, seed=s) trains members with seeds s*100 + i
(checkpoints models/noise_testing/deep_ensemble_noise0_seed{s}_m{i}.pt, resumed per member if they exist), then
pipeline.evaluate_conformal writes main-format results/noise_testing/{tag}.json/.npz, then the add-on dumper writes per-member
probabilities (src/dump_member_probs.py).

  python src/de_train.py --seeds 0 1 2
  python src/de_train.py --seeds 0 --smoke        # 1 epoch, tiny subsets, writes only under results/de_smoke/
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
import pipeline  # noqa: E402
import common as C  # noqa: E402
from pipeline import DeepEnsemble, evaluate_conformal, train_with_noise  # noqa: E402
from dump_member_probs import dump_ensemble, roots  # noqa: E402

N_RESPLITS = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="ensemble seeds; members have seeds s*100 + i")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on tiny subsets, outputs under results/de_smoke/")
    a = ap.parse_args()
    R = roots(a.smoke)
    for s in a.seeds:
        run = train_with_noise(DeepEnsemble(n=5), noise_ratio=0, seed=s, smoke=a.smoke,
                               epochs=1 if a.smoke else C.EPOCHS, ckpt_dir=R["ckpt"])
        evaluate_conformal(run["model"], run["loaders"]["cal"], run["loaders"]["test"], seed=s,
                           n_resplits=N_RESPLITS, tag=run["tag"], config=run["config"], results_dir=R["main"])
        dump_ensemble(s, smoke=a.smoke)
        print(f"ensemble seed {s} done: {run['tag']}", flush=True)


if __name__ == "__main__":
    main()
