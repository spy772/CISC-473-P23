"""Add-on dumper: per-member softmax probabilities of a pipeline-trained Deep Ensemble.

main's evaluate_conformal only saves ensemble-level cal/test probabilities (and its .npz files are git-ignored), so the DE add-ons
(mutual information, single-member baselines, OOD on SVHN, ablation over members) need per-member probabilities. This loads the
checkpoints written by pipeline.train.train_with_noise (models/noise_testing/{tag}_m{i}.pt) and writes, for ensemble seed s:

  results/noise_testing/de_addon/{tag}/m{i}.npz     test, cal, ood_svhn probabilities (float32, (n, 10))
  results/noise_testing/de_addon/{tag}/labels.npz   test_labels, cal_labels
  results/noise_testing/de_addon/{tag}/meta.json    seeds, ckpt names, train minutes, hardware, git commit, inference time

{tag} = pipeline.train.run_tag(...) = deep_ensemble_noise0_seed{s} (members _m0.._m4 have seeds s*100 + i).
No model/split/metric code is duplicated: splits, loaders, model and softmax come from pipeline/ and common.py.
OOD set = SVHN official test split (26,032 images), CIFAR-10 normalisation, no augmentation.

  python src/dump_member_probs.py --seeds 0 1 2            # real runs
  python src/dump_member_probs.py --seeds 0 --smoke        # smoke checkpoints (results/de_smoke/)
"""
import argparse
import json
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
import pipeline  # noqa: E402,F401  (puts models/deep_ensembles on sys.path)
import common as C  # noqa: E402
from pipeline.data import build_train_aug, make_loaders  # noqa: E402
from pipeline.models import DeepEnsemble, _softmax_probs  # noqa: E402
from pipeline.train import run_tag  # noqa: E402

SVHN_TEST = REPO / "data_svhn_test" / "test_32x32.mat"
SVHN_URL = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
N_SVHN_TEST, N_SVHN_SMOKE = 26032, 2000
N_MEMBERS = 5


def roots(smoke=False):
    """Where checkpoints, main-format results and add-on probabilities live."""
    if smoke:
        base = REPO / "results" / "de_smoke"          # git-ignored; never mixed with real results
        return dict(ckpt=base / "ckpt", main=base / "main", addon=base / "de_addon", out=base / "out", fig=base / "figures")
    main = REPO / "results" / "noise_testing"
    return dict(ckpt=REPO / "models" / "noise_testing", main=main, addon=main / "de_addon",
                out=REPO / "results" / "de", fig=REPO / "figures" / "de")


def tag_for(seed, smoke=False, epochs=None):
    epochs = (1 if smoke else C.EPOCHS) if epochs is None else epochs
    return run_tag(DeepEnsemble(N_MEMBERS), 0, seed, seed, None, epochs, smoke)


def load_svhn_test(smoke=False):
    import scipy.io as sio
    if not SVHN_TEST.exists():
        SVHN_TEST.parent.mkdir(exist_ok=True)
        print("downloading SVHN test split ->", SVHN_TEST, flush=True)
        urllib.request.urlretrieve(SVHN_URL, SVHN_TEST)
    X = sio.loadmat(str(SVHN_TEST))["X"].transpose(3, 0, 1, 2)
    assert X.shape == (N_SVHN_TEST, 32, 32, 3), X.shape
    return np.ascontiguousarray(X[:N_SVHN_SMOKE] if smoke else X)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return None


def hardware():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else (platform.processor() or platform.machine())


def dump_ensemble(seed, smoke=False, force=False):
    R = roots(smoke)
    tag = tag_for(seed, smoke)
    out = R["addon"] / tag
    if (out / "meta.json").exists() and not force:
        print(f"[{tag}] add-on probs already dumped, skipping")
        return out
    model = DeepEnsemble(N_MEMBERS).build(seed)
    cks = []
    for i, net in enumerate(model.nets):
        path = R["ckpt"] / f"{tag}_m{i}.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: train it first (src/de_train.py --seeds {seed})")
        ck = torch.load(path, map_location=C.DEVICE, weights_only=False)
        net.load_state_dict(ck["state_dict"])
        cks.append(ck)
    data = build_train_aug(noise_ratio=0, seed=seed)
    L = make_loaders(data, seed=seed, smoke=smoke)
    svhn = DataLoader(C.ArrayImages(load_svhn_test(smoke)), batch_size=256, shuffle=False, num_workers=0)
    loaders = {"test": L["test"], "cal": L["cal"], "ood_svhn": svhn}
    out.mkdir(parents=True, exist_ok=True)
    members, labels = [], {}
    for i, net in enumerate(model.nets):
        arrs, n_img, t_inf = {}, 0, 0.0
        for name, ld in loaders.items():
            t0 = time.time()
            p, y = _softmax_probs([net], ld)
            if C.DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_inf += time.time() - t0
            n_img += len(p)
            arrs[name] = p.astype(np.float32)
            if name != "ood_svhn":
                labels[name + "_labels"] = y.astype(np.int64)
        np.savez_compressed(out / f"m{i}.npz", **arrs)
        acc = float((arrs["test"].argmax(1) == labels["test_labels"]).mean())
        members.append(dict(member=i, member_seed=cks[i]["config"]["member_seed"], ckpt=f"{tag}_m{i}.pt",
                            train_minutes=cks[i].get("history", {}).get("train_minutes"), test_acc=acc,
                            inference_sec_per_1000_imgs=1000 * t_inf / n_img,
                            n_params=int(sum(p.numel() for p in net.parameters()))))
        print(f"[{tag}] m{i} (seed {members[-1]['member_seed']}): test acc {acc:.4f}", flush=True)
    np.savez_compressed(out / "labels.npz", **labels)
    meta = dict(tag=tag, ensemble_seed=seed, smoke=smoke, hardware=hardware(), device=str(C.DEVICE), torch=torch.__version__,
                git_commit=git_commit(), n_test=int(len(labels["test_labels"])), n_cal=int(len(labels["cal_labels"])),
                n_ood=int(len(load_svhn_test(smoke))), members=members)
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="ensemble seeds (members seed*100 + i)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for s in a.seeds:
        dump_ensemble(s, a.smoke, a.force)
