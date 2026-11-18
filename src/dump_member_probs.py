"""Step 2: dump softmax probabilities of each DE member (inference only).

For each seed s writes results/raw/de_members/seed{s}/{test,cal,ood_svhn}_probs.npy
(float32, (n, 10)) + meta.json, and results/raw/labels/{test,cal}_labels.npy +
split_indices.json. Every metric in src/de_eval.py is computed from these files.

  python src/dump_member_probs.py                 # all seeds with a checkpoint, skips finished ones
  python src/dump_member_probs.py --seeds 5 6 7   # specific seeds
  python src/dump_member_probs.py --force         # re-dump
OOD = SVHN official test split (26,032 imgs), CIFAR-10 normalisation, no augmentation.
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
sys.path.insert(0, str(REPO / "models" / "deep_ensembles"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
import common as C  # noqa: E402

RAW = REPO / "results" / "raw"
SVHN_TEST = REPO / "data_svhn_test" / "test_32x32.mat"
SVHN_URL = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
N_SVHN_TEST = 26032


def ckpt_path(seed):
    return C.BASELINE_CKPT if seed == 0 else C.ENS_DIR / f"resnet20_seed{seed}.pt"


def load_svhn_test():
    import scipy.io as sio
    if not SVHN_TEST.exists():
        SVHN_TEST.parent.mkdir(exist_ok=True)
        print("downloading SVHN test split ->", SVHN_TEST, flush=True)
        urllib.request.urlretrieve(SVHN_URL, SVHN_TEST)
    X = sio.loadmat(str(SVHN_TEST))["X"].transpose(3, 0, 1, 2)
    assert X.shape == (N_SVHN_TEST, 32, 32, 3), X.shape
    return X


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return None


def hardware():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return platform.processor() or platform.machine()


def train_time(seed):
    p = C.ENS_DIR / f"train_time_seed{seed}.json"
    if p.exists():
        return json.load(open(p))["wall_clock_sec"]
    return None  # seed 0 (baseline checkpoint): training time not recorded in the repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    seeds = a.seeds if a.seeds is not None else [s for s in range(0, 64) if ckpt_path(s).exists()]

    images, labels, (tr, te, ca) = C.get_splits()
    RAW.joinpath("labels").mkdir(parents=True, exist_ok=True)
    np.save(RAW / "labels" / "test_labels.npy", labels[te].astype(np.int64))
    np.save(RAW / "labels" / "cal_labels.npy", labels[ca].astype(np.int64))
    with open(RAW / "labels" / "split_indices.json", "w") as f:
        json.dump(dict(split_seed=C.SPLIT_SEED, fracs=[C.TRAIN_FRAC, C.TEST_FRAC, C.CAL_FRAC],
                       n_train=len(tr), n_test=len(te), n_cal=len(ca),
                       train=tr.tolist(), test=te.tolist(), cal=ca.tolist()), f)
    assert (len(tr), len(te), len(ca)) == (36000, 12000, 12000)

    svhn = load_svhn_test()
    loaders = {
        "test": DataLoader(C.CIFAR10Array(images, labels, te, False), batch_size=256, num_workers=0),
        "cal": DataLoader(C.CIFAR10Array(images, labels, ca, False), batch_size=256, num_workers=0),
        "ood_svhn": DataLoader(C.ArrayImages(svhn), batch_size=256, num_workers=0),
    }
    for s in seeds:
        out = RAW / "de_members" / f"seed{s}"
        if (out / "meta.json").exists() and not a.force:
            print(f"seed {s}: already dumped, skipping")
            continue
        if not ckpt_path(s).exists():
            print(f"seed {s}: checkpoint missing, skipping")
            continue
        out.mkdir(parents=True, exist_ok=True)
        model = C.build_model()
        model.load_state_dict(torch.load(ckpt_path(s), map_location=C.DEVICE))
        n_img, t_inf = 0, 0.0
        for name, ld in loaders.items():
            t0 = time.time()
            p, y = C.collect_probs_labels(model, ld)
            if C.DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_inf += time.time() - t0
            n_img += len(p)
            np.save(out / f"{name}_probs.npy", p.astype(np.float32))
            if name != "ood_svhn":
                assert (y == np.load(RAW / "labels" / f"{name}_labels.npy")).all()
        acc = float((np.load(out / "test_probs.npy").argmax(1) == np.load(RAW / "labels" / "test_labels.npy")).mean())
        meta = dict(seed=s, ckpt=str(ckpt_path(s).relative_to(REPO)).replace("\\", "/"),
                    train_time_sec=train_time(s), hardware=hardware(), device=str(C.DEVICE),
                    torch=torch.__version__, git_commit=git_commit(), test_acc=acc,
                    inference_sec_per_1000_imgs=1000 * t_inf / n_img,
                    n_params=int(sum(p.numel() for p in model.parameters())))
        json.dump(meta, open(out / "meta.json", "w"), indent=2)
        print(f"seed {s}: test acc {acc:.4f}, {meta['inference_sec_per_1000_imgs']:.3f}s/1k imgs", flush=True)


if __name__ == "__main__":
    main()
