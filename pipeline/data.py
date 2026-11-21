"""Data: the shared CIFAR-10 60/20/20 split plus SVHN 'noise' images with random labels."""
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import common  # models/deep_ensembles/common.py (path set in pipeline/__init__.py)
from common import CIFAR10Array, SVHN_DIR

NUM_CLASSES = 10
SMOKE_TRAIN, SMOKE_EVAL = 2048, 2000


@lru_cache(maxsize=1)
def get_cifar_splits():
    """(images NHWC uint8, labels, train_idx, test_idx, cal_idx). The split is fixed
    (split seed 0), identical to every other model notebook in the repo."""
    images, labels, (train_idx, test_idx, cal_idx) = common.get_splits()
    return images, labels, train_idx, test_idx, cal_idx


@lru_cache(maxsize=1)
def load_svhn_train():
    """All 73,257 SVHN *train* images (NHWC uint8) from the repo's chunked .mat files.
    The SVHN test set (data/test_32x32.mat) is deliberately not used for train_aug."""
    import scipy.io as sio
    X = np.concatenate([sio.loadmat(SVHN_DIR / f"train_chunk_{i}.mat")["X"] for i in range(4)],
                       axis=3)
    return np.ascontiguousarray(X.transpose(3, 0, 1, 2))


def build_train_aug(noise_ratio=0.25, n_svhn=None, seed=0):
    """train_aug = CIFAR-10 train split + n_svhn SVHN images with uniformly random labels 0-9.

    n_svhn defaults to round(noise_ratio * len(CIFAR train)) = round(noise_ratio * 36000).
    `seed` controls which SVHN images are drawn and the labels they get.

    Returns dict with:
      train_aug        - CIFAR10Array with augmentation (crop + flip + CIFAR normalisation)
      train_aug_noaug  - same images, no augmentation (for Laplace / SNGP post-hoc fits)
      test, cal        - clean CIFAR-10 test / calibration splits (identical for every run)
      is_noise         - bool mask over train_aug, True for SVHN images
      svhn_idx         - indices into the SVHN train set that were used
      n_svhn, n_cifar
    """
    images, labels, train_idx, test_idx, cal_idx = get_cifar_splits()
    n_cifar = len(train_idx)
    if n_svhn is None:
        n_svhn = int(round(noise_ratio * n_cifar))

    rng = np.random.RandomState(seed)
    if n_svhn > 0:
        svhn = load_svhn_train()
        if n_svhn > len(svhn):
            raise ValueError(f"n_svhn={n_svhn} exceeds the {len(svhn)} available SVHN images")
        svhn_idx = np.sort(rng.choice(len(svhn), size=n_svhn, replace=False))
        svhn_labels = rng.randint(0, NUM_CLASSES, size=n_svhn).astype(np.int64)
        aug_images = np.concatenate([images[train_idx], svhn[svhn_idx]], axis=0)
        aug_labels = np.concatenate([labels[train_idx], svhn_labels], axis=0)
    else:
        svhn_idx = np.array([], dtype=np.int64)
        aug_images, aug_labels = images[train_idx], labels[train_idx]

    is_noise = np.zeros(len(aug_labels), dtype=bool)
    is_noise[n_cifar:] = True
    all_idx = np.arange(len(aug_labels))

    return dict(
        train_aug=CIFAR10Array(aug_images, aug_labels, all_idx, train=True),
        train_aug_noaug=CIFAR10Array(aug_images, aug_labels, all_idx, train=False),
        test=CIFAR10Array(images, labels, test_idx, train=False),
        cal=CIFAR10Array(images, labels, cal_idx, train=False),
        is_noise=is_noise, svhn_idx=svhn_idx, n_svhn=int(n_svhn), n_cifar=int(n_cifar),
    )


class _WithIndex(torch.utils.data.Dataset):
    """Wraps a dataset so each item also returns its index (to look up is_noise)."""
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, y, i


def _smoke_subset(ds, n, seed, is_noise=None):
    """Small subset for smoke tests. For train_aug keep the CIFAR/SVHN proportion."""
    rng = np.random.RandomState(seed)
    if is_noise is None or not is_noise.any():
        idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    else:
        frac = is_noise.mean()
        n_noise = int(round(n * frac))
        idx = np.concatenate([
            rng.choice(np.where(~is_noise)[0], size=n - n_noise, replace=False),
            rng.choice(np.where(is_noise)[0], size=n_noise, replace=False),
        ])
    return np.sort(idx)


def make_loaders(data, batch_size=common.BATCH_SIZE, eval_batch_size=256, seed=0, smoke=False):
    """DataLoaders for a build_train_aug() dict. The training loader yields (x, y, idx) so
    the training loop can split stats into CIFAR vs SVHN; the others yield (x, y).
    The training loader's shuffle order is seeded by `seed`."""
    train_ds, noaug_ds, test_ds, cal_ds = (data["train_aug"], data["train_aug_noaug"],
                                           data["test"], data["cal"])
    is_noise = data["is_noise"]
    if smoke:
        tr_idx = _smoke_subset(train_ds, SMOKE_TRAIN, seed=0, is_noise=is_noise)
        train_ds, noaug_ds = Subset(train_ds, tr_idx), Subset(noaug_ds, tr_idx)
        is_noise = is_noise[tr_idx]
        test_ds = Subset(test_ds, _smoke_subset(test_ds, SMOKE_EVAL, seed=1))
        cal_ds = Subset(cal_ds, _smoke_subset(cal_ds, SMOKE_EVAL, seed=2))

    g = torch.Generator().manual_seed(seed)
    return dict(
        train=DataLoader(_WithIndex(train_ds), batch_size=batch_size, shuffle=True,
                         num_workers=0, generator=g),
        train_noaug=DataLoader(noaug_ds, batch_size=eval_batch_size, shuffle=False, num_workers=0),
        test=DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=0),
        cal=DataLoader(cal_ds, batch_size=eval_batch_size, shuffle=False, num_workers=0),
        is_noise=torch.from_numpy(is_noise),
    )
