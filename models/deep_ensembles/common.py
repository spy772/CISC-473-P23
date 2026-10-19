"""Shared pieces for the Deep Ensembles runs, ported verbatim from
baseline_resnet20_conformal.ipynb (which is NOT modified). Same data split,
model, augmentation, optimizer, schedule and metric functions."""
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data" / "cifar-10-batches-py"
SVHN_DIR = REPO / "data_svhn_chunks"
BASELINE_CKPT = REPO / "models" / "baseline" / "baseline_resnet20_seed0.pt"
ENS_DIR = REPO / "models" / "deep_ensembles"
RESULTS_DIR = REPO / "results_shravan" / "deep_ensembles"

EPOCHS = 40
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
TRAIN_FRAC, TEST_FRAC, CAL_FRAC = 0.6, 0.2, 0.2
SPLIT_SEED = 0          # baseline split used SEEDS[0] == 0 for every model
ALPHA = 0.1
ECE_BINS = 15
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_cifar10(data_dir: Path = DATA_DIR):
    images, labels = [], []
    for i in range(1, 6):
        batch = _unpickle(data_dir / f"data_batch_{i}")
        images.append(batch[b"data"])
        labels.extend(batch[b"labels"])
    tb = _unpickle(data_dir / "test_batch")
    images.append(tb[b"data"])
    labels.extend(tb[b"labels"])
    images = np.concatenate(images, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, np.array(labels, dtype=np.int64)


def stratified_split(labels, fracs, seed):
    assert abs(sum(fracs) - 1.0) < 1e-6
    rng = np.random.RandomState(seed)
    split_idxs = [[] for _ in fracs]
    for c in np.unique(labels):
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        n = len(c_idx)
        cuts = np.cumsum([int(round(n * f)) for f in fracs])
        cuts[-1] = n
        start = 0
        for i, end in enumerate(cuts):
            split_idxs[i].append(c_idx[start:end])
            start = end
    return [np.concatenate(idx) for idx in split_idxs]


class CIFAR10Array(Dataset):
    def __init__(self, images, labels, indices, train: bool):
        self.images = images[indices]
        self.labels = labels[indices]
        self.train = train
        self.mean = np.array(CIFAR_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(CIFAR_STD, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.train:
            padded = np.pad(img, ((4, 4), (4, 4), (0, 0)), mode="reflect")
            top = np.random.randint(0, 9)
            left = np.random.randint(0, 9)
            img = padded[top:top + 32, left:left + 32, :]
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :]
        img = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        img = (img - self.mean) / self.std
        return torch.from_numpy(img.copy()), int(self.labels[i])


def get_splits():
    images, labels = load_cifar10()
    return images, labels, stratified_split(
        labels, [TRAIN_FRAC, TEST_FRAC, CAL_FRAC], seed=SPLIT_SEED)


def build_model():
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20",
                           pretrained=False, trust_repo=True, verbose=False)
    return model.to(DEVICE)


def load_svhn(test_mat=None):
    """SVHN OOD images (NHWC uint8). Uses SVHN test .mat if given/available
    (test_32x32.mat), else the repo's train chunks."""
    import scipy.io as sio
    cands = [Path(test_mat)] if test_mat else [SVHN_DIR / "test_32x32.mat",
                                               REPO / "data_svhn" / "test_32x32.mat"]
    for p in cands:
        if p.exists():
            X = sio.loadmat(p)["X"]
            return X.transpose(3, 0, 1, 2), f"SVHN test ({p.name})"
    X = np.concatenate([sio.loadmat(SVHN_DIR / f"train_chunk_{i}.mat")["X"]
                        for i in range(4)], axis=3)
    return X.transpose(3, 0, 1, 2), "SVHN train chunks (test split not in repo)"


class ArrayImages(Dataset):
    """Un-augmented, CIFAR-normalised images (dummy labels) for OOD inference."""
    def __init__(self, images):
        self.images = images
        self.mean = np.array(CIFAR_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(CIFAR_STD, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i].astype(np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(((img - self.mean) / self.std).copy()), 0


@torch.no_grad()
def collect_probs_labels(model, loader):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        P.append(F.softmax(model(x.to(DEVICE)), dim=1).cpu().numpy())
        Y.append(np.asarray(y))
    return np.concatenate(P), np.concatenate(Y)


# ---------------- metrics (copied from baseline notebook) ----------------
def compute_nll(probs, labels, eps=1e-12):
    p_true = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))


def compute_ece(probs, labels, n_bins=ECE_BINS):
    conf = probs.max(axis=1)
    acc = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, stats = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        prop = m.mean()
        if prop > 0:
            ece += prop * abs(acc[m].mean() - conf[m].mean())
            stats.append((lo, hi, prop, acc[m].mean(), conf[m].mean()))
    return float(ece), stats


def compute_brier(probs, labels):
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


# ---------------- shared conformal layer (verbatim from notebook cell 14) ----------------
def aps_scores(probs, labels):
    order = np.argsort(-probs, axis=1)
    ranked = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(ranked, axis=1)
    true_rank = np.array([np.where(order[i] == labels[i])[0][0] for i in range(len(labels))])
    return cumsum[np.arange(len(labels)), true_rank]


def aps_predict_sets(probs, qhat):
    order = np.argsort(-probs, axis=1)
    ranked = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(ranked, axis=1)
    sets = []
    for i in range(probs.shape[0]):
        k = np.searchsorted(cumsum[i], qhat) + 1
        k = min(k, probs.shape[1])
        sets.append(set(order[i, :k].tolist()))
    return sets


def shared_split_conformal(cal_probs, cal_labels, test_probs, test_labels, alpha=ALPHA):
    """Exactly the notebook's cell 14 logic."""
    cal_scores = aps_scores(cal_probs, cal_labels)
    n = len(cal_scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat = np.quantile(cal_scores, q_level, method="higher")
    sets = aps_predict_sets(test_probs, qhat)
    cov = float(np.mean([test_labels[i] in sets[i] for i in range(len(test_labels))]))
    return dict(qhat=float(qhat), coverage=cov,
                avg_set_size=float(np.mean([len(s) for s in sets])))
