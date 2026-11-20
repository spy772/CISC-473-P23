"""train_with_noise(): train any UQ adapter on train_aug (CIFAR-10 + SVHN noise)."""
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common
from common import DEVICE

from . import REPO
from .data import build_train_aug, make_loaders
from .models import UQModel

DEFAULT_CKPT_DIR = REPO / "models" / "noise_testing"


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_tag(model, noise_ratio, seed, noise_seed, n_svhn=None, epochs=common.EPOCHS, smoke=False):
    """File-name stem identifying a run: model, amount of noise, seeds, epochs."""
    noise = f"nsvhn{n_svhn}" if n_svhn is not None else f"noise{noise_ratio:g}"
    tag = f"{model.name}_{noise}_seed{seed}"
    if noise_seed != seed:
        tag += f"_nseed{noise_seed}"
    if epochs != common.EPOCHS:
        tag += f"_ep{epochs}"
    if smoke:
        tag += "_smoke"
    return tag


@torch.no_grad()
def _eval_loss_acc(net, loader):
    net.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = net(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def _train_one(net, loaders, epochs, lr, log_every, label):
    """Shared recipe (baseline notebook): SGD Nesterov, cosine LR, cross-entropy."""
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=common.MOMENTUM,
                          weight_decay=common.WEIGHT_DECAY, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    is_noise_all = loaders["is_noise"]
    keys = ["train_loss", "train_acc", "train_acc_cifar", "train_acc_svhn", "cal_loss", "cal_acc"]
    hist = {k: [] for k in keys}
    t0 = time.time()

    for epoch in range(epochs):
        net.train()
        loss_sum, n = 0.0, 0
        corr = {"all": 0, "cifar": 0, "svhn": 0}
        cnt = {"all": 0, "cifar": 0, "svhn": 0}
        for x, y, idx in loaders["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = net(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            hit = (logits.argmax(1) == y).cpu()
            noise = is_noise_all[idx]
            loss_sum += loss.item() * y.size(0)
            for k, m in (("all", torch.ones_like(noise)), ("cifar", ~noise), ("svhn", noise)):
                corr[k] += hit[m].sum().item()
                cnt[k] += m.sum().item()
        sched.step()

        cal_loss, cal_acc = _eval_loss_acc(net, loaders["cal"])
        row = [loss_sum / cnt["all"], corr["all"] / cnt["all"],
               corr["cifar"] / max(cnt["cifar"], 1),
               corr["svhn"] / cnt["svhn"] if cnt["svhn"] else float("nan"),
               cal_loss, cal_acc]
        for k, v in zip(keys, row):
            hist[k].append(v)
        if log_every and ((epoch + 1) % log_every == 0 or epoch + 1 == epochs):
            print(f"[{label}] epoch {epoch + 1:3d}/{epochs}  loss={row[0]:.4f} "
                  f"acc={row[1]:.4f} (cifar {row[2]:.4f}, svhn {row[3]:.4f})  "
                  f"cal_loss={cal_loss:.4f} cal_acc={cal_acc:.4f}  "
                  f"{(time.time() - t0) / 60:.1f}min")
    hist["train_minutes"] = (time.time() - t0) / 60
    return hist


def train_with_noise(model, noise_ratio=0.25, n_svhn=None, epochs=common.EPOCHS, seed=0,
                     noise_seed=None, lr=common.LR, smoke=False, ckpt_dir=DEFAULT_CKPT_DIR,
                     resume=True, deterministic=False, log_every=1):
    """Train a UQ adapter (Baseline(), MCDropout(), LaplaceLL(), SNGP(), DeepEnsemble())
    on train_aug = CIFAR-10 train split + SVHN images with random labels.

    model       : an adapter from pipeline.models. Its networks are (re)built here from `seed`.
    noise_ratio : SVHN images added, as a fraction of the 36k CIFAR training images.
    n_svhn      : absolute number of SVHN images (overrides noise_ratio).
    seed        : run seed -> weight init, batch order, augmentation, eval-time sampling.
    noise_seed  : which SVHN images / which random labels (defaults to `seed`).
    smoke       : tiny subsets for a quick end-to-end check (pass epochs=1 too).
    resume      : load an existing checkpoint for this exact run instead of retraining.

    Returns dict(model, history, loaders, data, config, tag). The post-hoc step
    (Laplace / SNGP covariance) is run on the un-augmented train_aug.
    """
    noise_seed = seed if noise_seed is None else noise_seed
    set_seed(seed, deterministic)
    data = build_train_aug(noise_ratio=noise_ratio, n_svhn=n_svhn, seed=noise_seed)
    loaders = make_loaders(data, seed=seed, smoke=smoke)

    tag = run_tag(model, noise_ratio, seed, noise_seed, n_svhn, epochs, smoke)
    config = dict(model=model.name, model_config=model.config(), noise_ratio=noise_ratio,
                  n_svhn=data["n_svhn"], n_cifar=data["n_cifar"], seed=seed,
                  noise_seed=noise_seed, epochs=epochs, lr=lr, smoke=smoke,
                  n_train=len(loaders["train"].dataset))
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # if we use multiple seeds, model.nets will have multiple models.
    model.build(seed)

    member_seeds = model.member_seeds(seed)
    multi = len(model.nets) > 1
    history = []
    for i, (net, ms) in enumerate(zip(model.nets, member_seeds)):
        path = ckpt_dir / (f"{tag}_m{i}.pt" if multi else f"{tag}.pt")
        label = f"{tag}" + (f" m{i}" if multi else "")
        if resume and path.exists():
            ck = torch.load(path, map_location=DEVICE, weights_only=False)
            if ck.get("config", {}).get("model_config") != config["model_config"]:
                raise ValueError(f"{path} was trained with a different model config; "
                                 f"delete it or pass resume=False")
            net.load_state_dict(ck["state_dict"])
            history.append(ck.get("history", {}))
            print(f"[{label}] loaded checkpoint {path.name}")
            continue
        set_seed(ms, deterministic)
        loaders["train"].generator.manual_seed(ms)

        # _train_one trains a specific network (e.g. MCDropout, seed 2) on a full train (e.g. 40 epochs)
        h = _train_one(net, loaders, epochs, lr, log_every, label)
        history.append(h)
        torch.save({"state_dict": net.state_dict(), "history": h,
                    "config": {**config, "member": i, "member_seed": ms}}, path)

    # Evaluate the models that we created with the diffetent seeds.
    for net in model.nets:
        net.eval()

    t0 = time.time()
    model.post_fit(loaders["train_noaug"])
    if type(model).post_fit is not UQModel.post_fit:
        print(f"[{tag}] post-hoc fit done in {time.time() - t0:.1f}s")

    # Return the trained model
    return dict(model=model, history=history if multi else history[0], loaders=loaders,
                data=data, config=config, tag=tag)


def run_experiment(model_factory, noise_ratios, seeds, alpha=common.ALPHA, n_resplits=0,
                   smoke=False, **train_kwargs):
    """Train + evaluate `model_factory()` for every (noise_ratio, seed) pair.
    model_factory is a class or zero-arg callable, e.g. `Baseline` or `lambda: MCDropout(T=20)`.
    Existing checkpoints are reused, so adding seeds later only trains the new runs.
    Returns a list of result dicts (one per run) from evaluate_conformal."""
    from .evaluate import evaluate_conformal
    rows = []
    for r in noise_ratios:
        for s in seeds:

            # Train the model using the specified (seed, noise_ratio) pair
            run = train_with_noise(model_factory(), noise_ratio=r, seed=s, smoke=smoke,
                                   **train_kwargs)

            # Test the model. We get its ECE, NLL, and coverage metrics.
            res = evaluate_conformal(run["model"], run["loaders"]["cal"], run["loaders"]["test"],
                                     alpha=alpha, seed=s, n_resplits=n_resplits,
                                     tag=run["tag"], config=run["config"])
            rows.append(res)
    return rows
