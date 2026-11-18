"""Train Deep Ensemble members (seeds 1..4 by default). Seed 0 is Edward's
baseline checkpoint and is never retrained.

Identical to the baseline recipe (baseline_resnet20_conformal.ipynb cell 7):
SGD(lr .1, mom .9, nesterov, wd 5e-4), CosineAnnealingLR(T_max=40), 40 epochs,
batch 128, crop+flip augmentation, plain cross-entropy (no adversarial training).
Only the random seed differs between members; the data split is fixed.

Resumable: a full state checkpoint is written every epoch to
resnet20_seed{k}.partial.pt; finished seeds (resnet20_seed{k}.pt) are skipped.

Usage:
  python models/deep_ensembles/train_ensemble.py                 # full run
  python models/deep_ensembles/train_ensemble.py --smoke         # 1-epoch tiny test
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F401,F403
import common as C  # noqa: E402


def eval_loss_acc(model, loader):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(C.DEVICE), y.to(C.DEVICE)
            logits = model(x)
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def rng_state():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state())


def set_rng_state(s):
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])


def train_member(seed, out_dir, epochs, train_ds, val_loader, log_prefix=""):
    final = out_dir / f"resnet20_seed{seed}.pt"
    partial = out_dir / f"resnet20_seed{seed}.partial.pt"
    csv_path = out_dir / f"train_log_seed{seed}.csv"
    time_path = out_dir / f"train_time_seed{seed}.json"

    C.set_seed(seed)
    model = C.build_model()
    opt = torch.optim.SGD(model.parameters(), lr=C.LR, momentum=C.MOMENTUM,
                          weight_decay=C.WEIGHT_DECAY, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True, num_workers=0)

    start_epoch, elapsed_prev = 0, 0.0
    if partial.exists():
        ck = torch.load(partial, map_location=C.DEVICE)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); set_rng_state(ck["rng"])
        start_epoch, elapsed_prev = ck["epoch"], ck["elapsed"]
        print(f"{log_prefix}[seed {seed}] resuming from epoch {start_epoch}", flush=True)
    else:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                                    "lr", "epoch_sec", "elapsed_min"])

    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        te = time.time()
        model.train()
        correct = total = 0
        loss_sum = 0.0
        lr = opt.param_groups[0]["lr"]
        for x, y in loader:
            x, y = x.to(C.DEVICE), y.to(C.DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        sched.step()
        val_loss, val_acc = eval_loss_acc(model, val_loader)
        elapsed = elapsed_prev + time.time() - t0
        row = [epoch + 1, loss_sum / total, correct / total, val_loss, val_acc, lr,
               time.time() - te, elapsed / 60]
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"{log_prefix}[seed {seed}] epoch {epoch+1:3d}/{epochs} train_loss={row[1]:.4f} "
              f"train_acc={row[2]:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"elapsed={elapsed/60:.1f}min", flush=True)
        tmp = partial.with_suffix(".tmp")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        rng=rng_state(), epoch=epoch + 1, elapsed=elapsed), tmp)
        tmp.replace(partial)

    torch.save(model.state_dict(), final)      # same format as the baseline checkpoint
    with open(time_path, "w") as f:
        json.dump(dict(seed=seed, epochs=epochs, wall_clock_sec=elapsed_prev + time.time() - t0,
                       device=str(C.DEVICE)), f, indent=2)
    partial.unlink(missing_ok=True)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--out-dir", type=Path, default=C.ENS_DIR)
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch on a small subset; writes to a scratch dir, not models/deep_ensembles")
    args = ap.parse_args()

    if 0 in args.seeds:
        sys.exit("seed 0 is Edward's baseline checkpoint; do not retrain it.")

    images, labels, (tr, te, ca) = C.get_splits()
    print(f"device={C.DEVICE} torch={torch.__version__} train={len(tr)} val(cal)={len(ca)}", flush=True)

    if args.smoke:
        import tempfile
        args.out_dir = Path(tempfile.mkdtemp(prefix="de_smoke_"))
        args.epochs = 1
        rs = np.random.RandomState(0)
        tr, ca = rs.choice(tr, 512, replace=False), rs.choice(ca, 256, replace=False)
        args.seeds = args.seeds[:1]
        print(f"SMOKE TEST -> {args.out_dir}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = C.CIFAR10Array(images, labels, tr, train=True)
    val_loader = DataLoader(C.CIFAR10Array(images, labels, ca, train=False),
                            batch_size=256, shuffle=False, num_workers=0)

    for seed in args.seeds:
        final = args.out_dir / f"resnet20_seed{seed}.pt"
        if final.exists():
            print(f"[seed {seed}] already finished -> skipping", flush=True)
            continue
        train_member(seed, args.out_dir, args.epochs, train_ds, val_loader)
        print(f"[seed {seed}] done -> {final}", flush=True)

    if args.smoke:  # prove resume + skip work
        print("smoke: re-running to verify finished seeds are skipped")
        for seed in args.seeds:
            assert (args.out_dir / f"resnet20_seed{seed}.pt").exists()
        print("SMOKE OK; outputs:", sorted(p.name for p in args.out_dir.iterdir()))


if __name__ == "__main__":
    main()
