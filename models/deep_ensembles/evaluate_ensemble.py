"""Evaluate individual members and Deep Ensembles (M = 1..5).

Member 0 is Edward's baseline checkpoint (loaded from models/baseline/); members
1..4 are models/deep_ensembles/resnet20_seed{k}.pt. Ensemble = mean of member
softmax probabilities. Missing members are skipped, so this can be run
(partially) before training finishes.

Writes results_shravan/deep_ensembles/{results.csv,results.json,comparison_table.md,
sanity_checks.md} and probability .npy files under .../probs/.

  python models/deep_ensembles/evaluate_ensemble.py [--svhn-test-mat path/to/test_32x32.mat]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F401,F403
import common as C  # noqa: E402
import conformal_fixed as cf  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

K = 10
N_OOD = 12000  # SVHN subsample size so ID/OOD are balanced (AUPR chance = 0.5)


def member_path(seed):
    return C.BASELINE_CKPT if seed == 0 else C.ENS_DIR / f"resnet20_seed{seed}.pt"


def entropy(p):
    return -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)


def member_probs(seed, loaders, probs_dir):
    """Cached softmax outputs of one member on cal / test / svhn."""
    names = list(loaders)
    files = {n: probs_dir / f"member_seed{seed}_{n}.npy" for n in names}
    if all(f.exists() for f in files.values()):
        return {n: np.load(f) for n, f in files.items()}
    model = C.build_model()
    model.load_state_dict(torch.load(member_path(seed), map_location=C.DEVICE))
    out = {}
    for n in names:
        p, _ = C.collect_probs_labels(model, loaders[n])
        out[n] = p.astype(np.float32)
        np.save(files[n], out[n])
    return out


def shared_mondrian(cal_p, cal_y, test_p, test_y, alpha=C.ALPHA):
    """Mondrian built from the *shared* APS score + shared set rule (one qhat per
    class). The shared notebook has no Mondrian code, so this is my construction."""
    qh = []
    for c in range(K):
        s = C.aps_scores(cal_p[cal_y == c], cal_y[cal_y == c])
        n = len(s)
        qh.append(np.quantile(s, min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0), method="higher"))
    order = np.argsort(-test_p, axis=1)
    cs = np.cumsum(np.take_along_axis(test_p, order, 1), axis=1)
    covered, sizes = [], []
    for i in range(len(test_y)):
        # class c in set(x) iff c is within the first k_c ranks, k_c from qhat_c
        s = set()
        for c in range(K):
            k = min(np.searchsorted(cs[i], qh[c]) + 1, K)
            if c in order[i, :k]:
                s.add(c)
        covered.append(test_y[i] in s)
        sizes.append(len(s))
    covered = np.array(covered)
    return dict(coverage=float(covered.mean()), avg_set_size=float(np.mean(sizes)),
                per_class_coverage=[float(covered[test_y == c].mean()) for c in range(K)])


def id_metrics(p, y):
    ece, _ = C.compute_ece(p, y)
    return dict(accuracy=float((p.argmax(1) == y).mean()), nll=C.compute_nll(p, y),
                ece=ece, brier=C.compute_brier(p, y))


def ood_metrics(p_id, p_ood, member_probs_id=None, member_probs_ood=None):
    y = np.r_[np.zeros(len(p_id)), np.ones(len(p_ood))]
    out = {}
    scores = {"msp": -np.r_[p_id.max(1), p_ood.max(1)],
              "entropy": np.r_[entropy(p_id), entropy(p_ood)]}
    if member_probs_id is not None:
        mi = lambda ps: entropy(np.mean(ps, 0)) - np.mean([entropy(q) for q in ps], 0)
        scores["mi"] = np.r_[mi(member_probs_id), mi(member_probs_ood)]
    for k, s in scores.items():
        out[f"auroc_{k}"] = float(roc_auc_score(y, s))
        out[f"aupr_{k}"] = float(average_precision_score(y, s))
    if member_probs_id is None:
        out["auroc_mi"] = out["aupr_mi"] = None  # N/A for a single model
    return out


def evaluate_config(name, kind, M, p, member_list, cal_p, cal_y, y, ood_p, mem):
    row = dict(config=name, kind=kind, M=M, members=member_list)
    row.update(id_metrics(p["test"], y))
    sh = C.shared_split_conformal(cal_p, cal_y, p["test"], y)
    row.update(shared_split_cov=sh["coverage"], shared_split_size=sh["avg_set_size"],
               shared_qhat=sh["qhat"])
    sm = shared_mondrian(cal_p, cal_y, p["test"], y)
    row.update(shared_mondrian_cov=sm["coverage"], shared_mondrian_size=sm["avg_set_size"],
               shared_mondrian_per_class=sm["per_class_coverage"])
    for sc, tag in (("aps_rand", "fixed"), ("lac", "lac")):
        s = cf.split_conformal(cal_p, cal_y, p["test"], y, C.ALPHA, sc)
        m = cf.mondrian_conformal(cal_p, cal_y, p["test"], y, C.ALPHA, sc)
        row.update({f"{tag}_split_cov": s["coverage"], f"{tag}_split_size": s["avg_set_size"],
                    f"{tag}_mondrian_cov": m["coverage"], f"{tag}_mondrian_size": m["avg_set_size"],
                    f"{tag}_mondrian_per_class": m["per_class_coverage"]})
    row.update(ood_metrics(p["test"], ood_p, *(mem if mem else (None, None))))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svhn-test-mat", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    RES = C.RESULTS_DIR
    probs_dir = RES / "probs"
    probs_dir.mkdir(parents=True, exist_ok=True)
    seeds = [s for s in args.seeds if member_path(s).exists()]
    print("members found:", seeds, "device:", C.DEVICE, flush=True)

    images, labels, (tr, te, ca) = C.get_splits()
    svhn, svhn_desc = C.load_svhn(args.svhn_test_mat)
    sub = np.random.RandomState(0).choice(len(svhn), min(N_OOD, len(svhn)), replace=False)
    print(f"OOD set: {svhn_desc}, n={len(svhn)}, subsampled to {len(sub)}", flush=True)
    mk = lambda ds: DataLoader(ds, batch_size=500, shuffle=False, num_workers=0)
    loaders = dict(cal=mk(C.CIFAR10Array(images, labels, ca, False)),
                   test=mk(C.CIFAR10Array(images, labels, te, False)),
                   svhn=mk(C.ArrayImages(svhn)))
    cal_y, y = labels[ca], labels[te]

    P = {s: member_probs(s, loaders, probs_dir) for s in seeds}
    rows = []
    for s in seeds:
        p = P[s]
        name = "baseline (seed 0, Edward ckpt)" if s == 0 else f"member seed {s}"
        rows.append(evaluate_config(name, "baseline" if s == 0 else "member", 1, p, [s],
                                    p["cal"], cal_y, y, p["svhn"][sub], None))
    ens_probs = {}
    for M in range(2, len(seeds) + 1):
        ms = seeds[:M]
        ens = {n: np.mean([P[s][n] for s in ms], axis=0) for n in ("cal", "test", "svhn")}
        ens_probs[M] = ens
        mem = (np.stack([P[s]["test"] for s in ms]), np.stack([P[s]["svhn"][sub] for s in ms]))
        rows.append(evaluate_config(f"ensemble M={M}", "ensemble", M, ens, ms, ens["cal"],
                                    cal_y, y, ens["svhn"][sub], mem))
        print(f"  evaluated M={M}", flush=True)

    # ---- save ensemble probabilities (baseline-compatible: float32 [N, 10]) ----
    if ens_probs:
        Mmax = max(ens_probs)
        e = ens_probs[Mmax]
        np.save(RES / "probs" / f"ensemble_M{Mmax}_cifar10_test_probs.npy", e["test"].astype(np.float32))
        np.save(RES / "probs" / "cifar10_test_labels.npy", y)
        np.save(RES / "probs" / f"ensemble_M{Mmax}_cifar10_cal_probs.npy", e["cal"].astype(np.float32))
        np.save(RES / "probs" / "cifar10_cal_labels.npy", cal_y)
        np.save(RES / "probs" / f"ensemble_M{Mmax}_svhn_probs.npy", e["svhn"].astype(np.float32))
        np.save(RES / "probs" / "svhn_ood_subsample_idx.npy", sub)
        # SVHN has no CIFAR labels; OOD "labels" = 1 (OOD) for every SVHN image
        np.save(RES / "probs" / "svhn_labels.npy", np.ones(len(e["svhn"]), dtype=np.int64))

    meta = dict(device=str(C.DEVICE), torch=torch.__version__, ood_set=svhn_desc,
                ood_n_total=int(len(svhn)), ood_n_used=int(len(sub)), alpha=C.ALPHA,
                ece_bins=C.ECE_BINS, n_test=int(len(y)), n_cal=int(len(cal_y)))
    json.dump(dict(meta=meta, rows=rows), open(RES / "results.json", "w"), indent=2)

    import pandas as pd
    flat = []
    for r in rows:
        d = {k: v for k, v in r.items() if not isinstance(v, list) or k == "members"}
        d["members"] = " ".join(map(str, r["members"]))
        for tag in ("shared_mondrian", "fixed_mondrian", "lac_mondrian"):
            for c, v in enumerate(r[f"{tag}_per_class"]):
                d[f"{tag}_cov_class{c}"] = v
        flat.append(d)
    pd.DataFrame(flat).to_csv(RES / "results.csv", index=False)

    write_tables(rows, meta, RES)
    print("wrote results to", RES)


def fmt(x, n=4):
    return "N/A" if x is None else f"{x:.{n}f}"


def write_tables(rows, meta, RES):
    by = {r["config"]: r for r in rows}
    base = rows[0]
    ens = [r for r in rows if r["kind"] == "ensemble"]
    top = ens[-1] if ens else None
    ed = json.load(open(C.REPO / "baseline_resnet20_results.json"))
    lines = ["# Baseline vs Deep Ensemble", "",
             f"CIFAR-10 test n={meta['n_test']}, calibration n={meta['n_cal']}, alpha={meta['alpha']}. "
             f"OOD = {meta['ood_set']} (subsampled to {meta['ood_n_used']}).", "",
             "| Metric | Edward (reported) | Shravan repro (reported) | Seed-0 ckpt (recomputed) | "
             f"Deep Ensemble {'M=%d' % top['M'] if top else '(n/a)'} |", "|---|---|---|---|---|"]
    e = lambda k, n=4: fmt(top[k], n) if top else "N/A"
    em, rm = ed["baseline_metrics"], ed["conformal_metrics"]
    L = [
        ("Accuracy", fmt(em["test_accuracy"]), "0.9039", fmt(base["accuracy"]), e("accuracy")),
        ("NLL", fmt(em["nll"]), "0.2992", fmt(base["nll"]), e("nll")),
        ("ECE (15 bins)", fmt(em["ece"]), "0.0284", fmt(base["ece"]), e("ece")),
        ("Brier", "n/a", "n/a", fmt(base["brier"]), e("brier")),
        ("Conformal coverage, shared code (split APS)", fmt(rm["empirical_coverage"]), "1.0000",
         fmt(base["shared_split_cov"]), e("shared_split_cov")),
        ("Avg set size, shared code", fmt(rm["avg_set_size"], 2), "6.32",
         fmt(base["shared_split_size"], 2), e("shared_split_size", 2)),
        ("Conformal coverage, fixed (rand. APS, split)", "n/a", "n/a",
         fmt(base["fixed_split_cov"]), e("fixed_split_cov")),
        ("Avg set size, fixed", "n/a", "n/a", fmt(base["fixed_split_size"], 2), e("fixed_split_size", 2)),
        ("Mondrian coverage, fixed", "n/a", "n/a", fmt(base["fixed_mondrian_cov"]), e("fixed_mondrian_cov")),
        ("Mondrian avg set size, fixed", "n/a", "n/a", fmt(base["fixed_mondrian_size"], 2),
         e("fixed_mondrian_size", 2)),
        ("OOD AUROC (max-softmax)", "n/a", "n/a", fmt(base["auroc_msp"]), e("auroc_msp")),
        ("OOD AUROC (entropy)", "n/a", "n/a", fmt(base["auroc_entropy"]), e("auroc_entropy")),
        ("OOD AUROC (mutual info)", "n/a", "n/a", "N/A", e("auroc_mi")),
        ("OOD AUPR (max-softmax)", "n/a", "n/a", fmt(base["aupr_msp"]), e("aupr_msp")),
        ("OOD AUPR (entropy)", "n/a", "n/a", fmt(base["aupr_entropy"]), e("aupr_entropy")),
        ("OOD AUPR (mutual info)", "n/a", "n/a", "N/A", e("aupr_mi")),
    ]
    lines += [f"| {a} | {b} | {c} | {d} | {f} |" for a, b, c, d, f in L]
    lines += ["", "\"Shravan repro (reported)\" values are those from the independent retrain "
              "quoted in the project notes (not recomputed here). The seed-0 column is the "
              "same checkpoint Edward reported, re-evaluated by this pipeline."]
    (RES / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf8")

    # ---- sanity checks ----
    sc = ["# Sanity checks", ""]
    flag = lambda ok: "PASS" if ok else "**FLAG**"
    if top:
        sc.append(f"- {flag(top['nll'] <= base['nll'] and top['ece'] <= base['ece'])} "
                  f"Ensemble M={top['M']} vs baseline: NLL {top['nll']:.4f} vs {base['nll']:.4f}, "
                  f"ECE {top['ece']:.4f} vs {base['ece']:.4f}")
        sc.append(f"- {flag(top['accuracy'] > base['accuracy'])} Ensemble accuracy "
                  f"{top['accuracy']:.4f} vs baseline {base['accuracy']:.4f}")
    mem = [r for r in rows if r["kind"] in ("baseline", "member")]
    if len(mem) > 1:
        a = [r["accuracy"] for r in mem]
        sc.append(f"- {flag(max(a) - min(a) <= 0.005)} Member accuracy spread: "
                  f"{min(a):.4f}-{max(a):.4f} (range {max(a)-min(a):.4f}, limit 0.005)")
    for r in rows:
        c = r["fixed_split_cov"]
        sc.append(f"- {flag(0.88 <= c <= 0.93)} {r['config']}: fixed split coverage {c:.4f} (0.88-0.93)")
    (RES / "sanity_checks.md").write_text("\n".join(sc) + "\n", encoding="utf8")
    print("\n".join(sc))


if __name__ == "__main__":
    main()
