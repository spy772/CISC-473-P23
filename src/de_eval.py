"""Deep Ensembles final evaluation. One command: python src/de_eval.py

Reads ONLY the dumped files in results/raw/ (see src/dump_member_probs.py); never runs a model.
Ensembles: A = seeds 0-4, B = 5-9, C = 10-14 (each used only if all 5 members are present).
Writes results/de/{tables,failure}/, results/de/{RESULTS,MISSING}.md and figures/de/.
"""
import argparse
import warnings
import itertools
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
warnings.filterwarnings("ignore", category=FutureWarning)
from de_conformal import ConformalSets, resplit_eval, METHODS  # noqa: E402

RAW = REPO / "results" / "raw"
OUT = REPO / "results" / "de"
TAB, FAIL, FIG = OUT / "tables", OUT / "failure", REPO / "figures" / "de"
ENSEMBLES = {"A": range(0, 5), "B": range(5, 10), "C": range(10, 15)}
ALPHA, N_BINS, N_RESPLITS, N_BOOT, N_SUBSETS = 0.1, 15, 100, 10000, 10
K = 10
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
SANITY = {"seed0": (0.9028, 0.3026, 0.0312), "A": (0.9273, 0.2232, 0.0164)}
TODO = []  # strings collected for MISSING.md
MD = []    # RESULTS.md sections


# ----------------------------------------------------------------------------- metrics
def entropy(p):
    p = p.astype(np.float64)
    return -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)


def ece_equal_width(conf, correct, n_bins=N_BINS):
    b = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)  # bins (lo, hi]
    sc = np.bincount(b, weights=conf, minlength=n_bins)
    sa = np.bincount(b, weights=correct.astype(float), minlength=n_bins)
    return float(np.abs(sa - sc).sum() / len(conf))


def ece_adaptive(conf, correct, n_bins=N_BINS):
    o = np.argsort(conf, kind="stable")
    tot = sum(len(i) * abs(correct[i].mean() - conf[i].mean()) for i in np.array_split(o, n_bins))
    return float(tot / len(conf))


def id_metrics(p, y, mi=None):
    p = p.astype(np.float64)
    conf, correct = p.max(1), p.argmax(1) == y
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))
    brier = ((p - np.eye(K)[y]) ** 2).sum(1)
    return dict(accuracy=correct.mean(), ece=ece_equal_width(conf, correct),
                adaptive_ece=ece_adaptive(conf, correct), nll=nll.mean(), brier=brier.mean(),
                mean_conf=conf.mean(), mean_entropy=entropy(p).mean(),
                mean_mi=float(mi.mean()) if mi is not None else np.nan,
                overconf_gap=conf.mean() - correct.mean())


def mutual_info(member_probs):
    """H(mean of members) - mean of member entropies."""
    mem = np.stack(member_probs).astype(np.float64)
    return entropy(mem.mean(0)) - np.mean([entropy(m) for m in mem], axis=0)


def ood_score_dict(p, mi, sizes):
    """OOD scores, higher = more OOD (mi is None for single models)."""
    return dict(msp=-p.max(1), entropy=entropy(p), mi=mi, raps_size=sizes.astype(float))


def ood_metrics(s_id, s_ood):
    y = np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))]
    s = np.r_[s_id, s_ood]
    fpr, tpr, _ = roc_curve(1 - y, -s)           # ID = positive class
    fpr95 = float(fpr[np.searchsorted(tpr, 0.95, side="left")])
    return dict(auroc=roc_auc_score(y, s), aupr_in=average_precision_score(1 - y, -s),
                aupr_out=average_precision_score(y, s), fpr95=fpr95)


def selective(conf_score, correct):
    """conf_score: higher = more confident. Returns AURC / acc@coverage and the accuracy-coverage curve."""
    o = np.argsort(-conf_score, kind="stable")
    c = correct[o].astype(float)
    n = len(c)
    cum_acc = np.cumsum(c) / np.arange(1, n + 1)
    out = dict(aurc=float((1 - cum_acc).mean()))
    for cov in (0.8, 0.9, 0.95):
        out["acc@%d" % int(cov * 100)] = float(cum_acc[int(np.ceil(cov * n)) - 1])
    return out, cum_acc


# ----------------------------------------------------------------------------- loading
class Unit:
    """A single member or an ensemble (mean of member softmaxes), built from dumped probs."""

    def __init__(self, name, seeds, mem):
        self.name, self.seeds = name, list(seeds)
        self.is_ens = len(self.seeds) > 1
        self.test = np.mean([mem[s]["test"] for s in seeds], axis=0)
        self.cal = np.mean([mem[s]["cal"] for s in seeds], axis=0)
        self.ood = np.mean([mem[s]["ood"] for s in seeds], axis=0)
        self.mi_test = mutual_info([mem[s]["test"] for s in seeds]) if self.is_ens else None
        self.mi_ood = mutual_info([mem[s]["ood"] for s in seeds]) if self.is_ens else None

    def fit_raps(self, y_cal):
        """RAPS (k_reg=5, lam=0.01), qhat from the original 12k cal split, applied to test and SVHN."""
        self.cs = ConformalSets("raps", alpha=ALPHA, k_reg=5, lam=0.01, seed=0).fit(self.cal, y_cal)
        self.size_test = self.cs.predict_sets(self.test).sum(1)
        self.size_ood = self.cs.predict_sets(self.ood).sum(1)
        return self


def load_members():
    mem, meta = {}, {}
    for d in sorted((RAW / "de_members").glob("seed*"), key=lambda p: int(p.name[4:])):
        if not (d / "meta.json").exists():
            continue
        s = int(d.name[4:])
        mem[s] = dict(test=np.load(d / "test_probs.npy"), cal=np.load(d / "cal_probs.npy"),
                      ood=np.load(d / "ood_svhn_probs.npy"))
        meta[s] = json.load(open(d / "meta.json"))
    return mem, meta


# ----------------------------------------------------------------------------- formatting / IO
def ms(vals, nd=4):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "n/a"
    if len(v) == 1:
        return "%.*f (n=1)" % (nd, v[0])
    return "%.*f ± %.*f" % (nd, v.mean(), nd, v.std(ddof=1))


def to_md(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(x) for x in r.values) + " |")
    return "\n".join(lines)


def save_table(df, name, title, note=None):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / (name + ".csv"), index=False, encoding="utf-8")
    tex = df.to_latex(index=False, escape=True, column_format="l" + "r" * (len(df.columns) - 1),
                      caption=title, label="tab:de_" + name).replace("±", r"$\pm$").replace("Δ", r"$\Delta$")
    (TAB / (name + ".tex")).write_text(tex, encoding="utf-8")
    MD.append("### %s\n\n%s\n%s\n" % (title, to_md(df), ("\n" + note + "\n") if note else ""))


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(FIG / (name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def ens_cols(r, fn, ens):
    """Add per-ensemble columns Ens A/B/C (TODO if not available)."""
    for name in ENSEMBLES:
        r["Ens " + name] = ("%.4f" % fn(name)) if name in ens else "TODO"
    return r


# ----------------------------------------------------------------------------- bootstrap
def boot_counts(n, B, rng):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float32)


def boot_weighted_auroc(s_id, s_ood, cid, cood):
    """AUROC (OOD positive, higher score = OOD, ties count 0.5) for each row of the resample-count
    matrices cid (B, n_id) and cood (B, n_ood), via cumulative weights over globally sorted scores."""
    s = np.r_[s_id, s_ood]
    n_id = len(s_id)
    o = np.argsort(s, kind="stable")
    ss = s[o]
    new = np.r_[True, ss[1:] != ss[:-1]]
    starts = np.flatnonzero(new)
    gid = np.cumsum(new) - 1
    gs = starts[gid]                                          # group start per position
    ge = np.r_[starts[1:] - 1, len(ss) - 1][gid]              # group end per position
    is_id = o < n_id
    W = np.concatenate([cid, cood], axis=1)[:, o]
    Wid = np.where(is_id[None, :], W, 0.0)
    cum = np.cumsum(Wid, axis=1, dtype=np.float64)
    below_eq = 0.5 * ((cum - Wid)[:, gs] + cum[:, ge])
    num = (np.where(is_id[None, :], 0.0, W) * below_eq).sum(1)
    return num / (cid.sum(1) * cood.sum(1))


def bootstrap_deltas(ens, members, y, rng, B=N_BOOT, chunk=500):
    """Paired bootstrap over test images (and SVHN images for OOD): delta = ensemble - mean(members)."""
    n, n_ood = len(y), len(ens.ood)

    def per_image(u):
        p = u.test.astype(np.float64)
        conf, corr = p.max(1), (p.argmax(1) == y).astype(np.float64)
        oh = np.eye(N_BINS)[np.clip(np.ceil(conf * N_BINS).astype(int) - 1, 0, N_BINS - 1)]
        return dict(acc=corr, nll=-np.log(np.clip(p[np.arange(n), y], 1e-12, 1)),
                    brier=((p - np.eye(K)[y]) ** 2).sum(1), size=u.size_test.astype(np.float64),
                    bin_acc=oh * corr[:, None], bin_conf=oh * conf[:, None])

    def stats(pi, C):
        return dict(accuracy=C @ pi["acc"] / n, nll=C @ pi["nll"] / n, brier=C @ pi["brier"] / n,
                    raps_size=C @ pi["size"] / n, ece=np.abs(C @ pi["bin_acc"] - C @ pi["bin_conf"]).sum(1) / n)

    pi_e, pi_m = per_image(ens), [per_image(m) for m in members]
    ent = lambda u: (entropy(u.test), entropy(u.ood))  # noqa: E731
    ent_e, ent_m = ent(ens), [ent(m) for m in members]
    res = {k: [] for k in ("accuracy", "ece", "nll", "brier", "raps_size", "ood_auroc_entropy")}
    for st in range(0, B, chunk):
        b = min(chunk, B - st)
        C, Co = boot_counts(n, b, rng), boot_counts(n_ood, b, rng)
        se, sm = stats(pi_e, C), [stats(p, C) for p in pi_m]
        for k in ("accuracy", "ece", "nll", "brier", "raps_size"):
            res[k].append(se[k] - np.mean([d[k] for d in sm], axis=0))
        res["ood_auroc_entropy"].append(boot_weighted_auroc(*ent_e, C, Co) -
                                        np.mean([boot_weighted_auroc(*e, C, Co) for e in ent_m], axis=0))
    return {k: np.concatenate(v) for k, v in res.items()}


# ----------------------------------------------------------------------------- figures
def reliability(ax, p, y, label, color, n_bins=N_BINS):
    conf, corr = p.max(1), (p.argmax(1) == y).astype(float)
    b = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    pts = [(conf[b == i].mean(), corr[b == i].mean()) for i in range(n_bins) if (b == i).any()]
    ax.plot(*zip(*pts), "o-", color=color, ms=4, label=label)


def image_grid(imgs, titles, name, suptitle, ncols=4):
    nrows = int(np.ceil(len(imgs) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.6 * nrows))
    axs = np.atleast_1d(axs).ravel()
    for a in axs:
        a.axis("off")
    for a, im, t in zip(axs, imgs, titles):
        a.imshow(im)
        a.set_title(t, fontsize=7)
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    savefig(fig, name)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-resplits", type=int, default=N_RESPLITS)
    ap.add_argument("--skip-images", action="store_true", help="skip failure image grids")
    args = ap.parse_args()
    t_start = time.time()
    for d in (TAB, FAIL, FIG):
        d.mkdir(parents=True, exist_ok=True)

    y_test = np.load(RAW / "labels" / "test_labels.npy")
    y_cal = np.load(RAW / "labels" / "cal_labels.npy")
    mem, meta = load_members()
    seeds = sorted(mem)
    if not seeds:
        sys.exit("no dumped members in results/raw/de_members/ -- run src/dump_member_probs.py")
    singles = {s: Unit("seed%d" % s, [s], mem).fit_raps(y_cal) for s in seeds}
    ens = {}
    for name, rg in ENSEMBLES.items():
        if all(s in mem for s in rg):
            ens[name] = Unit(name, rg, mem).fit_raps(y_cal)
        else:
            TODO.append("Ensemble %s (seeds %d-%d): missing seeds %s -> all ensemble-%s numbers are TODO." % (
                name, rg[0], rg[-1], [s for s in rg if s not in mem], name))
    pool10 = [s for s in range(10) if s in mem]
    if len(pool10) < 10:
        TODO.append("Ensemble-size ablation covers M=1..%d only (needs seeds 0-9; have %s); M>%d is TODO." % (
            len(pool10), pool10, len(pool10)))
    if len(ens) < 3:
        TODO.append("Only %d/3 independent ensembles available: 'mean ± std over ensembles' has n=%d; the paired t-test "
                    "across ensembles is TODO (needs >=3)." % (len(ens), len(ens)))
    no_time = [s for s, m in meta.items() if m.get("train_time_sec") is None]
    if no_time:
        TODO.append("Training time not recorded for seeds %s (seed 0 is the baseline checkpoint trained by a teammate); "
                    "total train-time figures exclude them." % no_time)
    everyone = list(singles.values()) + list(ens.values())
    fname = "A" if "A" in ens else (next(iter(ens)) if ens else None)
    fe = ens[fname] if fname else None

    # ---------------- sanity check
    sanity_rows, bad = [], False
    for key, u in [("seed0", singles.get(0)), ("A", ens.get("A"))]:
        if u is None:
            continue
        m = id_metrics(u.test, y_test, u.mi_test)
        got = (m["accuracy"], m["nll"], m["ece"])
        # compare to 4 d.p. with half-unit tolerance (ensemble A acc = 11127/12000 = 0.92725 exactly, reported as 0.9273 half-up)
        ok = all(abs(g - e) <= 0.00005 + 1e-9 for g, e in zip(got, SANITY[key]))
        got = tuple(round(g, 5) for g in got)
        bad |= not ok
        sanity_rows.append("%s: got Acc/NLL/ECE = %s, expected %s -> %s" % (key, got, SANITY[key], "OK" if ok else "MISMATCH"))
    print("\n".join(sanity_rows))
    if bad:
        (OUT / "RESULTS.md").write_text("# SANITY CHECK FAILED\n\n" + "\n".join(sanity_rows) + "\n", encoding="utf-8")
        sys.exit("SANITY MISMATCH -- stopping (see results/de/RESULTS.md)")

    # ---------------- main metrics + per-seed raw
    ID_KEYS = ["accuracy", "ece", "adaptive_ece", "nll", "brier", "mean_conf", "mean_entropy", "mean_mi", "overconf_gap"]
    idm = {u.name: id_metrics(u.test, y_test, u.mi_test) for u in everyone}
    rows = []
    for k in ID_KEYS:
        r = {"Metric": k, "Single ResNet-20 (mean ± std, %d members)" % len(singles): ms([idm[s.name][k] for s in singles.values()]),
             "DE M=5 (mean ± std, %d ensembles)" % len(ens): ms([idm[e][k] for e in ens])}
        rows.append(ens_cols(r, lambda n: idm[n][k], ens))
    save_table(pd.DataFrame(rows), "main_metrics", "Accuracy and calibration, CIFAR-10 test (n=12000)",
               "Sanity check: " + "; ".join(sanity_rows) + ". mean_mi is n/a for single models. Single-model std is the seed variance.")
    rows = []
    for u in everyone:
        r = {"unit": u.name, "seeds": "-".join(map(str, u.seeds)) if u.is_ens else u.seeds[0]}
        r.update({k: "%.4f" % v for k, v in idm[u.name].items()})
        r["raps_size_test"] = "%.4f" % u.size_test.mean()
        rows.append(r)
    save_table(pd.DataFrame(rows), "per_seed_raw", "Per-member and per-ensemble raw metrics (fixed cal/test split)")

    # ---------------- conformal, 100 re-splits
    print("conformal re-splits ...", flush=True)
    cres = {u.name: resplit_eval(u.cal, y_cal, u.test, y_test, n_splits=args.n_resplits) for u in everyone}
    CST = [("coverage", "coverage"), ("avg_size", "avg set size"), ("median_size", "median set size"),
           ("pc_cov_min", "per-class cov min"), ("pc_cov_max", "per-class cov max"),
           ("size_correct", "set size (correct preds)"), ("size_incorrect", "set size (incorrect preds)"), ("frac_empty", "frac empty sets")]
    rows = []
    for m in METHODS:
        for k, lab in CST:
            r = {"method": m, "stat": lab,
                 "Single (mean ± std, %d members)" % len(singles): ms([cres[s.name][m][k].mean() for s in singles.values()]),
                 "DE M=5 (mean ± std, %d ens)" % len(ens): ms([cres[e][m][k].mean() for e in ens])}
            r = ens_cols(r, lambda n: cres[n][m][k].mean(), ens)
            r["Ens A: std over splits"] = "%.4f" % cres["A"][m][k].std(ddof=1) if "A" in ens else "TODO"
            rows.append(r)
    save_table(pd.DataFrame(rows), "conformal", "Conformal prediction, alpha=0.1, mean over %d random 12k/12k re-splits of pooled cal+test" % args.n_resplits,
               "Set rule: class y in set iff score(x,y) <= qhat, so sets can be empty (frac reported). Per-class coverage min/max are across the 10 classes, averaged over splits.")

    # ---------------- OOD
    print("ood ...", flush=True)
    oodm, extra = {}, {}
    for u in everyone:
        sid, sood = ood_score_dict(u.test, u.mi_test, u.size_test), ood_score_dict(u.ood, u.mi_ood, u.size_ood)
        for sname in sid:
            oodm[(u.name, sname)] = (ood_metrics(sid[sname], sood[sname]) if sid[sname] is not None
                                     else dict(auroc=np.nan, aupr_in=np.nan, aupr_out=np.nan, fpr95=np.nan))
        c = u.ood.max(1)
        extra[u.name] = dict(svhn_mean_conf=float(c.mean()), svhn_frac_conf_gt_0_9=float((c > 0.9).mean()))
    rows = []
    for sname in ("msp", "entropy", "mi", "raps_size"):
        for k in ("auroc", "aupr_in", "aupr_out", "fpr95"):
            r = {"score": sname, "metric": k,
                 "Single (mean ± std, %d members)" % len(singles): ms([oodm[(s.name, sname)][k] for s in singles.values()]),
                 "DE M=5 (mean ± std, %d ens)" % len(ens): ms([oodm[(e, sname)][k] for e in ens])}
            rows.append(ens_cols(r, lambda n: oodm[(n, sname)][k], ens))
    for k in ("svhn_mean_conf", "svhn_frac_conf_gt_0_9"):
        r = {"score": "-", "metric": k, "Single (mean ± std, %d members)" % len(singles): ms([extra[s.name][k] for s in singles.values()]),
             "DE M=5 (mean ± std, %d ens)" % len(ens): ms([extra[e][k] for e in ens])}
        rows.append(ens_cols(r, lambda n: extra[n][k], ens))
    save_table(pd.DataFrame(rows), "ood", "OOD detection: CIFAR-10 test (12000) vs SVHN test (26032)",
               "ID is positive for AUPR-In and FPR@95TPR (fraction of SVHN accepted when 95% of CIFAR test is kept); OOD is positive for AUROC and AUPR-Out. "
               "raps_size = RAPS (k_reg=5, lam=0.01) set size, qhat from the cal split. MI is n/a for single models. "
               "OOD prevalence is 0.68, so AUPR chance level is 0.68 (Out) / 0.32 (In).")

    # ---------------- selective prediction
    sel, curves = {}, {}
    for u in everyone:
        corr = u.test.argmax(1) == y_test
        scores = {"msp": u.test.max(1), "entropy": -entropy(u.test)}
        if u.mi_test is not None:
            scores["mi"] = -u.mi_test
        for sname, sc in scores.items():
            sel[(u.name, sname)], curves[(u.name, sname)] = selective(sc, corr)
    rows = []
    for sname in ("msp", "entropy", "mi"):
        for k in ("aurc", "acc@80", "acc@90", "acc@95"):
            r = {"score": sname, "metric": k,
                 "Single (mean ± std, %d members)" % len(singles): ms([sel[(s.name, sname)][k] for s in singles.values() if (s.name, sname) in sel]),
                 "DE M=5 (mean ± std, %d ens)" % len(ens): ms([sel[(e, sname)][k] for e in ens])}
            for name in ENSEMBLES:
                r["Ens " + name] = "%.4f" % sel[(name, sname)][k] if name in ens else "TODO"
            rows.append(r)
    save_table(pd.DataFrame(rows), "selective", "Selective prediction (accuracy at coverage 80/90/95%; AURC = mean selective risk, lower is better)",
               "MI is n/a for single models. Full-coverage accuracy is in the main table.")

    # ---------------- ensemble-size ablation
    print("ablation ...", flush=True)
    rng = np.random.RandomState(0)
    AB = ("accuracy", "ece", "nll", "brier", "mean_conf", "raps_size", "ood_auroc_entropy", "ood_auroc_mi")
    abl_raw = []
    for M in range(1, len(pool10) + 1):
        combos = list(itertools.combinations(pool10, M))
        if len(combos) > N_SUBSETS:
            combos = [combos[i] for i in sorted(rng.choice(len(combos), N_SUBSETS, replace=False))]
        vals = []
        for c in combos:
            u = singles[c[0]] if M == 1 else Unit("M%d" % M, c, mem).fit_raps(y_cal)
            m = id_metrics(u.test, y_test, u.mi_test)
            m["raps_size"] = float(u.size_test.mean())
            m["ood_auroc_entropy"] = ood_metrics(entropy(u.test), entropy(u.ood))["auroc"]
            m["ood_auroc_mi"] = ood_metrics(u.mi_test, u.mi_ood)["auroc"] if u.is_ens else np.nan
            vals.append(m)
        r = {"M": M, "n_subsets": len(combos)}
        for k in AB:
            v = np.array([x[k] for x in vals], dtype=float)
            r[k + "_mean"] = float(np.mean(v))
            r[k + "_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        abl_raw.append(r)
    abl = pd.DataFrame(abl_raw)
    fmt = lambda r, k: "n/a" if np.isnan(r[k + "_mean"]) else "%.4f ± %.4f" % (r[k + "_mean"], r[k + "_std"])  # noqa: E731
    save_table(pd.DataFrame([dict([("M", r["M"]), ("subsets", r["n_subsets"])] + [(k, fmt(r, k)) for k in AB]) for r in abl_raw]),
               "ablation_size", "Ensemble-size ablation (mean ± std over random member subsets of seeds 0-9)",
               "Up to %d distinct subsets per M drawn with RandomState(0); M=%d has a single subset (std 0)." % (N_SUBSETS, len(pool10)))

    # ---------------- DE vs single: paired deltas
    print("bootstrap ...", flush=True)
    DK = [("accuracy", "Δ accuracy"), ("ece", "Δ ECE"), ("nll", "Δ NLL"), ("brier", "Δ Brier"),
          ("raps_size", "Δ RAPS set size"), ("ood_auroc_entropy", "Δ OOD AUROC (entropy)")]
    sig_rows, delta = [], {}
    for name, e in ens.items():
        mus = [singles[s] for s in e.seeds]
        b = bootstrap_deltas(e, mus, y_test, np.random.RandomState(0), B=args.n_boot)
        mean_m = lambda k: np.mean([idm[m.name][k] for m in mus])  # noqa: E731
        delta[name] = dict(accuracy=idm[name]["accuracy"] - mean_m("accuracy"), ece=idm[name]["ece"] - mean_m("ece"),
                           nll=idm[name]["nll"] - mean_m("nll"), brier=idm[name]["brier"] - mean_m("brier"),
                           raps_size=e.size_test.mean() - np.mean([m.size_test.mean() for m in mus]),
                           ood_auroc_entropy=oodm[(name, "entropy")]["auroc"] - np.mean([oodm[(m.name, "entropy")]["auroc"] for m in mus]))
        for k, lab in DK:
            lo, hi = np.percentile(b[k], [2.5, 97.5])
            sig_rows.append({"ensemble": name, "metric": lab, "Δ (ens − mean of its members)": "%.4f" % delta[name][k],
                             "95% bootstrap CI": "[%.4f, %.4f]" % (lo, hi), "excludes 0": "yes" if (lo > 0 or hi < 0) else "no"})
    for k, lab in DK:
        ds = [delta[n][k] for n in ens]
        if len(ds) >= 3:
            t, p = sps.ttest_1samp(ds, 0.0)
            sig_rows.append({"ensemble": "paired t-test, %d ensembles" % len(ds), "metric": lab, "Δ (ens − mean of its members)": ms(ds),
                             "95% bootstrap CI": "t=%.3f" % t, "excludes 0": "p=%.4g" % p})
        else:
            sig_rows.append({"ensemble": "paired t-test", "metric": lab, "Δ (ens − mean of its members)": "TODO (need >=3 ensembles, have %d)" % len(ds),
                             "95% bootstrap CI": "TODO", "excludes 0": "TODO"})
    save_table(pd.DataFrame(sig_rows), "significance", "DE vs single models: deltas, paired bootstrap (%d resamples) and t-test across ensembles" % args.n_boot,
               "Delta = ensemble metric minus the mean of the same metric over its own 5 members (ECE/NLL/size: negative = DE better). "
               "Bootstrap resamples the 12000 test images (same resample for ensemble and members); the OOD row also resamples the 26032 SVHN images.")

    # ---------------- cost
    t_rec = [meta[s]["train_time_sec"] for s in seeds if meta[s].get("train_time_sec") is not None]
    inf = [meta[s]["inference_sec_per_1000_imgs"] for s in seeds]
    npar = meta[seeds[0]]["n_params"]
    cost = pd.DataFrame([
        {"quantity": "params, single ResNet-20", "value": "{:,}".format(npar)},
        {"quantity": "params, DE M=5 (5x)", "value": "{:,}".format(5 * npar)},
        {"quantity": "params, DE M=10 (10x)", "value": "{:,}".format(10 * npar)},
        {"quantity": "train time per member, min (mean ± std over %d members with a record; CPU and GPU runs mixed)" % len(t_rec),
         "value": ms([t / 60 for t in t_rec], 1)},
        {"quantity": "total recorded train time, h (%d members)" % len(t_rec), "value": "%.2f" % (sum(t_rec) / 3600)},
        {"quantity": "inference s / 1000 images, single (mean ± std over dumps)", "value": ms(inf, 3)},
        {"quantity": "inference s / 1000 images, DE M=5 (= 5 x single, members run sequentially)", "value": "%.3f" % (5 * np.mean(inf))},
        {"quantity": "inference s / 1000 images, DE M=10 (= 10 x single)", "value": "%.3f" % (10 * np.mean(inf))},
        {"quantity": "inference hardware(s)", "value": "; ".join(sorted(set(meta[s]["hardware"] for s in seeds)))},
    ])
    save_table(cost, "cost", "Cost", "Inference time was measured at dump time (batch 256, includes data loading/normalisation) on the hardware listed; GPU dumps are not comparable to CPU dumps. Train times are wall-clock from train_time_seed*.json; seed 3 includes ~3.5 h of laptop sleep (actual compute ~100 min, see models/deep_ensembles/README.md), so the mean/std above are inflated.")

    # ---------------- failure analysis (ensemble A)
    if fe is not None:
        pred, conf = fe.test.argmax(1), fe.test.max(1)
        wrong = np.flatnonzero(pred != y_test)
        cm = np.zeros((K, K), int)
        np.add.at(cm, (y_test, pred), 1)
        off = sorted(((int(cm[i, j]), CLASSES[i], CLASSES[j]) for i in range(K) for j in range(K) if i != j), reverse=True)
        top_pairs = [dict(count=c, true=t, pred=p) for c, t, p in off[:5]]
        top_wrong = wrong[np.argsort(-conf[wrong], kind="stable")][:16]
        svhn_conf = fe.ood.max(1)
        top_svhn = np.argsort(-svhn_conf, kind="stable")[:16]
        big = np.lexsort((conf, -fe.size_test))[:16]          # largest set, ties -> lowest confidence
        top_mi = np.argsort(-fe.mi_test, kind="stable")[:16]
        tg = np.array(json.load(open(RAW / "labels" / "split_indices.json"))["test"])
        lab = lambda ix: dict(idx=ix.tolist(), global_idx=tg[ix].tolist(), true=[CLASSES[i] for i in y_test[ix]], pred=[CLASSES[i] for i in pred[ix]])  # noqa: E731
        outs = dict(
            confusion_matrix=cm.tolist(), top5_confused_pairs=top_pairs,
            confident_wrong=dict(lab(top_wrong), conf=conf[top_wrong].round(4).tolist()),
            confident_svhn=dict(idx=top_svhn.tolist(), pred=[CLASSES[i] for i in fe.ood.argmax(1)[top_svhn]], conf=svhn_conf[top_svhn].round(4).tolist()),
            largest_sets=dict(lab(big), size=fe.size_test[big].tolist()),
            highest_mi=dict(lab(top_mi), mi=fe.mi_test[top_mi].round(4).tolist()))
        for k, v in outs.items():
            json.dump(dict(ensemble=fname, note="idx = position in the 12k test split (SVHN: in the 26032 test set); global_idx = index into the pooled 60k CIFAR-10 array", **{k: v}),
                      open(FAIL / (k + ".json"), "w"), indent=1)
        MD.append("### Top 5 confused pairs (ensemble %s, test)\n\n%s\n" % (fname, to_md(pd.DataFrame(top_pairs))))

    # ---------------- figures
    print("figures ...", flush=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    s0 = singles.get(0) or singles[seeds[0]]
    big_ens = Unit("M10", pool10, mem) if len(pool10) == 10 else None
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    reliability(ax, s0.test, y_test, "single (seed %d)" % s0.seeds[0], "#1f77b4")
    if fe is not None:
        reliability(ax, fe.test, y_test, "DE M=5 (%s)" % fname, "#d95f02")
    if big_ens is not None:
        reliability(ax, big_ens.test, y_test, "DE M=10", "#1b9e77")
    ax.set(xlabel="confidence", ylabel="accuracy", title="Reliability (15 bins)")
    ax.legend()
    savefig(fig, "reliability")

    fig, axs = plt.subplots(2, 4, figsize=(12, 5.2))
    for ax, (k, t) in zip(axs.ravel(), zip(AB, ["accuracy", "ECE", "NLL", "Brier", "mean confidence", "RAPS set size", "OOD AUROC (entropy)", "OOD AUROC (MI)"])):
        ax.errorbar(abl["M"], abl[k + "_mean"], yerr=abl[k + "_std"], marker="o", ms=3, capsize=2, color="#d95f02")
        ax.set(title=t, xlabel="ensemble size M")
    fig.tight_layout()
    savefig(fig, "ablation_size")

    fig, ax = plt.subplots(figsize=(5, 4))
    cov = np.arange(1, len(y_test) + 1) / len(y_test)
    ax.plot(cov, curves[(s0.name, "msp")], color="#1f77b4", label="single (seed %d) MSP" % s0.seeds[0])
    if fe is not None:
        for sname, c in (("msp", "#d95f02"), ("entropy", "#7570b3"), ("mi", "#1b9e77")):
            ax.plot(cov, curves[(fname, sname)], color=c, label="DE M=5 %s" % sname)
    ax.set(xlim=(0.3, 1), ylim=(0.9, 1.002), xlabel="coverage", ylabel="accuracy")
    ax.legend()
    savefig(fig, "accuracy_vs_coverage")

    fig, ax = plt.subplots(figsize=(5, 3.6))
    for i, (u, lab_, c) in enumerate([(s0, "single seed %d" % s0.seeds[0], "#1f77b4")] + ([(fe, "DE M=5", "#d95f02")] if fe is not None else [])):
        ax.bar(np.arange(11) + (i - 0.5) * 0.38, np.bincount(u.size_test, minlength=11)[:11] / len(u.size_test), 0.38, label=lab_, color=c)
    ax.set(yscale="log", xlabel="RAPS set size (alpha=0.1)", ylabel="fraction of test images")
    ax.legend()
    savefig(fig, "setsize_hist")

    if fe is not None:
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, (t, a, b) in zip(axs[:2], [("entropy", entropy(fe.test), entropy(fe.ood)), ("mutual information", fe.mi_test, fe.mi_ood)]):
            bins = np.linspace(0, max(a.max(), b.max()), 60)
            ax.hist(a, bins, alpha=0.6, density=True, label="CIFAR-10 test", color="#1f77b4")
            ax.hist(b, bins, alpha=0.6, density=True, label="SVHN test", color="#d95f02")
            ax.set(yscale="log", xlabel=t + " (DE M=5)")
            ax.legend()
        ys = np.r_[np.zeros(len(fe.test)), np.ones(len(fe.ood))]
        sid, sood = ood_score_dict(fe.test, fe.mi_test, fe.size_test), ood_score_dict(fe.ood, fe.mi_ood, fe.size_ood)
        for sname, c in (("msp", "#1f77b4"), ("entropy", "#d95f02"), ("mi", "#1b9e77")):
            f_, t_, _ = roc_curve(ys, np.r_[sid[sname], sood[sname]])
            axs[2].plot(f_, t_, color=c, label="%s (AUROC %.3f)" % (sname, oodm[(fname, sname)]["auroc"]))
        axs[2].plot([0, 1], [0, 1], "k--", lw=1)
        axs[2].set(xlabel="FPR", ylabel="TPR (SVHN detected)")
        axs[2].legend()
        fig.tight_layout()
        savefig(fig, "ood_hist_roc")

        fig, ax = plt.subplots(figsize=(5.6, 5))
        ax.imshow(np.log10(cm + 1), cmap="Blues")
        for i in range(K):
            for j in range(K):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6, color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(CLASSES, rotation=60, ha="right"); ax.set_yticklabels(CLASSES)
        ax.set(xlabel="predicted", ylabel="true", title="DE %s confusion (test)" % fname)
        ax.grid(False)
        savefig(fig, "confusion_matrix")

        if not args.skip_images:
            try:
                sys.path.insert(0, str(REPO / "models" / "deep_ensembles"))
                import common as C
                import scipy.io as sio
                images, _ = C.load_cifar10()
                svhn = sio.loadmat(str(REPO / "data_svhn_test" / "test_32x32.mat"))["X"].transpose(3, 0, 1, 2)
                cls = lambda i: CLASSES[i]  # noqa: E731
                image_grid([images[tg[i]] for i in top_wrong], ["true %s\npred %s (%.3f)" % (cls(y_test[i]), cls(pred[i]), conf[i]) for i in top_wrong],
                           "failure_confident_wrong", "16 most confident wrong test predictions (DE %s)" % fname)
                image_grid([svhn[i] for i in top_svhn], ["pred %s\nconf %.3f" % (cls(fe.ood[i].argmax()), svhn_conf[i]) for i in top_svhn],
                           "failure_confident_svhn", "16 most confident SVHN images (DE %s)" % fname)
                image_grid([images[tg[i]] for i in big], ["true %s pred %s\nsize %d" % (cls(y_test[i]), cls(pred[i]), fe.size_test[i]) for i in big],
                           "failure_largest_sets", "16 largest RAPS sets (DE %s)" % fname)
                image_grid([images[tg[i]] for i in top_mi], ["true %s pred %s\nMI %.3f" % (cls(y_test[i]), cls(pred[i]), fe.mi_test[i]) for i in top_mi],
                           "failure_highest_mi", "16 highest mutual-information test images (DE %s)" % fname)
            except Exception as ex:
                TODO.append("Failure image grids not produced (%r); indices are still in results/de/failure/." % (ex,))

    # ---------------- raw observations
    obs, flags = ["Members used: %s; ensembles present: %s." % (seeds, sorted(ens))], []
    if fe is not None:
        a = idm[fname]
        sm = {k: np.mean([idm[s.name][k] for s in singles.values()]) for k in ID_KEYS if k != "mean_mi"}
        sing = lambda f: np.mean([f(s) for s in singles.values()])  # noqa: E731
        obs += [
            "Single mean vs DE %s: acc %.4f -> %.4f, ECE %.4f -> %.4f, NLL %.4f -> %.4f, Brier %.4f -> %.4f." % (
                fname, sm["accuracy"], a["accuracy"], sm["ece"], a["ece"], sm["nll"], a["nll"], sm["brier"], a["brier"]),
            "Overconfidence gap (mean conf - acc): single %.4f, DE %.4f." % (sm["overconf_gap"], a["overconf_gap"]),
            "RAPS re-split mean: DE %s coverage %.4f, size %.3f; single-member mean size %.3f." % (
                fname, cres[fname]["raps"]["coverage"].mean(), cres[fname]["raps"]["avg_size"].mean(), sing(lambda s: cres[s.name]["raps"]["avg_size"].mean())),
            "Non-randomized APS (score<=qhat rule) re-split mean, DE %s: size %.3f, coverage %.4f, empty-set fraction %.4f." % (
                fname, cres[fname]["aps"]["avg_size"].mean(), cres[fname]["aps"]["coverage"].mean(), cres[fname]["aps"]["frac_empty"].mean()),
            "OOD AUROC (entropy / MI / MSP / RAPS size), DE %s: %.4f / %.4f / %.4f / %.4f; single mean entropy %.4f, MSP %.4f." % (
                fname, oodm[(fname, "entropy")]["auroc"], oodm[(fname, "mi")]["auroc"], oodm[(fname, "msp")]["auroc"], oodm[(fname, "raps_size")]["auroc"],
                sing(lambda s: oodm[(s.name, "entropy")]["auroc"]), sing(lambda s: oodm[(s.name, "msp")]["auroc"])),
            "SVHN mean confidence: single %.4f, DE %.4f; fraction conf>0.9: single %.4f, DE %.4f." % (
                sing(lambda s: extra[s.name]["svhn_mean_conf"]), extra[fname]["svhn_mean_conf"],
                sing(lambda s: extra[s.name]["svhn_frac_conf_gt_0_9"]), extra[fname]["svhn_frac_conf_gt_0_9"]),
            "Member accuracy range: %.4f-%.4f." % (min(idm[s.name]["accuracy"] for s in singles.values()), max(idm[s.name]["accuracy"] for s in singles.values())),
            "Ablation NLL (mean over subsets): M=1 %.4f, M=2 %.4f, M=%d %.4f." % (abl["nll_mean"].iloc[0], abl["nll_mean"].iloc[min(1, len(abl) - 1)], len(abl), abl["nll_mean"].iloc[-1]),
            "Ablation ECE is lowest at M=%d (%.4f) and rises to %.4f at M=%d (ensemble becomes under-confident: mean conf < acc)." % (
                int(abl["M"].iloc[abl["ece_mean"].idxmin()]), abl["ece_mean"].min(), abl["ece_mean"].iloc[-1], len(abl)),
            "RAPS set size: DE %s %.3f vs single mean %.3f (fixed split; Δ bootstrap CI in the significance table)." % (fname, fe.size_test.mean(), sing(lambda s: singles[s.seeds[0]].size_test.mean())),
            "Top confused pair (DE %s): %s -> %s, %d test images." % (fname, top_pairs[0]["true"], top_pairs[0]["pred"], top_pairs[0]["count"]),
            "Best selective-prediction score for DE %s by AURC: %s." % (fname, min(("msp", "entropy", "mi"), key=lambda s_: sel[(fname, s_)]["aurc"])),
        ]
        if a["ece"] > sm["ece"]:
            flags.append("DE ECE is HIGHER than the single-model mean.")
        for n_, d in delta.items():
            if d["ood_auroc_entropy"] < 0:
                flags.append("Ensemble %s has LOWER entropy-OOD AUROC than the mean of its members." % n_)
        for s in singles.values():
            c_ = cres[s.name]["raps"]["coverage"].mean()
            if not 0.89 <= c_ <= 0.91:
                flags.append("seed %d RAPS re-split coverage %.4f outside [0.89, 0.91]." % (s.seeds[0], c_))
        if np.ptp([idm[s.name]["accuracy"] for s in singles.values()]) > 0.005:
            flags.append("Member accuracy range exceeds 0.5 pt.")
    MD.append("### Raw observations\n\n" + "\n".join("- " + o for o in obs) + "\n\n### Flags\n\n" + "\n".join("- " + f_ for f_ in flags or ["none"]) + "\n")

    head = "# Deep Ensembles results (auto-generated by `python src/de_eval.py`)\n\nRuntime %.0f s. Members: %s. Ensembles: %s. See NOTES.md for definitions and MISSING.md for TODOs.\n\n" % (
        time.time() - t_start, seeds, sorted(ens))
    (OUT / "RESULTS.md").write_text(head + "\n".join(MD), encoding="utf-8")
    (OUT / "MISSING.md").write_text("# Missing / TODO (auto-generated by src/de_eval.py)\n\n" + "\n".join("- " + t for t in TODO) + "\n", encoding="utf-8")
    print("done in %.0fs; TODO items: %d" % (time.time() - t_start, len(TODO)))


if __name__ == "__main__":
    main()
