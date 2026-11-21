"""Deep Ensembles final evaluation. One command: python src/de_eval.py   (add --smoke for the smoke-run files)

Inputs : per-member probabilities from src/dump_member_probs.py (results/noise_testing/de_addon/deep_ensemble_noise0_seed{s}/),
         produced by members trained through pipeline.train_with_noise(DeepEnsemble(5), noise_ratio=0, seed=s). Never runs a model.
Headline numbers use main's code unchanged: common.compute_ece/nll/brier, pipeline.evaluate.conformal_metrics (randomized APS, LAC,
Mondrian, 100 re-splits with ddof=0 std) and pipeline.evaluate.compare/summarize (across-seed std, ddof=1).
Add-ons (this file + src/de_conformal.py): non-randomized APS, RAPS, OOD vs SVHN test, selective prediction, ensemble-size ablation,
significance tests, failure analysis, mutual information, adaptive ECE, cost.
Ensemble seed s uses members s*100 + i (i=0..4); expected ensemble seeds 0, 1, 2 (15 members). Missing ones are listed in MISSING.md.
"""
import argparse
import itertools
import json
import math
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
warnings.filterwarnings("ignore", category=FutureWarning)
import pipeline  # noqa: E402,F401
import common  # noqa: E402
from pipeline.evaluate import conformal_metrics, compare, summarize, to_markdown, _sets  # noqa: E402
from de_conformal import ConformalSets, resplit_eval, METHODS  # noqa: E402
from dump_member_probs import roots, tag_for, load_svhn_test, N_MEMBERS  # noqa: E402

EXPECTED_ENS = (0, 1, 2)
ALPHA, N_BINS, N_RESPLITS, N_BOOT, N_SUBSETS = common.ALPHA, common.ECE_BINS, 100, 10000, 10
K = 10
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
TODO = []  # strings collected for MISSING.md
MD = []    # RESULTS.md sections
OUT = TAB = FAIL = FIG = None  # set in main() (real vs --smoke locations)


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
    """accuracy / ECE / NLL / Brier from main's common.compute_*; the rest are add-ons."""
    p32 = p.astype(np.float32)
    ece, _ = common.compute_ece(p32, y)
    pf = p.astype(np.float64)
    conf, correct = pf.max(1), pf.argmax(1) == y
    return dict(accuracy=float(correct.mean()), ece=ece, adaptive_ece=ece_adaptive(conf, correct),
                nll=common.compute_nll(p32, y), brier=common.compute_brier(p32, y),
                mean_conf=conf.mean(), mean_entropy=entropy(pf).mean(),
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
    """A single member or an ensemble (mean of member softmaxes) built from dumped probs.
    keys = [(ensemble_seed, member_index), ...]; run_seed = ensemble seed (main uses it to seed conformal draws)."""

    def __init__(self, name, keys, mem, run_seed):
        self.name, self.keys, self.run_seed = name, list(keys), run_seed
        self.is_ens = len(self.keys) > 1
        self.test = np.mean([mem[k]["test"] for k in keys], axis=0)
        self.cal = np.mean([mem[k]["cal"] for k in keys], axis=0)
        self.ood = np.mean([mem[k]["ood_svhn"] for k in keys], axis=0)
        self.mi_test = mutual_info([mem[k]["test"] for k in keys]) if self.is_ens else None
        self.mi_ood = mutual_info([mem[k]["ood_svhn"] for k in keys]) if self.is_ens else None

    def fit_raps(self, y_cal):
        """RAPS (k_reg=5, lam=0.01), qhat from the cal split (draw seed 0), applied to test and SVHN."""
        self.cs = ConformalSets("raps", alpha=ALPHA, k_reg=5, lam=0.01, seed=0).fit(self.cal, y_cal)
        self.size_test = self.cs.predict_sets(self.test).sum(1)
        self.size_ood = self.cs.predict_sets(self.ood).sum(1)
        return self


def load_addon(addon_dir, smoke):
    """-> mem {(s,i): {test,cal,ood_svhn}}, meta {s: meta.json}, labels (y_test, y_cal)."""
    mem, meta, labels = {}, {}, None
    for s in EXPECTED_ENS:
        d = Path(addon_dir) / tag_for(s, smoke)
        if not (d / "meta.json").exists():
            continue
        meta[s] = json.load(open(d / "meta.json"))
        lab = np.load(d / "labels.npz")
        cur = (lab["test_labels"], lab["cal_labels"])
        if labels is None:
            labels = cur
        else:
            assert all((a == b).all() for a, b in zip(labels, cur)), "label mismatch between ensemble seeds"
        for m in meta[s]["members"]:
            z = np.load(d / ("m%d.npz" % m["member"]))
            mem[(s, m["member"])] = {k: z[k] for k in ("test", "cal", "ood_svhn")}
    return mem, meta, labels


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
def single_split_stats(u, y_cal, y_test, score):
    """Extras for main's single-split sets (same sets as conformal_metrics: pipeline.evaluate._sets, seed = run seed)."""
    sets = _sets(u.cal, y_cal, u.test, ALPHA, score, u.run_seed)
    n = len(y_test)
    cov, sizes, corr = sets[np.arange(n), y_test], sets.sum(1), u.test.argmax(1) == y_test
    pc = [cov[y_test == c].mean() for c in range(K)]
    return dict(median_size=float(np.median(sizes)), pc_min=float(min(pc)), pc_max=float(max(pc)),
                size_correct=float(sizes[corr].mean()), size_incorrect=float(sizes[~corr].mean()) if (~corr).any() else np.nan)


def main():
    global OUT, TAB, FAIL, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-resplits", type=int, default=N_RESPLITS)
    ap.add_argument("--skip-images", action="store_true", help="skip failure image grids")
    ap.add_argument("--smoke", action="store_true", help="evaluate the 1-epoch smoke files under results/de_smoke/ (not real results)")
    args = ap.parse_args()
    t_start = time.time()
    R = roots(args.smoke)
    OUT, TAB, FAIL, FIG = R["out"], R["out"] / "tables", R["out"] / "failure", R["fig"]
    for d in (TAB, FAIL, FIG):
        d.mkdir(parents=True, exist_ok=True)
    banner = "**SMOKE RUN (1 epoch, tiny subsets): pipeline check only, these are not results.**\n\n" if args.smoke else ""

    mem, meta, labels = load_addon(R["addon"], args.smoke)
    ens_seeds = sorted(s for s in meta if all((s, i) in mem for i in range(N_MEMBERS)))
    for s in EXPECTED_ENS:
        if s not in ens_seeds:
            TODO.append("Ensemble seed %d (members %d-%d): not trained/dumped yet -> all ensemble-seed-%d numbers are TODO. "
                        "Train with src/de_train.py --seeds %d (GPU notebook train_jobs/de_train.ipynb)." % (s, s * 100, s * 100 + 4, s, s))
    if not ens_seeds:
        (OUT / "MISSING.md").write_text("# Missing / TODO (auto-generated by src/de_eval.py)\n\n" + "\n".join("- " + t for t in TODO) + "\n", encoding="utf-8")
        (OUT / "RESULTS.md").write_text("# Deep Ensembles results\n\nNo trained ensembles found yet. See MISSING.md.\n", encoding="utf-8")
        print("no ensembles found; wrote MISSING.md")
        return
    y_test, y_cal = labels
    nres = args.n_resplits
    singles = {k: Unit("s%d_m%d" % k, [k], mem, k[0]).fit_raps(y_cal) for k in sorted(mem)}
    ens = {s: Unit("ensemble seed %d" % s, [(s, i) for i in range(N_MEMBERS)], mem, s).fit_raps(y_cal) for s in ens_seeds}
    S, E = list(singles.values()), list(ens.values())
    everyone = S + E
    pool = sorted(mem)
    if len(pool) < 15:
        TODO.append("Ensemble-size ablation covers M=1..%d only (all 15 pipeline members need ensemble seeds 0, 1, 2); larger M is TODO." % len(pool))
    if len(ens) < 3:
        TODO.append("Only %d/3 ensemble seeds available: 'mean ± std over ensemble seeds' has n=%d and the paired t-test across ensembles is TODO (needs >=3)." % (len(ens), len(ens)))
    fe = E[0]
    nS = len(S)
    cn = "Single (mean ± std, %d members, ddof=1)" % nS
    cd = "DE M=5 (mean ± std, %d ens. seeds, ddof=1)" % len(E)
    have = {s: ens[s] for s in ens_seeds}

    def ens_cols(r, fn):
        for s in EXPECTED_ENS:
            r["Ens seed %d" % s] = ("%.4f" % fn(s)) if s in have else "TODO"
        return r

    # ---------------- headline metrics (main's common.compute_* ) ----------------
    ID_KEYS = ["accuracy", "ece", "nll", "brier", "adaptive_ece", "mean_conf", "mean_entropy", "mean_mi", "overconf_gap"]
    idm = {u.name: id_metrics(u.test, y_test, u.mi_test) for u in everyone}
    rows = []
    for k in ID_KEYS:
        r = {"Metric": k, cn: ms([idm[u.name][k] for u in S]), cd: ms([idm[u.name][k] for u in E])}
        rows.append(ens_cols(r, lambda s: idm[ens[s].name][k]))
    save_table(pd.DataFrame(rows), "main_metrics", "Accuracy and calibration, CIFAR-10 test (n=%d)" % len(y_test),
               "accuracy, ece, nll, brier use common.compute_*; adaptive_ece, mean_conf, mean_entropy, mean_mi, overconf_gap are add-ons. "
               "mean_mi is n/a for single models. Single-model std is the seed variance.")
    rows = []
    for u in everyone:
        r = {"unit": u.name, "member seeds": "-".join(str(a * 100 + b) for a, b in u.keys) if u.is_ens else u.keys[0][0] * 100 + u.keys[0][1]}
        r.update({k: "%.4f" % v for k, v in idm[u.name].items()})
        r["raps_size_test"] = "%.4f" % u.size_test.mean()
        rows.append(r)
    save_table(pd.DataFrame(rows), "per_seed_raw", "Per-member and per-ensemble raw metrics (fixed cal/test split)")

    # ---------------- conformal: main's conformal_metrics, unchanged ----------------
    print("conformal (main's conformal_metrics, %d re-splits) ..." % nres, flush=True)
    mc = {u.name: conformal_metrics(u.cal.astype(np.float32), y_cal, u.test.astype(np.float32), y_test, alpha=ALPHA,
                                    seed=u.run_seed, n_resplits=nres) for u in everyone}
    ss = {(u.name, sc): single_split_stats(u, y_cal, y_test, sc) for u in everyone for sc in ("aps_rand", "lac")}
    MAIN_STATS = []
    for sc in ("aps_rand", "lac"):
        MAIN_STATS += [(sc, "coverage", lambda m, sc=sc: m[sc]["coverage"]), (sc, "avg_set_size", lambda m, sc=sc: m[sc]["avg_set_size"]),
                       (sc, "frac_empty", lambda m, sc=sc: m[sc]["frac_empty"]), (sc, "singleton_frac", lambda m, sc=sc: m[sc]["singleton_frac"]),
                       (sc, "resplit coverage mean", lambda m, sc=sc: m[sc]["resplit"]["coverage_mean"]),
                       (sc, "resplit coverage std (ddof=0)", lambda m, sc=sc: m[sc]["resplit"]["coverage_std"]),
                       (sc, "resplit set size mean", lambda m, sc=sc: m[sc]["resplit"]["avg_set_size_mean"]),
                       (sc, "resplit set size std (ddof=0)", lambda m, sc=sc: m[sc]["resplit"]["avg_set_size_std"])]
    MAIN_STATS += [("mondrian_aps_rand", "coverage", lambda m: m["mondrian_aps_rand"]["coverage"]),
                   ("mondrian_aps_rand", "avg_set_size", lambda m: m["mondrian_aps_rand"]["avg_set_size"]),
                   ("mondrian_aps_rand", "min_class_coverage", lambda m: m["mondrian_aps_rand"]["min_class_coverage"])]
    rows = []
    for sc, lab, fn in MAIN_STATS:
        r = {"method": sc, "stat": lab, cn: ms([fn(mc[u.name]) for u in S]), cd: ms([fn(mc[u.name]) for u in E])}
        rows.append(ens_cols(r, lambda s: fn(mc[ens[s].name])))
    for sc in ("aps_rand", "lac"):
        for k, lab in (("median_size", "median set size (single split)"), ("pc_min", "per-class cov min (single split)"),
                       ("pc_max", "per-class cov max (single split)"), ("size_correct", "set size, correct preds (single split)"),
                       ("size_incorrect", "set size, incorrect preds (single split)")):
            r = {"method": sc, "stat": lab, cn: ms([ss[(u.name, sc)][k] for u in S]), cd: ms([ss[(u.name, sc)][k] for u in E])}
            rows.append(ens_cols(r, lambda s: ss[(ens[s].name, sc)][k]))
    save_table(pd.DataFrame(rows), "conformal_main", "Conformal prediction with main's pipeline.evaluate.conformal_metrics (alpha=0.1)",
               "Single split = cal/test split, draws seeded with the ensemble seed (members use their ensemble's seed). Re-splits: %d, permutation seed = seed*100000 + r, "
               "std over splits with ddof=0 (main's convention); mean ± std in the Single/DE columns is across members/ensemble seeds with ddof=1." % nres)
    rows = []
    for u in E:
        for sc in ("aps_rand", "lac"):
            h = mc[u.name][sc]
            rows.append({"unit": u.name, "method": sc, **{"size %d" % k: int(v) for k, v in enumerate(h["set_size_hist"])},
                         "coverage by set size": json.dumps({k: round(v, 4) for k, v in h["coverage_by_set_size"].items()})})
    save_table(pd.DataFrame(rows), "conformal_setsize_hist", "Set-size histogram (counts) and coverage by set size, single split (from conformal_metrics)")

    # main-format summary (pipeline.evaluate.compare/summarize, i.e. main's ddof=1 across-seed std)
    def as_main(u):
        m = idm[u.name]
        return {"model": "deep_ensemble" if u.is_ens else "deep_ensemble_member", "noise_ratio": 0, "n_svhn": 0, "seed": u.run_seed,
                "accuracy": m["accuracy"], "nll": m["nll"], "ece": m["ece"], "brier": m["brier"], "conformal": mc[u.name]}
    summ = summarize(compare([as_main(u) for u in everyone], path=None))
    rows = [{k: ("%.4f ± %.4f" % (r[k], r[k + "_std"]) if (k + "_std") in r and r["n_seeds"] > 1 else ("%.4f" % r[k] if isinstance(r[k], float) else r[k]))
             for k in r if not k.endswith("_std")} for r in summ]
    save_table(pd.DataFrame(rows), "main_format_summary", "Main-format summary (pipeline.evaluate.summarize: mean ± std across seeds, ddof=1)",
               "'deep_ensemble_member' = the %d individual members (each evaluated with its ensemble's seed); 'deep_ensemble' = ensemble seeds %s." % (nS, ens_seeds))

    # conformal add-ons: re-splits with the same convention (non-randomized APS, RAPS, richer stats for randomized APS / Mondrian)
    print("conformal add-ons ...", flush=True)
    ce = {u.name: resplit_eval(u.cal, y_cal, u.test, y_test, n_splits=nres, seed=u.run_seed) for u in everyone}
    CST = [("coverage", "coverage"), ("avg_size", "avg set size"), ("median_size", "median set size"), ("pc_cov_min", "per-class cov min"),
           ("pc_cov_max", "per-class cov max"), ("size_correct", "set size (correct preds)"), ("size_incorrect", "set size (incorrect preds)"),
           ("frac_empty", "frac empty sets")]
    rows = []
    for m in METHODS:
        for k, lab in CST:
            r = {"method": m, "stat": lab, cn: ms([ce[u.name][m][k].mean() for u in S]), cd: ms([ce[u.name][m][k].mean() for u in E])}
            r = ens_cols(r, lambda s: ce[ens[s].name][m][k].mean())
            r["First ens. seed: std over splits (ddof=0)"] = "%.4f" % ce[fe.name][m][k].std(ddof=0)
            rows.append(r)
    save_table(pd.DataFrame(rows), "conformal_extras", "Conformal add-ons: non-randomized APS and RAPS plus richer stats, mean over %d re-splits" % nres,
               "Same re-split convention as main (permutation seed*100000 + r, draws seeded by r). aps_rand/mondrian are also in main's tables; sets can be empty (score<=qhat rule).")

    # ---------------- consistency checks vs main
    cons = []
    d_cov = max(abs(ce[u.name]["aps_rand"]["coverage"].mean() - mc[u.name]["aps_rand"]["resplit"]["coverage_mean"]) for u in everyone)
    d_siz = max(abs(ce[u.name]["aps_rand"]["avg_size"].mean() - mc[u.name]["aps_rand"]["resplit"]["avg_set_size_mean"]) for u in everyone)
    d_std = max(abs(ce[u.name]["aps_rand"]["coverage"].std(ddof=0) - mc[u.name]["aps_rand"]["resplit"]["coverage_std"]) for u in everyone)
    cons.append("add-on randomized APS re-splits vs main's conformal_metrics (all %d units): max |diff| coverage mean %.2e, set size mean %.2e, coverage std (ddof=0) %.2e"
                % (len(everyone), d_cov, d_siz, d_std))
    for s in ens_seeds:
        jp = R["main"] / (tag_for(s, args.smoke) + ".json")
        if jp.exists():
            mj = json.load(open(jp))
            u = ens[s]
            diffs = dict(accuracy=abs(mj["accuracy"] - idm[u.name]["accuracy"]), ece=abs(mj["ece"] - idm[u.name]["ece"]),
                         nll=abs(mj["nll"] - idm[u.name]["nll"]), brier=abs(mj["brier"] - idm[u.name]["brier"]),
                         aps_cov=abs(mj["conformal"]["aps_rand"]["coverage"] - mc[u.name]["aps_rand"]["coverage"]),
                         aps_size=abs(mj["conformal"]["aps_rand"]["avg_set_size"] - mc[u.name]["aps_rand"]["avg_set_size"]),
                         lac_size=abs(mj["conformal"]["lac"]["avg_set_size"] - mc[u.name]["lac"]["avg_set_size"]))
            cons.append("ensemble seed %d: saved main result %s vs recomputed from add-on probs, max |diff|: %s" % (
                s, jp.name, ", ".join("%s %.1e" % kv for kv in diffs.items())))
        else:
            cons.append("ensemble seed %d: main result %s not present in this checkout (git-ignored .npz / not committed); cross-check skipped" % (s, jp.name))
    MD.append("### Consistency checks\n\n" + "\n".join("- " + c for c in cons) + "\n")

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
            r = {"score": sname, "metric": k, cn: ms([oodm[(u.name, sname)][k] for u in S]), cd: ms([oodm[(u.name, sname)][k] for u in E])}
            rows.append(ens_cols(r, lambda s: oodm[(ens[s].name, sname)][k]))
    for k in ("svhn_mean_conf", "svhn_frac_conf_gt_0_9"):
        r = {"score": "-", "metric": k, cn: ms([extra[u.name][k] for u in S]), cd: ms([extra[u.name][k] for u in E])}
        rows.append(ens_cols(r, lambda s: extra[ens[s].name][k]))
    n_ood = len(fe.ood)
    save_table(pd.DataFrame(rows), "ood", "OOD detection: CIFAR-10 test (%d) vs SVHN test (%d)" % (len(y_test), n_ood),
               "ID is positive for AUPR-In and FPR@95TPR (fraction of SVHN accepted when 95%% of CIFAR test is kept); OOD is positive for AUROC and AUPR-Out. "
               "raps_size = RAPS (k_reg=5, lam=0.01) set size, qhat from the cal split. MI is n/a for single models. OOD prevalence %.2f." % (n_ood / (n_ood + len(y_test))))

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
                 cn: ms([sel[(u.name, sname)][k] for u in S if (u.name, sname) in sel]), cd: ms([sel[(u.name, sname)][k] for u in E])}
            rows.append(ens_cols(r, lambda s: sel[(ens[s].name, sname)][k]))
    save_table(pd.DataFrame(rows), "selective", "Selective prediction (accuracy at coverage 80/90/95%; AURC = mean selective risk, lower is better)",
               "MI is n/a for single models. Full-coverage accuracy is in the main table.")

    # ---------------- ensemble-size ablation (members drawn from all available pipeline members)
    print("ablation ...", flush=True)
    rng = np.random.RandomState(0)
    AB = ("accuracy", "ece", "nll", "brier", "mean_conf", "raps_size", "ood_auroc_entropy", "ood_auroc_mi")
    abl_raw = []
    for M in range(1, len(pool) + 1):
        if math.comb(len(pool), M) <= N_SUBSETS:      # few enough: use every subset
            combos = list(itertools.combinations(pool, M))
        else:                                          # otherwise N_SUBSETS distinct random subsets
            seen = set()
            while len(seen) < N_SUBSETS:
                seen.add(tuple(sorted(rng.choice(len(pool), M, replace=False))))
            combos = [tuple(pool[i] for i in c) for c in sorted(seen)]
        vals = []
        for c in combos:
            u = singles[c[0]] if M == 1 else Unit("M%d" % M, c, mem, 0).fit_raps(y_cal)
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
               "ablation_size", "Ensemble-size ablation (mean ± std over random member subsets drawn from the %d available pipeline members)" % len(pool),
               "Up to %d distinct subsets per M (all if there are no more than that), RandomState(0); M=%d has one subset (std 0). Subsets may mix members from different ensemble seeds." % (N_SUBSETS, len(pool)))

    # ---------------- DE vs single: paired deltas
    print("bootstrap ...", flush=True)
    DK = [("accuracy", "Δ accuracy"), ("ece", "Δ ECE"), ("nll", "Δ NLL"), ("brier", "Δ Brier"),
          ("raps_size", "Δ RAPS set size"), ("ood_auroc_entropy", "Δ OOD AUROC (entropy)")]
    sig_rows, delta = [], {}
    for s, e in ens.items():
        mus = [singles[(s, i)] for i in range(N_MEMBERS)]
        b = bootstrap_deltas(e, mus, y_test, np.random.RandomState(0), B=args.n_boot)
        mean_m = lambda k: np.mean([idm[m.name][k] for m in mus])  # noqa: E731
        delta[s] = dict(accuracy=idm[e.name]["accuracy"] - mean_m("accuracy"), ece=idm[e.name]["ece"] - mean_m("ece"),
                        nll=idm[e.name]["nll"] - mean_m("nll"), brier=idm[e.name]["brier"] - mean_m("brier"),
                        raps_size=e.size_test.mean() - np.mean([m.size_test.mean() for m in mus]),
                        ood_auroc_entropy=oodm[(e.name, "entropy")]["auroc"] - np.mean([oodm[(m.name, "entropy")]["auroc"] for m in mus]))
        for k, lab in DK:
            lo, hi = np.percentile(b[k], [2.5, 97.5])
            sig_rows.append({"ensemble": e.name, "metric": lab, "Δ (ens − mean of its members)": "%.4f" % delta[s][k],
                             "95% bootstrap CI": "[%.4f, %.4f]" % (lo, hi), "excludes 0": "yes" if (lo > 0 or hi < 0) else "no"})
    for k, lab in DK:
        ds = [delta[s][k] for s in ens_seeds]
        if len(ds) >= 3:
            t, p = sps.ttest_1samp(ds, 0.0)
            sig_rows.append({"ensemble": "paired t-test, %d ensembles" % len(ds), "metric": lab, "Δ (ens − mean of its members)": ms(ds),
                             "95% bootstrap CI": "t=%.3f" % t, "excludes 0": "p=%.4g" % p})
        else:
            sig_rows.append({"ensemble": "paired t-test", "metric": lab, "Δ (ens − mean of its members)": "TODO (need >=3 ensembles, have %d)" % len(ds),
                             "95% bootstrap CI": "TODO", "excludes 0": "TODO"})
    save_table(pd.DataFrame(sig_rows), "significance", "DE vs single models: deltas, paired bootstrap (%d resamples) and t-test across ensembles" % args.n_boot,
               "Delta = ensemble metric minus the mean of the same metric over its own 5 members (ECE/NLL/size: negative = DE better). "
               "Bootstrap resamples the test images (same resample for the ensemble and its members); the OOD row also resamples the SVHN images.")

    # ---------------- cost
    allm = [m for s in ens_seeds for m in meta[s]["members"]]
    t_rec = [m["train_minutes"] for m in allm if m.get("train_minutes") is not None]
    inf = [m["inference_sec_per_1000_imgs"] for m in allm]
    npar = allm[0]["n_params"]
    cost = pd.DataFrame([
        {"quantity": "params, single ResNet-20", "value": "{:,}".format(npar)},
        {"quantity": "params, DE M=5 (5x)", "value": "{:,}".format(5 * npar)},
        {"quantity": "params, DE M=10 (10x)", "value": "{:,}".format(10 * npar)},
        {"quantity": "train time per member, min (mean ± std over %d members; see hardware row)" % len(t_rec), "value": ms(t_rec, 1)},
        {"quantity": "total train time, h (%d members)" % len(t_rec), "value": "%.2f" % (sum(t_rec) / 60)},
        {"quantity": "inference s / 1000 images, single (mean ± std over members)", "value": ms(inf, 3)},
        {"quantity": "inference s / 1000 images, DE M=5 (= 5 x single, sequential)", "value": "%.3f" % (5 * np.mean(inf))},
        {"quantity": "inference s / 1000 images, DE M=10 (= 10 x single)", "value": "%.3f" % (10 * np.mean(inf))},
        {"quantity": "hardware", "value": "; ".join(sorted(set(meta[s]["hardware"] for s in ens_seeds)))},
    ])
    save_table(cost, "cost", "Cost", "Train minutes come from main's checkpoint history (pipeline.train); inference time measured by the add-on dumper (batch 256, includes preprocessing). "
               "Timings from different hardware are not comparable.")

    # ---------------- failure analysis (first ensemble seed)
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
    from pipeline.data import get_cifar_splits, _smoke_subset, SMOKE_EVAL
    images, _, _, te_idx, _ = get_cifar_splits()
    tg = te_idx[_smoke_subset(range(len(te_idx)), SMOKE_EVAL, seed=1)] if args.smoke else te_idx
    assert len(tg) == len(y_test)
    lab = lambda ix: dict(idx=ix.tolist(), global_idx=tg[ix].tolist(), true=[CLASSES[i] for i in y_test[ix]], pred=[CLASSES[i] for i in pred[ix]])  # noqa: E731
    outs = dict(confusion_matrix=cm.tolist(), top5_confused_pairs=top_pairs,
                confident_wrong=dict(lab(top_wrong), conf=conf[top_wrong].round(4).tolist()),
                confident_svhn=dict(idx=top_svhn.tolist(), pred=[CLASSES[i] for i in fe.ood.argmax(1)[top_svhn]], conf=svhn_conf[top_svhn].round(4).tolist()),
                largest_sets=dict(lab(big), size=fe.size_test[big].tolist()),
                highest_mi=dict(lab(top_mi), mi=fe.mi_test[top_mi].round(4).tolist()))
    for k, v in outs.items():
        json.dump(dict(ensemble=fe.name, smoke=args.smoke, note="idx = position in the test split (SVHN: in the SVHN test set); global_idx = index into the pooled 60k CIFAR-10 array", **{k: v}),
                  open(FAIL / (k + ".json"), "w"), indent=1)
    MD.append("### Top 5 confused pairs (%s, test)\n\n%s\n" % (fe.name, to_md(pd.DataFrame(top_pairs))))

    # ---------------- figures
    print("figures ...", flush=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    s0 = singles[(ens_seeds[0], 0)]
    big_ens = Unit("M10", pool[:10], mem, 0) if len(pool) >= 10 else None
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    reliability(ax, s0.test, y_test, "single (%s)" % s0.name, "#1f77b4")
    reliability(ax, fe.test, y_test, "DE M=5 (%s)" % fe.name, "#d95f02")
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
    ax.plot(cov, curves[(s0.name, "msp")], color="#1f77b4", label="single (%s) MSP" % s0.name)
    for sname, c in (("msp", "#d95f02"), ("entropy", "#7570b3"), ("mi", "#1b9e77")):
        ax.plot(cov, curves[(fe.name, sname)], color=c, label="DE M=5 %s" % sname)
    ax.set(xlim=(0.3, 1), xlabel="coverage", ylabel="accuracy")
    if not args.smoke:
        ax.set_ylim(0.9, 1.002)
    ax.legend()
    savefig(fig, "accuracy_vs_coverage")

    fig, ax = plt.subplots(figsize=(5, 3.6))
    for i, (u, lab_, c) in enumerate([(s0, "single %s" % s0.name, "#1f77b4"), (fe, "DE M=5", "#d95f02")]):
        ax.bar(np.arange(11) + (i - 0.5) * 0.38, np.bincount(u.size_test, minlength=11)[:11] / len(u.size_test), 0.38, label=lab_, color=c)
    ax.set(yscale="log", xlabel="RAPS set size (alpha=0.1)", ylabel="fraction of test images")
    ax.legend()
    savefig(fig, "setsize_hist")

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (t, a, b) in zip(axs[:2], [("entropy", entropy(fe.test), entropy(fe.ood)), ("mutual information", fe.mi_test, fe.mi_ood)]):
        bins = np.linspace(0, max(a.max(), b.max()) + 1e-9, 60)
        ax.hist(a, bins, alpha=0.6, density=True, label="CIFAR-10 test", color="#1f77b4")
        ax.hist(b, bins, alpha=0.6, density=True, label="SVHN test", color="#d95f02")
        ax.set(yscale="log", xlabel=t + " (DE M=5)")
        ax.legend()
    ys = np.r_[np.zeros(len(fe.test)), np.ones(len(fe.ood))]
    sid, sood = ood_score_dict(fe.test, fe.mi_test, fe.size_test), ood_score_dict(fe.ood, fe.mi_ood, fe.size_ood)
    for sname, c in (("msp", "#1f77b4"), ("entropy", "#d95f02"), ("mi", "#1b9e77")):
        f_, t_, _ = roc_curve(ys, np.r_[sid[sname], sood[sname]])
        axs[2].plot(f_, t_, color=c, label="%s (AUROC %.3f)" % (sname, oodm[(fe.name, sname)]["auroc"]))
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
    ax.set(xlabel="predicted", ylabel="true", title="%s confusion (test)" % fe.name)
    ax.grid(False)
    savefig(fig, "confusion_matrix")

    if not args.skip_images:
        svhn = load_svhn_test(args.smoke)
        cls = lambda i: CLASSES[i]  # noqa: E731
        image_grid([images[tg[i]] for i in top_wrong], ["true %s\npred %s (%.3f)" % (cls(y_test[i]), cls(pred[i]), conf[i]) for i in top_wrong],
                   "failure_confident_wrong", "16 most confident wrong test predictions (%s)" % fe.name)
        image_grid([svhn[i] for i in top_svhn], ["pred %s\nconf %.3f" % (cls(fe.ood[i].argmax()), svhn_conf[i]) for i in top_svhn],
                   "failure_confident_svhn", "16 most confident SVHN images (%s)" % fe.name)
        image_grid([images[tg[i]] for i in big], ["true %s pred %s\nsize %d" % (cls(y_test[i]), cls(pred[i]), fe.size_test[i]) for i in big],
                   "failure_largest_sets", "16 largest RAPS sets (%s)" % fe.name)
        image_grid([images[tg[i]] for i in top_mi], ["true %s pred %s\nMI %.3f" % (cls(y_test[i]), cls(pred[i]), fe.mi_test[i]) for i in top_mi],
                   "failure_highest_mi", "16 highest mutual-information test images (%s)" % fe.name)

    # ---------------- raw observations + flags
    sm = {k: np.mean([idm[u.name][k] for u in S]) for k in ID_KEYS if k != "mean_mi"}
    a = idm[fe.name]
    sing = lambda f: np.mean([f(u) for u in S])  # noqa: E731
    obs = ["Members: %d (%s); ensemble seeds present: %s." % (nS, ", ".join(sorted(u.name for u in S)), ens_seeds),
           "Single mean vs %s: acc %.4f -> %.4f, ECE %.4f -> %.4f, NLL %.4f -> %.4f, Brier %.4f -> %.4f." % (
               fe.name, sm["accuracy"], a["accuracy"], sm["ece"], a["ece"], sm["nll"], a["nll"], sm["brier"], a["brier"]),
           "Overconfidence gap (mean conf - acc): single %.4f, %s %.4f." % (sm["overconf_gap"], fe.name, a["overconf_gap"]),
           "Randomized APS (main's conformal_metrics, single split): %s coverage %.4f, size %.4f; single-member mean size %.4f." % (
               fe.name, mc[fe.name]["aps_rand"]["coverage"], mc[fe.name]["aps_rand"]["avg_set_size"], sing(lambda u: mc[u.name]["aps_rand"]["avg_set_size"])),
           "LAC (main): %s coverage %.4f, size %.4f." % (fe.name, mc[fe.name]["lac"]["coverage"], mc[fe.name]["lac"]["avg_set_size"]),
           "Non-randomized APS re-split mean (add-on): %s size %.3f, coverage %.4f, empty-set fraction %.4f." % (
               fe.name, ce[fe.name]["aps"]["avg_size"].mean(), ce[fe.name]["aps"]["coverage"].mean(), ce[fe.name]["aps"]["frac_empty"].mean()),
           "OOD AUROC (entropy / MI / MSP / RAPS size), %s: %.4f / %.4f / %.4f / %.4f; single mean entropy %.4f, MSP %.4f." % (
               fe.name, oodm[(fe.name, "entropy")]["auroc"], oodm[(fe.name, "mi")]["auroc"], oodm[(fe.name, "msp")]["auroc"], oodm[(fe.name, "raps_size")]["auroc"],
               sing(lambda u: oodm[(u.name, "entropy")]["auroc"]), sing(lambda u: oodm[(u.name, "msp")]["auroc"])),
           "SVHN mean confidence: single %.4f, %s %.4f; fraction conf>0.9: single %.4f, %s %.4f." % (
               sing(lambda u: extra[u.name]["svhn_mean_conf"]), fe.name, extra[fe.name]["svhn_mean_conf"],
               sing(lambda u: extra[u.name]["svhn_frac_conf_gt_0_9"]), fe.name, extra[fe.name]["svhn_frac_conf_gt_0_9"]),
           "Member accuracy range: %.4f-%.4f." % (min(idm[u.name]["accuracy"] for u in S), max(idm[u.name]["accuracy"] for u in S)),
           "Ablation: ECE lowest at M=%d (%.4f), %.4f at M=%d; NLL M=1 %.4f -> M=%d %.4f." % (
               int(abl["M"].iloc[abl["ece_mean"].idxmin()]), abl["ece_mean"].min(), abl["ece_mean"].iloc[-1], len(abl),
               abl["nll_mean"].iloc[0], len(abl), abl["nll_mean"].iloc[-1]),
           "Top confused pair (%s): %s -> %s, %d test images." % (fe.name, top_pairs[0]["true"], top_pairs[0]["pred"], top_pairs[0]["count"])]
    flags = []
    if not args.smoke:
        for u in S:
            if not 0.88 <= idm[u.name]["accuracy"] <= 0.93:
                flags.append("%s accuracy %.4f outside the expected ~0.88-0.93 range." % (u.name, idm[u.name]["accuracy"]))
        for u in E:
            if idm[u.name]["accuracy"] < np.mean([idm[singles[k].name]["accuracy"] for k in u.keys]):
                flags.append("%s accuracy is below the mean of its members." % u.name)
            if idm[u.name]["ece"] > np.mean([idm[singles[k].name]["ece"] for k in u.keys]):
                flags.append("%s ECE is higher than the mean of its members." % u.name)
        for s, d in delta.items():
            if d["ood_auroc_entropy"] < 0:
                flags.append("Ensemble seed %d has LOWER entropy-OOD AUROC than its members' mean." % s)
    MD.append("### Raw observations\n\n" + "\n".join("- " + o for o in obs) + "\n\n### Flags\n\n" + "\n".join("- " + f_ for f_ in flags or ["none"]) + "\n")

    head = "# Deep Ensembles results (auto-generated by `python src/de_eval.py`)\n\n%sRuntime %.0f s. Members: %d. Ensemble seeds: %s. See NOTES.md for definitions and MISSING.md for TODOs.\n\n" % (
        banner, time.time() - t_start, nS, ens_seeds)
    (OUT / "RESULTS.md").write_text(head + "\n".join(MD), encoding="utf-8")
    (OUT / "MISSING.md").write_text("# Missing / TODO (auto-generated by src/de_eval.py)\n\n" + ("\n".join("- " + t for t in TODO) if TODO else "Nothing missing.") + "\n", encoding="utf-8")
    print("done in %.0fs; TODO items: %d" % (time.time() - t_start, len(TODO)))


if __name__ == "__main__":
    main()
