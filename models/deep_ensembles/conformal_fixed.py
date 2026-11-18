"""Corrected conformal layer (proposal -- the shared notebook code is untouched).

Primary fix: *randomized* APS (Romano et al., 2020) with a set rule that is
literally "include class y iff score(x, y) <= qhat", so the score used for
calibration and the rule used for prediction are the same function.
Also provides LAC (1 - p_true) as a reference, and Mondrian (class-conditional)
variants of both.

All functions take softmax probabilities (n, K) and integer labels.
Randomness comes from an explicit seed so results are reproducible.
"""
import numpy as np


# ------------------------------ scores ------------------------------------
def _aps_all_scores(probs, u):
    """score(x, y) for *every* class y: mass of classes ranked strictly above
    y plus u * p_y, with one U ~ Uniform(0,1) per point. Shape (n, K)."""
    order = np.argsort(-probs, axis=1)
    ranked = np.take_along_axis(probs, order, axis=1).astype(np.float64)
    before = np.cumsum(ranked, axis=1) - ranked
    scores_ranked = before + u[:, None] * ranked
    scores = np.empty_like(scores_ranked)
    np.put_along_axis(scores, order, scores_ranked, axis=1)
    return scores


def _lac_all_scores(probs, u=None):
    return 1.0 - probs.astype(np.float64)


SCORES = {"aps_rand": _aps_all_scores, "lac": _lac_all_scores}


def _true_scores(all_scores, labels):
    return all_scores[np.arange(len(labels)), labels]


def conformal_quantile(scores, alpha):
    """Upper empirical quantile: ceil((n+1)(1-alpha))/n (inf if > n)."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


# ------------------------- split conformal ---------------------------------
def split_conformal(cal_probs, cal_labels, test_probs, test_labels,
                    alpha=0.1, score="aps_rand", seed=0):
    f = SCORES[score]
    rng = np.random.RandomState(seed)
    u_cal = rng.rand(len(cal_labels))
    u_test = rng.rand(len(test_labels))
    qhat = conformal_quantile(_true_scores(f(cal_probs, u_cal), cal_labels), alpha)
    sets = f(test_probs, u_test) <= qhat            # (n, K) boolean membership
    covered = sets[np.arange(len(test_labels)), test_labels]
    sizes = sets.sum(1)
    return dict(qhat=float(qhat), coverage=float(covered.mean()),
                avg_set_size=float(sizes.mean()), frac_empty=float((sizes == 0).mean()))


# ------------------------- Mondrian (per-class) ----------------------------
def mondrian_conformal(cal_probs, cal_labels, test_probs, test_labels,
                       alpha=0.1, score="aps_rand", seed=0, n_classes=None):
    """Class-conditional: one qhat per true class, calibrated only on
    calibration points of that class. Class c enters the set of x iff
    score(x, c) <= qhat_c."""
    f = SCORES[score]
    K = n_classes or cal_probs.shape[1]
    rng = np.random.RandomState(seed)
    u_cal = rng.rand(len(cal_labels))
    u_test = rng.rand(len(test_labels))
    cal_true = _true_scores(f(cal_probs, u_cal), cal_labels)
    qhats = np.array([conformal_quantile(cal_true[cal_labels == c], alpha) for c in range(K)])
    sets = f(test_probs, u_test) <= qhats[None, :]
    covered = sets[np.arange(len(test_labels)), test_labels]
    per_class = [float(covered[test_labels == c].mean()) for c in range(K)]
    sizes = sets.sum(1)
    return dict(qhats=qhats.tolist(), coverage=float(covered.mean()),
                per_class_coverage=per_class, avg_set_size=float(sizes.mean()),
                frac_empty=float((sizes == 0).mean()))
