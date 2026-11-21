"""Reusable split-conformal layer (takes softmax-probability arrays; no model, no torch).

Teammates can call it:
    from de_conformal import ConformalSets, resplit_eval
    cs = ConformalSets("raps", alpha=0.1, k_reg=5, lam=0.01, seed=0).fit(cal_probs, cal_labels)
    member = cs.predict_sets(test_probs)        # bool (n, K); sizes = member.sum(1)

Methods ("aps", "aps_rand", "raps", "mondrian_aps_rand"). All use the SAME function for
calibration and prediction: class y is in the set iff score(x, y) <= qhat, so empty sets are
possible (frac_empty is reported). Scores (rank r is 1-indexed in descending probability):
    aps       : sum of probs of classes ranked <= r  (inclusive cumsum, no randomisation)
    aps_rand  : mass strictly above y + U * p_y
    raps      : aps_rand + lam * max(0, r - k_reg)
    mondrian_aps_rand : aps_rand score, one qhat per TRUE class (class-conditional)
qhat = ceil((n+1)(1-alpha))/n empirical quantile (inf -> full set if that rank exceeds n).
"""
import numpy as np

METHODS = ("aps", "aps_rand", "raps", "mondrian_aps_rand")


def sort_stats(probs):
    p = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-p, axis=1, kind="stable")
    ranked = np.take_along_axis(p, order, axis=1)
    before = np.cumsum(ranked, axis=1) - ranked
    return order, ranked, before


def all_scores(order, ranked, before, u, method, k_reg=5, lam=0.01):
    """score(x, y) for every class y, shape (n, K), from precomputed sort stats."""
    if method == "aps":
        s = before + ranked
    else:
        s = before + u[:, None] * ranked
        if method == "raps":
            ranks = np.arange(1, ranked.shape[1] + 1)[None, :]
            s = s + lam * np.maximum(0, ranks - k_reg)
    out = np.empty_like(s)
    np.put_along_axis(out, order, s, axis=1)
    return out


def conformal_quantile(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


class ConformalSets:
    def __init__(self, method="raps", alpha=0.1, k_reg=5, lam=0.01, seed=0):
        assert method in METHODS
        self.method, self.alpha, self.k_reg, self.lam, self.seed = method, alpha, k_reg, lam, seed

    def _scores(self, probs, rng):
        order, ranked, before = sort_stats(probs)
        return all_scores(order, ranked, before, rng.rand(len(probs)), self.method, self.k_reg, self.lam)

    def fit(self, cal_probs, cal_labels):
        rng = np.random.RandomState(self.seed)
        s = self._scores(cal_probs, rng)
        self._rng_state = rng.get_state()   # test U continues this stream (same draw order as conformal_fixed.py)
        ts = s[np.arange(len(cal_labels)), cal_labels]
        if self.method == "mondrian_aps_rand":
            K = cal_probs.shape[1]
            self.qhat = np.array([conformal_quantile(ts[cal_labels == c], self.alpha) for c in range(K)])
        else:
            self.qhat = conformal_quantile(ts, self.alpha)
        return self

    def predict_sets(self, probs, seed=None):
        rng = np.random.RandomState(0 if seed is None else seed)
        if seed is None:
            rng.set_state(self._rng_state)
        s = self._scores(probs, rng)
        return s <= (self.qhat[None, :] if self.method == "mondrian_aps_rand" else self.qhat)


def set_stats(member, labels, probs):
    """Coverage / size / per-class coverage / correct-vs-incorrect size for a bool (n,K) set matrix."""
    n, K = member.shape
    cov = member[np.arange(n), labels]
    sizes = member.sum(1)
    correct = probs.argmax(1) == labels
    pc = np.array([cov[labels == c].mean() for c in range(K)])
    return dict(coverage=float(cov.mean()), avg_size=float(sizes.mean()), median_size=float(np.median(sizes)),
                pc_cov_min=float(pc.min()), pc_cov_max=float(pc.max()),
                size_correct=float(sizes[correct].mean()), size_incorrect=float(sizes[~correct].mean()),
                frac_empty=float((sizes == 0).mean()))


def resplit_eval(cal_probs, cal_labels, test_probs, test_labels, n_splits=100, alpha=0.1,
                 k_reg=5, lam=0.01, methods=METHODS):
    """Pool cal+test (24k), re-split 12k/12k with seeds 0..n_splits-1. Split r: permutation from
    RandomState(r); U from a separate RandomState(r): first n_cal draws for cal, the next for test (identical
    to conformal_fixed.split_conformal(seed=r) and pipeline/evaluate.py). Returns {method: {stat: array(n_splits)}}."""
    P = np.concatenate([cal_probs, test_probs]).astype(np.float64)
    Y = np.concatenate([cal_labels, test_labels])
    n, n_cal = len(Y), len(cal_labels)
    order, ranked, before = sort_stats(P)
    K = P.shape[1]
    correct = P.argmax(1) == Y
    res = {m: {} for m in methods}
    for r in range(n_splits):
        rng = np.random.RandomState(r)
        perm = rng.permutation(n)
        ci, ti = perm[:n_cal], perm[n_cal:]
        ru = np.random.RandomState(r)
        u = np.empty(n)
        u[ci] = ru.rand(len(ci))
        u[ti] = ru.rand(len(ti))
        for m in methods:
            S = all_scores(order, ranked, before, u, m, k_reg, lam)
            ts = S[np.arange(n), Y]
            if m == "mondrian_aps_rand":
                q = np.array([conformal_quantile(ts[ci][Y[ci] == c], alpha) for c in range(K)])[None, :]
            else:
                q = conformal_quantile(ts[ci], alpha)
            mem = S[ti] <= q
            yt = Y[ti]
            cov = mem[np.arange(len(ti)), yt]
            sizes = mem.sum(1)
            pc = np.array([cov[yt == c].mean() for c in range(K)])
            ct = correct[ti]
            st = dict(coverage=cov.mean(), avg_size=sizes.mean(), median_size=np.median(sizes),
                      pc_cov_min=pc.min(), pc_cov_max=pc.max(), size_correct=sizes[ct].mean(),
                      size_incorrect=sizes[~ct].mean(), frac_empty=(sizes == 0).mean())
            for k, v in st.items():
                res[m].setdefault(k, []).append(float(v))
    return {m: {k: np.array(v) for k, v in d.items()} for m, d in res.items()}
