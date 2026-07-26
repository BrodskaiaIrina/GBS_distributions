"""
gbs_core.py — shared foundation for the GBS distribution-learning methods.

Contains:
  * NumPy 2.0 compatibility shim for thewalrus.
  * Target distributions (normal, mmodal, lognorm, expo, cauchy, bimodal_asym).
  * Baseline GBS utilities from the original GBS.py
    (bin_samples_from_dist, A_from_samples, train_WAW, gbs_sample,
     probability, prob_rn, dkl, ones_inds_in_bin, ...).
  * Analytic (Monte-Carlo-free) evaluation helpers
    (model_distribution_from_A, analytic_eval, _safe_kl, _safe_takagi_cap).

Every method module imports the pieces it needs from here so the code paths
match the names used in the notebooks.
"""

import numpy as np

# --- NumPy 2.0 compatibility shim -------------------------------------------
# thewalrus' Fock-space routines still call np.find_common_type (removed in
# NumPy 2.0). Restore it before importing thewalrus.
if not hasattr(np, "find_common_type"):
    np.find_common_type = (lambda array_types, scalar_types=():
                           np.result_type(*([*array_types, *scalar_types] or [np.float64])))

import thewalrus as tw
from thewalrus.quantum import Covmat, adj_to_qmat
from thewalrus._torontonian import threshold_detection_prob
from thewalrus.decompositions import takagi
from numpy.linalg import norm
from strawberryfields.apps import train


# ============================================================================
# Target distributions (defined over a 1-D variable x on [x0, x1])
# ============================================================================
def normal(x):
    p = np.exp(-((1.5 - x) ** 2))
    return p / np.sum(p)


def mmodal(x):
    p = np.exp(-((1 - x) ** 2) / 0.1) + np.exp(-((3 - x) ** 2) / 0.1)
    return p / np.sum(p)


def lognorm(x):
    sig, mu = 0.5, 0.0
    ln = 1.0 / (x * sig * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - mu) ** 2 / (2 * sig ** 2))
    return ln / np.sum(ln)


def expo(x):
    p = np.exp(-x)
    return p / np.sum(p)


def cauchy(x):
    p = 1.0 / (1.0 + ((x - 2.0) / 0.3) ** 2)
    return p / np.sum(p)


def bimodal_asym(x):
    p = 0.7 * np.exp(-((1 - x) ** 2) / 0.1) + 0.3 * np.exp(-((3 - x) ** 2) / 0.1)
    return p / np.sum(p)


TARGETS = {"normal": normal, "mmodal": mmodal, "lognorm": lognorm,
           "expo": expo, "cauchy": cauchy, "bimodal_asym": bimodal_asym}


# ============================================================================
# Baseline utilities (ported from the original GBS.py)
# ============================================================================
def _bin_search(arr, num, n):
    ind = int(2 ** (n - 1))
    for i in range(1, n + 1):
        if num <= arr[ind]:
            ind = ind - 2 ** (n - i - 1)
        else:
            ind = ind + 2 ** (n - i - 1)
    return int(np.ceil(ind)), arr[int(np.ceil(ind))]


def _sample_from_dist(dist):
    cdf = np.cumsum(dist)
    num = np.random.rand()
    n = int(np.log(len(dist)) / np.log(2))
    idx, val = _bin_search(cdf, num, n)
    return idx, val


def _bin_list(sbin):
    return [1 if ch == "1" else 0 for ch in sbin]


def ones_inds_in_bin(s):
    """Indices of the clicking (=1) modes in a binary pattern."""
    return [i for i in range(len(s)) if s[i] == 1]


def _edge_combinations(nodes):
    """Original GBS.py edge rule: a single clicking mode yields a self-loop,
    otherwise all unordered pairs."""
    if len(nodes) == 1:
        return [[nodes[0], nodes[0]]]
    combs = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            combs.append([nodes[i], nodes[j]])
    return combs


def _add_edges(A, edges):
    for e1, e2 in edges:
        A[e1, e2] += 1
    return A


def bin_samples_from_dist(p_dist, x0, x1, num_samples, modes):
    """Draw click-pattern samples that encode 1-D target bins as bit strings
    (standard binary encoding). x0, x1 delimit the target's support."""
    x = np.linspace(x0, x1, 2 ** modes - 2)
    d = p_dist(x)
    d = np.append(0, d)      # pad the all-zero and all-one patterns with 0 mass
    d = np.append(d, 0)
    out = []
    for _ in range(num_samples):
        idx = _sample_from_dist(d)[0]
        out.append(_bin_list(np.binary_repr(idx, width=modes)))
    return out


def A_from_samples(samples):
    """Baseline adjacency-matrix construction (original GBS.py)."""
    modes = len(samples[0])
    n_mean = modes / 2
    A = np.zeros([modes, modes])
    for s in samples:
        A = _add_edges(A, _edge_combinations(ones_inds_in_bin(s)))
    A = A + A.T
    for i in range(modes):
        A[i, i] /= 2
    return train.rescale_adjacency(A, n_mean, True)


def train_WAW(A, data_samples, steps=1000, rate=0.05):
    """Baseline WAW training (Banchi et al.), sample-based KL gradient."""
    modes = len(A)
    weights = train.Exp(modes)
    n_mean = modes / 2
    vgbs = train.VGBS(A, n_mean, weights, threshold=True)
    cost = train.KL(data_samples, vgbs)
    params = np.random.rand(modes)
    norms = []
    for i in range(steps):
        params -= rate * cost.grad(params)
        if i % 10 == 0:
            norms.append(norm(cost.grad(params)))
    return vgbs.A(params), norms, params


def gbs_sample(A, num_samples):
    """Threshold sampling from a graph adjacency matrix."""
    modes = len(A)
    return tw.samples.torontonian_sample_graph(A, n_mean=modes / 2,
                                               samples=num_samples, parallel=False)


def probability(samples):
    """Empirical distribution over all 2^modes patterns."""
    modes = len(samples[0])
    p = np.zeros([2 ** modes])
    for s in samples:
        n, t = 0, 0
        for b in s[-1::-1]:
            n += b * 2 ** t
            t += 1
        p[n] += 1
    return p / len(samples)


def prob_rn(prob):
    """Drop the all-0 and all-1 patterns and renormalise (with smoothing)."""
    prob = np.array(prob, dtype=float)
    prob[prob == 0] += 1e-4
    return prob[1:-1] / np.sum(prob[1:-1])


def dkl(P, Q):
    """KL(P || Q)."""
    return np.sum(P * np.log(P / Q))


# ============================================================================
# Analytic (Monte-Carlo-free) evaluation
# ============================================================================
def _safe_takagi_cap(A, max_lam=0.998):
    """Cap A's largest Takagi value below 1 so the state stays physical.
    Falls back through eigvalsh / SVD if the LAPACK routine fails."""
    m = None
    for attempt in (lambda: np.max(np.abs(takagi(A)[0])),
                    lambda: np.max(np.abs(np.linalg.eigvalsh(A))),
                    lambda: np.max(np.linalg.svd(A, compute_uv=False))):
        try:
            m = float(attempt())
            break
        except Exception:
            continue
    if m is None:
        m = float(np.max(np.abs(A))) * A.shape[0]
    return A * (max_lam / m) if m >= 0.999 else A


def model_distribution_from_A(A, modes, n_mean=None, hbar=2):
    """Exact threshold-detection distribution over all 2^modes patterns
    (torontonian via threshold_detection_prob). n_mean defaults to modes/2."""
    if n_mean is None:
        n_mean = modes / 2
    A_safe = _safe_takagi_cap(A)
    Q = adj_to_qmat(A_safe, n_mean)
    cov = np.real(Covmat(Q, hbar=hbar))
    mu = np.zeros(2 * modes)
    probs = np.zeros(2 ** modes)
    for idx in range(2 ** modes):
        pattern = np.array([int(b) for b in np.binary_repr(idx, width=modes)])
        probs[idx] = threshold_detection_prob(mu, cov, pattern, hbar=hbar)
    probs = np.maximum(probs, 1e-15)
    return probs / probs.sum()


def _safe_kl(p, q):
    """KL(p||q) with the 0*log(0)=0 convention (avoids NaN on hard zeros)."""
    p = np.asarray(p, dtype=float)
    q = np.maximum(np.asarray(q, dtype=float), 1e-15)
    mask = p > 1e-15
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def analytic_eval(A, modes, prob_target_rn):
    """Return (model prob over the 2^m-2 non-trivial bins, KL vs target).
    On numerical failure returns (target, +inf)."""
    try:
        p_full = model_distribution_from_A(A, modes)
    except Exception:
        return np.asarray(prob_target_rn).copy(), float("inf")
    p_rn = p_full[1:-1] / p_full[1:-1].sum()
    return p_rn, _safe_kl(prob_target_rn, p_rn)


def analytic_target(target_dist, x0, x1, modes, encoding="binary"):
    """Exact target distribution over the 2^m-2 non-trivial patterns, using
    the closed-form p(x) on bin centres. encoding in {'binary','gray'}."""
    from encoding.gray_encoding import to_gray  # local import to avoid cycle
    x = np.linspace(x0, x1, 2 ** modes - 2)
    p_x = target_dist(x)
    p_x = p_x / p_x.sum()
    p_full = np.zeros(2 ** modes)
    if encoding == "gray":
        for i in range(1, 2 ** modes - 1):
            p_full[to_gray(i)] += p_x[i - 1]
    else:
        for i in range(1, 2 ** modes - 1):
            p_full[i] = p_x[i - 1]
    return p_full[1:-1] / p_full[1:-1].sum()
