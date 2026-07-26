"""
pnr_readout.py

Photon-number-resolving (counting) readout. Threshold detection folds every
collision configuration into one click pattern, creating the factorial
suppression that all the construction heuristics compensate for. A counting
detector reads the photon pattern n directly, with

    P(n) proportional to  |Haf(A[n])|^2 / prod_i n_i! ,

where A[n] repeats row/column i n_i times. For a collision-free 0/1 pattern
this is simply |Haf(A_S)|^2 over the support submatrix — the clean signal the
heuristics were estimating.

Findings from the experiments: at fixed modes PNR adds no matrix parameters and
its usable outcomes are dominated by a steep photon-number magnitude hierarchy,
so it does NOT beat threshold in general; but its model family is intrinsically
more CONCENTRATED, so it can win on sharply-peaked / tightly-multimodal targets
while threshold wins on broad ones.
"""

import numpy as np
from math import factorial
from thewalrus import hafnian
from thewalrus.quantum import probabilities as tw_probabilities
from gbs_core import (_safe_takagi_cap, adj_to_qmat, Covmat,
                      model_distribution_from_A, _safe_kl)


# --- Full PNR distribution (all c^m patterns up to cutoff) ------------------
def model_pnr_distribution(A, modes, cutoff=4, n_mean=None, hbar=2):
    """Exact photon-number distribution P(n_1,...,n_m), shape [cutoff]*modes."""
    if n_mean is None:
        n_mean = modes / 2
    A_safe = _safe_takagi_cap(A)
    Q = adj_to_qmat(A_safe, n_mean)
    cov = np.real(Covmat(Q, hbar=hbar))
    mu = np.zeros(2 * modes)
    return np.real(tw_probabilities(mu, cov, cutoff, hbar=hbar))


def pnr_collision_mass(P):
    """(collision mass with some n_i>=2, parity-forbidden odd-total mass)."""
    total = P.sum()
    coll = sum(P[idx] for idx in np.ndindex(*P.shape) if any(n >= 2 for n in idx))
    odd = sum(P[idx] for idx in np.ndindex(*P.shape) if sum(idx) % 2 == 1)
    return coll / total, odd / total


# --- Per-pattern PNR weights (hafnian based, no cutoff) ---------------------
def pnr_pattern_weight(A, n):
    """Exact unnormalised PNR weight |Haf(A[n])|^2 / prod(n_i!)."""
    tot = sum(n)
    if tot == 0 or tot % 2 == 1:
        return 0.0
    idx = []
    for i, ni in enumerate(n):
        idx += [i] * ni
    M = A[np.ix_(idx, idx)]
    denom = 1
    for ni in n:
        denom *= factorial(ni)
    return abs(hafnian(M)) ** 2 / denom


def pnr_01_weights(A, patterns):
    """Unnormalised PNR weights for 0/1 patterns: |Haf(A_S)|^2."""
    A_safe = _safe_takagi_cap(A)
    w = np.zeros(len(patterns))
    for i, s in enumerate(patterns):
        supp = [j for j, b in enumerate(s) if b]
        if len(supp) % 2 == 1:
            continue
        w[i] = abs(hafnian(A_safe[np.ix_(supp, supp)])) ** 2
    return w


def pnr_model_over(A, patterns):
    """Normalised PNR probabilities over a list of photon patterns."""
    A_safe = _safe_takagi_cap(A)
    w = np.array([pnr_pattern_weight(A_safe, n) for n in patterns])
    s = w.sum()
    return w / s if s > 0 else w


def enumerate_even_pnr(modes, K, max_total=14):
    """First K non-vacuum even-total photon patterns, by (total, lexicographic)."""
    pats = []

    def comps(m, t):
        if m == 1:
            yield (t,)
            return
        for first in range(t + 1):
            for rest in comps(m - 1, t - first):
                yield (first,) + rest

    tot = 2
    while len(pats) < K and tot <= max_total:
        pats.extend(comps(modes, tot))
        tot += 2
    return pats[:K]


def thr_weights(A, patterns, modes):
    """Unnormalised threshold weights for a list of click patterns."""
    p_full = model_distribution_from_A(A, modes)
    return np.array([p_full[int("".join(map(str, s)), 2)] for s in patterns])


if __name__ == "__main__":
    from gbs_core import normal, bin_samples_from_dist, A_from_samples
    data = bin_samples_from_dist(normal, 0.01, 4, 1500, 4)
    A = A_from_samples(data)
    P = model_pnr_distribution(A, 4, cutoff=5)
    coll, odd = pnr_collision_mass(P)
    print(f"collision mass = {coll:.2%}, parity-forbidden (odd) mass = {odd:.3%}")
