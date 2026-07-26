"""
A_from_samples_perpattern.py

Per-pattern factorial weights. Instead of the leading-order parity rule, we sum
over ALL photon-number configurations for a k-click pattern under a uniform
reference matrix A0 = alpha * J, using

    Haf(alpha * J_{N x N}) = (N-1)!! * alpha^{N/2}   (N even, else 0),
    f_k(N) = [x^N] (e^x - 1)^k
           = (1/N!) * sum_{j=0}^{k} (-1)^j C(k,j) (k-j)^N.

The pattern weight w_k = R(2) / R(k) inverts the k-dependent factorial
structure (normalised so k=2 has weight 1). Under the uniform assumption the
weight depends only on k, so only m values are needed.

Note: empirically this OVER-suppresses high-k samples and was superseded by the
per-pair normalisation (A_from_samples_perpair.py). It is kept for completeness.
"""

import numpy as np
from itertools import combinations
from math import factorial, comb
from gbs_core import ones_inds_in_bin, train


def _f_kN(k, N):
    return sum((-1) ** j * comb(k, j) * (k - j) ** N for j in range(k + 1)) / factorial(N)


def _double_fact(n):
    if n <= 0:
        return 1
    r = 1
    for i in range(n, 0, -2):
        r *= i
    return r


def pattern_weight_uniform(k, alpha=0.4, N_max=None):
    """Factorial-correction weight for a k-click pattern under A0 = alpha*J,
    normalised so that k=2 has weight 1."""
    if k == 0:
        return 0.0
    if N_max is None:
        N_max = k + 8

    def R(kk):
        k_min = kk if kk % 2 == 0 else kk + 1
        total = sum(_double_fact(N - 1) ** 2 * alpha ** N * _f_kN(kk, N)
                    for N in range(k_min, N_max + 1, 2))
        return total / alpha ** k_min

    return R(2) / R(k)


def A_from_samples_perpattern(samples, alpha_ref=0.4):
    modes = len(samples[0])
    n_mean = modes / 2
    A = np.zeros([modes, modes])
    w_of_k = {k: pattern_weight_uniform(k, alpha=alpha_ref) for k in range(1, modes + 1)}

    for s in samples:
        node_inds = ones_inds_in_bin(s)
        k = len(node_inds)
        if k == 0:
            continue
        w = np.sqrt(w_of_k[k])                      # |A_ij|^2 ~ count => sqrt
        for idx in node_inds:
            A[idx, idx] += w
        for i, j in combinations(node_inds, 2):
            A[i, j] += w
            A[j, i] += w

    if np.max(A) > 0:
        A = A / np.max(A)
    return train.rescale_adjacency(A, n_mean, True)


if __name__ == "__main__":
    for k in range(1, 6):
        print(f"k={k}: uniform weight = {pattern_weight_uniform(k):.4f}")
