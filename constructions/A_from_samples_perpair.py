"""
A_from_samples_perpair.py

Per-pair normalisation + parity rule. A k-click sample deposits +1 into k
self-loops and C(k,2) edges, so high-k samples flood the matrix and over-weight
high-click patterns (the spurious 01111 peak). The fix: divide each sample's
contribution by how many entries it touches, so every sample deposits ~one unit
of evidence into the diagonal and ~one into the off-diagonal regardless of k:

    A_ii += w_s / k                for i in supp(s)
    A_ij += w_s / C(k,2)           for {i,j} in supp(s)

with the parity weight w_s = sqrt(2) (odd k) or 1 (even k) on top. This was the
best-performing initial construction across targets and mode counts.
"""

import numpy as np
from itertools import combinations
from math import comb
from gbs_core import ones_inds_in_bin, train


def A_from_samples_perpair(samples, odd_boost=np.sqrt(2)):
    modes = len(samples[0])
    n_mean = modes / 2
    A = np.zeros([modes, modes])

    for s in samples:
        node_inds = ones_inds_in_bin(s)
        k = len(node_inds)
        if k == 0:
            continue
        w = odd_boost if (k % 2 == 1) else 1.0
        for idx in node_inds:
            A[idx, idx] += w / k
        if k >= 2:
            npairs = comb(k, 2)
            for i, j in combinations(node_inds, 2):
                A[i, j] += w / npairs
                A[j, i] += w / npairs

    if np.max(A) > 0:
        A = A / np.max(A)
    return train.rescale_adjacency(A, n_mean, True)


if __name__ == "__main__":
    from gbs_core import normal, bin_samples_from_dist
    data = bin_samples_from_dist(normal, 0.01, 4, 1000, 5)
    A = A_from_samples_perpair(data)
    print("per-pair A shape:", A.shape)
