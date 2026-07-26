"""
A_from_samples_parity.py

Parity-rule construction. The hafnian vanishes on odd-dimensional matrices, so
every click pattern with an ODD number of clicks k must double at least one
mode to reach an even photon count, incurring a 1/2! factorial suppression
that even-k patterns do not. We therefore weight each sample by

    w_s = sqrt(2)  if k = |s| is odd,
          1        if k is even,

applied uniformly to every matrix element the sample touches (self-loops AND
edges). For k = 1 this reduces to the sqrt(2) diagonal boost of
A_from_samples_corrected, so the parity rule strictly generalises it.
"""

import numpy as np
from itertools import combinations
from gbs_core import ones_inds_in_bin, train


def A_from_samples_parity(samples, odd_boost=np.sqrt(2)):
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
            A[idx, idx] += w
        for i, j in combinations(node_inds, 2):
            A[i, j] += w
            A[j, i] += w

    if np.max(A) > 0:
        A = A / np.max(A)
    return train.rescale_adjacency(A, n_mean, True)


if __name__ == "__main__":
    from gbs_core import normal, bin_samples_from_dist
    data = bin_samples_from_dist(normal, 0.01, 4, 1000, 5)
    A = A_from_samples_parity(data)
    print("parity A shape:", A.shape)
