"""
A_from_samples_corrected.py

Corrected adjacency-matrix construction (the first fix, from
Corrected_matrix_construction.ipynb):

  1. Structural fix: count a self-loop for EVERY clicking mode, not only for
     single-click events (the original A_from_samples only adds loops for
     one-click samples).
  2. Factorial fix: multiply the diagonal by sqrt(2) to compensate for the
     bunching suppression P(2 photons in mode i) ~ |A_ii|^2 / 2!.

This is the precursor to the parity rule (see A_from_samples_parity.py), which
generalises the sqrt(2) boost to every odd-click pattern.
"""

import numpy as np
from gbs_core import ones_inds_in_bin, train


def A_from_samples_corrected(samples, diagonal_boost=np.sqrt(2)):
    modes = len(samples[0])
    n_mean = modes / 2
    A = np.zeros([modes, modes])

    for s in samples:
        node_inds = ones_inds_in_bin(s)
        for idx in node_inds:                      # self-loop for every click
            A[idx, idx] += 1
        for i in range(len(node_inds)):            # all clicking pairs
            for j in range(i + 1, len(node_inds)):
                A[node_inds[i], node_inds[j]] += 1

    A = A + A.T
    for i in range(modes):
        A[i, i] /= 2                               # undo diagonal double-count

    if np.max(A) > 0:
        A = A / np.max(A)
    for i in range(modes):
        A[i, i] *= diagonal_boost                  # factorial correction

    return train.rescale_adjacency(A, n_mean, True)


if __name__ == "__main__":
    from gbs_core import normal, bin_samples_from_dist
    data = bin_samples_from_dist(normal, 0.01, 4, 1000, 4)
    A = A_from_samples_corrected(data)
    print("corrected A diagonal:", np.round(np.diag(A), 3))
