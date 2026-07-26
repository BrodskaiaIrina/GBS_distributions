"""
A_from_samples_variance.py

Photon-number-aware construction (iterative click-count matching). rescale_adjacency
fixes only the MEAN photon number; the variance of the total photon number
(and hence of the click count k) is left uncontrolled. We match the whole
click-count histogram h(k) = Pr(|s| = k) by iterative proportional fitting:

    build A (per-pair + parity, scaled per shell by rho_k),
    measure h_model(k) from a probe sample,
    rho_k <- rho_k * clip( (h_data(k) / h_model(k))^alpha, 0.1, 10 ),
    repeat.

alpha is a tunable variance prefactor: alpha=0 recovers per-pair (no variance
correction), alpha=1 is full photon-count matching, intermediate values pull
partially toward the data's click-count variance. Different targets prefer
different alpha.
"""

import numpy as np
from itertools import combinations
from math import comb
from gbs_core import ones_inds_in_bin, gbs_sample, train


def click_hist(samples, modes):
    """Normalised histogram of the click count k over samples."""
    h = np.zeros(modes + 1)
    for s in samples:
        h[int(np.sum(s))] += 1
    return h / max(len(samples), 1)


def A_from_samples_variance(samples, alpha=0.5, n_iter=3, num_probe=800, clip=(0.1, 10.0)):
    modes = len(samples[0])
    n_mean = modes / 2
    h_data = click_hist(samples, modes)
    rho = np.ones(modes + 1)
    A = np.zeros([modes, modes])

    for _ in range(n_iter):
        A = np.zeros([modes, modes])
        for s in samples:
            node_inds = ones_inds_in_bin(s)
            k = len(node_inds)
            if k == 0:
                continue
            w = (np.sqrt(2) if k % 2 == 1 else 1.0) * rho[k]
            for idx in node_inds:
                A[idx, idx] += w / k
            if k >= 2:
                npairs = comb(k, 2)
                for i, j in combinations(node_inds, 2):
                    A[i, j] += w / npairs
                    A[j, i] += w / npairs
        if np.max(A) > 0:
            A = A / np.max(A)
        A = train.rescale_adjacency(A, n_mean, True)

        h_model = click_hist(gbs_sample(A, num_probe), modes)
        for k in range(modes + 1):
            if h_model[k] > 1e-9:
                rho[k] *= np.clip((h_data[k] / h_model[k]) ** alpha, *clip)
    return A


# Backwards-compatible alias: alpha=1 is the original "photon-matched" variant.
def A_from_samples_photon_matched(samples, n_iter=3, num_probe=800):
    return A_from_samples_variance(samples, alpha=1.0, n_iter=n_iter, num_probe=num_probe)


if __name__ == "__main__":
    from gbs_core import mmodal, bin_samples_from_dist
    data = bin_samples_from_dist(mmodal, 0.01, 4, 2000, 5)
    for a in (0.0, 0.5, 1.0):
        A = A_from_samples_variance(data, alpha=a)
        print(f"alpha={a}: A max = {np.max(A):.3f}")
