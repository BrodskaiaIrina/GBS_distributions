"""
train_WAW_bestkl.py

Keep-best WAW. The baseline train_WAW uses random weight initialisation, runs a
fixed number of steps, and returns the LAST iterate with no KL monitoring. On
Gray-encoded data this walked away from a good solution into a single-mode
collapse. train_WAW_bestkl fixes all three: zero initialisation, per-step KL
evaluation, and return of the lowest-KL iterate seen.
"""

import numpy as np
from gbs_core import train


def train_WAW_bestkl(A, data_samples, steps=400, rate=0.03, init="zeros"):
    modes = len(A)
    weights = train.Exp(modes)
    n_mean = modes / 2
    vgbs = train.VGBS(A, n_mean, weights, threshold=True)
    cost = train.KL(data_samples, vgbs)

    params = np.zeros(modes) if init == "zeros" else np.random.rand(modes)
    best_params = params.copy()
    best_cost = cost(params)
    kl_hist = [best_cost]

    for _ in range(steps):
        params = params - rate * cost.grad(params)
        c = cost(params)
        kl_hist.append(c)
        if c < best_cost:
            best_cost = c
            best_params = params.copy()

    return vgbs.A(best_params), kl_hist, best_params


if __name__ == "__main__":
    from gbs_core import normal, bin_samples_from_dist
    from constructions.A_from_samples_parity import A_from_samples_parity
    data = bin_samples_from_dist(normal, 0.01, 4, 1000, 4)
    A = A_from_samples_parity(data)
    Aw, hist, _ = train_WAW_bestkl(A, data, steps=100)
    print(f"best analytic KL cost: {min(hist):.4f}")
