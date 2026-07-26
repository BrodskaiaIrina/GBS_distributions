"""
train_displaced_gbs.py

Displaced GBS (full displacement vector). Standard GBS uses squeezed vacuum,
whose photon number is always even, so threshold-click parity is biased toward
even patterns. Replacing squeezed vacuum with DISPLACED squeezed vacuum
(mean vector mu != 0) populates both parities and removes the constraint.

This trains the WAW weights (theta) and the full displacement vector
mu in R^{2m} jointly by numerical-gradient descent on the exact KL (feasible
for small m, where all 2^m patterns can be enumerated with the loop
torontonian). Returns the trained matrix, mu, and a training history.
"""

import numpy as np
from gbs_core import (adj_to_qmat, Covmat, threshold_detection_prob, takagi,
                      probability, norm)


def adj_to_cov(A_mat, n_mean, hbar=2):
    Q = adj_to_qmat(A_mat, n_mean)
    return np.real(Covmat(Q, hbar=hbar))


def click_probs_all_patterns(cov, mu, modes, hbar=2):
    probs = np.zeros(2 ** modes)
    for idx in range(2 ** modes):
        pattern = np.array([int(b) for b in np.binary_repr(idx, width=modes)])
        probs[idx] = threshold_detection_prob(mu, cov, pattern, hbar=hbar)
    return probs


def model_distribution(A_mat, mu, n_mean, modes, hbar=2):
    cov = adj_to_cov(A_mat, n_mean, hbar=hbar)
    probs = np.maximum(click_probs_all_patterns(cov, mu, modes, hbar=hbar), 1e-15)
    return probs / probs.sum()


def kl_divergence(p_data, p_model):
    p_data = np.maximum(p_data, 1e-15)
    p_model = np.maximum(p_model, 1e-15)
    return np.sum(p_data * np.log(p_data / p_model))


def train_displaced_gbs(A_base, data_samples, n_mean, steps=300,
                        lr_w=0.05, lr_mu=0.02, eps=1e-4):
    modes = len(A_base)
    p_data = probability(data_samples)
    p_data = np.maximum(p_data, 1e-6)
    p_data = p_data / p_data.sum()

    theta = np.zeros(modes)
    mu = np.zeros(2 * modes)
    history = {"kl": [], "parity_gap": [], "mu_norm": []}

    def A_of(th):
        W = np.diag(np.sqrt(np.exp(th)))
        A = W @ A_base @ W
        lam, _ = takagi(A)
        if np.max(lam) >= 0.999:
            A = A * (0.998 / np.max(lam))
        return A

    for step in range(steps):
        A_w = A_of(theta)
        p_model = model_distribution(A_w, mu, n_mean, modes)
        kl = kl_divergence(p_data, p_model)

        even_model = sum(p_model[i] for i in range(2 ** modes) if bin(i).count("1") % 2 == 0)
        even_data = sum(p_data[i] for i in range(2 ** modes) if bin(i).count("1") % 2 == 0)
        history["kl"].append(kl)
        history["parity_gap"].append(abs(even_model - even_data))
        history["mu_norm"].append(norm(mu))
        if step % 50 == 0:
            print(f"Step {step:3d}: KL={kl:.4f}  parity_gap={abs(even_model-even_data):.4f}  |mu|={norm(mu):.4f}")

        grad_theta = np.zeros(modes)
        for k in range(modes):
            tp = theta.copy(); tp[k] += eps
            grad_theta[k] = (kl_divergence(p_data, model_distribution(A_of(tp), mu, n_mean, modes)) - kl) / eps
        grad_mu = np.zeros(2 * modes)
        for k in range(2 * modes):
            mp = mu.copy(); mp[k] += eps
            grad_mu[k] = (kl_divergence(p_data, model_distribution(A_w, mp, n_mean, modes)) - kl) / eps

        theta -= lr_w * grad_theta
        mu -= lr_mu * grad_mu

    return A_of(theta), mu, history


if __name__ == "__main__":
    from gbs_core import normal, bin_samples_from_dist, A_from_samples
    data = bin_samples_from_dist(normal, 0.01, 4, 1000, 4)
    A = A_from_samples(data)
    A_f, mu, hist = train_displaced_gbs(A, data, 2.0, steps=100)
    print(f"final KL={hist['kl'][-1]:.4f}  |mu|={norm(mu):.4f}")
