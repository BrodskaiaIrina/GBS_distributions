"""
train_waw_then_displace.py

Scalable scalar displacement. Training the full 2m displacement vector is only
feasible for small m. A cheaper, scalable alternative decouples the two jobs:

  Step 1: standard WAW to fit the pairwise correlations (sample-based gradient).
  Step 2: a single scalar displacement alpha, applied uniformly to all position
          quadratures (mu = [alpha, ..., alpha, 0, ..., 0]), tuned by BINARY
          SEARCH so the odd-parity fraction of the samples matches the data.

Increasing alpha monotonically raises the odd-parity fraction, so one scalar
knob suffices to fix the parity balance that squeezed-vacuum GBS cannot.
"""

import numpy as np
from gbs_core import train_WAW
from gbs_core import adj_to_qmat, Covmat
from thewalrus.samples import torontonian_sample_state


def adj_to_cov(A_mat, n_mean, hbar=2):
    Q = adj_to_qmat(A_mat, n_mean)
    return np.real(Covmat(Q, hbar=hbar))


def parity_fraction_from_samples(samples):
    return sum(1 for s in samples if sum(s) % 2 == 1) / len(samples)


def sample_displaced_gbs_scalar(A_trained, alpha, n_mean, num_samples, hbar=2):
    """Sample with uniform scalar displacement on the position quadratures."""
    modes = len(A_trained)
    cov = adj_to_cov(A_trained, n_mean, hbar=hbar)
    mu = np.zeros(2 * modes)
    mu[:modes] = alpha
    return np.array(torontonian_sample_state(cov, samples=num_samples, mu=mu, hbar=hbar))


def find_displacement_for_parity(A_trained, target_odd_fraction, n_mean,
                                 alpha_min=0.0, alpha_max=3.0,
                                 num_samples_per_step=300, tol=0.02,
                                 max_iter=20, verbose=True):
    """Binary search for the scalar alpha that matches the target odd fraction."""
    history = []
    f0 = parity_fraction_from_samples(
        sample_displaced_gbs_scalar(A_trained, 0.0, n_mean, num_samples_per_step))
    history.append((0.0, f0))
    if verbose:
        print(f"  alpha=0.00: odd_frac={f0:.3f} (target={target_odd_fraction:.3f})")
    if abs(f0 - target_odd_fraction) < tol:
        return 0.0, history

    lo, hi, alpha_best = alpha_min, alpha_max, 0.0
    for it in range(max_iter):
        alpha = (lo + hi) / 2
        odd = parity_fraction_from_samples(
            sample_displaced_gbs_scalar(A_trained, alpha, n_mean, num_samples_per_step))
        history.append((alpha, odd))
        if verbose:
            print(f"  iter {it+1:2d}: alpha={alpha:.4f}, odd_frac={odd:.3f}")
        if abs(odd - target_odd_fraction) < tol:
            return alpha, history
        if odd < target_odd_fraction:
            lo = alpha
        else:
            hi = alpha
        alpha_best = alpha
    return alpha_best, history


def train_waw_then_displace(A_base, data_samples, n_mean, waw_steps=1000,
                            waw_rate=0.05, search_samples=300, parity_tol=0.02,
                            verbose=True):
    """Full pipeline: WAW training, then scalar-displacement parity search."""
    A_trained, waw_norms, _ = train_WAW(A_base, data_samples, steps=waw_steps, rate=waw_rate)
    target_odd = parity_fraction_from_samples(data_samples)
    if verbose:
        print(f"Displacement search (target odd fraction = {target_odd:.3f})...")
    alpha, search_history = find_displacement_for_parity(
        A_trained, target_odd, n_mean, num_samples_per_step=search_samples,
        tol=parity_tol, verbose=verbose)
    return A_trained, alpha, waw_norms, search_history


if __name__ == "__main__":
    from gbs_core import normal, bin_samples_from_dist, A_from_samples
    data = bin_samples_from_dist(normal, 0.01, 4, 1000, 4)
    A = A_from_samples(data)
    A_t, alpha, _, _ = train_waw_then_displace(A, data, 2.0, waw_steps=200)
    print(f"optimal scalar displacement alpha = {alpha:.4f}")
