"""
select_nmean.py

Choosing the mean photon number n_mean. Lowering n_mean suppresses the central
spike (see train_threshold_nmean.py), but n_mean cannot be zero (vacuum: no
clicks), and going too low starves a broad target of the high-click patterns it
needs. There is therefore a target-dependent optimum, and the honest way to pick
it is to sweep n_mean and select the value that minimises the KL to the
EMPIRICAL target (samples only); the choice generalises to the true target.

Empirically the optimum tracks the target's concentration:
  * broad targets (normal, log-normal)  -> n_mean ~ 2 (below the default m/2);
  * concentrated targets (cauchy, bimodal) -> n_mean ~ 0.5.
The spike grows monotonically with n_mean, so one should never use the default
n_mean = m/2 without checking.
"""

import numpy as np
from encoding.rank_assign import rank_assign_with_map, empirical_target_from_data
from optimization.train_threshold_nmean import train_threshold_nmean, thr_weights_nmean
from constructions.A_from_samples_parity import A_from_samples_parity


def _fit_at_nmean(A, ph, tp, modes, n_mean, steps):
    """Fit the threshold model at a given n_mean; return the bin-ordered model,
    its KL to the empirical target, and the spike/neighbour ratio."""
    w0 = thr_weights_nmean(A, tp, modes, n_mean)
    w0 = w0 / max(w0.sum(), 1e-15)
    pa, b2p = rank_assign_with_map(ph, w0)
    q = train_threshold_nmean(A, pa, tp, modes, n_mean, steps=steps)
    K = len(ph)
    q_x = np.array([q[b2p[i]] for i in range(K)])
    from gbs_core import _safe_kl
    kl = _safe_kl(ph, q_x)                       # KL in bin order == pattern order
    pk = int(np.argmax(q_x))
    spike = q_x[pk] / max((q_x[pk - 1] + q_x[pk + 1]) / 2, 1e-9)
    return q_x, kl, spike


def select_nmean(data, modes, nmeans=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0), steps=100):
    """Sweep n_mean and pick the value minimising KL to the empirical target.

    Returns (best_nmean, best_model_bin_ordered, history) where history is a list
    of (n_mean, KL_empirical, spike_ratio)."""
    ph = empirical_target_from_data(data, modes)
    A = A_from_samples_parity(data)
    tp = [tuple(int(b) for b in np.binary_repr(i, modes)) for i in range(1, 2 ** modes - 1)]
    history, best, best_kl, best_q = [], None, float("inf"), None
    for nm in nmeans:
        q_x, kl, spike = _fit_at_nmean(A, ph, tp, modes, nm, steps)
        history.append((nm, kl, spike))
        if kl < best_kl:
            best_kl, best, best_q = kl, nm, q_x
    return best, best_q, history


if __name__ == "__main__":
    from gbs_core import normal, cauchy, bin_samples_from_dist
    for name, dist in (("normal (broad)", normal), ("cauchy (sharp)", cauchy)):
        data = bin_samples_from_dist(dist, 0.01, 4, 6000, 6)
        best, _, hist = select_nmean(data, 6, steps=80)
        print(f"{name}: best n_mean = {best}")
        for nm, kl, sp in hist:
            print(f"    n_mean={nm:.1f}  KL_emp={kl:.4f}  spike={sp:.2f}")
