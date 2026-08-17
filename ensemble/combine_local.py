"""
combine_local.py

Location-dependent (per-bin) mixture of the threshold and PNR readouts. The
global mixture (combine_threshold_pnr.py) uses one weight w for all bins; on a
target with both broad and sharp regions that single w is a compromise. Here the
weight varies with the LOCAL shape of the target:

    q_mix(x) = w(x) q_thr(x) + (1 - w(x)) q_pnr(x),   renormalised,
    w(x) = sigmoid(a + b * r(x)),

where r(x) = log( p_hat(x) / mean(neighbours) ) is the local peakiness of the
EMPIRICAL target (>0 at peaks, <0 in flat/valley regions). Only two parameters
(a, b) are fitted (by minimising the empirical KL), so the global mixture is the
special case b=0 and the local rule cannot overfit. The learned gate typically
leans PNR (concentrated) at peaks and threshold (spread) on broad regions.

Note: unlike the global convex mixture (realizable as a per-shot choice of
detector), the per-bin mixture is a post-hoc DENSITY-estimation combination --
appropriate for learning the distribution, but it is not a simple two-device
mixture.
"""

import numpy as np
from gbs_core import _safe_kl
from encoding.rank_assign import empirical_target_from_data
from ensemble.combine_threshold_pnr import _threshold_fit, _pnr_fit


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def local_peakiness(ph):
    """Log-ratio of each bin to its neighbour average (>0 at peaks)."""
    K = len(ph)
    r = np.zeros(K)
    for i in range(K):
        nb = 0.5 * (ph[max(i - 1, 0)] + ph[min(i + 1, K - 1)]) + 1e-9
        r[i] = np.log((ph[i] + 1e-9) / nb)
    return r


def _fit_gate(ph, q_thr, q_pnr, r):
    """Grid-then-refine fit of (a, b) minimising KL(ph || mix)."""
    def kl_of(a, b):
        w = _sigmoid(a + b * r)
        q = w * q_thr + (1 - w) * q_pnr
        s = q.sum()
        return _safe_kl(ph, q / s) if s > 0 else float("inf")

    best, best_kl = (0.0, 0.0), float("inf")
    grid_a = np.linspace(-5, 5, 21)
    grid_b = np.linspace(-5, 5, 21)
    for a in grid_a:
        for b in grid_b:
            k = kl_of(a, b)
            if k < best_kl:
                best_kl, best = k, (a, b)
    # local refinement
    a0, b0 = best
    for a in np.linspace(a0 - 0.5, a0 + 0.5, 11):
        for b in np.linspace(b0 - 0.5, b0 + 0.5, 11):
            k = kl_of(a, b)
            if k < best_kl:
                best_kl, best = k, (a, b)
    return best


def combine_local(data, modes, n_mean_thr=2.0, steps=100):
    """Fit the local (per-bin) threshold+PNR mixture on the empirical target.
    Returns (q_local, (a, b), w_per_bin, q_thr, q_pnr)."""
    ph = empirical_target_from_data(data, modes)
    q_thr = _threshold_fit(data, ph, modes, n_mean_thr, steps)
    q_pnr = _pnr_fit(data, ph, modes, steps)
    r = local_peakiness(ph)
    a, b = _fit_gate(ph, q_thr, q_pnr, r)
    w = _sigmoid(a + b * r)
    q = w * q_thr + (1 - w) * q_pnr
    q = q / q.sum()
    return q, (a, b), w, q_thr, q_pnr


if __name__ == "__main__":
    from gbs_core import sharp_plus_broad, bin_samples_from_dist
    from ensemble.combine_threshold_pnr import combine_threshold_pnr
    modes = 6
    xs = np.linspace(0.01, 4, 2 ** modes - 2)
    pt = sharp_plus_broad(xs); pt = pt / pt.sum()
    data = bin_samples_from_dist(sharp_plus_broad, 0.01, 4, 6000, modes)
    q_g, w_g, q_thr, q_pnr = combine_threshold_pnr(data, modes, steps=90)
    q_l, (a, b), w, _, _ = combine_local(data, modes, steps=90)
    print(f"sharp_plus_broad m={modes} (KL vs true):")
    print(f"  threshold      : {_safe_kl(pt, q_thr):.4f}")
    print(f"  PNR            : {_safe_kl(pt, q_pnr):.4f}")
    print(f"  global mixture : {_safe_kl(pt, q_g):.4f}  (w={w_g:.2f})")
    print(f"  local mixture  : {_safe_kl(pt, q_l):.4f}  (a={a:.2f}, b={b:.2f})")
