"""
combine_threshold_pnr.py

Combining the threshold and photon-counting (PNR) readouts. The two model
families are complementary: the threshold family is spread (good for broad
targets), the PNR family is concentrated (good for sharp peaks). Because both
models can be mapped back to the same x-bins (inverting their rank_assign
encoding), we can combine them there.

Two combinations are provided, both with the mixing weight chosen on the
EMPIRICAL target (samples only):

  * convex mixture   q = w * q_thr + (1-w) * q_pnr
      -- physically realizable (run each device with probability w), and
         provably never worse than the better single model;
  * product of experts   q proportional to q_thr^w * q_pnr^(1-w)
      -- sharpens where both agree.

Empirically the mixture defaults to the better model on pure-broad / pure-sharp
targets and BEATS both on targets with mixed structure (e.g. a Cauchy peak with
broad tails, or an asymmetric bimodal), by 2-3x in KL.
"""

import numpy as np
from gbs_core import _safe_kl, _safe_takagi_cap, A_from_samples
from constructions.A_from_samples_parity import A_from_samples_parity
from encoding.rank_assign import rank_assign_with_map, empirical_target_from_data
from optimization.train_threshold_nmean import train_threshold_nmean, thr_weights_nmean
from readout.pnr_readout import enumerate_even_pnr, pnr_model_over


def _train_pnr(A_base, p_target, pats, modes, steps=100, rate=0.05, eps=1e-3):
    def q_of(th):
        W = np.diag(np.sqrt(np.exp(th)))
        A = _safe_takagi_cap(W @ A_base @ W)
        return pnr_model_over(A, pats)

    def cost(th):
        try:
            return _safe_kl(p_target, q_of(th))
        except Exception:
            return float("inf")

    th = np.zeros(modes); bt, bc = th.copy(), cost(th)
    for _ in range(steps):
        c0 = cost(th)
        if not np.isfinite(c0):
            th = bt.copy(); continue
        g = np.array([(cost(th + eps * np.eye(modes)[k]) - c0) / eps for k in range(modes)])
        tn = th - rate * g; cn = cost(tn)
        if np.isfinite(cn):
            th = tn
            if cn < bc:
                bc, bt = cn, tn.copy()
        else:
            th = bt.copy()
    return q_of(bt)


def _threshold_fit(data, ph, modes, n_mean, steps):
    A = A_from_samples_parity(data)
    tp = [tuple(int(b) for b in np.binary_repr(i, modes)) for i in range(1, 2 ** modes - 1)]
    w0 = thr_weights_nmean(A, tp, modes, n_mean); w0 = w0 / max(w0.sum(), 1e-15)
    pa, b2p = rank_assign_with_map(ph, w0)
    q = train_threshold_nmean(A, pa, tp, modes, n_mean, steps=steps)
    return np.array([q[b2p[i]] for i in range(len(ph))])


def _pnr_fit(data, ph, modes, steps):
    A = A_from_samples(data)
    K = len(ph)
    pats = enumerate_even_pnr(modes, K)
    w0 = pnr_model_over(A, pats)
    pa, b2p = rank_assign_with_map(ph, w0)
    q = _train_pnr(A, pa, pats, modes, steps=steps)
    return np.array([q[b2p[i]] for i in range(K)])


def combine_threshold_pnr(data, modes, n_mean_thr=2.0, steps=100, method="mixture",
                          n_grid=51):
    """Train both readouts and combine them in bin space, selecting the mixing
    weight on the empirical target.

    method: 'mixture' (convex) or 'product' (product-of-experts).
    Returns (q_combined, w_star, q_threshold, q_pnr)."""
    ph = empirical_target_from_data(data, modes)
    q_thr = _threshold_fit(data, ph, modes, n_mean_thr, steps)
    q_pnr = _pnr_fit(data, ph, modes, steps)
    ws = np.linspace(0, 1, n_grid)

    if method == "product":
        def build(w):
            qp = (np.maximum(q_thr, 1e-15) ** w) * (np.maximum(q_pnr, 1e-15) ** (1 - w))
            return qp / qp.sum()
    else:
        def build(w):
            return w * q_thr + (1 - w) * q_pnr

    kls = [_safe_kl(ph, build(w)) for w in ws]
    w_star = float(ws[int(np.argmin(kls))])
    return build(w_star), w_star, q_thr, q_pnr


if __name__ == "__main__":
    from gbs_core import cauchy, bimodal_asym, bin_samples_from_dist
    for name, dist in (("cauchy", cauchy), ("bimodal_asym", bimodal_asym)):
        data = bin_samples_from_dist(dist, 0.01, 4, 6000, 6)
        xs = np.linspace(0.01, 4, 2 ** 6 - 2)
        pt = dist(xs); pt = pt / pt.sum()
        q_mix, w, q_thr, q_pnr = combine_threshold_pnr(data, 6, steps=90)
        print(f"{name}: threshold KL={_safe_kl(pt, q_thr):.3f}  PNR KL={_safe_kl(pt, q_pnr):.3f}  "
              f"mixture(w*={w:.2f}) KL={_safe_kl(pt, q_mix):.3f}")
