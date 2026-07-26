"""
train_threshold_nmean.py

Central-spike suppression via the mean photon number. When the target bins are
mapped by rank_assign, the model's single most-probable pattern (q0, with
q0/q1 ~ 2.3) lands on the peak bin and produces a spike ~2-3x its neighbours.
This is structural: reverse KL cannot flatten it (the m-weight WAW family simply
cannot make its top pattern smaller).

The effective knob is the MEAN PHOTON NUMBER n_mean. The default n_mean = m/2
grows with modes and over-squeezes, pushing the threshold distribution toward
one dominant click pattern. Lowering n_mean (~2) flattens the click
distribution, removes the spike, and — for broad targets — even improves KL.
Treat n_mean as a hyperparameter rather than hard-wiring it to m/2.
"""

import numpy as np
from gbs_core import _safe_takagi_cap, model_distribution_from_A, _safe_kl


def thr_weights_nmean(A, patterns, modes, n_mean):
    """Threshold probabilities of the given patterns at a chosen n_mean."""
    pf = model_distribution_from_A(A, modes, n_mean=n_mean)
    return np.array([pf[int("".join(map(str, s)), 2)] for s in patterns])


def train_threshold_nmean(A_base, p_target, patterns, modes, n_mean,
                          steps=100, rate=0.05, eps=1e-3):
    """WAW training under the threshold readout with a tunable n_mean."""
    def q_of(theta):
        W = np.diag(np.sqrt(np.exp(theta)))
        A = _safe_takagi_cap(W @ A_base @ W)
        w = thr_weights_nmean(A, patterns, modes, n_mean)
        s = w.sum()
        return w / s if s > 0 else w

    def cost(theta):
        try:
            return _safe_kl(p_target, q_of(theta))
        except Exception:
            return float("inf")

    theta = np.zeros(modes)
    best_t, best_c = theta.copy(), cost(theta)
    for _ in range(steps):
        c0 = cost(theta)
        if not np.isfinite(c0):
            theta = best_t.copy()
            continue
        g = np.zeros(modes)
        for k in range(modes):
            tp = theta.copy(); tp[k] += eps
            cp = cost(tp)
            g[k] = 0.0 if not np.isfinite(cp) else (cp - c0) / eps
        tn = theta - rate * g
        cn = cost(tn)
        if np.isfinite(cn):
            theta = tn
            if cn < best_c:
                best_c, best_t = cn, tn.copy()
        else:
            theta = best_t.copy()
    return q_of(best_t)


if __name__ == "__main__":
    from gbs_core import normal, bin_samples_from_dist
    from constructions.A_from_samples_parity import A_from_samples_parity
    from encoding.rank_assign import rank_assign_with_map, empirical_target_from_data
    m = 6; K = 2 ** m - 2
    data = bin_samples_from_dist(normal, 0.01, 4, 6000, m)
    ph = empirical_target_from_data(data, m); A = A_from_samples_parity(data)
    tp = [tuple(int(b) for b in np.binary_repr(i, m)) for i in range(1, 2 ** m - 1)]
    for nm in (m / 2, 2.0):
        w0 = thr_weights_nmean(A, tp, m, nm); w0 = w0 / max(w0.sum(), 1e-15)
        pa, b2p = rank_assign_with_map(ph, w0)
        q = train_threshold_nmean(A, pa, tp, m, nm, steps=60)
        thr_x = np.array([q[b2p[i]] for i in range(K)]); pk = int(np.argmax(thr_x))
        print(f"n_mean={nm:.1f}: spike/neighbor = "
              f"{thr_x[pk] / ((thr_x[pk-1] + thr_x[pk+1]) / 2):.2f}")
