"""
train_WAW_mixed.py

Reverse-KL outlier suppression. Forward KL(p||q) is zero-avoiding: where the
target p is ~0 (a valley), the term p*log(p/q) vanishes regardless of q, so the
model is never punished for stray mass there (e.g. the 10000 bump in the
multimodal valley). Adding a reverse-KL term KL(q||p), which diverges when q>0
at p~0, actively pushes such outliers down:

    cost = (1 - beta) * KL(p || q) + beta * KL(q || p).

beta = 0 is pure forward KL; larger beta trades a little mode-coverage for
outlier suppression. Note: outliers that are structural (the Gaussian state is
forced to emit them) barely move even at large beta — that residual is the
expressivity floor, not an objective artefact.
"""

import numpy as np
from gbs_core import _safe_takagi_cap, model_distribution_from_A, _safe_kl


def train_WAW_mixed(A_base, p_target, modes, beta=0.3, max_iter=120, rate=0.05,
                    eps=1e-3, p_floor=1e-3):
    p = np.asarray(p_target, dtype=float)
    p_rev = np.maximum(p, p_floor)
    p_rev = p_rev / p_rev.sum()

    def A_of(theta):
        W = np.diag(np.sqrt(np.exp(theta)))
        return W @ A_base @ W

    def cost(theta):
        try:
            A = _safe_takagi_cap(A_of(theta))
            pf = model_distribution_from_A(A, modes)
            q = pf[1:-1] / pf[1:-1].sum()
        except Exception:
            return float("inf")
        fwd = _safe_kl(p, q)
        q_safe = np.maximum(q, 1e-15)
        rev = float(np.sum(q_safe * np.log(q_safe / p_rev)))
        return (1 - beta) * fwd + beta * rev

    theta = np.zeros(modes)
    best_theta, best_c = theta.copy(), cost(theta)
    if not np.isfinite(best_c):
        return A_base, best_theta
    for _ in range(max_iter):
        c0 = cost(theta)
        if not np.isfinite(c0):
            theta = best_theta.copy()
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
                best_c, best_theta = cn, tn.copy()
        else:
            theta = best_theta.copy()
    return A_of(best_theta), best_theta


def outlier_mass(q, p_target, eps=0.01):
    """Total model mass landing on near-zero-target patterns."""
    mask = np.asarray(p_target) < eps
    return float(np.asarray(q)[mask].sum())


if __name__ == "__main__":
    from gbs_core import mmodal, bin_samples_from_dist, analytic_target, analytic_eval
    from constructions.A_from_samples_parity import A_from_samples_parity
    data = bin_samples_from_dist(mmodal, 0.01, 4, 2000, 5)
    A = A_from_samples_parity(data)
    pt = analytic_target(mmodal, 0.01, 4, 5)
    for beta in (0.0, 0.4):
        Aw, _ = train_WAW_mixed(A, pt, 5, beta=beta, max_iter=60)
        q, kl = analytic_eval(Aw, 5, pt)
        print(f"beta={beta}: KL={kl:.3f}  outlier_mass={outlier_mass(q, pt):.3f}")
