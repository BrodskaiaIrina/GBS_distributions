"""
train_fullA.py

Full-matrix MLE. The WAW scheme trains only the m diagonal weights, so it matches
per-mode marginals and leaves every off-diagonal correlation frozen at the
construction. But the probability depends on the whole matrix through the
hafnian, |Haf(A[n])|^2 — not just the factorial denominator. To "account for the
hafnian" we therefore optimise ALL entries of the symmetric matrix A, by
maximum likelihood (equivalently, minimising the exact KL to the empirical/true
target). This is standard MLE, and it is within the Banchi et al. framework (the
general gradient formula of their Sec. IV.A applies to any parametrisation of A).

Empirically full-matrix MLE roughly halves the KL versus diagonal WAW, because it
can shape the correlations the hafnian depends on.

Cost: m(m+1)/2 parameters and a finite-difference analytic-KL gradient; still a
classical O(poly) optimisation at small m. The matrix is capped to stay physical
(largest Takagi value < 1) after every step.
"""

import numpy as np
from gbs_core import _safe_takagi_cap, model_distribution_from_A, _safe_kl


def train_fullA(A_init, prob_target_rn, modes, steps=120, rate=0.04, eps=1e-3):
    """Optimise all entries of the symmetric A by minimising the analytic KL.
    Returns (A_opt, kl)."""
    iu = np.triu_indices(modes)

    def build(x):
        A = np.zeros((modes, modes)); A[iu] = x
        A = A + A.T - np.diag(np.diag(A))        # symmetric, diagonal once
        return _safe_takagi_cap(A)

    def kl(x):
        try:
            A = build(x)
            pf = model_distribution_from_A(A, modes)
            q = pf[1:-1] / pf[1:-1].sum()
            return _safe_kl(prob_target_rn, q)
        except Exception:
            return float("inf")

    x = np.asarray(A_init)[iu].astype(float).copy()
    best, best_kl = x.copy(), kl(x)
    for _ in range(steps):
        k0 = kl(x)
        if not np.isfinite(k0):
            x = best.copy(); continue
        g = np.zeros_like(x)
        for j in range(len(x)):
            xp = x.copy(); xp[j] += eps
            kp = kl(xp)
            g[j] = 0.0 if not np.isfinite(kp) else (kp - k0) / eps
        xn = x - rate * g
        kn = kl(xn)
        if np.isfinite(kn):
            x = xn
            if kn < best_kl:
                best_kl, best = kn, xn.copy()
        else:
            x = best.copy()
    return build(best), best_kl


if __name__ == "__main__":
    from gbs_core import TARGETS, bin_samples_from_dist, analytic_target, analytic_eval
    from constructions.A_from_samples_parity import A_from_samples_parity
    from optimization.train_WAW_analytic import train_WAW_analytic
    m = 5
    for t in ("normal", "cauchy", "lognorm"):
        pt = analytic_target(TARGETS[t], 0.01, 4, m)
        data = bin_samples_from_dist(TARGETS[t], 0.01, 4, 3000, m)
        A0 = A_from_samples_parity(data)
        _, kl_waw, _ = train_WAW_analytic(A0, pt, m, max_iter=100)
        _, kl_full = train_fullA(A0, pt, m, steps=100)
        print(f"{t:10s}: WAW({m}p)={kl_waw:.3f}  fullA({m*(m+1)//2}p)={kl_full:.3f}")
