"""
train_WAW_analytic.py

Analytic-KL WAW. The keep-best WAW selects on the strawberryfields VGBS cost,
but the reported metric is an empirical Monte-Carlo KL, and
torontonian_sample_graph silently rescales A at sampling time — so the
optimiser and the scoreboard could disagree and KL sometimes went UP after
training. train_WAW_analytic minimises EXACTLY the analytic KL we report
(threshold_detection_prob over all 2^m patterns) via numerical-gradient descent
on the log-weights theta, with keep-best return. It is robust to LAPACK
failures (any KL eval that raises is treated as +inf and skipped), so the KL
after training can never exceed the initial value.
"""

import numpy as np
from gbs_core import _safe_takagi_cap, model_distribution_from_A, _safe_kl


def train_WAW_analytic(A_base, prob_target_rn, modes, max_iter=120, rate=0.05, eps=1e-3):
    def A_of_theta(theta):
        W = np.diag(np.sqrt(np.exp(theta)))
        return W @ A_base @ W

    def kl_of_theta(theta):
        try:
            A = _safe_takagi_cap(A_of_theta(theta))
            p_full = model_distribution_from_A(A, modes)
            p_rn = p_full[1:-1] / p_full[1:-1].sum()
            return _safe_kl(prob_target_rn, p_rn)
        except Exception:
            return float("inf")

    theta = np.zeros(modes)
    best_theta, best_kl = theta.copy(), kl_of_theta(theta)
    if not np.isfinite(best_kl):
        return A_base, best_kl, best_theta

    for _ in range(max_iter):
        kl0 = kl_of_theta(theta)
        if not np.isfinite(kl0):
            theta = best_theta.copy()
            continue
        grad = np.zeros(modes)
        for k in range(modes):
            tp = theta.copy(); tp[k] += eps
            kl_p = kl_of_theta(tp)
            grad[k] = 0.0 if not np.isfinite(kl_p) else (kl_p - kl0) / eps
        theta_new = theta - rate * grad
        kl_new = kl_of_theta(theta_new)
        if np.isfinite(kl_new):
            theta = theta_new
            if kl_new < best_kl:
                best_kl, best_theta = kl_new, theta_new.copy()
        else:
            theta = best_theta.copy()
    return A_of_theta(best_theta), best_kl, best_theta


if __name__ == "__main__":
    from gbs_core import normal, bin_samples_from_dist, analytic_target
    from constructions.A_from_samples_parity import A_from_samples_parity
    data = bin_samples_from_dist(normal, 0.01, 4, 1000, 4)
    A = A_from_samples_parity(data)
    pt = analytic_target(normal, 0.01, 4, 4)
    Aw, kl, _ = train_WAW_analytic(A, pt, 4, max_iter=60)
    print(f"analytic KL after training: {kl:.4f}")
