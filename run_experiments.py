"""
run_experiments.py — convenience driver reproducing the main comparisons.

Run from the repository root, e.g.:

    python run_experiments.py constructions --target normal --modes 5 --encoding gray
    python run_experiments.py pnr          --target bimodal_asym --modes 6
    python run_experiments.py spike        --target normal --modes 6

Each method also has a standalone demo in its own file, runnable as
    python -m constructions.A_from_samples_parity
"""

import argparse
import numpy as np

from gbs_core import (TARGETS, bin_samples_from_dist, analytic_target,
                      analytic_eval, A_from_samples)
from encoding.gray_encoding import bin_samples_from_dist_gray
from encoding.rank_assign import rank_assign_with_map, empirical_target_from_data
from constructions.A_from_samples_parity import A_from_samples_parity
from constructions.A_from_samples_perpair import A_from_samples_perpair
from constructions.A_from_samples_variance import A_from_samples_variance
from optimization.train_WAW_analytic import train_WAW_analytic
from optimization.train_threshold_nmean import train_threshold_nmean, thr_weights_nmean
from readout.pnr_readout import enumerate_even_pnr, pnr_model_over, thr_weights

X0, X1 = 0.01, 4.0


def constructions(target="normal", modes=5, encoding="binary", num_samples=2000):
    """Compare the initial constructions, before and after analytic WAW."""
    dist = TARGETS[target]
    sampler = bin_samples_from_dist_gray if encoding == "gray" else bin_samples_from_dist
    data = sampler(dist, X0, X1, num_samples, modes)
    pt = analytic_target(dist, X0, X1, modes, encoding=encoding)
    builders = {
        "original": A_from_samples,
        "parity": A_from_samples_parity,
        "perpair": A_from_samples_perpair,
        "variance(a=.5)": lambda s: A_from_samples_variance(s, alpha=0.5),
    }
    print(f"\n{target} m={modes} [{encoding}] — analytic KL (before / after WAW)")
    print("-" * 48)
    for name, build in builders.items():
        A = build(data)
        _, kl_b = analytic_eval(A, modes, pt)
        _, kl_a, _ = train_WAW_analytic(A, pt, modes, max_iter=100)
        print(f"  {name:16s}  {kl_b:6.3f}  ->  {kl_a:6.3f}")


def pnr(target="bimodal_asym", modes=5, num_samples=4000):
    """Threshold vs full-PNR at the same modes, empirical (pretrained) encoding."""
    dist = TARGETS[target]
    K = 2 ** modes - 2
    xs = np.linspace(X0, X1, K)
    p_true = dist(xs); p_true = p_true / p_true.sum()
    data = bin_samples_from_dist(dist, X0, X1, num_samples, modes)
    ph = empirical_target_from_data(data, modes)

    # threshold branch (parity base)
    A_thr = A_from_samples_parity(data)
    tp = [tuple(int(b) for b in np.binary_repr(i, modes)) for i in range(1, 2 ** modes - 1)]
    w0 = thr_weights(A_thr, tp, modes); w0 = w0 / max(w0.sum(), 1e-15)
    pa, b2p = rank_assign_with_map(ph, w0)
    from optimization.train_threshold_nmean import train_threshold_nmean as _tt
    q_t = _tt(A_thr, pa, tp, modes, modes / 2)
    pt_as = np.zeros(K)
    for i in range(K):
        pt_as[b2p[i]] = p_true[i]
    from gbs_core import _safe_kl
    kl_thr = _safe_kl(pt_as, q_t)

    # PNR branch (plain base)
    A_pnr = A_from_samples(data)
    pats = enumerate_even_pnr(modes, K)
    w0p = pnr_model_over(A_pnr, pats)
    pap, b2pp = rank_assign_with_map(ph, w0p)
    from optimization.train_WAW_analytic import train_WAW_analytic as _dummy  # keep imports tidy
    q_p = _train_pnr(A_pnr, pap, pats, modes)
    pt_as_p = np.zeros(K)
    for i in range(K):
        pt_as_p[b2pp[i]] = p_true[i]
    kl_pnr = _safe_kl(pt_as_p, q_p)

    print(f"\n{target} m={modes} — same-modes readout comparison (KL vs true)")
    print(f"  threshold : {kl_thr:.3f}")
    print(f"  PNR       : {kl_pnr:.3f}")


def _train_pnr(A_base, p_target, pats, modes, steps=100, rate=0.05, eps=1e-3):
    from gbs_core import _safe_takagi_cap, _safe_kl
    def q_of(theta):
        W = np.diag(np.sqrt(np.exp(theta)))
        A = _safe_takagi_cap(W @ A_base @ W)
        return pnr_model_over(A, pats)
    def cost(theta):
        try:
            return _safe_kl(p_target, q_of(theta))
        except Exception:
            return float("inf")
    theta = np.zeros(modes); bt, bc = theta.copy(), cost(theta)
    for _ in range(steps):
        c0 = cost(theta)
        if not np.isfinite(c0):
            theta = bt.copy(); continue
        g = np.array([(cost(theta + eps * np.eye(modes)[k]) - c0) / eps for k in range(modes)])
        tn = theta - rate * g; cn = cost(tn)
        if np.isfinite(cn):
            theta = tn
            if cn < bc:
                bc, bt = cn, tn.copy()
        else:
            theta = bt.copy()
    return q_of(bt)


def spike(target="normal", modes=6, num_samples=6000):
    """Show the central spike suppressed by lowering n_mean."""
    dist = TARGETS[target]
    K = 2 ** modes - 2
    data = bin_samples_from_dist(dist, X0, X1, num_samples, modes)
    ph = empirical_target_from_data(data, modes)
    A = A_from_samples_parity(data)
    tp = [tuple(int(b) for b in np.binary_repr(i, modes)) for i in range(1, 2 ** modes - 1)]
    print(f"\n{target} m={modes} — spike vs n_mean")
    for nm in (modes / 2, 2.0):
        w0 = thr_weights_nmean(A, tp, modes, nm); w0 = w0 / max(w0.sum(), 1e-15)
        pa, b2p = rank_assign_with_map(ph, w0)
        q = train_threshold_nmean(A, pa, tp, modes, nm, steps=100)
        thr_x = np.array([q[b2p[i]] for i in range(K)]); pk = int(np.argmax(thr_x))
        ratio = thr_x[pk] / ((thr_x[pk - 1] + thr_x[pk + 1]) / 2)
        print(f"  n_mean={nm:.1f}: spike/neighbor = {ratio:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["constructions", "pnr", "spike"])
    ap.add_argument("--target", default="normal", choices=list(TARGETS))
    ap.add_argument("--modes", type=int, default=5)
    ap.add_argument("--encoding", default="binary", choices=["binary", "gray"])
    ap.add_argument("--num_samples", type=int, default=2000)
    a = ap.parse_args()
    if a.experiment == "constructions":
        constructions(a.target, a.modes, a.encoding, a.num_samples)
    elif a.experiment == "pnr":
        pnr(a.target, a.modes, a.num_samples)
    elif a.experiment == "spike":
        spike(a.target, a.modes, a.num_samples)
