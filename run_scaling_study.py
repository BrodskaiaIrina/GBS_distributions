"""
run_scaling_study.py — scaling study over nine targets and m in {4,5,6}.

Prints a KL table (threshold / PNR / global mix / local mix / convex-hull bound),
identifies the worst cases, and saves the report figures
(scaling_heatmap, worstcase_*, local_vs_global_cauchy) into report/figures/.
Run from the repository root:  python run_scaling_study.py
"""
import os, sys, matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gbs_core import TARGETS, bin_samples_from_dist, _safe_kl
from encoding.rank_assign import empirical_target_from_data
from ensemble.combine_threshold_pnr import _threshold_fit, _pnr_fit
from ensemble.combine_local import combine_local, local_peakiness, _sigmoid

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report", "figures")
os.makedirs(RES, exist_ok=True)
X0, X1 = 0.01, 4.0
TARGET_LIST = ["normal", "lognorm", "mmodal", "cauchy", "bimodal_asym",
               "skewed", "narrow", "heavy_tail", "sharp_plus_broad"]


def eff_bins(p):
    return float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-15)))))


def fit_all(target, m, ns, steps=70):
    xs = np.linspace(X0, X1, 2 ** m - 2)
    pt = TARGETS[target](xs); pt = pt / pt.sum()
    data = bin_samples_from_dist(TARGETS[target], X0, X1, ns, m)
    ph = empirical_target_from_data(data, m)
    # threshold: select n_mean in {0.5, 2.0} by empirical KL
    cands = {}
    for nm in (0.5, 2.0):
        q = _threshold_fit(data, ph, m, nm, steps)
        cands[nm] = (q, _safe_kl(ph, q))
    nm_best = min(cands, key=lambda k: cands[k][1])
    q_thr = cands[nm_best][0]
    q_pnr = _pnr_fit(data, ph, m, steps)
    # global mixture (empirical selection)
    ws = np.linspace(0, 1, 51)
    wg = ws[int(np.argmin([_safe_kl(ph, w * q_thr + (1 - w) * q_pnr) for w in ws]))]
    q_glob = wg * q_thr + (1 - wg) * q_pnr
    # local mixture
    r = local_peakiness(ph)
    from ensemble.combine_local import _fit_gate
    a, b = _fit_gate(ph, q_thr, q_pnr, r)
    w_loc = _sigmoid(a + b * r)
    q_loc = w_loc * q_thr + (1 - w_loc) * q_pnr; q_loc = q_loc / q_loc.sum()
    # per-bin convex-hull oracle: clip target between the two curves (best any mixture could do)
    lo = np.minimum(q_thr, q_pnr); hi = np.maximum(q_thr, q_pnr)
    q_orc = np.clip(pt, lo, hi); q_orc = q_orc / q_orc.sum()
    return dict(xs=xs, pt=pt, nm=nm_best, wg=wg,
                thr=_safe_kl(pt, q_thr), pnr=_safe_kl(pt, q_pnr),
                glob=_safe_kl(pt, q_glob), loc=_safe_kl(pt, q_loc),
                orc=_safe_kl(pt, q_orc), q_thr=q_thr, q_pnr=q_pnr,
                q_glob=q_glob, q_loc=q_loc, effbins=eff_bins(pt))


def main():
    modes_list = [4, 5, 6]
    ns_of = {4: 2000, 5: 3000, 6: 6000}
    results = {}
    print(f"{'target':18s} {'m':>2} {'effb':>6} {'thr':>6} {'pnr':>6} {'glob':>6} {'local':>6} {'hull*':>6} {'best':>6}")
    print("-" * 78)
    for t in TARGET_LIST:
        for m in modes_list:
            r = fit_all(t, m, ns_of[m])
            results[(t, m)] = r
            best = min(r['thr'], r['pnr'], r['glob'], r['loc'])
            print(f"{t:18s} {m:>2} {r['effbins']:>6.1f} {r['thr']:>6.3f} {r['pnr']:>6.3f} "
                  f"{r['glob']:>6.3f} {r['loc']:>6.3f} {r['orc']:>6.3f} {best:>6.3f}")

    # ---- worst cases (by best-achievable KL) ----
    worst = sorted(results.items(), key=lambda kv: -min(kv[1]['thr'], kv[1]['pnr'], kv[1]['glob'], kv[1]['loc']))[:5]
    print("\nWORST CASES (highest best-achievable KL):")
    for (t, m), r in worst:
        best = min(r['thr'], r['pnr'], r['glob'], r['loc'])
        print(f"  {t} m={m}: best KL={best:.3f} (effbins={r['effbins']:.1f})")

    # ---- Figure 1: heatmap of best-achievable KL ----
    M = np.zeros((len(TARGET_LIST), len(modes_list)))
    for i, t in enumerate(TARGET_LIST):
        for j, m in enumerate(modes_list):
            r = results[(t, m)]
            M[i, j] = min(r['thr'], r['pnr'], r['glob'], r['loc'])
    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(modes_list))); ax.set_xticklabels([f"m={m}" for m in modes_list])
    ax.set_yticks(range(len(TARGET_LIST))); ax.set_yticklabels(TARGET_LIST)
    for i in range(len(TARGET_LIST)):
        for j in range(len(modes_list)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Best-achievable KL (min over methods)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="KL")
    plt.tight_layout(); plt.savefig(f"{RES}/scaling_heatmap.png", dpi=110, bbox_inches="tight"); plt.close()

    # ---- Figure 2: local vs global vs hull-oracle on cauchy (m=6) ----
    r = results[("cauchy", 6)]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(r['xs'], r['pt'], width=(r['xs'][1]-r['xs'][0])*0.9, color="#1f77b4", alpha=0.3, label="Target")
    ax.plot(r['xs'], r['q_glob'], "-o", ms=3, color="#9467bd", label=f"Global mix (KL={r['glob']:.3f})")
    ax.plot(r['xs'], r['q_loc'], "-^", ms=3, color="#ff7f0e", label=f"Local mix (KL={r['loc']:.3f})")
    ax.plot(r['xs'], np.clip(r['pt'], np.minimum(r['q_thr'], r['q_pnr']), np.maximum(r['q_thr'], r['q_pnr'])),
            ":", color="gray", label=f"Per-bin convex-hull bound (KL={r['orc']:.3f})")
    ax.set_xlabel("x"); ax.set_ylabel("p(x)"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_title("Cauchy (m=6): local vs global mixture, with per-bin convex-hull bound")
    plt.tight_layout(); plt.savefig(f"{RES}/local_vs_global_cauchy.png", dpi=110, bbox_inches="tight"); plt.close()

    # ---- Figure 3: worst case fit ----
    (tw, mw), rw = worst[0]
    best_q = min([('thr', rw['q_thr']), ('pnr', rw['q_pnr']), ('glob', rw['q_glob']), ('loc', rw['q_loc'])],
                 key=lambda kv: _safe_kl(rw['pt'], kv[1]))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(rw['xs'], rw['pt'], width=(rw['xs'][1]-rw['xs'][0])*0.9, color="#1f77b4", alpha=0.35, label="Target")
    ax.plot(rw['xs'], best_q[1], "-o", ms=3, color="#d62728",
            label=f"Best model ({best_q[0]}, KL={_safe_kl(rw['pt'], best_q[1]):.3f})")
    ax.set_xlabel("x"); ax.set_ylabel("p(x)"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_title(f"Worst case: {tw} (m={mw})")
    plt.tight_layout(); plt.savefig(f"{RES}/worstcase_{tw}_m{mw}.png", dpi=110, bbox_inches="tight"); plt.close()

    print(f"\nSaved: scaling_heatmap.png, local_vs_global_cauchy.png, worstcase_{tw}_m{mw}.png")
    print("DONE")


if __name__ == "__main__":
    main()
