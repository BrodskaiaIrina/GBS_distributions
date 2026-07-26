"""
rank_assign.py

Probability-matched encoding and empirical target estimation.

`rank_assign` chooses which physical outcome represents which target bin by
matching the target's mass profile to the model's natural probability profile:
the largest target mass goes to the most-probable pattern, and so on. It is the
analogue, for readouts whose natural axis is probability magnitude (e.g. PNR),
of the Gray-code choice for threshold (whose natural axis is Hamming distance).

`empirical_target_from_data` estimates the (unknown) target shape from samples
alone (decoding click samples back to bin indices and histogramming), so the
encoding can be built without ever peeking at the analytic target — this is the
honest, realizable version used in the pretrained experiments.
"""

import numpy as np


def rank_assign(target_vals, w0):
    """Return the target re-aligned to pattern order: biggest target mass on the
    most-probable pattern. w0 = base model probabilities over the patterns."""
    target_vals = np.asarray(target_vals)
    K = len(target_vals)
    order = np.argsort(-np.asarray(w0))         # patterns by descending base prob
    st = np.sort(target_vals)[::-1]             # target by descending value
    p_assigned = np.zeros(K)
    p_assigned[order] = st
    return p_assigned


def rank_assign_with_map(target_vals, w0):
    """Like rank_assign but also returns the explicit bin->pattern index map,
    so the assignment can be inverted for an x-ordered (real-distribution) plot."""
    target_vals = np.asarray(target_vals)
    K = len(target_vals)
    pat_order = np.argsort(-np.asarray(w0))
    bin_order = np.argsort(-target_vals)
    p_assigned = np.zeros(K)
    bin2pat = np.zeros(K, dtype=int)
    for r in range(K):
        b, pat = bin_order[r], pat_order[r]
        p_assigned[pat] = target_vals[b]
        bin2pat[b] = pat
    return p_assigned, bin2pat


def empirical_target_from_data(data, modes):
    """Empirical target over the K = 2**modes - 2 bins, from samples only:
    decode each click sample to its bin index and histogram."""
    K = 2 ** modes - 2
    h = np.zeros(K)
    for s in data:
        idx = int("".join(str(int(b)) for b in s), 2)   # 1 .. 2**modes - 2
        if 1 <= idx <= K:
            h[idx - 1] += 1
    h = np.maximum(h, 1e-6)
    return h / h.sum()


if __name__ == "__main__":
    tgt = np.array([0.05, 0.30, 0.10, 0.40, 0.15])
    w0 = np.array([0.1, 0.5, 0.2, 0.05, 0.15])
    p_as, b2p = rank_assign_with_map(tgt, w0)
    recovered = np.array([p_as[b2p[i]] for i in range(len(tgt))])
    print("inversion exact:", np.allclose(recovered, tgt))
