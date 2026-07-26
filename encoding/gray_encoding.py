"""
gray_encoding.py

Gray-code encoding of the target bins. The default `bin_samples_from_dist`
maps a smooth p(x) to click patterns via the STANDARD binary representation of
the bin index, which is not Hamming-smooth: adjacent x-bins can differ in all m
bits (e.g. 01111 <-> 10000). A smooth target then becomes highly non-smooth in
click-pattern space, which a pairwise (hafnian) model cannot represent.

A reflected Gray code, gray(n) = n XOR (n >> 1), makes consecutive bins differ
in exactly one bit, so a smooth target stays piecewise-smooth in pattern space.
This is the single highest-leverage encoding change (it roughly halved the KL
on the multimodal target).
"""

import numpy as np
from gbs_core import _bin_list, _sample_from_dist


def to_gray(n):
    """Reflected binary Gray code of a non-negative integer."""
    return n ^ (n >> 1)


def inv_gray(g):
    """Inverse reflected Gray code (Gray integer -> original bin index)."""
    mask = g >> 1
    while mask:
        g ^= mask
        mask >>= 1
    return g


def bin_samples_from_dist_gray(p_dist, x0, x1, num_samples, modes):
    """Like bin_samples_from_dist, but the bin index is Gray-coded so that
    adjacent x-bins map to click patterns at Hamming distance 1."""
    x = np.linspace(x0, x1, 2 ** modes - 2)
    d = p_dist(x)
    d = np.append(0, d)
    d = np.append(d, 0)
    out = []
    for _ in range(num_samples):
        idx = _sample_from_dist(d)[0]
        out.append(_bin_list(np.binary_repr(to_gray(idx), width=modes)))
    return out


if __name__ == "__main__":
    for n in range(8):
        g = to_gray(n)
        assert inv_gray(g) == n
        print(f"{n:03b} -> gray {g:03b} -> inv {inv_gray(g):03b}")
