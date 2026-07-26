"""
train_parity_mixing.py

Parity-aware mixing (classical workaround). Instead of modifying the quantum
circuit, split the data by click parity and train two standard GBS circuits:

  * even circuit: standard WAW on the even-parity samples;
  * odd circuit: flip one mode in the odd-parity samples (odd -> even), train a
    standard WAW, then flip that mode back when sampling.

At sampling time, draw from the even circuit with probability lambda (the data's
even fraction) and from the odd circuit otherwise. This reproduces the correct
even/odd balance by construction, at the cost of two circuits.
"""

import numpy as np
from gbs_core import A_from_samples, train_WAW, gbs_sample


def split_by_parity(samples):
    even = [s for s in samples if sum(s) % 2 == 0]
    odd = [s for s in samples if sum(s) % 2 == 1]
    return even, odd


def flip_mode(samples, mode_idx):
    out = []
    for s in samples:
        s_new = list(s)
        s_new[mode_idx] = 1 - s_new[mode_idx]
        out.append(s_new)
    return out


def train_parity_mixing(A_base, data_samples, flip_mode_idx=0):
    even_data, odd_data = split_by_parity(data_samples)
    lambda_even = len(even_data) / len(data_samples)
    print(f"Parity split: {len(even_data)} even ({lambda_even:.2%}), "
          f"{len(odd_data)} odd ({1-lambda_even:.2%})")

    if len(even_data) > 10:
        A_even, norms_even, _ = train_WAW(A_from_samples(even_data), even_data, steps=1000)
    else:
        A_even, norms_even = A_base, []

    if len(odd_data) > 10:
        odd_as_even = flip_mode(odd_data, flip_mode_idx)
        A_odd, norms_odd, _ = train_WAW(A_from_samples(odd_as_even), odd_as_even, steps=1000)
    else:
        A_odd, norms_odd = A_base, []

    return A_even, A_odd, lambda_even, norms_even, norms_odd


def sample_parity_mixed(A_even, A_odd, lambda_even, flip_mode_idx, num_samples):
    n_even = int(round(lambda_even * num_samples))
    n_odd = num_samples - n_even
    samples = []
    if n_even > 0:
        samples.extend(gbs_sample(A_even, n_even).tolist())
    if n_odd > 0:
        s_odd = gbs_sample(A_odd, n_odd).tolist()
        samples.extend(flip_mode(s_odd, flip_mode_idx))
    np.random.shuffle(samples)
    return samples


if __name__ == "__main__":
    from gbs_core import normal, bin_samples_from_dist, A_from_samples
    data = bin_samples_from_dist(normal, 0.01, 4, 1000, 4)
    A = A_from_samples(data)
    A_e, A_o, lam, _, _ = train_parity_mixing(A, data)
    s = sample_parity_mixed(A_e, A_o, lam, 0, 500)
    odd = sum(1 for x in s if sum(x) % 2 == 1) / len(s)
    print(f"sampled odd fraction = {odd:.2%}")
