# GBS distribution learning

Methods for training a Gaussian Boson Sampling (GBS) device to reproduce
one-dimensional classical distributions encoded as bit strings. Each method
from the research notebooks lives in its own file, named after the method it
implements, so the code paths match the report.

The full narrative (motivation + results) is in
[`report/GBS_distribution_learning.tex`](report/GBS_distribution_learning.tex).

## Layout

```
gbs_core.py                         shared foundation: targets, baseline GBS
                                    utilities, analytic (MC-free) evaluation
constructions/                      building the initial adjacency matrix A
  A_from_samples_corrected.py         sqrt(2) diagonal boost (first fix)
  A_from_samples_parity.py            parity rule (generalised sqrt(2))
  A_from_samples_perpattern.py        per-pattern factorial weights (superseded)
  A_from_samples_perpair.py           per-pair normalisation (best construction)
  A_from_samples_variance.py          photon-number / click-count matching
encoding/                           mapping target bins to click patterns
  gray_encoding.py                    Gray-code encoding (to_gray / bin sampler)
  rank_assign.py                      probability-matched encoding + empirical
                                      target estimation from samples
optimization/                       WAW training variants
  train_WAW_bestkl.py                 zero-init, keep-best-KL (sample-based)
  train_WAW_analytic.py               minimises the exact analytic KL
  train_WAW_mixed.py                  reverse-KL outlier suppression
  train_threshold_nmean.py            central-spike suppression via n_mean
displacement/                       breaking photon-number parity
  train_displaced_gbs.py              full 2m displacement vector
  train_waw_then_displace.py          scalable scalar displacement
  train_parity_mixing.py              two-circuit classical mixing
readout/
  pnr_readout.py                      photon-number-resolving readout
run_experiments.py                  convenience driver (constructions / pnr / spike)
report/                             LaTeX report + figures
```

## Install

```bash
pip install -r requirements.txt
```

`gbs_core.py` contains a NumPy 2.0 shim so recent NumPy works with `thewalrus`.
The code was developed against the same conda environment as the notebooks
(python 3.9, `thewalrus` + `strawberryfields`).

## Running

All commands are run **from the repository root** so the package imports
resolve (Python 3 namespace packages).

### Each method has a self-contained demo

```bash
python -m constructions.A_from_samples_parity
python -m constructions.A_from_samples_perpair
python -m constructions.A_from_samples_variance
python -m encoding.gray_encoding
python -m encoding.rank_assign
python -m optimization.train_WAW_analytic
python -m optimization.train_WAW_mixed
python -m optimization.train_threshold_nmean
python -m displacement.train_waw_then_displace
python -m displacement.train_parity_mixing
python -m readout.pnr_readout
```

### The driver reproduces the main comparisons

```bash
# initial constructions, before/after analytic WAW
python run_experiments.py constructions --target normal --modes 5 --encoding gray

# threshold vs photon-counting readout at the same modes
python run_experiments.py pnr --target bimodal_asym --modes 6 --num_samples 6000

# central-spike suppression by lowering n_mean
python run_experiments.py spike --target normal --modes 6
```

Targets: `normal`, `mmodal`, `lognorm`, `expo`, `cauchy`, `bimodal_asym`.

## Typical usage in code

```python
from gbs_core import normal, analytic_target, analytic_eval
from encoding.gray_encoding import bin_samples_from_dist_gray
from constructions.A_from_samples_perpair import A_from_samples_perpair
from optimization.train_WAW_analytic import train_WAW_analytic

modes = 5
data = bin_samples_from_dist_gray(normal, 0.01, 4, 2000, modes)   # Gray encoding
A    = A_from_samples_perpair(data)                               # initial matrix
target = analytic_target(normal, 0.01, 4, modes, encoding="gray")
_, kl_before = analytic_eval(A, modes, target)
A_opt, kl_after, _ = train_WAW_analytic(A, target, modes)         # analytic WAW
print(kl_before, "->", kl_after)
```

## Method map (what to read for what)

| Goal | File |
| --- | --- |
| Fix the initial matrix bias | `constructions/A_from_samples_parity.py`, `A_from_samples_perpair.py` |
| Match the photon-number variance | `constructions/A_from_samples_variance.py` |
| Make a smooth target representable | `encoding/gray_encoding.py` |
| Reliable, monotone optimisation | `optimization/train_WAW_analytic.py` |
| Remove valley outliers | `optimization/train_WAW_mixed.py` |
| Remove the central peak spike | `optimization/train_threshold_nmean.py` |
| Break photon-number parity | `displacement/*.py` |
| Photon-counting detectors | `readout/pnr_readout.py` |

## Summary of findings

- **Parity / per-pair construction** corrects the factorial bias in the initial
  matrix; per-pair is the most robust.
- **Gray encoding** is decisive for multimodal targets (~2x KL reduction).
- **Analytic keep-best WAW** makes optimisation monotone (the baseline could
  make KL worse through a silent rescale at sampling time).
- **`n_mean`** should be treated as a hyperparameter, not hard-wired to `m/2`;
  lowering it suppresses the central spike and can lower the KL.
- **Threshold vs PNR**: match the detector to the target's concentration —
  threshold for broad targets, photon-counting for sharply-peaked ones.
- The residual error on multimodal targets is an expressivity floor of a single
  Gaussian state; displacement or a GBS mixture is the next step.
