# Speaker notes — "Learning Classical Distributions with GBS"

Target length ~25–30 min for a prepared audience. One block per slide, in
delivery order. Square-bracket notes are cues, not to be read aloud.

---

## 1 — Title
Good [morning/afternoon]. I'll talk about using Gaussian Boson Sampling as a
*trainable generative model* for classical probability distributions. Part one
is the theory — how a GBS device is programmed by a matrix and how the standard
WAW scheme trains it. Part two is our own work: a chain of fixes to that
pipeline, with results, and an honest discussion of where it breaks.

## 2 — Outline
Two parts. Theory first, in the matrix-centric language of Banchi, Quesada and
Arrazola. Then our contributions: the construction of the matrix, the encoding,
the optimiser, the detector, and a scaling study that exposes the hard cases.

## 3 — GBS in one equation
I'll start where the applications literature starts: not with the optics, but
with a matrix. A GBS device is *programmed* by a symmetric m-by-m matrix A, and
the photon-number distribution is this boxed formula — the probability of an
outcome vector n (bold n) is the modulus-squared hafnian of A-sub-n, over the
product of factorials, normalised by Z. The notation A-sub-n means: repeat row
and column i exactly n_i times, delete it if n_i is zero. Two notation notes,
because they bite later. First, bold n is the *outcome vector*; n-bar, later, is
the *mean photon number* — different objects, don't mix them. Second, on all the
main slides A is the small m-by-m "graph" block; the physical state actually
lives on a 2m-by-2m kernel — I'll write that one in bold A — and the two are
related in the appendix. Identity matrices I write as capital I. The operational fact to remember is
that the device preferentially samples configurations with *large* hafnian —
that's exactly why a cleverly chosen A solves graph problems: dense subgraphs,
similarity, point processes, vibronic spectra. Our question is the inverse one:
can we *train* A so the samples reproduce a target distribution we specify?

## 4 — The adjacency matrix A
The m-by-m symmetric block A has a Takagi–Autonne decomposition, U diag(lambda)
U-transpose, with singular values strictly below one. Two things to carry
forward. First, the hafnian — the sum over perfect matchings — is #P-hard, and
crucially it vanishes on odd-dimensional matrices. So any photon configuration
that contributes must have an *even* total photon number: parity. Second, the
mean photon number is the sum of lambda-squared over one-minus-lambda-squared,
and we tune it by rescaling A by a scalar c. Keep that scale in mind — it comes
back as a real hyperparameter in Part two.

## 5 — Threshold detectors
Most hardware uses *threshold* detectors: they don't count photons, they only
report "click or no click". Mathematically that marginalises the photon
distribution over all collision configurations with the same support, and the
result is a torontonian instead of a hafnian — this inclusion–exclusion sum,
exponential in the number of clicks. The one conceptual point: that
marginalisation folds all the collision terms into a single click probability.
That fold is harmless for physics but, as we'll see, it's the origin of the
biases we have to correct when we train.

## 6 — The task
Concretely: take a one-dimensional target density, chop it into 2^m minus 2
bins, and encode each bin as a click pattern — by default, the binary
representation of the bin index. We get data samples, and we minimise the KL
divergence between the data and the GBS model. For evaluation we mostly use the
*analytic* distribution — we compute the probability of all 2^m patterns exactly
with the torontonian — so our KL numbers have no sampling noise. That matters;
one of our findings is a bug that only the analytic evaluation exposed.

## 7 — Building the base matrix from data
Before we can train anything we need a starting matrix A, and it's worth being
explicit about how it's built, because everything in Part two modifies exactly
this step. The baseline — the `A_from_samples` routine — is just co-occurrence
counting: for each sample, you put an *edge between every pair of clicking
modes*, so A_ij is proportional to how often modes i and j click together. Then
symmetrise, normalise by the largest entry, and rescale to the target mean
photon number. The important point, which sets up the whole talk: this step fixes
the model's *correlation structure*. WAW, coming next, only reweights rows and
columns — it cannot create or move correlations, only the base A can. And the
baseline has two flaws we'll fix in Part two: it adds self-loops only for
single-click events, and it has no parity correction at all.

## 8 — WAW parametrization
This is the training scheme from the 2020 paper. Fix the base matrix A, and make
the per-mode *weights* trainable: A goes to W A W with W diagonal, the weights
reparametrised as exp of minus theta-dot-f so they stay in zero to one — which
keeps the state physical. The magic is that the hafnian factorises: hafnian of
W A W equals hafnian of A times determinant of W. So the weights pull out of the
hafnian and the distribution becomes this clean exponential family —
modulus-squared hafnian of A, times a product of w_i to the n_i, over
factorials. The hafnian itself is independent of the parameters. In practice one
uses the simplest diagonal choice, w_k equals exp theta_k.

## 9 — Training: analytic KL gradient
Because the hafnian doesn't depend on the weights, the derivative is strikingly
simple: the log-derivative with respect to w_k is (n_k minus the mean photon
number) over w_k. Plug that into the KL and everything collapses to *marginal
matching* — the gradient is the sum over modes of (model mean photon number minus
data mean) times f-k. That's computable in O(m) classically, from the covariance
matrix, even when sampling the device is itself intractable — a genuinely elegant
result. But note the flip side, which is the thesis of Part two: WAW only matches
*per-mode marginals*. Every *correlation* in the model comes from the base matrix
A — the one we just built. So if that construction is biased, no amount of WAW
training fixes it. That's why Part two is about the construction, the encoding,
and the readout — not the optimiser alone.

## 10 — The factorial / parity bias
Here's the core bias. Expand a k-click pattern at low squeezing. If k is even,
the minimal configuration — one photon per clicked mode — already has an even
total, so no penalty. If k is *odd*, that configuration has an odd total, the
hafnian vanishes, and you're forced to double one mode to reach an even count —
which costs a one-over-two-factorial. So every odd-click pattern is universally
suppressed relative to even ones. The naive construction ignores this, and
because the suppression is a *correlation* effect, WAW — marginals only — cannot
repair it. We have to fix the construction.

## 11 — Construction fixes
So we weight each sample's contribution to the base matrix. A ladder of ideas.
"Corrected" just boosts the diagonal by root-two — fixes only single clicks.
"Parity" generalises that: weight root-two for every odd-click sample, applied to
all entries it touches. "Per-pattern" tries to derive exact per-k weights from a
uniform reference matrix — but it over-suppresses high-k samples, so we dropped
it. "Per-pair" is the robust one: divide each sample's contribution by the number
of entries it touches, so a 4-click and a 2-click sample deposit comparable
evidence per entry. And "variance" iteratively matches the whole click-count
histogram, to fix not just the mean photon budget but its spread.

## 12 — Construction: the figure
This is per-pair against the baseline and per-pattern on the multimodal target,
before and after WAW. Look at the baseline in red: it over-produces the
high-click patterns — the big spurious bars. Per-pattern in purple does the
opposite, over-producing low-click ones. Per-pair in green sits closest to the
blue target across the board. This is the single most robust construction we
found.

## 13 — Encoding matters: Gray code
Now a completely different lever: the encoding. Standard binary is *not*
Hamming-smooth — bin fifteen is 01111, bin sixteen is 10000, adjacent in value
but maximally far in Hamming distance. So a smooth density becomes jagged in
pattern space, and a pairwise hafnian model simply cannot represent that. The fix
is a reflected Gray code — n XOR n-shifted — so consecutive bins differ by a
single bit. On the left, binary; on the right, Gray. For the multimodal target
this roughly halves the KL. For a smooth target the encoding matters less; for
multimodal it's the dominant factor.

## 14 — Optimisation: two fixes
Two problems in the optimiser. First, the baseline WAW uses random
initialisation, a fixed number of steps, and returns the last iterate — on
Gray-encoded data that collapses onto a single mode. Fix: zero initialisation,
monitor the KL, return the best iterate. Second — and this one is a genuine bug —
the sampling routine silently *rescales* the matrix at sampling time, so the
distribution you optimised is not the one you evaluate, and the reported KL could
actually go *up* after training. We only caught it because we switched to the
analytic KL. The fix is to optimise the exact analytic KL directly, with
keep-best, which makes the KL provably non-increasing.

## 15 — Construction × encoding: results
Putting construction and encoding together, after WAW, at five modes. Two
readings. For the smooth targets — normal, log-normal — the *construction*
dominates: a factor of three KL reduction from baseline to parity or variance.
For multimodal, the *encoding* dominates: Gray takes it from 0.59 down to 0.24.
Different targets, different bottleneck.

## 16 — Photon-counting readout
Next lever: the detector. If instead of threshold we use photon-counting, then
for a collision-free pattern the probability is just the modulus-squared hafnian
of the support submatrix — no factorial, no fold. In other words, PNR hands us
directly the clean signal that all those construction heuristics were trying to
estimate. On the right you can see the picture literally: patterns 200, 400 and
so on all collapse to a single threshold click.

## 17 — PNR: concentration, not a free win
But — and this is the honest part — PNR is not a free lunch. At fixed modes it
adds no matrix parameters, and the usable outcomes are dominated by a steep
photon-number magnitude hierarchy, so you need a probability-matched encoding or
the KL blows up. What PNR *does* give is a more *concentrated* model family —
lower entropy, more mass in the top few patterns — whereas threshold is more
spread. The table makes the consequence concrete: threshold wins on the broad
normal, PNR wins on the concentrated bimodal. The slogan is: match the detector
to the target's concentration.

## 18 — Selecting n_mean
That concentration idea resurfaces in the mean photon number. The usual default,
m-over-two, is arbitrary. n_mean is really a *concentration knob*: low n_mean
gives a concentrated model; high n_mean spreads the mass but inflates a central
spike. The honest recipe is to *select* it by minimum empirical KL. The left
panel shows the fit quality has a target-dependent minimum — broad targets near
two, concentrated targets near one-half — while the spike, on the right, grows
monotonically with n_mean. The optimum is always below the default.

## 19 — Combining threshold + PNR
Since threshold and PNR are complementary, combine them. Both models map back to
the same x-bins, so we form a convex mixture, weight chosen by minimum empirical
KL. This is physically realizable — pick a detector per shot — and being convex
it can never be worse than the better single model. On Cauchy it's dramatic:
threshold overshoots the tails, PNR overshoots the peak, and a fifty-fifty
mixture lands on the target — a three-fold KL improvement over either. Same story
on the asymmetric bimodal. The mixture wins two-to-three-fold whenever the target
has both broad and sharp structure.

## 20 — Scaling study
To stress-test, we added harder targets — strongly skewed, very narrow,
heavy-tailed, and a sharp-plus-broad mixture — and swept four, five and six
modes. This heatmap is the best achievable KL over all methods. The headline:
the pipeline is robust — eight of nine targets land under 0.08 at every mode
count, including the heavy-tailed and skewed ones, and more modes help rather
than hurt. There's exactly one dark row.

## 21 — Worst case: the concentration ceiling
That dark row is the very *narrow* target, and it's the most interesting failure.
The model finds the location correctly but it *leaks* — here about 0.13 of the
mass onto a distant pattern. The reason is a concentration *ceiling*: a GBS click
distribution's most-probable pattern simply cannot carry the roughly one-half of
the mass a two-bin target demands. This is the mirror image of the multimodal
limit: there the model can't *spread* into a deep valley; here it can't *pile up*
into a narrow peak. Interestingly, it eases with more modes, because finer
binning makes the same physical peak span more bins.

## 22 — Honesty checks
Two checks so we're not fooling ourselves. One: the probability-matched encoding
uses only the *empirical* histogram, never the true target — and the train and
generalisation KL agree, and match the oracle encoding to within about 0.01, so
it's not a hidden cheat. Two: we tried a *local*, per-bin mixture, gating the
weight by local peakiness. It matches the global mixture but doesn't beat it; and
the per-bin convex-hull bound shows headroom *does* exist, but a simple gate
can't capture it and richer per-bin fitting overfits. So the global mixture stays
our recommendation — a clean negative result.

## 23 — Summary
To summarise the levers: the parity and per-pair constructions fix the factorial
bias; variance matching fixes the photon budget; Gray coding fixes the encoding
for multimodal; analytic keep-best makes the optimiser trustworthy and caught a
real bug; and the threshold–PNR mixture exploits detector complementarity. Net,
about a three-fold KL reduction on smooth targets, two-and-a-half on multimodal,
and eight of nine targets under 0.08.

## 24 — Limits and scope
Now the honest framing, because a prepared audience will ask. For a
one-dimensional density, a histogram or inverse-CDF sampler is exact — GBS does
*not* compete as a sampler, and at these sizes everything is classically
simulable, so there's no quantum advantage here. The value is different: a
careful *characterisation* of GBS as a trainable model, plus reusable evaluation
methodology. We now understand three structural limits — the multimodal valley,
the concentration ceiling, and the unrealised per-bin-mixture headroom. The way
forward is added expressivity — displacement or a small mixture of GBS states,
which attack both the valley and the ceiling — and, more importantly, pivoting to
GBS-*native* targets like graph distributions, where the adjacency matrix is the
natural object rather than an awkward encoding of a scalar.

## 25 — References
The core references: Banchi–Quesada–Arrazola for the training scheme, Quesada et
al. for threshold detection, Hamilton et al. and Kruse et al. for GBS itself.

## 26 — Thank you
Thank you — I'm happy to take questions. [Likely: why KL and not total variation;
whether displacement closes the ceiling; what a graph-native target would look
like; is any of this hard to simulate.]

---

## Appendix cues (only if asked)
- **Device**: squeezed vacuum has only even photon numbers — that's the physical
  root of the parity bias; interferometer U gives the block A = U diag(tanh r)
  U-transpose.
- **Gaussian formalism / the two A's**: the 2m-by-2m *kernel* is bold-A =
  X(1 − (V + ½)⁻¹); for a pure state bold-A = A ⊕ A*, with the m-by-m block A
  used on the main slides. Per-mode photon number ⟨n_k⟩ = V_kk + V_{k+m,k+m} − ½
  is what the WAW gradient uses.
- **Displacement**: μ ≠ 0 lifts parity; uses the loop torontonian; a scalable
  surrogate is "train WAW, then one scalar displacement by binary search".
- **Gradient derivation**: the one-slide sketch of how the marginal-matching
  gradient drops out of the WAW exponential family.
