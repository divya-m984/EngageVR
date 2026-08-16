# EngageVR -- Method References

Every algorithm implemented in this repository is listed here with its
primary reference.  A method is not implemented until its original paper
(or an authoritative implementation) has been consulted.

## Remote photoplethysmography (Milestone 3)

### GREEN -- green-channel baseline

> W. Verkruysse, L. O. Svaasand, J. S. Nelson (2008).
> "Remote plethysmographic imaging using ambient light."
> *Optics Express* 16(26):21434-21445.
> DOI: [10.1364/OE.16.021434](https://doi.org/10.1364/OE.16.021434)

Establishes that a plethysmographic signal is recoverable from an
ordinary colour camera under ambient light, and that the modulation is
strongest in the green channel — haemoglobin absorbs green more strongly
than red or blue while green still penetrates the dermis.  Also
establishes spatial averaging over a facial skin region (forehead,
cheek) as the extraction step.

**Implemented in:** `src/engagevr/rppg/methods.py::extract_green`

```
G_n(t) = G(t) / mean(G) - 1
```
followed by linear detrending and band-pass filtering.

Role: an interpretable baseline, not a recommended production method.
It applies no colour-space projection, so illumination change affecting
the green channel is indistinguishable from a pulse.

---

### CHROM -- chrominance-based rPPG

> G. de Haan, V. Jeanne (2013).
> "Robust pulse rate from chrominance-based rPPG."
> *IEEE Transactions on Biomedical Engineering* 60(10):2878-2886.
> DOI: [10.1109/TBME.2013.2266196](https://doi.org/10.1109/TBME.2013.2266196)

**Implemented in:** `src/engagevr/rppg/methods.py::extract_chrom`

Per sliding window of length `l`:

1. Temporal normalization:
   ```
   Rn = R / mean(R),   Gn = G / mean(G),   Bn = B / mean(B)
   ```
2. Chrominance projection:
   ```
   Xs = 3*Rn - 2*Gn
   Ys = 1.5*Rn + Gn - 1.5*Bn
   ```
3. Band-pass filter both to the pulse band, giving `Xf`, `Yf`.
4. Alpha tuning — this is what cancels the motion-induced specular
   component:
   ```
   alpha = std(Xf) / std(Yf)
   S     = Xf - alpha * Yf
   ```
5. Overlap-add: mean-remove `S`, apply a Hann taper, add into the output
   at its offset, 50 % hop.

**Window:** 1.6 s (the paper's 32 samples at 20 fps), 50 % overlap, Hann
taper.  The sample count is derived from the actual sampling rate so the
physical window length is preserved at any frame rate.

**Documented deviations:**
- The paper's fixed 32-point FIR band-pass is replaced by the configured
  Butterworth SOS band-pass, so all three methods share one documented
  pulse band from configuration.
- The paper's optional fixed skin-tone standardisation step is **not**
  applied.  Per-window temporal normalization already removes the
  stationary colour, and a fixed reference skin tone would embed an
  assumption about skin colour that this project deliberately avoids.

This project's pulse band (0.7-4.0 Hz = 42-240 BPM) follows the 40-240
BPM range used in this paper.

---

### POS -- plane orthogonal to skin

> W. Wang, A. C. den Brinker, S. Stuijk, G. de Haan (2017).
> "Algorithmic principles of remote PPG."
> *IEEE Transactions on Biomedical Engineering* 64(7):1479-1491.
> DOI: [10.1109/TBME.2016.2609282](https://doi.org/10.1109/TBME.2016.2609282)

**Implemented in:** `src/engagevr/rppg/methods.py::extract_pos`

Algorithm 1, as published:

```
l = round(1.6 * fs)
H = zeros(N)
for n in 0 .. N-1:
    m = n - l + 1
    if m >= 0:
        Cn = C[m:n+1] / mean(C[m:n+1], axis=0)     # 1. normalize
        S  = P @ Cn.T                              # 2. project
        h  = S[0] + (std(S[0]) / std(S[1])) * S[1] # 3. tune
        H[m:n+1] += h - mean(h)                    # 4. overlap-add
```

with

```
P = [[ 0, 1, -1],
     [-2, 1,  1]]
```

Both rows of `P` annihilate the `[1, 1, 1]` direction, so the projection
is insensitive to intensity-only variation from specular reflection and
subject-camera distance change.  Step 3's ratio-of-standard-deviations
weight cancels the residual motion component present on both axes.

**Windowing:** unit stride, so consecutive windows overlap by `l - 1`
samples; each window's mean-removed result is *added* into the output at
its own offset with no taper, exactly as in Algorithm 1.  The first
`l - 1` output samples receive fewer contributions; this edge effect is
inherent to the published algorithm and is not corrected.

**Documented deviations:** none to the algorithm.  The paper's separate
band-pass stage is applied by the calling layer rather than inside the
function, so all three methods share one configured filter.

Also provides the skin-reflection model that the synthetic trace
generator's structure is based on.

---

### Spectral heart-rate estimation

> P. D. Welch (1967).
> "The use of fast Fourier transform for the estimation of power spectra:
> a method based on time averaging over short, modified periodograms."
> *IEEE Transactions on Audio and Electroacoustics* 15(2):70-73.
> DOI: [10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901)

**Implemented in:** `src/engagevr/rppg/heart_rate.py` via
`scipy.signal.welch`.  Segment averaging trades frequency resolution for
variance reduction, which is the right trade for a noisy camera-derived
signal where a single periodogram is dominated by its own estimation
variance.  The resulting resolution is reported with every estimate,
because it bounds how precisely any BPM value can be stated.

---

### Filter design

> Butterworth filter design and second-order-section realisation, via
> `scipy.signal.butter(..., output="sos")` and
> `scipy.signal.sosfiltfilt`.

SOS is used rather than direct-form `(b, a)` coefficients: direct-form
realisations of even moderate-order band-pass filters are numerically
ill-conditioned at the narrow relative bandwidths involved here (0.7-4.0
Hz against a 15 Hz Nyquist), and fail silently rather than loudly.

`sosfiltfilt` is zero-phase, so the pulse waveform is not shifted
relative to its timestamps.  It is valid for **offline window
processing** only — it is non-causal and cannot be used for streaming.

---

## Behavioural proxies (Milestone 2)

### Eye Aspect Ratio

> T. Soukupova, J. Cech (2016).
> "Real-Time Eye Blink Detection using Facial Landmarks."
> *21st Computer Vision Winter Workshop.*

**Implemented in:** `src/engagevr/face/features.py`

### Blur / focus measure

> S. Pertuz, D. Puig, M. A. Garcia (2013).
> "Analysis of focus measure operators for shape-from-focus."
> *Pattern Recognition* 46(5):1415-1432.

**Implemented in:** `src/engagevr/capture/quality.py` (Laplacian
variance).

### Head pose

OpenCV `solvePnP` with a canonical 3D face model.

**Implemented in:** `src/engagevr/head_pose/estimator.py`

### Facial landmarks

MediaPipe Face Mesh / FaceLandmarker Tasks Vision API (Apache 2.0).

**Implemented in:** `src/engagevr/face/landmarker.py`

---

## Deferred, with the reason

### Heart-rate variability (HRV)

**Not implemented.**  Time-domain HRV (SDNN, RMSSD, pNN50) and
inter-beat intervals require beat-to-beat timing accurate to
milliseconds, which requires validated individual peak detection on a
waveform whose morphology is trustworthy.  A spectral pulse-rate
estimate provides no per-beat timing at all.

Before any HRV feature is implemented, the following must be established
from primary literature and recorded here:

- minimum recording duration per HRV feature,
- minimum sampling rate for adequate IBI resolution,
- waveform-quality criteria for accepting a beat,
- peak-validation and artifact-correction procedure,
- the error that camera-derived IBIs introduce relative to contact PPG
  or ECG.

Deriving intervals from an unvalidated camera waveform would produce
numbers that look precise and mean nothing.

### Learning-based rPPG

**Not implemented.**  The project rule is interpretable baselines before
deep learning.  Classical methods must be evaluated on a real public
dataset first.
</content>

## Milestone 4 — Protocol, Timing, and Task

### Clock-offset estimation from round trips

The heartbeat diagnostic uses the standard four-timestamp formulation. For a
request sent at client time `t0`, received by the server at `t1`, answered at
`t2`, and received back by the client at `t3`:

```
rtt    = (t3 - t0) - (t2 - t1)
offset = ((t1 - t0) + (t2 - t3)) / 2
```

with `|true offset − offset| ≤ rtt / 2` **under an assumption of symmetric
path delay**.

Reference: D. Mills, J. Martin (ed.), J. Burbank, W. Kasch, *Network Time
Protocol Version 4: Protocol and Algorithms Specification*, RFC 5905, IETF,
June 2010, §8 ("On-Wire Protocol").
DOI: [10.17487/RFC5905](https://doi.org/10.17487/RFC5905)

**Deviations and scope in this repository:**

- No clock is ever adjusted or corrected. The formula is used for
  *diagnostics only*.
- The symmetric-delay assumption is **not verifiable** here, so `rtt / 2` is
  recorded as a bound *under an unverified assumption* rather than as an
  error bar, and `symmetric_delay_assumed = True` travels with every
  estimate.
- No filtering, clock discipline, or stratum logic from RFC 5905 is
  implemented. A single round trip yields a single estimate.
- Monotonic clocks from different processes are never compared: their
  origins are unrelated.

### JSON Lines

The event stream uses the JSON Lines convention — one complete JSON value per
line, UTF-8, newline-delimited. See <https://jsonlines.org/>. The property
this project relies on is that records are independent, so a torn final line
from an interrupted write affects only itself and every previously flushed
line remains readable.

### Reaction-time distribution used by the simulator

The simulator draws fabricated reaction times from a lognormal shape.

**This is not a model of human reaction times.** No parameter was fitted to
any dataset, no distributional claim is made, and the values are not
comparable to any published reaction-time norm. The shape was chosen only
because it is strictly positive and right-skewed, so that fabricated values
do not look like a symmetric artefact. A literature-grounded response model
would require an experimental design, a pilot, and approval, none of which
exist — see `docs/LIMITATIONS.md`.

### Task paradigm

The desktop task is a **neutral stimulus–response task**: three abstract
shapes, one response key each, a configurable response deadline. The
specification lists sequence-classification, working-memory, visual-search,
and response-inhibition paradigms as candidates for a future, properly
designed task.

No published paradigm is claimed to be implemented here. The current task is
a **software telemetry source** used to exercise the protocol, the backend,
the storage layer, and the replay path. Selecting and justifying an actual
cognitive paradigm from primary literature is future work that must precede
any psychological interpretation of its output.

## Milestone 5 — Evaluation methodology

### Grouped cross-validation

Grouped k-fold and stratified grouped k-fold are used because several
windows share a session and several sessions can share a participant.

> scikit-learn developers. "Cross-validation iterators for grouped data."
> *scikit-learn User Guide*, §3.1.2.3.
> https://scikit-learn.org/stable/modules/cross_validation.html

**Implementation note.** `StratifiedGroupKFold` is used only when every
class appears in at least `n_splits` distinct groups; feasibility is
checked in `engagevr.training.splits._stratification_feasible` rather than
left to the splitter's behaviour, and the fallback to `GroupKFold` records
its reason. Ungrouped splitters are deliberately not imported (DEC-041).

### Probability calibration

Platt (sigmoid) scaling:

> J. Platt (1999). "Probabilistic outputs for support vector machines and
> comparisons to regularized likelihood methods." *Advances in Large
> Margin Classifiers*, MIT Press, pp. 61-74.

Isotonic calibration:

> B. Zadrozny, C. Elkan (2002). "Transforming classifier scores into
> accurate multiclass probability estimates." *Proceedings of the 8th ACM
> SIGKDD International Conference on Knowledge Discovery and Data Mining*.
> DOI 10.1145/775047.775151

**Implementation note.** `CalibratedClassifierCV(cv="prefit")` was
deprecated in scikit-learn 1.6 and removed in 1.8; this project wraps the
fitted estimator in `sklearn.frozen.FrozenEstimator`, which is the
supported API on the installed 1.9.0 (DEC-042). The minimum-data
thresholds for isotonic calibration (50 rows, 10 per class) are this
project's engineering choice, not a published threshold, and are
documented as such.

### Expected calibration error

> C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger (2017). "On calibration of
> modern neural networks." *Proceedings of the 34th International
> Conference on Machine Learning (ICML)*, PMLR 70:1321-1330.

The equal-width binned estimator used here follows the definition in that
paper:

```
ECE = Σ_m (|B_m| / n) · | accuracy(B_m) − confidence(B_m) |
```

**Deviations, stated.** Bins are equal-width over the maximum predicted
class probability, `M = 10` by default and recorded on every document.
Empty bins contribute nothing and are retained with `count = 0` so the
binning is reconstructible from the stored result. ECE is a *binned
estimator*: its value depends on the bin count, which is why the count is
persisted next to the number.

### Brier score

> G. W. Brier (1950). "Verification of forecasts expressed in terms of
> probability." *Monthly Weather Review* 78(1):1-3.
> DOI 10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

**Convention, stated.** The multiclass form used here is
`mean_i ‖ p_i − onehot(y_i) ‖²`, i.e. the mean squared Euclidean distance
between the predicted probability vector and the one-hot true vector,
range `[0, 2]`. The alternative convention (half this value) is not used.
The definition is carried in the schema field that stores the number, so a
persisted result is interpretable without this document.

### Permutation importance

> L. Breiman (2001). "Random forests." *Machine Learning* 45(1):5-32.
> DOI 10.1023/A:1010933404324

> A. Fisher, C. Rudin, F. Dominici (2019). "All models are wrong, but many
> are useful: learning a variable's importance by studying an entire class
> of prediction models simultaneously." *Journal of Machine Learning
> Research* 20(177):1-81.

**Implementation note.** Importance is computed on **held-out fold data**,
never on training data, and impurity-based importance is not used: it is
biased toward high-cardinality features and is computed on the training
set. Permutation is applied to the estimator's post-preprocessing input
matrix; the fitted transformer is deterministic, so this is equivalent to
permuting through the pipeline and avoids re-running the
`ColumnTransformer` for every (feature × repeat) permutation.

Fisher, Rudin and Dominici document the correlated-feature caveat this
project repeats: with near-duplicate inputs, permuting one alone
understates both.

### Histogram-based gradient boosting

> T. Chen, C. Guestrin (2016). "XGBoost: a scalable tree boosting system."
> *Proceedings of the 22nd ACM SIGKDD International Conference on
> Knowledge Discovery and Data Mining*. DOI 10.1145/2939672.2939785

> G. Ke et al. (2017). "LightGBM: a highly efficient gradient boosting
> decision tree." *Advances in Neural Information Processing Systems 30*.

scikit-learn's `HistGradientBoosting*` estimators implement the
histogram-based algorithm these papers describe. Neither XGBoost nor
LightGBM is a dependency of this project (DEC-037); the references are
given because they are the primary description of the algorithm actually
being used.

### Deferred, with the reason

**SHAP** (Lundberg & Lee, 2017) is not used in this milestone. Shapley
attributions are expensive, are frequently over-read as causal, and would
add a dependency to explain models that have not been evaluated against
any real label. Coefficients and permutation importance answer the
question this milestone actually has: which inputs did a model lean on in
held-out folds.

## Milestone 6: multimodal fusion

No published fusion algorithm is reimplemented in this milestone. The
strategies here are textbook combination rules, implemented directly and
documented in full in `docs/MULTIMODAL_FUSION.md`:

| Strategy | What it is |
|---|---|
| early feature fusion | concatenation of modality feature blocks before one estimator is fitted |
| uniform late fusion | unweighted mean of the available experts' outputs |
| quality-aware late fusion | weighted mean, weights from availability and recorded signal quality, per the equation in `docs/MULTIMODAL_FUSION.md` |
| validation-derived late fusion | weighted mean, weights from bounded skill scores computed on inner validation groups |
| stacked fusion | an interpretable meta-estimator fitted on out-of-fold expert predictions |

The stacking design follows the standard leakage-safe construction: a
meta-learner is fitted on predictions generated for held-out inner folds,
never on in-sample predictions. `assert_out_of_fold` enforces the property
independently rather than trusting the construction.

Every scikit-learn estimator used — `LogisticRegression`, `Ridge`,
`StratifiedGroupKFold`, `GroupKFold`, `CalibratedClassifierCV`,
`FrozenEstimator`, `SimpleImputer`, `StandardScaler`, `ColumnTransformer`,
`Pipeline` — was verified against the installed scikit-learn 1.9.0. No new
dependency was added for this milestone.

**No claim is made that any of these strategies is superior on real data.**
Nothing here has been evaluated against a real participant label.

### Personalization

No published personalization algorithm is reimplemented either. Both
corrections are elementary and are stated in full in
`docs/MULTIMODAL_FUSION.md` and on the persisted records themselves.

| Component | What it is |
|---|---|
| personal-baseline normalization | the standard score `z = (x - mu) / sigma`, with the statistics restricted to one subject's earlier windows |
| few-shot regression correction | a per-subject additive bias, `b_s = mean(y_calibration - y_population_prediction)` |
| few-shot classification correction | a Laplace-smoothed, shrunk per-subject log-odds shift applied to the population probabilities and renormalised |

The classification correction is a prior-shift adjustment: it moves the
population probabilities toward the class frequencies the subject actually
exhibited during calibration. Smoothing both the observed and the expected
term identically gives the property the design depends on — the shift is
exactly zero when they agree — and the shrinkage factor `n / (n + kappa)`
is the usual way of trusting a small sample less. Neither `kappa = 5.0` nor
`alpha = 1.0` is taken from a published result; both are engineering
defaults, stated as such, and untuned.

The leave-subject-out protocol with a within-subject chronological
calibration/evaluation boundary is the standard construction for
personalization evaluation. What is enforced here beyond the usual is that
the boundary is in wall-clock time rather than row position, so overlapping
windows cannot straddle it unnoticed.

`docs/RESEARCH_QUESTIONS.md` RQ2 asks whether personalized baselines
outperform population models. **Nothing in this milestone answers it.** No
personalized model here has been fitted to a real participant label, and a
synthetic self-check cannot support a claim in either direction.
