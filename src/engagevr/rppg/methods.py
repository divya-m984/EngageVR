"""Classical rPPG extraction methods: GREEN, CHROM, and POS.

All three share one interface: an ``(N, 3)`` RGB trace sampled at a known
uniform rate in, a 1-D pulse waveform out.  None of them holds mutable
global state, and all are deterministic for deterministic input.

No method is universally superior.  GREEN is the simplest and the most
sensitive to illumination change; CHROM and POS both use colour-space
projections designed to suppress specular and motion-induced intensity
variation, and they differ in which plane they project onto.  Which one
performs best depends on illumination, motion, camera, and subject.
Nothing in this repository establishes a ranking between them.

References
----------
GREEN
    W. Verkruysse, L. O. Svaasand, J. S. Nelson (2008).
    "Remote plethysmographic imaging using ambient light."
    Optics Express 16(26):21434-21445. DOI 10.1364/OE.16.021434

CHROM
    G. de Haan, V. Jeanne (2013).
    "Robust pulse rate from chrominance-based rPPG."
    IEEE Trans. Biomed. Eng. 60(10):2878-2886.
    DOI 10.1109/TBME.2013.2266196

POS
    W. Wang, A. C. den Brinker, S. Stuijk, G. de Haan (2017).
    "Algorithmic principles of remote PPG."
    IEEE Trans. Biomed. Eng. 64(7):1479-1491.
    DOI 10.1109/TBME.2016.2609282
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from engagevr.config import RppgPreprocessingConfig
from engagevr.rppg.errors import RppgUnavailable
from engagevr.rppg.preprocessing import (
    bandpass_filter,
    design_bandpass_sos,
    detrend_linear,
    require_finite,
    require_non_constant,
)
from engagevr.schemas.rppg import RppgMethod, RppgWaveform, UnavailableReason

#: Sliding-window length used by CHROM and POS, in seconds.
#:
#: Both papers use 32 samples at 20 fps = 1.6 s.  The window must be long
#: enough to contain at least one full pulse cycle at the lowest plausible
#: rate while remaining short enough that the skin-tone normalization
#: stays locally valid.
DEFAULT_METHOD_WINDOW_SECONDS = 1.6

#: POS projection matrix (Wang et al. 2017, Algorithm 1).
#:
#: Row 1 = [0, 1, -1] gives G - B; row 2 = [-2, 1, 1] gives -2R + G + B.
#: These two vectors span the plane orthogonal to the normalized skin-tone
#: direction [1, 1, 1], which is what makes the projection insensitive to
#: intensity-only (specular / motion) variation.
POS_PROJECTION: npt.NDArray[np.float64] = np.array(
    [[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]],
    dtype=np.float64,
)

_R, _G, _B = 0, 1, 2
_EPS = 1e-12


def _validate_rgb(rgb: npt.NDArray[np.float64]) -> None:
    """Shared input validation for every method."""
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise RppgUnavailable(
            UnavailableReason.MISSING_CHANNEL,
            f"expected an (N, 3) RGB trace, got shape {rgb.shape}",
        )
    if rgb.shape[0] < 2:
        raise RppgUnavailable(
            UnavailableReason.WINDOW_TOO_SHORT,
            f"{rgb.shape[0]} samples is too few",
        )
    require_finite(rgb)
    for name, col in (("R", _R), ("G", _G), ("B", _B)):
        if float(np.ptp(rgb[:, col])) <= _EPS:
            raise RppgUnavailable(
                UnavailableReason.CONSTANT_SIGNAL,
                f"{name} channel is constant",
            )


def extract_green(
    rgb: npt.NDArray[np.float64],
    fs: float,
) -> npt.NDArray[np.float64]:
    """GREEN baseline: the mean-normalized, detrended green channel.

    Verkruysse et al. (2008) found the plethysmographic modulation to be
    strongest in the green channel, because haemoglobin absorbs green
    light more strongly than red or blue while green still penetrates the
    dermis.  The method takes the green trace, divides by its temporal
    mean to express it as a relative modulation, and removes a linear
    trend::

        G_n(t) = G(t) / mean(G) - 1

    This is deliberately the least sophisticated method available.  It
    applies no colour-space projection, so any illumination change that
    affects the green channel is indistinguishable from a pulse.  It
    exists as an interpretable reference point against which CHROM and
    POS can be compared, not as a recommended production method.
    """
    _validate_rgb(rgb)
    del fs  # GREEN is rate-independent; kept for interface symmetry.
    green = rgb[:, _G]
    mean = float(np.mean(green))
    if abs(mean) < _EPS:
        raise RppgUnavailable(
            UnavailableReason.CONSTANT_SIGNAL,
            "green channel has a zero temporal mean",
        )
    normalized = green / mean - 1.0
    return detrend_linear(normalized.reshape(-1, 1)).reshape(-1)


def extract_chrom(
    rgb: npt.NDArray[np.float64],
    fs: float,
    *,
    band_low_hz: float,
    band_high_hz: float,
    filter_order: int,
    window_seconds: float = DEFAULT_METHOD_WINDOW_SECONDS,
) -> npt.NDArray[np.float64]:
    """CHROM (de Haan & Jeanne, 2013).

    Equations as implemented, per sliding window of length ``l``:

    1. Temporal normalization -- each channel divided by its mean over
       the window::

           Rn = R / mean(R),  Gn = G / mean(G),  Bn = B / mean(B)

    2. Chrominance projection (Section III of the paper)::

           Xs = 3*Rn - 2*Gn
           Ys = 1.5*Rn + Gn - 1.5*Bn

       The coefficients follow from the standardised skin-tone axis
       assumption; the two signals isolate two chrominance directions
       that respond differently to specular reflection.

    3. Band-pass filter both projections to the pulse band, giving
       ``Xf`` and ``Yf``.

    4. Alpha tuning, which is what cancels the motion-induced specular
       component::

           alpha = std(Xf) / std(Yf)
           S = Xf - alpha * Yf

    5. Overlap-add reconstruction: the window signal is mean-removed,
       multiplied by a Hann window, and added into the output at its
       original offset, with a 50% hop.

    Windowing and overlap assumptions
    ---------------------------------
    The paper uses 32-sample windows at 20 fps (1.6 s) with 50% overlap
    and a Hann taper.  This implementation keeps 1.6 s and 50% overlap
    but derives the sample count from the actual sampling rate, so the
    physical window length is preserved at any frame rate.

    Deviations from the paper
    -------------------------
    - The paper's fixed 32-point FIR band-pass is replaced by the
      configured Butterworth SOS band-pass, so that the pulse band is a
      single documented configuration value shared across all three
      methods rather than being hard-coded per method.
    - The paper's optional skin-tone standardisation step (dividing by a
      fixed reference skin colour before projection) is **not** applied.
      Per-window temporal normalization already removes the stationary
      colour, and applying a fixed reference skin tone would embed an
      assumption about skin colour that this project explicitly avoids.
    """
    _validate_rgb(rgb)
    n = rgb.shape[0]
    step = _window_length(fs, window_seconds, n)
    sos = design_bandpass_sos(band_low_hz, band_high_hz, fs, filter_order)

    hop = max(1, step // 2)
    window_fn = np.hanning(step)
    output = np.zeros(n, dtype=np.float64)
    contributed = False

    for start in range(0, n - step + 1, hop):
        stop = start + step
        block = rgb[start:stop, :]

        means = np.mean(block, axis=0)
        if np.any(np.abs(means) < _EPS):
            continue
        norm = block / means

        xs = 3.0 * norm[:, _R] - 2.0 * norm[:, _G]
        ys = 1.5 * norm[:, _R] + norm[:, _G] - 1.5 * norm[:, _B]

        try:
            xf = bandpass_filter(xs.reshape(-1, 1), sos).reshape(-1)
            yf = bandpass_filter(ys.reshape(-1, 1), sos).reshape(-1)
        except RppgUnavailable:
            # Window shorter than the filter pad: no CHROM window can be
            # filtered at this rate/order combination.
            raise

        std_y = float(np.std(yf))
        if std_y < _EPS:
            continue
        alpha = float(np.std(xf)) / std_y
        signal = xf - alpha * yf
        signal = signal - float(np.mean(signal))

        output[start:stop] += window_fn * signal
        contributed = True

    if not contributed:
        raise RppgUnavailable(
            UnavailableReason.CONSTANT_SIGNAL,
            "no CHROM window produced a usable chrominance signal",
        )
    return output


def extract_pos(
    rgb: npt.NDArray[np.float64],
    fs: float,
    *,
    window_seconds: float = DEFAULT_METHOD_WINDOW_SECONDS,
) -> npt.NDArray[np.float64]:
    """POS -- plane orthogonal to skin (Wang et al., 2017, Algorithm 1).

    Implemented as published, with a window sliding one sample at a time::

        l = round(1.6 * fs)
        H = zeros(N)
        for n in 0 .. N-1:
            m = n - l + 1
            if m >= 0:
                Cn = C[m:n+1] / mean(C[m:n+1], axis=0)   # step 1: normalize
                S  = P @ Cn.T                            # step 2: project
                h  = S[0] + (std(S[0]) / std(S[1])) * S[1]   # step 3: tune
                H[m:n+1] += h - mean(h)                  # step 4: overlap-add

    with the projection matrix::

        P = [[ 0, 1, -1],
             [-2, 1,  1]]

    Step 2 projects the temporally normalized RGB onto the plane
    orthogonal to the ``[1, 1, 1]`` skin-tone direction, which is what
    removes intensity-only variation caused by specular reflection and
    subject-camera distance change.  Step 3 combines the two projection
    axes with a ratio-of-standard-deviations weight, so that the
    remaining motion component present in both axes cancels.

    Windowing and overlap-add
    -------------------------
    The stride is one sample, so consecutive windows overlap by
    ``l - 1`` samples.  Each window's mean-removed result is *added* into
    the output at its own offset -- no taper is applied, matching
    Algorithm 1.  The first ``l - 1`` output samples receive fewer
    contributions than later samples; this edge effect is inherent to the
    published algorithm and is not corrected here.

    Deviations from the paper
    -------------------------
    None. The projection matrix, normalization, weighting, window length
    (1.6 s), unit stride, and overlap-add are exactly as published. The
    paper's separate pre/post band-pass stage is applied by the caller
    (:func:`extract_waveform`) rather than inside this function, so that
    all three methods share one configured filter.
    """
    _validate_rgb(rgb)
    n = rgb.shape[0]
    length = _window_length(fs, window_seconds, n)

    output = np.zeros(n, dtype=np.float64)
    contributed = False

    for end in range(length - 1, n):
        start = end - length + 1
        block = rgb[start : end + 1, :]

        means = np.mean(block, axis=0)
        if np.any(np.abs(means) < _EPS):
            continue
        norm = block / means

        projected = POS_PROJECTION @ norm.T  # shape (2, length)
        s1 = projected[0, :]
        s2 = projected[1, :]

        std2 = float(np.std(s2))
        if std2 < _EPS:
            # Degenerate window: fall back to the first axis alone rather
            # than dividing by ~0, which would inject a huge artifact.
            h = s1
        else:
            h = s1 + (float(np.std(s1)) / std2) * s2

        output[start : end + 1] += h - float(np.mean(h))
        contributed = True

    if not contributed:
        raise RppgUnavailable(
            UnavailableReason.CONSTANT_SIGNAL,
            "no POS window produced a usable projection",
        )
    return output


def _window_length(fs: float, window_seconds: float, n_samples: int) -> int:
    """Convert a window duration in seconds to a sample count."""
    if fs <= 0.0:
        raise RppgUnavailable(
            UnavailableReason.SAMPLING_RATE_UNKNOWN,
            "sampling rate must be positive",
        )
    length = round(window_seconds * fs)
    if length < 2:
        raise RppgUnavailable(
            UnavailableReason.WINDOW_TOO_SHORT,
            f"method window of {window_seconds}s at {fs} Hz is under 2 samples",
        )
    if length > n_samples:
        raise RppgUnavailable(
            UnavailableReason.WINDOW_TOO_SHORT,
            f"trace of {n_samples} samples is shorter than the "
            f"{length}-sample method window",
        )
    return length


def extract_waveform(
    method: RppgMethod,
    rgb: npt.NDArray[np.float64],
    timestamps: npt.NDArray[np.float64],
    fs: float,
    cfg: RppgPreprocessingConfig,
    *,
    method_window_seconds: float = DEFAULT_METHOD_WINDOW_SECONDS,
) -> RppgWaveform:
    """Run one rPPG method and band-pass the result to the pulse band.

    Returns an ``available=False`` waveform carrying the reason whenever
    the input or the configuration makes a defensible estimate
    impossible.  It never returns a NaN-filled waveform.
    """
    params: dict[str, float | int | str | bool] = {
        "method_window_seconds": method_window_seconds,
        "band_low_hz": cfg.pulse_band_low_hz,
        "band_high_hz": cfg.pulse_band_high_hz,
        "filter_order": cfg.filter_order,
        "detrend": cfg.detrend,
        "normalize": cfg.normalize,
        "sampling_rate_hz": fs,
    }

    def unavailable(reason: UnavailableReason) -> RppgWaveform:
        return RppgWaveform(
            method=method,
            available=False,
            reason=reason,
            sampling_rate_hz=fs if fs > 0 else None,
            band_low_hz=cfg.pulse_band_low_hz,
            band_high_hz=cfg.pulse_band_high_hz,
            filter_order=cfg.filter_order,
            method_params=params,
        )

    try:
        if method is RppgMethod.GREEN:
            raw = extract_green(rgb, fs)
        elif method is RppgMethod.CHROM:
            raw = extract_chrom(
                rgb,
                fs,
                band_low_hz=cfg.pulse_band_low_hz,
                band_high_hz=cfg.pulse_band_high_hz,
                filter_order=cfg.filter_order,
                window_seconds=method_window_seconds,
            )
        elif method is RppgMethod.POS:
            raw = extract_pos(rgb, fs, window_seconds=method_window_seconds)
        else:  # pragma: no cover - StrEnum is exhaustive
            raise RppgUnavailable(
                UnavailableReason.MISSING_CHANNEL, f"unsupported method {method}"
            )

        sos = design_bandpass_sos(
            cfg.pulse_band_low_hz,
            cfg.pulse_band_high_hz,
            fs,
            cfg.filter_order,
        )
        filtered = bandpass_filter(raw.reshape(-1, 1), sos).reshape(-1)
        require_finite(filtered)
        require_non_constant(filtered)
    except RppgUnavailable as exc:
        return unavailable(exc.reason)

    duration = float(timestamps[-1] - timestamps[0]) if timestamps.size >= 2 else 0.0
    return RppgWaveform(
        method=method,
        available=True,
        reason=None,
        sampling_rate_hz=fs,
        timestamps=[float(t) for t in timestamps],
        values=[float(v) for v in filtered],
        n_samples=int(filtered.size),
        duration_s=duration,
        band_low_hz=cfg.pulse_band_low_hz,
        band_high_hz=cfg.pulse_band_high_hz,
        filter_order=cfg.filter_order,
        method_params=params,
    )


__all__ = [
    "DEFAULT_METHOD_WINDOW_SECONDS",
    "POS_PROJECTION",
    "extract_chrom",
    "extract_green",
    "extract_pos",
    "extract_waveform",
]
