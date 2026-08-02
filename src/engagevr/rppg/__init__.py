"""Remote photoplethysmography (rPPG) signal-processing pipeline.

Classical, interpretable methods only: GREEN (Verkruysse et al., 2008),
CHROM (de Haan & Jeanne, 2013), and POS (Wang et al., 2017).  No deep
learning, no learned model, no HRV.

Heart-rate outputs are signal-processing estimates from camera data.
They are not medical measurements and they are not engagement or
cognitive-load values.
"""

from engagevr.rppg.errors import RppgError, RppgUnavailable
from engagevr.rppg.evaluation import aggregate_metrics, per_subject_metrics
from engagevr.rppg.heart_rate import estimate_heart_rate
from engagevr.rppg.methods import (
    extract_chrom,
    extract_green,
    extract_pos,
    extract_waveform,
)
from engagevr.rppg.quality import assess_window_quality
from engagevr.rppg.roi import extract_combined, extract_region, extract_regions
from engagevr.rppg.trace import (
    build_synthetic_window,
    build_window,
    generate_synthetic_rgb_trace,
    iter_windows,
)
from engagevr.rppg.window import prepare_window, process_window

__all__ = [
    "RppgError",
    "RppgUnavailable",
    "aggregate_metrics",
    "assess_window_quality",
    "build_synthetic_window",
    "build_window",
    "estimate_heart_rate",
    "extract_chrom",
    "extract_combined",
    "extract_green",
    "extract_pos",
    "extract_region",
    "extract_regions",
    "extract_waveform",
    "generate_synthetic_rgb_trace",
    "iter_windows",
    "per_subject_metrics",
    "prepare_window",
    "process_window",
]
