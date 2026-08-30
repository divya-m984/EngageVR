"""Canonical hashing for configuration, splits, and feature schemas.

One hashing convention is used everywhere in this repository: SHA-256
over a canonical UTF-8 rendering, with no wall clock and no absolute path
participating.  Milestone 5 established it for datasets and artifacts
(:mod:`engagevr.training.artifacts`,
:mod:`engagevr.features.assembly`); this module extends the same
convention to the three things Milestone 10 needs to identify and does
not invent a second one.

Canonical rendering is ``json.dumps(payload, sort_keys=True,
separators=(",", ":"))``.  Sorting keys means a reordered document
fingerprints identically; the compact separators mean whitespace does
not participate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from engagevr.config import EngageVRConfig
from engagevr.schemas.mlops import ConfigurationVersion
from engagevr.training.artifacts import engagevr_version, sha256_file

#: Repository root, resolved the same way :mod:`engagevr.config` resolves it.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

#: Configuration paths removed before fingerprinting, and why.
#:
#: These describe *where this machine keeps things* rather than *what the
#: pipeline computed.  Leaving them in would give two identical runs on
#: two machines different configuration fingerprints, which would make the
#: fingerprint useless for the one job it has.  Nothing that can change a
#: number is excluded.
EXCLUDED_CONFIG_PATHS: dict[str, str] = {
    "rppg.datasets.ubfc_rppg_root": (
        "an absolute path to a locally obtained public dataset. It differs "
        "on every machine, is never fetched by this software, and takes no "
        "part in any synthetic pipeline."
    ),
    "logging.file": (
        "a local log destination. It changes where diagnostics are written "
        "and nothing about what was computed."
    ),
    "capture.camera_index": (
        "a device index that identifies one webcam on one machine. No "
        "modelling, tracking, or packaging stage reads it."
    ),
}

#: Sections snapshotted alongside the fingerprint, so a reviewer can see
#: the settings that shaped a run without re-resolving the whole file.
SNAPSHOT_SECTIONS: tuple[str, ...] = (
    "project",
    "features",
    "training",
    "fusion",
    "personalization",
    "uncertainty",
    "adaptation",
    "mlops",
)


def canonical_json(payload: object) -> str:
    """Canonical UTF-8 rendering used for every fingerprint here."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    """SHA-256 of a string's UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_payload(payload: object) -> str:
    """SHA-256 over the canonical rendering of ``payload``."""
    return sha256_text(canonical_json(payload))


def repository_relative(path: Path | str) -> str:
    """Render ``path`` relative to the repository, for storage in a document.

    Absolute paths are machine facts, so they are never persisted in a
    record whose identity must survive being copied to another machine.
    The repository root is tried first and the working directory second;
    a path under neither degrades to its own file name rather than
    leaking a home directory into an artifact.
    """
    resolved = Path(path).resolve()
    for base in (REPOSITORY_ROOT, Path.cwd().resolve()):
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            continue
    return resolved.name


def _remove_path(document: dict[str, Any], dotted: str) -> None:
    parts = dotted.split(".")
    cursor: Any = document
    for part in parts[:-1]:
        if not isinstance(cursor, dict) or part not in cursor:
            return
        cursor = cursor[part]
    if isinstance(cursor, dict):
        cursor.pop(parts[-1], None)


def normalize_config(config: EngageVRConfig) -> dict[str, Any]:
    """The effective configuration, JSON-rendered, environment stripped.

    ``config`` is the *resolved* model, so every default that was not
    written in the YAML file is present here.  That is the point: the
    fingerprint must describe the settings the code actually ran under,
    not the subset somebody happened to type.
    """
    document: dict[str, Any] = config.model_dump(mode="json")
    for dotted in EXCLUDED_CONFIG_PATHS:
        _remove_path(document, dotted)
    return document


def config_fingerprint(config: EngageVRConfig) -> str:
    """SHA-256 over the normalized effective configuration."""
    return sha256_payload(normalize_config(config))


def build_configuration_version(
    config: EngageVRConfig,
    *,
    sections: Sequence[str] = SNAPSHOT_SECTIONS,
) -> ConfigurationVersion:
    """Build the persisted configuration-version record."""
    normalized = normalize_config(config)
    snapshots = {name: normalized[name] for name in sections if name in normalized}
    return ConfigurationVersion(
        config_fingerprint=sha256_payload(normalized),
        excluded_paths=tuple(sorted(EXCLUDED_CONFIG_PATHS)),
        exclusion_reasons=dict(EXCLUDED_CONFIG_PATHS),
        engagevr_version=engagevr_version(),
        project_config_version=config.project.version,
        section_snapshots=snapshots,
    )


def split_fingerprint(splits: Mapping[str, Any]) -> str:
    """SHA-256 over the group membership of every fold.

    Row counts and target distributions are deliberately excluded: they
    are consequences of the group assignment, and including them would
    make the fingerprint change when a dataset grew without the split
    design changing at all.  Groups are sorted, because a fold is a set.
    """
    folds = splits.get("folds", ())
    payload = {
        "strategy": splits.get("strategy"),
        "group_field": splits.get("group_field"),
        "n_splits": splits.get("n_splits"),
        "random_seed": splits.get("random_seed"),
        "folds": [
            {
                "fold_index": fold.get("fold_index"),
                "train_groups": sorted(fold.get("train_groups", ())),
                "calibration_groups": sorted(fold.get("calibration_groups", ())),
                "test_groups": sorted(fold.get("test_groups", ())),
            }
            for fold in folds
        ],
    }
    return sha256_payload(payload)


def feature_schema_fingerprint(
    feature_set: Sequence[str], *, catalog_version: str
) -> str:
    """SHA-256 over the ordered predictor columns and the catalog version.

    Order participates: a reordered predictor matrix is a different
    schema, and a linear model's coefficients are read positionally.
    """
    return sha256_payload(
        {
            "feature_catalog_version": catalog_version,
            "feature_set": list(feature_set),
        }
    )


def file_checksum(path: Path) -> str:
    """SHA-256 of a file, via the Milestone 5 implementation."""
    return sha256_file(path)


__all__ = [
    "EXCLUDED_CONFIG_PATHS",
    "REPOSITORY_ROOT",
    "SNAPSHOT_SECTIONS",
    "build_configuration_version",
    "canonical_json",
    "config_fingerprint",
    "feature_schema_fingerprint",
    "file_checksum",
    "normalize_config",
    "repository_relative",
    "sha256_payload",
    "sha256_text",
    "split_fingerprint",
]
