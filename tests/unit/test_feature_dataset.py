"""Dataset assembly, fingerprinting, and privacy tests.

The fingerprint is the mechanism that makes "reproducible from the same
data, configuration, and seed" checkable rather than asserted, so these
tests pin down exactly what it does and does not respond to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engagevr.features.assembly import (
    DatasetAssemblyError,
    build_dataset_metadata,
    catalog_path,
    compute_fingerprint,
    dataset_columns,
    metadata_path,
    read_dataset_metadata,
    read_dataset_table,
    rows_to_table,
    sort_rows,
    write_dataset,
)
from engagevr.features.catalog import FEATURE_CATALOG
from engagevr.features.synthetic import (
    SyntheticDatasetConfig,
    generate_synthetic_dataset,
)
from engagevr.features.validation import (
    DatasetValidationError,
    assert_no_identifier_values,
    assert_no_identity_columns,
    validate_feature_windows,
)
from engagevr.schemas.features import (
    FEATURE_SCHEMA_VERSION,
    SYNTHETIC_LABEL,
    FeatureCatalog,
    FeatureWindow,
    SubjectKind,
)
from engagevr.schemas.session import DataSource
from engagevr.schemas.targets import TargetName

TARGETS = list(TargetName)


def fingerprint(rows: tuple[FeatureWindow, ...], **overrides: object) -> str:
    kwargs: dict[str, object] = {
        "window_duration_seconds": 10.0,
        "window_step_seconds": 10.0,
        "windows_overlap": False,
    }
    kwargs.update(overrides)
    return compute_fingerprint(rows, FEATURE_CATALOG, TARGETS, **kwargs)  # type: ignore[arg-type]


class TestDeterministicBuild:
    def test_two_builds_of_one_seed_are_identical(
        self, m5_synthetic_config: SyntheticDatasetConfig
    ) -> None:
        first = generate_synthetic_dataset(m5_synthetic_config)
        second = generate_synthetic_dataset(m5_synthetic_config)
        assert fingerprint(first) == fingerprint(second)

    def test_parquet_bytes_round_trip(
        self, m5_dataset: Path, m5_synthetic_rows: tuple[FeatureWindow, ...]
    ) -> None:
        table = read_dataset_table(m5_dataset)
        assert table.num_rows == len(m5_synthetic_rows)
        assert list(table.schema.names) == list(
            dataset_columns(FEATURE_CATALOG, TARGETS)
        )

    def test_schema_round_trip_preserves_nulls_and_values(
        self, m5_dataset: Path, m5_synthetic_rows: tuple[FeatureWindow, ...]
    ) -> None:
        table = read_dataset_table(m5_dataset).to_pandas()
        ordered = sort_rows(m5_synthetic_rows)
        for position in (0, len(ordered) // 2, len(ordered) - 1):
            row = ordered[position]
            record = table.iloc[position]
            assert record["window_id"] == row.window_id
            for name, value in row.features.items():
                stored = record[f"feat__{name}"]
                if value is None:
                    assert stored != stored or stored is None  # NaN or None
                else:
                    assert float(stored) == pytest.approx(value)

    def test_metadata_document_round_trips(self, m5_dataset: Path) -> None:
        metadata = read_dataset_metadata(m5_dataset)
        assert metadata.dataset_schema_version == FEATURE_SCHEMA_VERSION
        assert metadata.feature_catalog_version == FEATURE_CATALOG.version
        assert metadata.row_count > 0
        assert len(metadata.dataset_fingerprint) == 64

    def test_catalog_snapshot_is_written_beside_the_dataset(
        self, m5_dataset: Path
    ) -> None:
        snapshot = catalog_path(m5_dataset)
        assert snapshot.exists()
        with snapshot.open() as handle:
            restored = FeatureCatalog.model_validate(json.load(handle))
        assert restored.names() == FEATURE_CATALOG.names()

    def test_origin_is_inspectable_without_opening_a_model_file(
        self, m5_dataset: Path
    ) -> None:
        with metadata_path(m5_dataset).open() as handle:
            document = json.load(handle)
        assert document["data_source_counts"] == {"synthetic": document["row_count"]}
        assert document["random_seed"] == 42
        assert document["scientific_evaluation_eligible"] is False
        assert any("SYNTHETIC" in d for d in document["disclaimers"])


class TestFingerprintSensitivity:
    def test_fingerprint_changes_when_a_value_changes(
        self, m5_synthetic_rows: tuple[FeatureWindow, ...]
    ) -> None:
        original = fingerprint(m5_synthetic_rows)
        row = m5_synthetic_rows[0]
        name = next(k for k, v in row.features.items() if v is not None)
        mutated = row.model_copy(
            update={
                "features": {**row.features, name: float(row.features[name] or 0.0) + 1}
            }
        )
        changed = (mutated, *m5_synthetic_rows[1:])
        assert fingerprint(changed) != original

    def test_fingerprint_changes_on_a_one_bit_float_difference(
        self, m5_synthetic_rows: tuple[FeatureWindow, ...]
    ) -> None:
        row = m5_synthetic_rows[0]
        name = next(k for k, v in row.features.items() if v is not None)
        value = float(row.features[name] or 0.0)
        nudged = row.model_copy(
            update={"features": {**row.features, name: value * (1 + 2**-52)}}
        )
        assert fingerprint((nudged, *m5_synthetic_rows[1:])) != fingerprint(
            m5_synthetic_rows
        )

    def test_fingerprint_changes_when_feature_order_changes(
        self, m5_synthetic_rows: tuple[FeatureWindow, ...]
    ) -> None:
        reordered = FeatureCatalog(
            version=FEATURE_CATALOG.version,
            entries=tuple(reversed(FEATURE_CATALOG.entries)),
        )
        original = fingerprint(m5_synthetic_rows)
        shuffled = compute_fingerprint(
            m5_synthetic_rows,
            reordered,
            TARGETS,
            window_duration_seconds=10.0,
            window_step_seconds=10.0,
            windows_overlap=False,
        )
        assert shuffled != original

    def test_fingerprint_changes_when_the_target_set_changes(
        self, m5_synthetic_rows: tuple[FeatureWindow, ...]
    ) -> None:
        subset = compute_fingerprint(
            m5_synthetic_rows,
            FEATURE_CATALOG,
            [TargetName.ENGAGEMENT_CLASS],
            window_duration_seconds=10.0,
            window_step_seconds=10.0,
            windows_overlap=False,
        )
        assert subset != fingerprint(m5_synthetic_rows)

    def test_fingerprint_changes_when_window_geometry_changes(
        self, m5_synthetic_rows: tuple[FeatureWindow, ...]
    ) -> None:
        assert fingerprint(m5_synthetic_rows, window_step_seconds=5.0) != fingerprint(
            m5_synthetic_rows
        )
        assert fingerprint(m5_synthetic_rows, windows_overlap=True) != fingerprint(
            m5_synthetic_rows
        )

    def test_fingerprint_is_insensitive_to_row_order(
        self, m5_synthetic_rows: tuple[FeatureWindow, ...]
    ) -> None:
        shuffled = tuple(reversed(m5_synthetic_rows))
        assert fingerprint(shuffled) == fingerprint(m5_synthetic_rows)

    def test_fingerprint_excludes_the_creation_timestamp(
        self, m5_synthetic_rows: tuple[FeatureWindow, ...]
    ) -> None:
        from datetime import UTC, datetime

        first = build_dataset_metadata(
            m5_synthetic_rows,
            FEATURE_CATALOG,
            TARGETS,
            window_duration_seconds=10.0,
            window_step_seconds=10.0,
            windows_overlap=False,
            created_at_utc=datetime(2020, 1, 1, tzinfo=UTC),
        )
        second = build_dataset_metadata(
            m5_synthetic_rows,
            FEATURE_CATALOG,
            TARGETS,
            window_duration_seconds=10.0,
            window_step_seconds=10.0,
            windows_overlap=False,
            created_at_utc=datetime(2030, 6, 1, tzinfo=UTC),
        )
        assert first.created_at_utc != second.created_at_utc
        assert first.dataset_fingerprint == second.dataset_fingerprint


class TestPrivacyInvariants:
    def test_no_column_name_suggests_identity_or_raw_media(
        self, m5_dataset: Path
    ) -> None:
        table = read_dataset_table(m5_dataset)
        assert_no_identity_columns(table.schema.names)

    def test_no_raw_media_column_exists(self, m5_dataset: Path) -> None:
        """No column can hold a frame, an image, or a landmark array.

        ``feat__rppg_valid_frame_pct`` and ``feat__capture_dropped_frame_pct``
        are counts *about* frames, which is why the check targets the
        raw-media tokens rather than the word "frame" on its own.
        """
        names = {name.lower() for name in read_dataset_table(m5_dataset).schema.names}
        media_tokens = (
            "raw_frame",
            "frame_bytes",
            "frame_data",
            "landmark",
            "image",
            "pixel_array",
            "video",
            "photo",
            "thumbnail",
        )
        for token in media_tokens:
            assert not any(token in name for name in names), token

    def test_no_cell_value_looks_like_an_email(self, m5_dataset: Path) -> None:
        table = read_dataset_table(m5_dataset).to_pandas()
        for column in table.columns:
            if table[column].dtype == object:
                assert_no_identifier_values(table[column].tolist())

    def test_subject_identifiers_are_pseudonymous(self, m5_dataset: Path) -> None:
        table = read_dataset_table(m5_dataset).to_pandas()
        for value in set(table["subject_id"]):
            assert value.startswith("synthetic-subject-")
            assert "participant 1" not in value.lower()

    def test_identifier_columns_reject_an_email_value(self) -> None:
        with pytest.raises(DatasetValidationError, match="email address"):
            assert_no_identifier_values(["ok", "someone@example.com"])

    def test_a_forbidden_column_name_is_caught(self) -> None:
        with pytest.raises(DatasetValidationError, match="forbidden fragment"):
            assert_no_identity_columns(["feat__ok", "participant_email"])


class TestSyntheticLabelling:
    def test_every_row_is_labelled_synthetic(self, m5_dataset: Path) -> None:
        table = read_dataset_table(m5_dataset).to_pandas()
        assert set(table["data_source"]) == {DataSource.SYNTHETIC.value}
        assert set(table["synthetic_label"]) == {SYNTHETIC_LABEL}
        assert set(table["subject_kind"]) == {SubjectKind.SYNTHETIC_SUBJECT.value}

    def test_every_target_is_labelled_and_prohibited(self, m5_dataset: Path) -> None:
        table = read_dataset_table(m5_dataset).to_pandas()
        for target in TARGETS:
            label = f"target_meta__{target.value}__synthetic_label"
            permitted = f"target_meta__{target.value}__scientific_evaluation_permitted"
            assert set(table[label]) == {SYNTHETIC_LABEL}
            assert set(table[permitted]) == {False}

    def test_dataset_is_ineligible_for_scientific_evaluation(
        self, m5_dataset: Path
    ) -> None:
        metadata = read_dataset_metadata(m5_dataset)
        assert metadata.scientific_evaluation_eligible is False
        for summary in metadata.targets:
            assert summary.scientific_evaluation_permitted is False


class TestValidationRefusals:
    def test_duplicate_window_ids_are_rejected(
        self, m5_synthetic_rows: tuple[FeatureWindow, ...]
    ) -> None:
        duplicate = m5_synthetic_rows[1].model_copy(
            update={"window_id": m5_synthetic_rows[0].window_id}
        )
        with pytest.raises(DatasetValidationError, match="duplicate window id"):
            validate_feature_windows((m5_synthetic_rows[0], duplicate), FEATURE_CATALOG)

    def test_an_empty_dataset_is_rejected(self) -> None:
        with pytest.raises(DatasetValidationError, match="at least one window"):
            validate_feature_windows([], FEATURE_CATALOG)

    def test_a_reversed_utc_range_is_rejected_at_the_row_schema(
        self, m5_synthetic_rows: tuple[FeatureWindow, ...]
    ) -> None:
        row = m5_synthetic_rows[0]
        with pytest.raises(ValueError, match="strictly after"):
            row.model_copy(update={"window_end_utc": row.window_start_utc})
            FeatureWindow.model_validate(
                {
                    **row.model_dump(),
                    "window_end_utc": row.window_start_utc,
                }
            )

    def test_a_non_finite_feature_is_rejected(
        self, m5_synthetic_rows: tuple[FeatureWindow, ...]
    ) -> None:
        row = m5_synthetic_rows[0]
        name = next(k for k, v in row.features.items() if v is not None)
        broken = row.model_copy(
            update={"features": {**row.features, name: float("nan")}}
        )
        with pytest.raises(DatasetValidationError, match="non-finite"):
            validate_feature_windows((broken,), FEATURE_CATALOG)

    def test_a_feature_outside_the_catalog_is_rejected(
        self, m5_synthetic_rows: tuple[FeatureWindow, ...]
    ) -> None:
        row = m5_synthetic_rows[0]
        polluted = row.model_copy(
            update={
                "features": {**row.features, "invented": 1.0},
                "feature_available": {**row.feature_available, "invented": True},
            }
        )
        with pytest.raises(DatasetValidationError, match="not in feature catalog"):
            validate_feature_windows((polluted,), FEATURE_CATALOG)

    def test_mismatched_availability_flags_are_rejected(
        self, m5_synthetic_rows: tuple[FeatureWindow, ...]
    ) -> None:
        row = m5_synthetic_rows[0]
        name = next(k for k, v in row.features.items() if v is not None)
        with pytest.raises(ValueError, match="marked unavailable"):
            row.model_copy(
                update={"feature_available": {**row.feature_available, name: False}}
            ).model_validate(
                {
                    **row.model_dump(),
                    "feature_available": {**row.feature_available, name: False},
                }
            )

    def test_writing_an_empty_dataset_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetAssemblyError, match="no rows"):
            write_dataset(
                [],
                tmp_path / "empty.parquet",
                target_names=TARGETS,
                window_duration_seconds=10.0,
                window_step_seconds=10.0,
                windows_overlap=False,
            )

    def test_a_catalog_version_mismatch_is_refused(
        self, tmp_path: Path, m5_synthetic_rows: tuple[FeatureWindow, ...]
    ) -> None:
        stale = m5_synthetic_rows[0].model_copy(
            update={"feature_catalog_version": "0.9"}
        )
        with pytest.raises(DatasetAssemblyError, match="built against feature catalog"):
            write_dataset(
                (stale,),
                tmp_path / "stale.parquet",
                target_names=TARGETS,
                window_duration_seconds=10.0,
                window_step_seconds=10.0,
                windows_overlap=False,
            )

    def test_a_dataset_without_metadata_cannot_be_read(self, tmp_path: Path) -> None:
        rows = generate_synthetic_dataset(
            SyntheticDatasetConfig(
                subjects=2, sessions_per_subject=1, windows_per_session=2
            )
        )
        path = tmp_path / "orphan.parquet"
        table = rows_to_table(rows, FEATURE_CATALOG, TARGETS)
        import pyarrow.parquet as pq

        pq.write_table(table, path)
        with pytest.raises(DatasetAssemblyError, match="no metadata document"):
            read_dataset_metadata(path)


class TestMetadataContent:
    def test_missingness_is_reported_per_feature(self, m5_dataset: Path) -> None:
        metadata = read_dataset_metadata(m5_dataset)
        names = {entry.feature_name for entry in metadata.missingness}
        assert names == set(FEATURE_CATALOG.names())
        assert 0.0 <= metadata.overall_missing_pct <= 100.0
        assert metadata.overall_missing_pct > 0.0

    def test_class_distributions_and_numeric_summaries_are_present(
        self, m5_dataset: Path
    ) -> None:
        metadata = read_dataset_metadata(m5_dataset)
        by_name = {summary.target_name: summary for summary in metadata.targets}
        classification = by_name["engagement_class"]
        regression = by_name["engagement_score"]
        assert classification.class_distribution is not None
        assert sum(classification.class_distribution.values()) == metadata.row_count
        assert regression.value_minimum is not None
        assert regression.value_maximum is not None

    def test_counts_match_the_generator(
        self, m5_dataset: Path, m5_synthetic_config: SyntheticDatasetConfig
    ) -> None:
        metadata = read_dataset_metadata(m5_dataset)
        assert metadata.subject_count == m5_synthetic_config.subjects
        assert (
            metadata.session_count
            == m5_synthetic_config.subjects * m5_synthetic_config.sessions_per_subject
        )
        assert metadata.window_duration_seconds == pytest.approx(
            m5_synthetic_config.window_duration_seconds
        )

    def test_creation_configuration_and_seed_are_recorded(
        self, m5_dataset: Path
    ) -> None:
        metadata = read_dataset_metadata(m5_dataset)
        assert metadata.random_seed == 42
        assert metadata.creation_configuration["seed"] == 42
