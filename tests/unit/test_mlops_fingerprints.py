"""Configuration, split, and feature-schema fingerprints.

A fingerprint has one job: two runs of the same thing must produce the
same value, and two runs of different things must not.  These tests
pursue both halves, and check that what is deliberately excluded from the
identity really is excluded — otherwise the same pipeline on two machines
would get two different fingerprints and the fingerprint would be useless.
"""

from __future__ import annotations

import json

import pytest

from engagevr.config import EngageVRConfig, load_config
from engagevr.mlops.fingerprints import (
    EXCLUDED_CONFIG_PATHS,
    REPOSITORY_ROOT,
    build_configuration_version,
    canonical_json,
    config_fingerprint,
    feature_schema_fingerprint,
    normalize_config,
    repository_relative,
    sha256_payload,
    sha256_text,
    split_fingerprint,
)


@pytest.fixture
def config() -> EngageVRConfig:
    return load_config()


class TestCanonicalRendering:
    def test_key_order_does_not_change_the_digest(self) -> None:
        assert sha256_payload({"a": 1, "b": 2}) == sha256_payload({"b": 2, "a": 1})

    def test_whitespace_does_not_participate(self) -> None:
        assert "\n" not in canonical_json({"a": [1, 2]})
        assert ", " not in canonical_json({"a": [1, 2]})

    def test_a_different_value_changes_the_digest(self) -> None:
        assert sha256_payload({"a": 1}) != sha256_payload({"a": 2})

    def test_the_digest_is_a_lowercase_sha256(self) -> None:
        digest = sha256_text("engagevr")
        assert len(digest) == 64
        assert digest == digest.lower()


class TestConfigurationFingerprint:
    def test_the_same_configuration_fingerprints_identically(
        self, config: EngageVRConfig
    ) -> None:
        assert config_fingerprint(config) == config_fingerprint(load_config())

    def test_a_changed_setting_changes_the_fingerprint(
        self, config: EngageVRConfig
    ) -> None:
        changed = config.model_copy(
            update={"training": config.training.model_copy(update={"folds": 7})}
        )
        assert config_fingerprint(changed) != config_fingerprint(config)

    def test_every_default_is_present_not_only_what_the_yaml_states(
        self, config: EngageVRConfig
    ) -> None:
        # The fingerprint must describe the settings the code ran under,
        # not the subset somebody happened to type into the YAML file.
        document = normalize_config(config)
        assert "mlops" in document
        assert "uncertainty" in document
        assert document["training"]["random_seed"] is not None

    @pytest.mark.parametrize("dotted", sorted(EXCLUDED_CONFIG_PATHS))
    def test_each_environment_specific_path_is_removed(
        self, config: EngageVRConfig, dotted: str
    ) -> None:
        document = normalize_config(config)
        cursor: object = document
        parts = dotted.split(".")
        for part in parts[:-1]:
            assert isinstance(cursor, dict)
            cursor = cursor[part]
        assert isinstance(cursor, dict)
        assert parts[-1] not in cursor

    def test_a_local_dataset_root_does_not_change_the_fingerprint(
        self, config: EngageVRConfig
    ) -> None:
        # Two machines with the dataset in different places must agree.
        elsewhere = config.model_copy(
            update={
                "rppg": config.rppg.model_copy(
                    update={
                        "datasets": config.rppg.datasets.model_copy(
                            update={"ubfc_rppg_root": "/somewhere/else"}
                        )
                    }
                )
            }
        )
        assert config_fingerprint(elsewhere) == config_fingerprint(config)

    def test_a_camera_index_does_not_change_the_fingerprint(
        self, config: EngageVRConfig
    ) -> None:
        other = config.model_copy(
            update={"capture": config.capture.model_copy(update={"camera_index": 3})}
        )
        assert config_fingerprint(other) == config_fingerprint(config)

    def test_every_exclusion_is_recorded_with_a_reason(
        self, config: EngageVRConfig
    ) -> None:
        version = build_configuration_version(config)
        assert set(version.excluded_paths) == set(EXCLUDED_CONFIG_PATHS)
        for path in version.excluded_paths:
            assert version.exclusion_reasons[path]

    def test_the_snapshot_carries_the_sections_that_shaped_the_run(
        self, config: EngageVRConfig
    ) -> None:
        version = build_configuration_version(config)
        for section in ("features", "training", "uncertainty", "mlops"):
            assert section in version.section_snapshots

    def test_the_snapshot_is_json_serialisable(self, config: EngageVRConfig) -> None:
        version = build_configuration_version(config)
        json.dumps(version.model_dump(mode="json"))


class TestSplitFingerprint:
    def _splits(self, **overrides: object) -> dict[str, object]:
        document: dict[str, object] = {
            "strategy": "stratified_group_k_fold",
            "group_field": "subject_id",
            "n_splits": 3,
            "random_seed": 42,
            "folds": [
                {
                    "fold_index": 0,
                    "train_groups": ["s1", "s2"],
                    "calibration_groups": ["s2"],
                    "test_groups": ["s3"],
                    "train_row_count": 20,
                }
            ],
        }
        document.update(overrides)
        return document

    def test_group_order_does_not_change_the_digest(self) -> None:
        # A fold is a set of groups. Listing them in a different order is
        # the same split.
        reordered = self._splits(
            folds=[
                {
                    "fold_index": 0,
                    "train_groups": ["s2", "s1"],
                    "calibration_groups": ["s2"],
                    "test_groups": ["s3"],
                }
            ]
        )
        assert split_fingerprint(reordered) == split_fingerprint(self._splits())

    def test_row_counts_do_not_participate(self) -> None:
        # A dataset that grew without the split design changing must not
        # look like a different split.
        grown = self._splits(
            folds=[
                {
                    "fold_index": 0,
                    "train_groups": ["s1", "s2"],
                    "calibration_groups": ["s2"],
                    "test_groups": ["s3"],
                    "train_row_count": 9999,
                }
            ]
        )
        assert split_fingerprint(grown) == split_fingerprint(self._splits())

    def test_a_moved_group_changes_the_digest(self) -> None:
        moved = self._splits(
            folds=[
                {
                    "fold_index": 0,
                    "train_groups": ["s1"],
                    "calibration_groups": [],
                    "test_groups": ["s2", "s3"],
                }
            ]
        )
        assert split_fingerprint(moved) != split_fingerprint(self._splits())

    def test_a_different_seed_changes_the_digest(self) -> None:
        assert split_fingerprint(self._splits(random_seed=7)) != split_fingerprint(
            self._splits()
        )


class TestFeatureSchemaFingerprint:
    def test_predictor_order_participates(self) -> None:
        # A linear model's coefficients are read positionally, so a
        # reordered predictor matrix is a different schema.
        first = feature_schema_fingerprint(["a", "b"], catalog_version="1.0")
        second = feature_schema_fingerprint(["b", "a"], catalog_version="1.0")
        assert first != second

    def test_the_catalog_version_participates(self) -> None:
        first = feature_schema_fingerprint(["a"], catalog_version="1.0")
        second = feature_schema_fingerprint(["a"], catalog_version="2.0")
        assert first != second

    def test_the_same_schema_fingerprints_identically(self) -> None:
        first = feature_schema_fingerprint(["a", "b"], catalog_version="1.0")
        second = feature_schema_fingerprint(("a", "b"), catalog_version="1.0")
        assert first == second


class TestRepositoryRelativePaths:
    def test_a_repository_path_is_rendered_relative(self) -> None:
        assert (
            repository_relative(REPOSITORY_ROOT / "configs" / "defaults.yaml")
            == "configs/defaults.yaml"
        )

    def test_an_outside_path_does_not_leak_a_home_directory(self) -> None:
        rendered = repository_relative("/etc/hostname")
        assert rendered == "hostname"
        assert "/" not in rendered
