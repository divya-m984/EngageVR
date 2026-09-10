"""The accepted numerical reference, and the fail-open hole it closes.

The hole
--------
DEC-106 removed floating-point values from portable identity, which was
right.  The first version of its check then compared two executions of
the **same** CI runner against each other, which was wrong: both sides
come from one CPU, so both can be wrong in the same way.

Measured on this repository before the repair, and reproduced by
``test_the_documented_hole_is_closed`` below: mutating every float in
``metrics.json`` — structure, keys, dtypes, ordering, and labels
untouched — left ``dvc.lock`` **byte-identical** and the same-runner
comparison **exited zero**.

The repair is a committed accepted reference: a fresh execution is held
to the numbers in the repository, not to another execution of itself.

What these tests hold
---------------------
- an accepted-scale (cross-kernel) difference passes;
- a materially different number fails, including the 0.72 -> 0.91 case;
- every structural change fails, with no tolerance reaching it;
- a missing, unlisted, stale, or modified reference fails;
- an unclassified new numerical artifact fails closed;
- raw-digest integrity is still checked separately.

Reproducing a reference means the software is portable. It is not
evidence that any number in it is correct, and nothing in this repository
has been evaluated against a participant-provided label.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from engagevr.mlops.numeric_contract import (
    MEASURED_CROSS_KERNEL_DEVIATION,
    NUMERIC_PORTABILITY_ATOL,
    NUMERIC_PORTABILITY_RTOL,
    NumericTolerance,
    iter_floats,
    structure_digest,
)
from engagevr.mlops.numeric_reference import (
    MANIFEST_NAME,
    NumericReferenceError,
    build_reference,
    check_against_references,
    compare_to_reference,
    read_manifest,
    read_reference,
    reference_file_name,
    verify_manifest,
    verify_tolerance_agreement,
    write_manifest,
    write_reference,
)
from engagevr.schemas.mlops import CpuDependentNumericArtifact, NumericReference

ARTIFACT = "experiments/baseline-engagement_class/metrics.json"
TABLE = "experiments/baseline-engagement_class/predictions.parquet"


def write_metrics(path: Path, *, value: float, label: str = "low") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "run_id": "engagement_class-selfcheck-abc123",
        "random_seed": 42,
        "fold_count": 3,
        "results": [
            {
                "model_name": "logistic_regression",
                "chosen_label": label,
                "aggregate": [{"mean": value, "fold_values": [value, value * 0.5]}],
            }
        ],
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def write_table(
    path: Path,
    *,
    probabilities: list[float],
    labels: list[str] | None = None,
    widths: list[Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(probabilities)
    pd.DataFrame(
        {
            "window_id": [f"w{i:03d}" for i in range(n)],
            "fold_index": list(range(n)),
            "predicted_label": labels or ["low", "high", "medium"][:n],
            "probability__high": probabilities,
            "interval_width": widths if widths is not None else [0.1, None, 0.3][:n],
        }
    ).to_parquet(path, index=False)
    return path


def _declared(
    path: str,
    structure: str,
    *,
    atol: float = NUMERIC_PORTABILITY_ATOL,
    rtol: float = NUMERIC_PORTABILITY_RTOL,
) -> CpuDependentNumericArtifact:
    """One stage-record entry, as the pipeline would produce it."""
    return CpuDependentNumericArtifact(
        path=path,
        structure_sha256=structure,
        excluded_from_portable_identity="measured to differ across CPUs",
        atol=atol,
        rtol=rtol,
        numeric_tolerance=f"|a-b| <= {atol:g} + {rtol:g}*|b|",
    )


@pytest.fixture
def accepted(tmp_path: Path) -> tuple[Path, Path, list[CpuDependentNumericArtifact]]:
    """A pipeline root, a committed reference directory, and the declared set."""
    root = tmp_path / "pipeline"
    write_metrics(root / ARTIFACT, value=0.72)
    write_table(root / TABLE, probabilities=[0.25, 0.5, 0.75])
    references = tmp_path / "references"
    references.mkdir()
    declared: list[CpuDependentNumericArtifact] = []
    for artifact_path in (ARTIFACT, TABLE):
        reference = build_reference(root / artifact_path, artifact_path)
        write_reference(reference, references)
        declared.append(_declared(artifact_path, reference.structure_sha256))
    write_manifest(references, target="engagement_class")
    return root, references, declared


def run_check(
    accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]],
) -> list[str]:
    root, references, declared = accepted
    problems, _compared = check_against_references(root, references, declared)
    return problems


# ---------------------------------------------------------------------------
# The hole itself
# ---------------------------------------------------------------------------


class TestTheDocumentedHoleIsClosed:
    def test_a_same_runner_pair_agrees_while_both_are_wrong(
        self, tmp_path: Path
    ) -> None:
        # Both "executions" carry the same wrong number, so comparing them
        # to each other proves nothing. This is the failure mode, asserted
        # rather than described.
        a = write_metrics(tmp_path / "a.json", value=0.91)
        b = write_metrics(tmp_path / "b.json", value=0.91)
        assert a.read_bytes() == b.read_bytes()
        assert structure_digest(a) == structure_digest(b)

    def test_the_structure_digest_cannot_see_the_change(self, tmp_path: Path) -> None:
        # And this is why dvc.lock cannot see it either: the stage record
        # pins structure, and structure is unmoved.
        accepted_doc = write_metrics(tmp_path / "accepted.json", value=0.72)
        wrong = write_metrics(tmp_path / "wrong.json", value=0.91)
        assert structure_digest(accepted_doc) == structure_digest(wrong)

    def test_the_accepted_reference_does_see_it(self, tmp_path: Path) -> None:
        accepted_doc = write_metrics(tmp_path / "accepted.json", value=0.72)
        reference = build_reference(accepted_doc, ARTIFACT)
        wrong = write_metrics(tmp_path / "wrong.json", value=0.91)
        problems = compare_to_reference(wrong, reference)
        assert problems, "0.72 -> 0.91 must fail against the accepted reference"
        assert "exceed the tolerance" in problems[0]


# ---------------------------------------------------------------------------
# 1-3. Tolerance behaviour against the accepted reference
# ---------------------------------------------------------------------------


class TestNumericalAgreement:
    def test_a_one_ulp_difference_passes(self, tmp_path: Path) -> None:
        reference = build_reference(
            write_metrics(tmp_path / "a.json", value=0.72), ARTIFACT
        )
        nudged = write_metrics(
            tmp_path / "b.json", value=math.nextafter(0.72, math.inf)
        )
        assert compare_to_reference(nudged, reference) == []

    def test_an_observed_cross_kernel_difference_passes(self, tmp_path: Path) -> None:
        reference = build_reference(
            write_metrics(tmp_path / "a.json", value=0.72), ARTIFACT
        )
        moved = write_metrics(
            tmp_path / "b.json", value=0.72 + MEASURED_CROSS_KERNEL_DEVIATION
        )
        assert compare_to_reference(moved, reference) == []

    def test_a_difference_beyond_tolerance_fails(self, tmp_path: Path) -> None:
        reference = build_reference(
            write_metrics(tmp_path / "a.json", value=0.72), ARTIFACT
        )
        moved = write_metrics(tmp_path / "b.json", value=0.72 + 1e-5)
        problems = compare_to_reference(moved, reference)
        assert problems and "exceed the tolerance" in problems[0]

    def test_the_documented_case_fails(self, tmp_path: Path) -> None:
        reference = build_reference(
            write_metrics(tmp_path / "a.json", value=0.72), ARTIFACT
        )
        problems = compare_to_reference(
            write_metrics(tmp_path / "b.json", value=0.91), reference
        )
        assert problems
        assert "0.91" in " ".join(problems)

    def test_a_parquet_value_beyond_tolerance_fails(self, tmp_path: Path) -> None:
        reference = build_reference(
            write_table(tmp_path / "a.parquet", probabilities=[0.25, 0.5, 0.75]),
            TABLE,
        )
        moved = write_table(tmp_path / "b.parquet", probabilities=[0.25, 0.9, 0.75])
        assert compare_to_reference(moved, reference)


# ---------------------------------------------------------------------------
# 4-8. Structural changes, none of which a tolerance may reach
# ---------------------------------------------------------------------------


class TestStructuralChangesFailAgainstTheReference:
    def _reference(self, tmp_path: Path) -> NumericReference:
        return build_reference(
            write_table(tmp_path / "a.parquet", probabilities=[0.25, 0.5, 0.75]),
            TABLE,
        )

    def test_a_schema_change_fails(self, tmp_path: Path) -> None:
        reference = self._reference(tmp_path)
        frame = pd.read_parquet(tmp_path / "a.parquet")
        frame["extra"] = 1
        frame.to_parquet(tmp_path / "b.parquet", index=False)
        problems = compare_to_reference(tmp_path / "b.parquet", reference)
        assert problems and "structure differs" in problems[0]

    def test_a_dtype_change_fails(self, tmp_path: Path) -> None:
        reference = self._reference(tmp_path)
        frame = pd.read_parquet(tmp_path / "a.parquet")
        frame["fold_index"] = frame["fold_index"].astype("float64")
        frame.to_parquet(tmp_path / "b.parquet", index=False)
        assert compare_to_reference(tmp_path / "b.parquet", reference)

    def test_a_row_order_change_fails(self, tmp_path: Path) -> None:
        reference = self._reference(tmp_path)
        frame = pd.read_parquet(tmp_path / "a.parquet")
        frame.iloc[::-1].reset_index(drop=True).to_parquet(
            tmp_path / "b.parquet", index=False
        )
        problems = compare_to_reference(tmp_path / "b.parquet", reference)
        assert problems and "structure differs" in problems[0]

    def test_a_predicted_label_change_fails(self, tmp_path: Path) -> None:
        reference = self._reference(tmp_path)
        write_table(
            tmp_path / "b.parquet",
            probabilities=[0.25, 0.5, 0.75],
            labels=["low", "low", "medium"],
        )
        problems = compare_to_reference(tmp_path / "b.parquet", reference)
        assert problems and "structure differs" in problems[0]

    def test_a_missingness_position_change_fails(self, tmp_path: Path) -> None:
        reference = self._reference(tmp_path)
        write_table(
            tmp_path / "b.parquet",
            probabilities=[0.25, 0.5, 0.75],
            widths=[None, 0.1, 0.3],
        )
        problems = compare_to_reference(tmp_path / "b.parquet", reference)
        assert problems and "structure differs" in problems[0]

    def test_a_json_identifier_change_fails(self, tmp_path: Path) -> None:
        reference = build_reference(
            write_metrics(tmp_path / "a.json", value=0.72), ARTIFACT
        )
        document = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
        document["run_id"] = "another-run"
        (tmp_path / "b.json").write_text(json.dumps(document, indent=2))
        assert compare_to_reference(tmp_path / "b.json", reference)


# ---------------------------------------------------------------------------
# 9-11. Fail closed on the reference set itself
# ---------------------------------------------------------------------------


class TestTheReferenceSetFailsClosed:
    def test_the_honest_pipeline_passes(
        self, accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]]
    ) -> None:
        assert run_check(accepted) == []

    def test_a_new_unclassified_numerical_artifact_fails(
        self, accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]]
    ) -> None:
        # A newly classified artifact with no accepted reference must be an
        # error, not an absence. This is the check being driven by the
        # stage records rather than by the reference directory.
        root, references, declared = accepted
        new_path = "experiments/baseline-engagement_class/brand_new.json"
        write_metrics(root / new_path, value=0.5)
        declared = [*declared, _declared(new_path, "0" * 64)]
        problems, _ = check_against_references(root, references, declared)
        assert problems and "no accepted reference" in problems[0]

    def test_a_missing_reference_file_fails(
        self, accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]]
    ) -> None:
        _root, references, _declared = accepted
        (references / reference_file_name(ARTIFACT)).unlink()
        problems = run_check(accepted)
        assert problems
        assert any("missing on disk" in p for p in problems)

    def test_a_modified_reference_fails(
        self, accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]]
    ) -> None:
        # Editing an accepted number without re-accepting it is exactly
        # the tampering this manifest exists to catch.
        _root, references, _declared = accepted
        path = references / reference_file_name(ARTIFACT)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["values"][0] = 0.91
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        problems = run_check(accepted)
        assert problems
        assert any("does not match" in p for p in problems)

    def test_a_corrupted_reference_fails(
        self, accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]]
    ) -> None:
        _root, references, _declared = accepted
        (references / reference_file_name(ARTIFACT)).write_text("{not json")
        assert run_check(accepted)

    def test_an_unlisted_reference_fails(
        self, accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]]
    ) -> None:
        _root, references, _declared = accepted
        shutil.copy(
            references / reference_file_name(ARTIFACT),
            references / "sneaked_in.reference.json",
        )
        problems = verify_manifest(references)
        assert problems and "absent from" in problems[0]

    def test_a_stale_reference_fails(
        self, accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]]
    ) -> None:
        # A reference the pipeline no longer declares checks nothing, and
        # silently checking nothing is the failure mode being avoided.
        root, references, declared = accepted
        problems, _ = check_against_references(root, references, declared[:1])
        assert problems and "no longer declares" in problems[0]

    def test_a_missing_manifest_fails(
        self, accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]]
    ) -> None:
        _root, references, _declared = accepted
        (references / MANIFEST_NAME).unlink()
        with pytest.raises(NumericReferenceError):
            verify_manifest(references)

    def test_a_missing_produced_artifact_fails(
        self, accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]]
    ) -> None:
        root, _references, _declared = accepted
        (root / ARTIFACT).unlink()
        problems = run_check(accepted)
        assert problems and "produced no artifact" in problems[0]

    def test_a_reference_describing_another_artifact_fails(
        self, accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]]
    ) -> None:
        root, references, declared = accepted
        path = references / reference_file_name(ARTIFACT)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["artifact_path"] = "experiments/elsewhere/metrics.json"
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        write_manifest(references, target="engagement_class")
        problems, _ = check_against_references(root, references, declared)
        assert problems and "describes" in problems[0]


# ---------------------------------------------------------------------------
# 12. Raw integrity remains a separate, exact check
# ---------------------------------------------------------------------------


class TestRawIntegrityIsSeparate:
    def test_a_raw_digest_change_is_detected_even_when_numbers_agree(
        self, tmp_path: Path
    ) -> None:
        # An in-tolerance difference is portable but NOT byte-identical.
        # The numerical check passes and the raw digest still moves, which
        # is the whole point of keeping both.
        first = write_metrics(tmp_path / "a.json", value=0.72)
        reference = build_reference(first, ARTIFACT)
        second = write_metrics(tmp_path / "b.json", value=0.72 + 1e-12)
        assert compare_to_reference(second, reference) == []
        assert (
            hashlib.sha256(first.read_bytes()).hexdigest()
            != hashlib.sha256(second.read_bytes()).hexdigest()
        )

    def test_the_reference_holds_no_raw_digest(self, tmp_path: Path) -> None:
        # The reference pins structure and values. Raw bytes stay in the
        # integrity sidecar, which is execution-specific evidence.
        reference = build_reference(
            write_metrics(tmp_path / "a.json", value=0.72), ARTIFACT
        )
        rendered = json.dumps(reference.model_dump(mode="json"))
        assert "sha256" not in rendered.replace("structure_sha256", "")


# ---------------------------------------------------------------------------
# The reference document's own contract
# ---------------------------------------------------------------------------


def _version_stub() -> Any:
    """The fields verify_model_version_portability reads, and no more."""
    from types import SimpleNamespace

    return SimpleNamespace(
        source_run_directory="run",
        model_artifact_path="models/x.joblib",
        referenced_checksums={},
        referenced_structure_digests={"metrics.json": "0" * 64},
    )


class TestModelVersionIdentityDoesNotCertifyPortability:
    """The separation the model-version note states, asserted."""

    def _run(self, tmp_path: Path, *, value: float) -> Path:
        from engagevr.config import load_config
        from engagevr.mlops.model_version import build_model_versions

        del load_config, build_model_versions
        run = tmp_path / "run"
        write_metrics(run / "metrics.json", value=value)
        return run

    def test_the_manifest_says_identity_is_not_portability(self) -> None:
        from engagevr.schemas.mlops import ModelVersionManifest

        default = ModelVersionManifest.model_fields[
            "numerical_portability_note"
        ].default
        assert "DOES NOT CERTIFY NUMERICAL PORTABILITY" in default
        assert "numeric-check" in default

    def test_portability_verification_fails_on_a_beyond_tolerance_number(
        self, tmp_path: Path
    ) -> None:
        # verify_model_version passes (structure is intact); the
        # portability entry point does not. That is the separation.
        from engagevr.mlops.model_version import verify_model_version_portability

        run = self._run(tmp_path, value=0.72)
        references = tmp_path / "refs"
        references.mkdir()
        write_reference(
            build_reference(run / "metrics.json", "run/metrics.json"), references
        )
        write_manifest(references, target="engagement_class")

        write_metrics(run / "metrics.json", value=0.91)

        problems = verify_model_version_portability(
            _version_stub(),
            run_directory=run,
            reference_directory=references,
            integrity=(),
        )
        assert any("exceed the tolerance" in p for p in problems)

    def test_portability_verification_passes_within_tolerance(
        self, tmp_path: Path
    ) -> None:
        from engagevr.mlops.model_version import verify_model_version_portability

        run = self._run(tmp_path, value=0.72)
        references = tmp_path / "refs"
        references.mkdir()
        write_reference(
            build_reference(run / "metrics.json", "run/metrics.json"), references
        )
        write_manifest(references, target="engagement_class")

        write_metrics(run / "metrics.json", value=0.72 + 1e-12)

        problems = verify_model_version_portability(
            _version_stub(),
            run_directory=run,
            reference_directory=references,
            integrity=(),
        )
        assert not any("exceed the tolerance" in p for p in problems)


class TestTheCommittedReferencesInThisRepository:
    """The real ``references/numeric/`` directory, as committed.

    Runnable from a clean checkout: the references are tracked files, not
    pipeline output, which is the property that makes them usable as a
    cross-environment baseline in the first place.
    """

    @pytest.fixture
    def directory(self) -> Path:
        from engagevr.mlops.numeric_reference import REFERENCE_DIRECTORY

        root = Path(__file__).resolve().parents[2]
        return root / REFERENCE_DIRECTORY

    def test_the_directory_is_committed_and_populated(self, directory: Path) -> None:
        assert directory.is_dir(), "the accepted references must be in the tree"
        assert list(directory.glob("*.reference.json"))

    def test_every_committed_reference_matches_its_recorded_digest(
        self, directory: Path
    ) -> None:
        assert verify_manifest(directory) == []

    def test_every_committed_reference_validates(self, directory: Path) -> None:
        for path in sorted(directory.glob("*.reference.json")):
            reference = read_reference(path)
            assert reference.value_count == len(reference.values)
            assert reference.scientific_evaluation_eligible is False
            assert reference.is_synthetic is True

    def test_every_classified_artifact_name_has_a_reference(
        self, directory: Path
    ) -> None:
        # Fail closed at the repository level: classifying a new artifact
        # without accepting its numbers must not be possible to forget.
        from engagevr.mlops.stage_record import CPU_DEPENDENT_NUMERIC_REASONS

        covered = {
            read_reference(path).artifact_path
            for path in directory.glob("*.reference.json")
        }
        assert set(CPU_DEPENDENT_NUMERIC_REASONS) == covered, (
            "every cpu-dependent numerical artifact this repository "
            "classifies must have an accepted reference committed beside "
            "it, matched by exact path rather than by file name"
        )

    def test_the_references_carry_the_self_check_banner(self, directory: Path) -> None:
        from engagevr.schemas.experiments import SOFTWARE_SELF_CHECK_BANNER

        for path in sorted(directory.glob("*.reference.json")):
            reference = read_reference(path)
            assert any(
                SOFTWARE_SELF_CHECK_BANNER in d for d in reference.disclaimers
            ), path.name


class TestTheCompleteCliPath:
    """The path CI actually runs, through ``run_numeric_check``."""

    def _args(self, **overrides: Any) -> Any:
        import argparse

        defaults = {
            "reference": None,
            "accepted": None,
            "candidate": None,
            "target": None,
            "atol": None,
            "rtol": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_it_refuses_when_given_nothing_to_compare_against(
        self, tmp_path: Path
    ) -> None:
        from engagevr.cli_milestone10 import run_numeric_check

        assert run_numeric_check(self._args(candidate=str(tmp_path))) == 2

    def test_a_beyond_tolerance_mutation_fails_through_the_cli(
        self, tmp_path: Path, m10_baseline_run: Path
    ) -> None:
        # The complete path: real stage records, real classification, real
        # accepted references, real comparison. Not the comparator alone.
        from engagevr.cli_milestone10 import run_numeric_check
        from engagevr.mlops.pipeline import (
            build_stages,
            default_layout,
            load_parameters,
        )
        from engagevr.mlops.stage_record import (
            build_stage_record,
            run_identity,
            write_stage_record,
        )

        parameters = load_parameters()
        layout = default_layout(tmp_path / "pipeline", parameters.target)
        layout.experiments.mkdir(parents=True)
        shutil.copytree(m10_baseline_run, layout.baseline_run)
        stage = next(
            s for s in build_stages(layout, parameters) if s.name == "baseline"
        )
        record, _integrity = build_stage_record(
            stage_name=stage.name,
            stage_kind=stage.kind,
            command=stage.command,
            logical_identity=run_identity(layout.baseline_run),
            targets=list(stage.recorded_targets),
            root=layout.root,
        )
        write_stage_record(record, stage.record)

        references = tmp_path / "refs"
        references.mkdir()
        for artifact in record.cpu_dependent_numeric_artifacts:
            write_reference(
                build_reference(layout.root / artifact.path, artifact.path),
                references,
            )
        write_manifest(references, target=parameters.target)

        args = self._args(candidate=str(layout.root), accepted=str(references))
        assert run_numeric_check(args) == 0, "the honest run must pass"

        # 0.72 -> 0.91, structure untouched.
        metrics = layout.baseline_run / "metrics.json"
        document = json.loads(metrics.read_text(encoding="utf-8"))

        def bump(node: Any) -> Any:
            if isinstance(node, bool):
                return node
            if isinstance(node, float):
                return node + 0.19
            if isinstance(node, dict):
                return {key: bump(value) for key, value in node.items()}
            if isinstance(node, list):
                return [bump(value) for value in node]
            return node

        metrics.write_text(json.dumps(bump(document), indent=2), encoding="utf-8")
        assert structure_digest(metrics) == next(
            a.structure_sha256
            for a in record.cpu_dependent_numeric_artifacts
            if a.path.endswith("metrics.json")
        ), "the mutation must leave structure identical, or it proves nothing"
        assert run_numeric_check(args) == 1, (
            "a materially different number must fail the complete CI path"
        )


class TestTheReferenceDocument:
    def test_it_records_every_finite_float(self, tmp_path: Path) -> None:
        artifact = write_table(tmp_path / "a.parquet", probabilities=[0.1, 0.2, 0.3])
        reference = build_reference(artifact, TABLE)
        assert reference.value_count == len(iter_floats(artifact))
        assert reference.value_count > 0

    def test_a_truncated_reference_is_refused(self, tmp_path: Path) -> None:
        reference = build_reference(
            write_metrics(tmp_path / "a.json", value=0.72), ARTIFACT
        )
        payload = reference.model_dump(mode="json")
        payload["values"] = payload["values"][:-1]
        with pytest.raises(ValueError, match="value_count"):
            NumericReference.model_validate(payload)

    def test_it_can_never_be_scientifically_eligible(self, tmp_path: Path) -> None:
        reference = build_reference(
            write_metrics(tmp_path / "a.json", value=0.72), ARTIFACT
        )
        payload = reference.model_dump(mode="json")
        payload["scientific_evaluation_eligible"] = True
        with pytest.raises(ValueError, match="never be scientifically eligible"):
            NumericReference.model_validate(payload)

    def test_it_holds_no_non_finite_value(self, tmp_path: Path) -> None:
        reference = build_reference(
            write_metrics(tmp_path / "a.json", value=0.72), ARTIFACT
        )
        payload = reference.model_dump(mode="json")
        payload["values"] = [float("inf"), *payload["values"][1:]]
        with pytest.raises(ValueError, match="finite values only"):
            NumericReference.model_validate(payload)

    def test_it_round_trips(self, tmp_path: Path) -> None:
        reference = build_reference(
            write_table(tmp_path / "a.parquet", probabilities=[0.25, 0.5, 0.75]),
            TABLE,
        )
        directory = tmp_path / "refs"
        directory.mkdir()
        path = write_reference(reference, directory)
        assert read_reference(path) == reference

    def test_the_manifest_digests_the_committed_bytes(self, tmp_path: Path) -> None:
        reference = build_reference(
            write_metrics(tmp_path / "a.json", value=0.72), ARTIFACT
        )
        directory = tmp_path / "refs"
        directory.mkdir()
        path = write_reference(reference, directory)
        write_manifest(directory, target="engagement_class")
        manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert (
            manifest["references"][path.name]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )
        assert verify_manifest(directory) == []


class TestTheAcceptedToleranceIsAuthoritative:
    """One tolerance, declared in one place, and not widenable ad hoc.

    Three records carry it — the accepted manifest, every accepted
    reference, and every stage record — and the running code is held to
    the manifest as well. Changing it is a deliberate repository contract
    change, which means editing the code, re-accepting the references,
    and reviewing the diff. See DEC-107.
    """

    def test_a_matching_tolerance_passes(
        self, accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]]
    ) -> None:
        root, references, declared = accepted
        problems, compared = check_against_references(
            root,
            references,
            declared,
            tolerance=NumericTolerance(
                atol=NUMERIC_PORTABILITY_ATOL, rtol=NUMERIC_PORTABILITY_RTOL
            ),
        )
        assert problems == []
        assert compared == 2

    def test_an_attempted_widening_is_refused(
        self, accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]]
    ) -> None:
        # The gate must not be loosenable from the command line.
        root, references, declared = accepted
        problems, compared = check_against_references(
            root, references, declared, tolerance=NumericTolerance(atol=0.5, rtol=0.5)
        )
        assert problems and "refused" in problems[0]
        assert compared == 0, "a refused comparison must compare nothing"

    def test_a_reference_disagreeing_with_the_manifest_fails(
        self, accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]]
    ) -> None:
        _root, references, _declared = accepted
        path = references / reference_file_name(ARTIFACT)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["atol"] = 1e-3
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        write_manifest(references, target="engagement_class")
        problems = verify_tolerance_agreement(references)
        assert problems and "MANIFEST.json declares" in problems[0]

    def test_a_stage_record_disagreeing_with_the_manifest_fails(
        self, accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]]
    ) -> None:
        # The repository must not simultaneously claim two tolerances.
        root, references, _declared = accepted
        loose = [_declared_loose(ARTIFACT)]
        problems = verify_tolerance_agreement(references, declared=loose)
        assert problems and "the stage record declares" in problems[0]
        reported, _ = check_against_references(root, references, loose)
        assert any("the stage record declares" in p for p in reported)

    def test_changing_the_code_default_without_re_accepting_fails(
        self,
        accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Widening DEFAULT_TOLERANCE while the committed manifest still
        # says 1e-6 is exactly the drift this check exists to refuse.
        import engagevr.mlops.numeric_reference as module

        _root, references, _declared = accepted
        monkeypatch.setattr(
            module, "DEFAULT_TOLERANCE", NumericTolerance(atol=1e-2, rtol=1e-2)
        )
        problems = verify_tolerance_agreement(references)
        assert problems
        assert "DEFAULT_TOLERANCE" in problems[0]
        assert "deliberate contract change" in problems[0]

    def test_a_one_ulp_variation_still_passes(
        self, accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]]
    ) -> None:
        root, references, declared = accepted
        write_metrics(root / ARTIFACT, value=math.nextafter(0.72, math.inf))
        problems, _ = check_against_references(root, references, declared)
        assert problems == []

    def test_the_documented_case_still_fails(
        self, accepted: tuple[Path, Path, list[CpuDependentNumericArtifact]]
    ) -> None:
        root, references, declared = accepted
        write_metrics(root / ARTIFACT, value=0.91)
        problems, _ = check_against_references(root, references, declared)
        assert problems and "exceed the tolerance" in problems[0]

    def test_a_reference_cannot_declare_a_non_positive_tolerance(self) -> None:
        with pytest.raises(ValueError):
            NumericReference(
                artifact_path=ARTIFACT,
                structure_sha256="a" * 64,
                value_count=0,
                values=(),
                atol=0.0,
                rtol=1e-6,
                numeric_tolerance="spec",
            )

    def test_the_committed_repository_states_one_tolerance(self) -> None:
        # The real references, from a clean checkout.
        from engagevr.mlops.numeric_reference import REFERENCE_DIRECTORY

        root = Path(__file__).resolve().parents[2] / REFERENCE_DIRECTORY
        assert verify_tolerance_agreement(root) == []
        manifest = read_manifest(root)
        assert manifest.atol == NUMERIC_PORTABILITY_ATOL
        assert manifest.rtol == NUMERIC_PORTABILITY_RTOL


def _declared_loose(path: str) -> CpuDependentNumericArtifact:
    """A stage-record entry claiming a wider tolerance than the accepted one."""
    return _declared(path, "a" * 64, atol=1e-2, rtol=1e-2)
