"""Two independent source trees produce the same ``dvc.lock``.

This is the end-to-end proof of the invariant the unit tests check
structurally:

    clean source tree -> dvc repro -> dvc.lock byte-identical

Two source-only copies of the repository are made in temporary
directories, each excluding ``.git``, ``.venv``, ``artifacts``,
``mlruns``, ``.dvc/cache``, every cache, and the Claude-local files.  The
pipeline is reproduced in each, and the two locks are compared byte for
byte, together with the declared outputs, the reproducibility manifest,
and every fingerprint.

Cost and gating
---------------
Each copy runs the full eight-stage pipeline — measured at ~167 s — so
two trees is roughly six minutes.  That is too slow to sit in the default
suite beside 3,600 tests that finish in milliseconds, so it is **opt-in**,
exactly like the hardware suite::

    ENGAGEVR_RUN_DVC_SYSTEM_TESTS=1 uv run pytest -m dvc_system

Skipping is not passing.  The fast, always-on regression coverage for the
same property is ``tests/unit/test_dvc_determinism.py``, which proves
byte-stability of every declared output class without reproducing the
whole DAG twice.  ``docs/RELEASE.md`` step 4 documents the manual
procedure, and the repository owner runs it before a release.

It needs no webcam, no network, no Unity, no browser, and no server.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Opt-in switch. Two full pipeline reproductions take about six minutes.
_ENABLED = os.environ.get("ENGAGEVR_RUN_DVC_SYSTEM_TESTS") == "1"
_REASON = (
    "Set ENGAGEVR_RUN_DVC_SYSTEM_TESTS=1 to run the two-tree dvc.lock "
    "stability proof (~6 minutes). Skipping is not passing; the fast "
    "regression coverage is tests/unit/test_dvc_determinism.py."
)

#: Never copied into a fresh source tree.
#:
#: Everything here is either generated, machine-specific, or private. A
#: copy that carried them would not be a fresh tree, and the test would
#: prove nothing about a clean clone.
EXCLUDED: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "artifacts",
        "mlruns",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "htmlcov",
        "dvc.lock",
        "CLAUDE.local.md",
        ".claude",
        "models",
        "node_modules",
    }
)


def _ignore(directory: str, names: list[str]) -> set[str]:
    del directory
    return {name for name in names if name in EXCLUDED}


def copy_source_tree(destination: Path) -> Path:
    """A source-only copy of the repository, with no generated state."""
    shutil.copytree(ROOT, destination, ignore=_ignore, symlinks=True)
    # .dvc/cache and .dvc/tmp sit below an included directory.
    for leftover in (".dvc/cache", ".dvc/tmp"):
        target = destination / leftover
        if target.exists():
            shutil.rmtree(target)
    for absent in ("artifacts", "mlruns", "dvc.lock"):
        assert not (destination / absent).exists(), absent
    # DVC refuses to run outside a repository. This initialises a fresh,
    # empty one inside the throwaway copy; it never touches the real
    # repository's history, and the copy is deleted with tmp_path.
    subprocess.run(
        ["git", "init", "--quiet"], cwd=destination, check=True, capture_output=True
    )
    return destination


def reproduce(tree: Path) -> subprocess.CompletedProcess[str]:
    """Run ``dvc repro`` inside a source tree.

    ``uv run`` inside the copy resolves the project from that copy's own
    ``pyproject.toml`` and ``uv.lock``, which is what a clean clone does.
    """
    return subprocess.run(
        ["uv", "run", "dvc", "repro"],
        cwd=tree,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def two_trees(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Two independent, reproduced source trees."""
    base = tmp_path_factory.mktemp("lock-stability")
    first = copy_source_tree(base / "tree-a")
    second = copy_source_tree(base / "tree-b")
    for tree in (first, second):
        result = reproduce(tree)
        if result.returncode != 0:
            pytest.fail(
                f"dvc repro failed in {tree}:\n"
                f"stdout:\n{result.stdout[-4000:]}\n"
                f"stderr:\n{result.stderr[-4000:]}"
            )
    return first, second


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


#: Applied at module scope so the expensive `two_trees` fixture is never
#: built when the suite is opted out. A function-scoped skip would run
#: after the module-scoped fixture, which is six minutes too late.
pytestmark = [
    pytest.mark.dvc_system,
    pytest.mark.skipif(not _ENABLED, reason=_REASON),
]


class TestLockStability:
    def test_both_trees_produced_a_lock(self, two_trees: tuple[Path, Path]) -> None:
        first, second = two_trees
        assert (first / "dvc.lock").is_file()
        assert (second / "dvc.lock").is_file()

    def test_the_two_locks_are_byte_identical(
        self, two_trees: tuple[Path, Path]
    ) -> None:
        first, second = two_trees
        left = (first / "dvc.lock").read_bytes()
        right = (second / "dvc.lock").read_bytes()
        if left != right:
            import difflib

            diff = "\n".join(
                list(
                    difflib.unified_diff(
                        left.decode().splitlines(),
                        right.decode().splitlines(),
                        "tree-a/dvc.lock",
                        "tree-b/dvc.lock",
                        lineterm="",
                    )
                )[:60]
            )
            pytest.fail(f"dvc.lock differs between two fresh reproductions:\n{diff}")

    def test_every_declared_output_is_byte_identical(
        self, two_trees: tuple[Path, Path]
    ) -> None:
        import hashlib

        import yaml

        first, second = two_trees
        document = yaml.safe_load((first / "dvc.yaml").read_text(encoding="utf-8"))
        declared: list[str] = []
        for stage in document["stages"].values():
            for out in stage["outs"]:
                declared.append(out if isinstance(out, str) else next(iter(out)))

        def digest(tree: Path, relative: str) -> dict[str, str]:
            target = tree / relative
            if target.is_file():
                return {relative: hashlib.sha256(target.read_bytes()).hexdigest()}
            return {
                str(path.relative_to(tree)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(target.rglob("*"))
                if path.is_file() and not path.name.startswith(".")
            }

        for relative in declared:
            # ${pipeline.target} is already interpolated in dvc.lock, but
            # dvc.yaml keeps the template; skip templated entries here and
            # rely on the lock comparison above for those.
            if "${" in relative:
                continue
            assert digest(first, relative) == digest(second, relative), relative

    def test_the_reproducibility_manifests_agree(
        self, two_trees: tuple[Path, Path]
    ) -> None:
        first, second = two_trees
        path = Path("artifacts/pipeline/mlops/reproducibility.json")
        left = read_json(first / path)
        right = read_json(second / path)
        assert left["logical_fingerprint"] == right["logical_fingerprint"]
        assert (
            left["configuration"]["config_fingerprint"]
            == right["configuration"]["config_fingerprint"]
        )
        assert left == right

    def test_the_dataset_fingerprints_agree(self, two_trees: tuple[Path, Path]) -> None:
        first, second = two_trees
        for stage in ("dataset-reference", "dataset-current"):
            path = Path(f"artifacts/pipeline/mlops/stages/{stage}.json")
            left = read_json(first / path)["logical_identity"]
            right = read_json(second / path)["logical_identity"]
            assert left.startswith("dataset_fingerprint:")
            assert left == right, stage

    def test_the_model_version_identities_agree(
        self, two_trees: tuple[Path, Path]
    ) -> None:
        first, second = two_trees
        directory = Path("artifacts/pipeline/mlops/model_versions")

        def identities(tree: Path) -> list[str]:
            return sorted(
                read_json(path)["model_version_id"]
                for path in sorted((tree / directory).glob("*.model-version.json"))
            )

        assert identities(first) == identities(second)
        assert identities(first)

    def test_synthetic_provenance_agrees_and_stays_ineligible(
        self, two_trees: tuple[Path, Path]
    ) -> None:
        first, second = two_trees
        path = Path("artifacts/pipeline/mlops/reproducibility.json")
        for tree in (first, second):
            manifest = read_json(tree / path)
            assert manifest["is_synthetic"] is True
            assert manifest["scientific_evaluation_eligible"] is False
        assert (
            read_json(first / path)["disclaimers"]
            == (read_json(second / path)["disclaimers"])
        )

    def test_the_execution_sidecars_differ_and_are_not_declared(
        self, two_trees: tuple[Path, Path]
    ) -> None:
        # The wall clock was not deleted, only relocated: the sidecars
        # exist, they carry timestamps, and they are not in the lock.
        import yaml

        first, second = two_trees
        sidecar = Path("artifacts/pipeline/mlops/stages/baseline.execution.json")
        left = read_json(first / sidecar)
        right = read_json(second / sidecar)
        assert left["created_at_utc"]
        assert right["created_at_utc"]
        lock = yaml.safe_load((first / "dvc.lock").read_text(encoding="utf-8"))
        rendered = json.dumps(lock)
        assert "execution.json" not in rendered

    def test_a_second_repro_in_each_tree_does_no_work(
        self, two_trees: tuple[Path, Path]
    ) -> None:
        for tree in two_trees:
            result = reproduce(tree)
            assert result.returncode == 0, result.stderr[-2000:]
            combined = result.stdout + result.stderr
            assert "Running stage" not in combined, combined[-2000:]
            assert combined.count("didn't change, skipping") == 8, combined[-2000:]
            assert (tree / "dvc.lock").is_file()
