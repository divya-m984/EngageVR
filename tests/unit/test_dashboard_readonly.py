"""The read-only boundary, the layering, and privacy.

Most of these are AST tests over the dashboard's own source.  A comment
saying "this module never writes" is not a guarantee; a test that fails
when someone imports ``open`` in write mode is.

Four properties are checked here:

* the dashboard never writes, deletes, or renames an artifact;
* it never trains, calibrates, or re-runs a milestone's pipeline;
* it never dispatches an adaptation or opens a transport;
* it never loads a model file, because those are executable pickles.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import engagevr.dashboard as dashboard_package

DASHBOARD = Path(dashboard_package.__file__).parent
CLI = DASHBOARD.parent / "cli_milestone9.py"
SCHEMA = DASHBOARD.parent / "schemas" / "dashboard.py"
SESSION_SCHEMA = DASHBOARD.parent / "schemas" / "dashboard_session.py"

#: Every dashboard source file, plus the Milestone 9 CLI and schemas.
SOURCES: tuple[Path, ...] = (
    *sorted(DASHBOARD.glob("*.py")),
    CLI,
    SCHEMA,
    SESSION_SCHEMA,
)

#: Modules that may import Streamlit. Everything else must stay
#: framework-free so the unit tests need no server.
STREAMLIT_LAYER: frozenset[str] = frozenset(
    {"components.py", "pages.py", "session_pages.py", "app.py"}
)

#: Modules that must never appear in a dashboard import statement.
FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {
        "joblib",
        "pickle",
        "sklearn",
        "mlflow",
        "dvc",
        "docker",
        "subprocess",
        "engagevr.training.runner",
        "engagevr.training.fusion_runner",
        "engagevr.training.personalization_runner",
        "engagevr.training.uncertainty_runner",
        "engagevr.adaptation.runner",
        "engagevr.adaptation.policy",
        "engagevr.adaptation.command",
        "engagevr.adaptation.lifecycle",
        "engagevr.transport",
        "engagevr.api",
        "websockets",
        "fastapi",
        "uvicorn",
        # The live and replay modes read recordings. They never produce
        # one, never re-emit one, and never drive a task.
        "engagevr.replay.player",
        "engagevr.replay.clock",
        "engagevr.simulator",
        "engagevr.task.simulator",
        "engagevr.task.generator",
        "engagevr.storage.jsonl",
        "asyncio",
    }
)

#: Function names that would mean the dashboard is doing modelling.
FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "fit",
        "fit_transform",
        "fit_predict",
        "partial_fit",
        "predict",
        "predict_proba",
        "calibrate",
        "run_training",
        "run_fusion",
        "run_personalization",
        "run_uncertainty",
        "run_adaptation",
        "evaluate_policy",
        "evaluate_adaptation_gate",
        "build_adaptation_command",
        "dispatch",
        "send",
        "acknowledge",
        "write_json_atomic",
        "write_parquet_atomic",
        "unlink",
        "rmtree",
        "remove",
        "rename",
        "replace",
        "mkdir",
        "touch",
        "write_text",
        "write_bytes",
        "write_table",
        "save_model",
        # Milestone 4 session writing and re-emission. A dashboard that
        # observes a recording must never become a participant in it.
        "SessionRecorder",
        "JsonlWriter",
        "ReplayPlayer",
        "TaskSimulator",
        "open_recorder",
        "record_drop",
        "publish",
        "broadcast",
        "connect",
        "run_simulation",
    }
)


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imported_modules(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def called_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


class TestSourcesExist:
    def test_every_dashboard_module_is_covered(self) -> None:
        assert len(SOURCES) >= 10
        assert all(path.is_file() for path in SOURCES)

    def test_the_session_modules_are_covered(self) -> None:
        names = {path.name for path in SOURCES}
        for expected in (
            "session_reader.py",
            "session_catalogue.py",
            "session_report.py",
            "session_pages.py",
            "views_session.py",
            "dashboard_session.py",
        ):
            assert expected in names, f"{expected} escapes the read-only tests"


class TestSessionCodeStaysReadOnly:
    """The live and replay modes read recordings; they never join one."""

    @pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
    def test_no_module_constructs_a_recorder_or_a_writer(self, path: Path) -> None:
        offending = called_names(parse(path)) & {
            "SessionRecorder",
            "JsonlWriter",
            "ReplayPlayer",
            "open_recorder",
            "record_drop",
        }
        assert not offending, (
            f"{path.name} constructs {sorted(offending)}. A dashboard that "
            "observes a recording must never write to one, re-emit from one, "
            "or become a participant in the session it is reading."
        )

    def test_the_session_reader_imports_no_writer(self) -> None:
        from engagevr.dashboard import session_reader

        source = Path(session_reader.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.update(alias.name for alias in node.names)
        assert "SessionRecorder" not in imported
        assert "JsonlWriter" not in imported
        assert "write_json_atomic" not in imported

    def test_the_session_reader_opens_files_for_reading_only(self) -> None:
        from engagevr.dashboard import session_reader

        text = Path(session_reader.__file__).read_text(encoding="utf-8")
        assert 'open("rb")' in text or '.open("rb")' in text
        for banned in ('"w"', '"a"', '"x"', '"r+"', '"w+"', '"a+"'):
            assert banned not in text, f"session_reader.py mentions mode {banned}"

    def test_no_session_module_starts_a_server_or_a_socket(self) -> None:
        """Checked through imports and calls, not prose.

        The package docstring says the tests need no socket; matching on
        the word would fail on the sentence that promises it.
        """
        for path in SOURCES:
            tree = parse(path)
            modules = imported_modules(tree)
            for banned in ("socket", "socketserver", "http.server", "asyncio"):
                assert not any(
                    name == banned or name.startswith(f"{banned}.") for name in modules
                ), f"{path.name} imports {banned}"
            offending = called_names(tree) & {"serve", "run_server", "create_task"}
            assert not offending, f"{path.name} calls {sorted(offending)}"

    def test_the_download_control_is_the_only_export_path(self) -> None:
        """A browser download is not a write to a source artifact."""
        from engagevr.dashboard import session_pages

        text = Path(session_pages.__file__).read_text(encoding="utf-8")
        assert "st.download_button" in text
        assert "write_text" not in text
        assert "write_bytes" not in text


class TestNoWrites:
    @pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
    def test_no_module_calls_a_write_or_delete(self, path: Path) -> None:
        offending = called_names(parse(path)) & FORBIDDEN_CALLS
        assert not offending, (
            f"{path.name} calls {sorted(offending)}. The dashboard displays "
            "what a run recorded; it never writes, deletes, retrains, "
            "recalibrates, or dispatches anything."
        )

    @pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
    def test_no_module_opens_a_file_for_writing(self, path: Path) -> None:
        tree = parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func
            is_open = (isinstance(name, ast.Name) and name.id == "open") or (
                isinstance(name, ast.Attribute) and name.attr == "open"
            )
            if not is_open:
                continue
            modes = [
                arg.value
                for arg in node.args[1:]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
            modes += [
                kw.value.value
                for kw in node.keywords
                if kw.arg == "mode"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ]
            for mode in modes:
                assert set(mode) <= {"r", "b", "t"}, (
                    f"{path.name} opens a file in mode {mode!r}"
                )

    @pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
    def test_no_module_imports_a_runner_or_a_model_loader(self, path: Path) -> None:
        modules = imported_modules(parse(path))
        offending = {
            name
            for name in modules
            if name in FORBIDDEN_IMPORTS
            or any(name.startswith(f"{banned}.") for banned in FORBIDDEN_IMPORTS)
        }
        # launch.py starts the Streamlit process and is the one place a
        # subprocess is legitimate; it still never writes an artifact.
        if path.name == "launch.py":
            offending -= {"subprocess"}
        assert not offending, f"{path.name} imports {sorted(offending)}"

    def test_only_launch_uses_a_subprocess(self) -> None:
        users = {
            path.name
            for path in SOURCES
            if "subprocess" in imported_modules(parse(path))
        }
        assert users == {"launch.py"}

    def test_no_module_performs_a_git_operation(self) -> None:
        for path in SOURCES:
            text = path.read_text(encoding="utf-8").lower()
            for banned in ("git commit", "git push", "git add", "gitpython", "pygit2"):
                assert banned not in text, f"{path.name} mentions {banned}"


class TestLayering:
    def test_only_the_top_layer_imports_streamlit(self) -> None:
        for path in sorted(DASHBOARD.glob("*.py")):
            imports = imported_modules(parse(path))
            uses_streamlit = any(
                name == "streamlit" or name.startswith("streamlit.") for name in imports
            )
            if path.name in STREAMLIT_LAYER:
                continue
            assert not uses_streamlit, (
                f"{path.name} imports Streamlit. The catalogue, loaders, "
                "views, formatting, and aggregation stay framework-free so "
                "the tests need no server."
            )

    def test_the_package_import_does_not_pull_in_streamlit(self) -> None:
        imports = imported_modules(parse(DASHBOARD / "__init__.py"))
        assert not any(name.startswith("streamlit") for name in imports)

    def test_the_cli_does_not_import_streamlit_at_module_level(self) -> None:
        tree = parse(CLI)
        top_level: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.add(node.module)
        assert not any(name.startswith("streamlit") for name in top_level)

    def test_the_view_modules_never_import_streamlit(self) -> None:
        for path in sorted(DASHBOARD.glob("views_*.py")):
            assert "streamlit" not in path.read_text(encoding="utf-8")


class TestNoModelLoading:
    @pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
    def test_no_module_mentions_a_model_file(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        assert ".joblib" not in text or "never" in text.lower(), (
            f"{path.name} references a .joblib file outside a prohibition"
        )

    def test_no_module_calls_joblib_load(self) -> None:
        for path in SOURCES:
            assert "joblib.load" not in path.read_text(encoding="utf-8")

    def test_no_module_unpickles(self) -> None:
        for path in SOURCES:
            text = path.read_text(encoding="utf-8")
            assert "pickle.load" not in text
            assert "pickle.loads" not in text


class TestNoRawMedia:
    @pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
    def test_no_module_reads_an_image_or_a_frame(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8").lower()
        for banned in ("cv2", "imread", "st.image", "st.video", "st.camera_input"):
            assert banned not in text, f"{path.name} mentions {banned}"

    def test_no_module_imports_a_capture_module(self) -> None:
        for path in SOURCES:
            modules = imported_modules(parse(path))
            for name in modules:
                assert not name.startswith("engagevr.capture")
                assert not name.startswith("engagevr.face")
                assert not name.startswith("engagevr.rppg")


class TestPrivacy:
    #: Fixture and source files that must carry no personal data.
    FILES: tuple[Path, ...] = (
        *SOURCES,
        Path(__file__).parent / "dashboard_fixtures.py",
    )

    @pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
    def test_no_email_address_appears(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        assert "@gmail" not in text
        assert "@hotmail" not in text
        assert "mailto:" not in text

    def test_fixture_subject_ids_are_obviously_synthetic(self) -> None:
        from tests.unit import dashboard_fixtures as fx

        for subject in fx.SUBJECTS:
            assert subject.startswith("synthetic-subject-")

    def test_no_module_resolves_an_identifier_to_a_person(self) -> None:
        for path in SOURCES:
            text = path.read_text(encoding="utf-8").lower()
            for banned in (
                "participant_name",
                "full_name",
                "real_name",
                "profile_picture",
                "avatar",
            ):
                assert banned not in text, f"{path.name} mentions {banned}"

    def test_the_subject_note_states_the_boundary(self) -> None:
        from engagevr.dashboard.presentation import SUBJECT_ID_NOTE

        lowered = SUBJECT_ID_NOTE.lower()
        assert "pseudonymous" in lowered
        assert "does not rank subjects" in lowered
        assert "personal information" in lowered


class TestConfigurationCannotOverrideProvenance:
    def test_the_settings_carry_no_provenance_field(self) -> None:
        from engagevr.config import DashboardConfig

        fields = set(DashboardConfig.model_fields)
        for banned in (
            "scientific_evaluation_eligible",
            "is_synthetic",
            "confidence_threshold",
            "treat_as_real",
            "mark_validated",
            "policy_mapping",
        ):
            assert banned not in fields

    def test_the_settings_reject_an_unknown_field(self) -> None:
        from pydantic import ValidationError

        from engagevr.config import DashboardConfig

        with pytest.raises(ValidationError):
            DashboardConfig(mark_as_scientifically_validated=True)

    def test_an_absolute_artifact_root_is_refused(self) -> None:
        from pydantic import ValidationError

        from engagevr.config import DashboardConfig

        with pytest.raises(ValidationError, match="repository-relative"):
            DashboardConfig(artifact_root="/etc")

    def test_an_escaping_artifact_root_is_refused(self) -> None:
        from pydantic import ValidationError

        from engagevr.config import DashboardConfig

        with pytest.raises(ValidationError, match=r"escape the repository"):
            DashboardConfig(artifact_root="../../elsewhere")

    def test_an_unknown_default_family_is_refused(self) -> None:
        from pydantic import ValidationError

        from engagevr.config import DashboardConfig

        with pytest.raises(ValidationError, match="must be one of"):
            DashboardConfig(default_run_family="champion")

    def test_a_non_positive_table_limit_is_refused(self) -> None:
        from pydantic import ValidationError

        from engagevr.config import DashboardConfig

        with pytest.raises(ValidationError):
            DashboardConfig(max_table_rows=0)

    def test_the_defaults_file_carries_the_section(self) -> None:
        from engagevr.config import load_config

        settings = load_config().dashboard
        assert settings.artifact_root == "artifacts/experiments"
        assert settings.validate_checksums is True
        assert settings.max_table_rows == 1000


class TestSessionConfiguration:
    def test_the_session_root_is_configurable(self) -> None:
        from engagevr.config import DashboardConfig

        assert DashboardConfig().session_root == "artifacts/sessions"

    def test_the_session_root_is_not_the_artifact_root(self) -> None:
        from engagevr.config import DashboardConfig

        settings = DashboardConfig()
        assert settings.session_root != settings.artifact_root

    def test_an_absolute_session_root_is_refused(self) -> None:
        from pydantic import ValidationError

        from engagevr.config import DashboardConfig

        with pytest.raises(ValidationError, match="repository-relative"):
            DashboardConfig(session_root="/var/lib")

    def test_an_escaping_session_root_is_refused(self) -> None:
        from pydantic import ValidationError

        from engagevr.config import DashboardConfig

        with pytest.raises(ValidationError, match=r"escape the repository"):
            DashboardConfig(session_root="../../elsewhere")

    def test_a_high_frequency_refresh_is_refused(self) -> None:
        from pydantic import ValidationError

        from engagevr.config import DashboardConfig

        with pytest.raises(ValidationError):
            DashboardConfig(live_refresh_seconds=0.1)

    def test_the_export_switch_cannot_alter_provenance(self) -> None:
        from engagevr.config import DashboardConfig

        fields = set(DashboardConfig.model_fields)
        assert "enable_session_report_export" in fields
        for banned in ("strip_provenance", "export_as_validated", "hide_synthetic"):
            assert banned not in fields

    def test_the_defaults_file_carries_the_session_settings(self) -> None:
        from engagevr.config import load_config

        settings = load_config().dashboard
        assert settings.session_root == "artifacts/sessions"
        assert settings.live_refresh_seconds >= 2.0
        assert settings.enable_session_report_export is True
