"""Packaging contracts: DVC, Docker, GitHub Actions, and .gitignore.

These are static checks on files the test suite cannot execute: a unit
test cannot run a GitHub-hosted workflow, and requiring Docker to run
``pytest`` would make the suite unrunnable on a machine that does not
have it.  What they *can* check is that the contracts hold — that every
DVC stage calls a real subcommand, that no image bakes in generated
state, that CI uses the lock file and never runs the hardware suite, and
that no generated artifact is on its way into Git.

They are deliberately insensitive to formatting: the YAML is parsed, not
grepped, wherever a structural check is possible.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from engagevr.mlops.pipeline import (
    STAGE_NAMES,
    PipelineParameters,
    build_stages,
    default_layout,
    load_parameters,
)

ROOT = Path(__file__).resolve().parents[2]
DVC_YAML = ROOT / "dvc.yaml"
PARAMS_YAML = ROOT / "params.yaml"
DOCKERIGNORE = ROOT / ".dockerignore"
DOCKER_COMPOSE = ROOT / "docker-compose.yml"
BACKEND_DOCKERFILE = ROOT / "Dockerfile.backend"
DASHBOARD_DOCKERFILE = ROOT / "Dockerfile.dashboard"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
GITIGNORE = ROOT / ".gitignore"
DVC_CONFIG = ROOT / ".dvc" / "config"
MAKEFILE = ROOT / "Makefile"


def load_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path} does not hold a mapping"
    return document


def cli_subcommands() -> set[str]:
    """Every subcommand the project's argument parser actually registers."""
    from engagevr.__main__ import _build_parser

    parser = _build_parser()
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if action.dest == "command" and choices:
            return set(choices)
    raise AssertionError("the CLI registers no subcommands")


def workflow_steps(job: dict[str, Any]) -> list[str]:
    return [str(step.get("run", "")) for step in job.get("steps", [])]


# ---------------------------------------------------------------------------
# DVC
# ---------------------------------------------------------------------------


class TestDvcPipeline:
    @pytest.fixture(scope="class")
    def pipeline(self) -> dict[str, Any]:
        return load_yaml(DVC_YAML)

    @pytest.fixture(scope="class")
    def stages(self, pipeline: dict[str, Any]) -> dict[str, Any]:
        return pipeline["stages"]

    def test_the_pipeline_and_its_parameters_are_source_controlled(self) -> None:
        assert DVC_YAML.is_file()
        assert PARAMS_YAML.is_file()
        assert DVC_CONFIG.is_file()
        assert (ROOT / ".dvcignore").is_file()

    def test_the_declared_stages_match_the_python_definition(
        self, stages: dict[str, Any]
    ) -> None:
        assert tuple(stages) == STAGE_NAMES

    def test_every_stage_calls_a_real_engagevr_subcommand(
        self, stages: dict[str, Any]
    ) -> None:
        available = cli_subcommands()
        for name, stage in stages.items():
            command = " ".join(str(stage["cmd"]).split())
            match = re.search(r"python -m engagevr ([a-z0-9-]+)", command)
            assert match, f"stage {name!r} does not invoke the engagevr CLI"
            assert match.group(1) in available, (
                f"stage {name!r} invokes {match.group(1)!r}, which the CLI "
                "does not register"
            )

    def test_no_stage_reimplements_modelling(self, stages: dict[str, Any]) -> None:
        for name, stage in stages.items():
            command = str(stage["cmd"])
            assert "python -c" not in command, name
            assert "sklearn" not in command, name
            assert "import " not in command, name

    def test_every_stage_declares_dependencies(self, stages: dict[str, Any]) -> None:
        for name, stage in stages.items():
            assert stage.get("deps"), f"stage {name!r} declares no dependencies"

    def test_every_stage_declares_outputs(self, stages: dict[str, Any]) -> None:
        for name, stage in stages.items():
            assert stage.get("outs"), f"stage {name!r} declares no outputs"

    def test_the_generating_stages_declare_their_parameters(
        self, stages: dict[str, Any]
    ) -> None:
        for name in ("dataset-reference", "dataset-current", "baseline", "uncertainty"):
            assert stages[name].get("params"), f"stage {name!r} declares no params"

    def test_every_referenced_parameter_exists(self, pipeline: dict[str, Any]) -> None:
        params = load_yaml(PARAMS_YAML)
        rendered = DVC_YAML.read_text(encoding="utf-8")
        for reference in set(re.findall(r"\$\{([a-z_.]+)\}", rendered)):
            cursor: Any = params
            for part in reference.split("."):
                assert isinstance(cursor, dict) and part in cursor, (
                    f"dvc.yaml references ${{{reference}}}, which params.yaml "
                    "does not define"
                )
                cursor = cursor[part]

    def test_every_declared_parameter_exists(self, stages: dict[str, Any]) -> None:
        params = load_yaml(PARAMS_YAML)
        for name, stage in stages.items():
            for dotted in stage.get("params", []):
                cursor: Any = params
                for part in str(dotted).split("."):
                    assert isinstance(cursor, dict) and part in cursor, (
                        f"stage {name!r} declares parameter {dotted!r}, which "
                        "params.yaml does not define"
                    )
                    cursor = cursor[part]

    def test_params_yaml_validates_against_the_pipeline_model(self) -> None:
        parameters = load_parameters(PARAMS_YAML)
        assert isinstance(parameters, PipelineParameters)
        assert parameters.current_seed != parameters.reference_seed

    def test_every_output_lands_in_the_gitignored_pipeline_root(
        self, stages: dict[str, Any]
    ) -> None:
        for name, stage in stages.items():
            for out in stage["outs"]:
                path = out if isinstance(out, str) else next(iter(out))
                assert str(path).startswith("artifacts/pipeline/"), (
                    f"stage {name!r} writes {path!r} outside the pipeline root"
                )

    def test_no_output_is_cached_by_dvc(self, stages: dict[str, Any]) -> None:
        # The demo is regenerated from source, not restored: there is no
        # remote to pull from and no binary belongs in the repository.
        for name, stage in stages.items():
            for out in stage["outs"]:
                assert isinstance(out, dict), (
                    f"stage {name!r} declares an output without cache: false"
                )
                options = next(iter(out.values()))
                assert options.get("cache") is False, name

    def test_no_stage_starts_a_long_running_server(
        self, stages: dict[str, Any]
    ) -> None:
        for name, stage in stages.items():
            command = str(stage["cmd"])
            for forbidden in (
                "streamlit",
                "uvicorn",
                " serve",
                "mlflow ui",
                "mlflow server",
            ):
                assert forbidden not in command, f"stage {name!r} runs {forbidden!r}"

    def test_no_stage_reaches_the_network_or_a_participant_dataset(
        self, stages: dict[str, Any]
    ) -> None:
        for name, stage in stages.items():
            command = str(stage["cmd"])
            for forbidden in (
                "curl",
                "wget",
                "http://",
                "https://",
                "dvc pull",
                "git ",
            ):
                assert forbidden not in command, f"stage {name!r} runs {forbidden!r}"
            assert "rppg-evaluate" not in command, name

    def test_the_dvc_configuration_disables_telemetry_and_auto_staging(self) -> None:
        text = DVC_CONFIG.read_text(encoding="utf-8")
        assert re.search(r"analytics\s*=\s*false", text)
        assert re.search(r"autostage\s*=\s*false", text)

    def test_no_dvc_remote_is_required(self) -> None:
        text = DVC_CONFIG.read_text(encoding="utf-8")
        assert "[core]" in text
        assert "remote" not in text.replace("# ", "").split("[core]")[0]
        assert "['remote" not in text and '["remote' not in text

    def test_the_python_stage_definition_and_dvc_agree_on_the_commands(self) -> None:
        parameters = load_parameters(PARAMS_YAML)
        layout = default_layout("artifacts/pipeline", parameters.target)
        declared = load_yaml(DVC_YAML)["stages"]
        for stage in build_stages(layout, parameters):
            rendered = " ".join(str(declared[stage.name]["cmd"]).split())
            rendered = rendered.replace("${pipeline.target}", parameters.target)
            for token in ("--output", "--dataset", "--run", "--reference"):
                if token in stage.command:
                    assert token in rendered, (
                        f"stage {stage.name!r} differs between dvc.yaml and "
                        "engagevr.mlops.pipeline"
                    )


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------


class TestDockerfiles:
    def test_both_dockerfiles_and_the_compose_file_exist(self) -> None:
        assert BACKEND_DOCKERFILE.is_file()
        assert DASHBOARD_DOCKERFILE.is_file()
        assert DOCKER_COMPOSE.is_file()
        assert DOCKERIGNORE.is_file()

    def test_the_backend_image_runs_the_existing_backend(self) -> None:
        text = BACKEND_DOCKERFILE.read_text(encoding="utf-8")
        assert '"serve"' in text
        assert "engagevr" in text
        # No new model-serving API was invented for a packaging milestone.
        # Instructions only: the prose above them says there is no
        # inference endpoint, and that sentence must stay sayable.
        instructions = " ".join(
            line.lower()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        for forbidden in ("predict", "inference", "model-server", "/invocations"):
            assert forbidden not in instructions

    def test_the_dashboard_image_runs_the_existing_dashboard(self) -> None:
        text = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")
        assert '"dashboard"' in text
        assert "--artifact-root" in text

    def test_the_dashboard_image_opens_no_browser(self) -> None:
        text = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")
        assert "STREAMLIT_SERVER_HEADLESS=true" in text
        assert "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false" in text

    @pytest.mark.parametrize(
        "path", [BACKEND_DOCKERFILE, DASHBOARD_DOCKERFILE], ids=["backend", "dashboard"]
    )
    def test_the_image_pins_python_312(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        assert "python:3.12" in text

    @pytest.mark.parametrize(
        "path", [BACKEND_DOCKERFILE, DASHBOARD_DOCKERFILE], ids=["backend", "dashboard"]
    )
    def test_the_image_installs_from_the_lock_file(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        assert "uv.lock" in text
        assert "uv sync --locked" in text
        assert "pip install" not in text

    @pytest.mark.parametrize(
        "path", [BACKEND_DOCKERFILE, DASHBOARD_DOCKERFILE], ids=["backend", "dashboard"]
    )
    def test_the_image_runs_as_a_non_root_user(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        assert "useradd" in text
        assert re.search(r"^USER engagevr$", text, re.MULTILINE)

    @pytest.mark.parametrize(
        "path", [BACKEND_DOCKERFILE, DASHBOARD_DOCKERFILE], ids=["backend", "dashboard"]
    )
    def test_the_image_declares_a_health_check(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        assert "HEALTHCHECK" in text

    @pytest.mark.parametrize(
        "path", [BACKEND_DOCKERFILE, DASHBOARD_DOCKERFILE], ids=["backend", "dashboard"]
    )
    def test_the_image_exposes_exactly_one_port(self, path: Path) -> None:
        exposed = re.findall(r"^EXPOSE (.+)$", path.read_text(encoding="utf-8"), re.M)
        assert len(exposed) == 1
        assert len(exposed[0].split()) == 1

    @pytest.mark.parametrize(
        "path", [BACKEND_DOCKERFILE, DASHBOARD_DOCKERFILE], ids=["backend", "dashboard"]
    )
    def test_no_secret_or_credential_is_baked_in(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8").lower()
        for forbidden in (
            "api_key",
            "apikey",
            "secret_key",
            "password",
            "token=",
            "aws_access",
            "-----begin",
        ):
            assert forbidden not in text, forbidden

    @pytest.mark.parametrize(
        "path", [BACKEND_DOCKERFILE, DASHBOARD_DOCKERFILE], ids=["backend", "dashboard"]
    )
    def test_no_generated_state_is_copied_into_the_image(self, path: Path) -> None:
        copies = re.findall(
            r"^COPY (?!--from)(.+)$", path.read_text(encoding="utf-8"), re.M
        )
        for line in copies:
            sources = line.replace("--chown=engagevr:engagevr", "").split()[:-1]
            for source in sources:
                assert not source.startswith("artifacts"), source
                assert not source.startswith("data"), source
                assert not source.startswith("models"), source
                assert source not in {".", "./", "mlruns", "dvc.lock"}, source

    def test_the_backend_binds_a_loopback_published_port_only(self) -> None:
        compose = load_yaml(DOCKER_COMPOSE)
        for name, service in compose["services"].items():
            for published in service.get("ports", []):
                assert str(published).startswith("127.0.0.1:"), (
                    f"service {name!r} publishes {published!r} beyond loopback"
                )

    def test_the_dashboard_mounts_its_roots_read_only(self) -> None:
        compose = load_yaml(DOCKER_COMPOSE)
        volumes = compose["services"]["dashboard"]["volumes"]
        assert volumes
        for volume in volumes:
            assert str(volume).endswith(":ro"), volume

    def test_the_compose_file_carries_no_secret(self) -> None:
        text = DOCKER_COMPOSE.read_text(encoding="utf-8").lower()
        for forbidden in ("password", "api_key", "secret:", "token:"):
            assert forbidden not in text
        compose = load_yaml(DOCKER_COMPOSE)
        assert "secrets" not in compose
        for service in compose["services"].values():
            assert "secrets" not in service
            assert "env_file" not in service

    def test_the_compose_file_defines_only_the_two_existing_services(self) -> None:
        compose = load_yaml(DOCKER_COMPOSE)
        assert set(compose["services"]) == {"backend", "dashboard"}

    def test_both_services_declare_a_health_check(self) -> None:
        compose = load_yaml(DOCKER_COMPOSE)
        for name, service in compose["services"].items():
            assert "healthcheck" in service, name


class TestDockerignore:
    @pytest.fixture(scope="class")
    def patterns(self) -> set[str]:
        return {
            line.strip()
            for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

    @pytest.mark.parametrize(
        "pattern",
        [
            ".git/",
            ".venv/",
            "artifacts/",
            "models/",
            "mlruns/",
            "*.joblib",
            "*.pkl",
            ".env",
            "secrets/",
            ".pytest_cache/",
            ".mypy_cache/",
            "CLAUDE.local.md",
            ".claude/",
            "*.mp4",
            "*.npy",
            "data/raw/",
            "*.db",
        ],
    )
    def test_the_forbidden_path_is_excluded(
        self, patterns: set[str], pattern: str
    ) -> None:
        assert pattern in patterns


# ---------------------------------------------------------------------------
# GitHub Actions
# ---------------------------------------------------------------------------


class TestContinuousIntegration:
    @pytest.fixture(scope="class")
    def workflow(self) -> dict[str, Any]:
        return load_yaml(WORKFLOW)

    @pytest.fixture(scope="class")
    def jobs(self, workflow: dict[str, Any]) -> dict[str, Any]:
        return workflow["jobs"]

    def test_the_three_jobs_exist(self, jobs: dict[str, Any]) -> None:
        assert {"check", "smoke", "docker"} <= set(jobs)

    def test_python_312_is_the_version_under_test(self, jobs: dict[str, Any]) -> None:
        for name in ("check", "smoke"):
            assert any("3.12" in step for step in workflow_steps(jobs[name])), name

    def test_the_lock_file_is_checked_and_used(self, jobs: dict[str, Any]) -> None:
        check = " ".join(workflow_steps(jobs["check"]))
        assert "uv lock --check" in check
        assert "uv sync --locked" in check
        assert "uv sync --locked" in " ".join(workflow_steps(jobs["smoke"]))

    def test_no_ad_hoc_pip_install_appears(self, workflow: dict[str, Any]) -> None:
        rendered = WORKFLOW.read_text(encoding="utf-8")
        assert "pip install" not in rendered

    def test_ruff_mypy_and_pytest_all_run(self, jobs: dict[str, Any]) -> None:
        check = " ".join(workflow_steps(jobs["check"]))
        assert "ruff format --check" in check
        assert "ruff check" in check
        assert "mypy src" in check
        assert "pytest" in check

    def test_the_protocol_drift_check_runs(self, jobs: dict[str, Any]) -> None:
        check = " ".join(workflow_steps(jobs["check"]))
        assert "generate_protocol_artifacts.py" in check
        assert "git diff --exit-code -- protocol/" in check

    def test_the_smoke_and_dvc_steps_run(self, jobs: dict[str, Any]) -> None:
        smoke = " ".join(workflow_steps(jobs["smoke"]))
        assert "system-smoke" in smoke
        assert "dvc dag" in smoke
        assert "dvc repro" in smoke

    def test_ci_gates_on_dvc_lock_byte_stability(self, jobs: dict[str, Any]) -> None:
        smoke = " ".join(workflow_steps(jobs["smoke"]))
        assert "sha256sum dvc.lock" in smoke
        assert "git diff --exit-code -- dvc.lock" in smoke

    def test_ci_runs_the_two_source_tree_proof(self, jobs: dict[str, Any]) -> None:
        smoke = " ".join(workflow_steps(jobs["smoke"]))
        assert "ENGAGEVR_RUN_DVC_SYSTEM_TESTS=1" in smoke
        assert "-m dvc_system" in smoke

    def test_the_docker_job_builds_both_images_and_validates_compose(
        self, jobs: dict[str, Any]
    ) -> None:
        docker = " ".join(workflow_steps(jobs["docker"]))
        assert "docker compose config" in docker
        assert "Dockerfile.backend" in docker
        assert "Dockerfile.dashboard" in docker
        assert "docker compose down" in docker

    def test_the_hardware_suite_is_never_required(
        self, workflow: dict[str, Any]
    ) -> None:
        rendered = WORKFLOW.read_text(encoding="utf-8")
        assert "ENGAGEVR_RUN_HARDWARE_TESTS=1" not in rendered
        assert "-m hardware" not in rendered

    def test_no_secret_is_required(self, workflow: dict[str, Any]) -> None:
        rendered = WORKFLOW.read_text(encoding="utf-8")
        assert "secrets." not in rendered
        assert "${{ secrets" not in rendered

    def test_no_step_depends_on_a_contributor_machine(self) -> None:
        rendered = WORKFLOW.read_text(encoding="utf-8")
        for forbidden in ("~/.claude", "/home/", "C:\\", "$HOME/.claude"):
            assert forbidden not in rendered

    def test_no_external_dataset_is_fetched(self) -> None:
        rendered = WORKFLOW.read_text(encoding="utf-8")
        for forbidden in ("wget ", "curl -O", "dvc pull", "UBFC"):
            assert forbidden not in rendered

    def test_permissions_are_read_only(self, workflow: dict[str, Any]) -> None:
        assert workflow.get("permissions", {}).get("contents") == "read"

    def test_uploaded_artifacts_are_synthetic_reports_only(
        self, jobs: dict[str, Any]
    ) -> None:
        uploads = [
            step
            for step in jobs["smoke"]["steps"]
            if "upload-artifact" in str(step.get("uses", ""))
        ]
        for step in uploads:
            paths = str(step["with"]["path"])
            assert ".joblib" not in paths
            assert ".parquet" not in paths
            assert "sessions" not in paths
            assert int(step["with"]["retention-days"]) <= 30


# ---------------------------------------------------------------------------
# Generated output stays out of Git
# ---------------------------------------------------------------------------


class TestGitIgnore:
    @pytest.fixture(scope="class")
    def lines(self) -> set[str]:
        return {
            line.strip()
            for line in GITIGNORE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }

    @pytest.mark.parametrize(
        "pattern",
        [
            "artifacts/",
            "models/",
            "mlruns/",
            "mlflow.db",
            ".dvc/cache/",
            ".dvc/tmp/",
            ".dvc/config.local",
            ".env",
            "secrets/",
            ".pytest_cache/",
            ".ruff_cache/",
        ],
    )
    def test_the_generated_path_is_ignored(self, lines: set[str], pattern: str) -> None:
        assert pattern in lines

    @pytest.mark.parametrize(
        "path",
        [
            "dvc.yaml",
            "params.yaml",
            # Tracked, because DVC refuses to run with an ignored lock.
            # It is a record of one execution, not a guarantee: see
            # DEC-100 and docs/MLOPS.md section 6.
            "dvc.lock",
            ".dvcignore",
            ".dvc/config",
            "Dockerfile.backend",
            "Dockerfile.dashboard",
            "docker-compose.yml",
            ".dockerignore",
            "configs/defaults.yaml",
        ],
    )
    def test_the_source_controlled_file_is_not_ignored(self, path: str) -> None:
        import subprocess

        result = subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0, f"{path} is gitignored but is source"

    def test_no_generated_binary_is_tracked(self) -> None:
        import subprocess

        result = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        for name in result.stdout.splitlines():
            assert not name.endswith((".joblib", ".pkl", ".parquet", ".db")), name


class TestMakefile:
    @pytest.fixture(scope="class")
    def text(self) -> str:
        return MAKEFILE.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "target",
        [
            "check",
            "lint",
            "typecheck",
            "test",
            "smoke",
            "dvc-repro",
            "dvc-verify",
            "docker-build",
            "release-check",
        ],
    )
    def test_the_target_exists(self, text: str, target: str) -> None:
        assert re.search(rf"^{re.escape(target)}:", text, re.MULTILINE)

    def test_the_existing_check_target_is_unchanged(self, text: str) -> None:
        assert re.search(r"^check: format lint typecheck test$", text, re.MULTILINE)

    def test_the_scoped_clean_never_removes_milestone_5_to_9_evidence(
        self, text: str
    ) -> None:
        clean = text.split("clean-mlops:")[1]
        removal = clean.split("@echo")[0]
        assert "artifacts/pipeline" in removal
        assert "artifacts/smoke" in removal
        assert "artifacts/experiments" not in removal
        assert "artifacts/sessions" not in removal
        # dvc.lock is tracked; a clean target must not delete it.
        assert "dvc.lock" not in removal
        assert "git clean" not in text
