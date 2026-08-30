.PHONY: install format lint typecheck test test-cov check clean protocol \
        smoke mlops-demo dvc-dag dvc-repro dvc-verify docker-build docker-up docker-down \
        release-check

install:
	uv sync

format:
	uv run ruff format src tests

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest

test-cov:
	uv run pytest --cov --cov-report=term-missing

protocol:
	uv run python scripts/generate_protocol_artifacts.py

check: format lint typecheck test
	@echo "All checks passed."

# --- Milestone 10: MLOps, packaging, reproducibility ---
#
# Every target below is a SOFTWARE SELF-CHECK over SYNTHETIC data. None
# produces evidence, and none makes any model validated, approved, or
# production-ready.

# The integrated software self-check. Much smaller than the full suite:
# no webcam, no network, no Unity, no browser, no external dataset, no
# server. Exits non-zero if any component fails to interoperate.
smoke:
	uv run python -m engagevr system-smoke

# The whole deterministic pipeline in one process, without DVC.
mlops-demo:
	uv run python -m engagevr mlops-demo

dvc-dag:
	uv run dvc dag

# The same pipeline as a DAG. Reproduces from source; needs no remote,
# no account, and no network.
dvc-repro:
	uv run dvc repro

# Prove the invariant DEC-100 states: a fresh reproduction from the same
# source, lock, configuration, seed, and parameters leaves the tracked
# dvc.lock byte-identical. A difference here is a defect, not churn.
dvc-verify:
	@uv run dvc repro
	@sha256sum dvc.lock > .dvc-verify-first
	@rm -rf artifacts/pipeline
	@uv run dvc repro
	@sha256sum dvc.lock > .dvc-verify-second
	@if diff -q .dvc-verify-first .dvc-verify-second > /dev/null; then \
		echo "dvc.lock is byte-stable across a fresh reproduction:"; \
		cat .dvc-verify-first; \
		rm -f .dvc-verify-first .dvc-verify-second; \
	else \
		echo "FAIL: dvc.lock changed across a fresh reproduction."; \
		echo "A declared output stopped being byte-stable. See DEC-104."; \
		diff .dvc-verify-first .dvc-verify-second || true; \
		rm -f .dvc-verify-first .dvc-verify-second; \
		exit 1; \
	fi

docker-build:
	docker build -f Dockerfile.backend -t engagevr-backend:local .
	docker build -f Dockerfile.dashboard -t engagevr-dashboard:local .

# Both ports are published to 127.0.0.1 only. Neither service has
# authentication; see docker-compose.yml.
docker-up:
	docker compose up -d

docker-down:
	docker compose down

# The full pre-release gate, in the order docs/RELEASE.md documents.
# It does not tag, push, or publish anything: the repository owner does
# that by hand.
release-check:
	uv lock --check
	uv run ruff format --check src tests
	uv run ruff check src tests
	uv run mypy src
	uv run pytest
	uv run python scripts/generate_protocol_artifacts.py
	git diff --exit-code -- protocol/
	$(MAKE) dvc-verify
	uv run python -m engagevr system-smoke
	@echo "Release checks passed. This is a SOFTWARE SELF-CHECK, not"
	@echo "scientific evaluation. See docs/RELEASE.md for the remaining"
	@echo "manual steps, including the Docker validation."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Remove ONLY the generated Milestone 10 outputs. Scoped deliberately:
# it never touches artifacts/experiments, artifacts/sessions, or
# artifacts/datasets, which hold Milestone 5-9 evidence a reader may
# still be looking at.
.PHONY: clean-mlops
clean-mlops:
	rm -rf artifacts/pipeline artifacts/smoke mlruns
	@echo "Removed artifacts/pipeline, artifacts/smoke, mlruns."
	@echo "Left untouched: artifacts/experiments, artifacts/sessions,"
	@echo "artifacts/datasets, and dvc.lock (which is tracked)."
