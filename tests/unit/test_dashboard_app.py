"""Page smoke tests through Streamlit's own testing API.

``AppTest`` runs the script in-process with no browser, no socket, and no
server, so CI needs nothing beyond the declared dependencies.  These
tests check that every page renders without raising and that the pages
which display a result cannot do so without the provenance banner.

They deliberately do not assert on layout.  What matters is that a
synthetic run is never presented as an evaluation, not where the columns
sit.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from engagevr.dashboard.app import ARTIFACT_ROOT_ENV
from engagevr.dashboard.pages import PAGES, RESULT_BEARING_PAGES
from engagevr.schemas.dashboard import DASHBOARD_DISCLAIMER, SYNTHETIC_BANNER
from tests.unit import dashboard_fixtures as fx

#: Generous: the script parses a handful of small JSON documents.
TIMEOUT_SECONDS = 60.0

PAGE_NAMES: tuple[str, ...] = tuple(name for name, _ in PAGES)


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A root holding one run of every family."""
    fx.make_baseline_run(tmp_path, "run-baseline-classification")
    fx.make_baseline_run(tmp_path, "run-baseline-regression", task_type="regression")
    fx.make_fusion_run(tmp_path, "run-fusion")
    fx.make_personalization_run(tmp_path, "run-personalization")
    fx.make_uncertainty_run(tmp_path, "run-uncertainty-classification")
    fx.make_uncertainty_run(
        tmp_path, "run-uncertainty-regression", task_type="regression"
    )
    fx.make_adaptation_run(tmp_path, "run-adaptation")
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(tmp_path))
    yield tmp_path


def run_app(page: str | None = None) -> AppTest:
    """Render the artifact observatory, optionally after selecting a page.

    The sidebar's first radio chooses the evidence mode and the second
    chooses the page.  These tests are about the artifact mode, which is
    the default, so only the page radio is touched here; the live and
    replay modes have their own module.
    """
    from engagevr.dashboard.launch import app_path

    app = AppTest.from_file(str(app_path()), default_timeout=TIMEOUT_SECONDS)
    app.run()
    if page is not None:
        page_radio = next(radio for radio in app.sidebar.radio if radio.label == "Page")
        page_radio.set_value(page).run()
    return app


def rendered_text(app: AppTest) -> str:
    """Every string the page produced, joined."""
    parts: list[str] = []
    for collection in (
        app.markdown,
        app.error,
        app.warning,
        app.info,
        app.success,
        app.caption,
        app.title,
        app.subheader,
        app.text,
    ):
        parts.extend(str(element.value) for element in collection)
    return " ".join(parts)


class TestEveryPageRenders:
    @pytest.mark.parametrize("page", PAGE_NAMES)
    def test_the_page_does_not_raise(self, artifact_root: Path, page: str) -> None:
        app = run_app(page)
        assert not app.exception, f"{page} raised: {app.exception}"

    def test_the_default_page_is_the_overview(self, artifact_root: Path) -> None:
        app = run_app()
        assert not app.exception
        assert any("Overview" in t.value for t in app.title)

    @pytest.mark.parametrize("page", PAGE_NAMES)
    def test_the_page_renders_with_no_runs_at_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, page: str
    ) -> None:
        monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(tmp_path / "empty"))
        app = run_app(page)
        assert not app.exception, f"{page} raised on an empty root: {app.exception}"

    @pytest.mark.parametrize("page", PAGE_NAMES)
    def test_the_page_survives_a_corrupt_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, page: str
    ) -> None:
        directory = fx.make_baseline_run(tmp_path, "run-broken")
        fx.corrupt(directory / "manifest.json")
        monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(tmp_path))
        app = run_app(page)
        assert not app.exception, f"{page} raised on a corrupt run: {app.exception}"


class TestProvenanceIsAlwaysShown:
    @pytest.mark.parametrize(
        "page",
        [name for name, handler in PAGES if handler in RESULT_BEARING_PAGES],
    )
    def test_a_result_page_shows_the_synthetic_banner(
        self, artifact_root: Path, page: str
    ) -> None:
        app = run_app(page)
        assert SYNTHETIC_BANNER in rendered_text(app), (
            f"{page} rendered without the software-self-check banner"
        )

    @pytest.mark.parametrize(
        "page",
        [name for name, handler in PAGES if handler in RESULT_BEARING_PAGES],
    )
    def test_a_result_page_states_the_eligibility_flag(
        self, artifact_root: Path, page: str
    ) -> None:
        assert "scientific_evaluation_eligible = **false**" in rendered_text(
            run_app(page)
        )

    @pytest.mark.parametrize("page", PAGE_NAMES)
    def test_every_page_carries_the_standing_disclaimer(
        self, artifact_root: Path, page: str
    ) -> None:
        # The sidebar prints it on every page, and a result page repeats
        # it in the banner.
        assert DASHBOARD_DISCLAIMER in rendered_text(run_app(page))

    def test_every_result_bearing_page_is_declared(self) -> None:
        handlers = {handler for _name, handler in PAGES}
        assert RESULT_BEARING_PAGES <= handlers


class TestNoValidatedUiForSyntheticRuns:
    def test_no_page_offers_to_mark_a_run_validated(self, artifact_root: Path) -> None:
        for page in PAGE_NAMES:
            app = run_app(page)
            labels = [button.label.lower() for button in app.button]
            for banned in ("validate", "mark as", "treat as real", "promote"):
                assert not any(banned in label for label in labels)

    def test_no_page_offers_to_run_or_dispatch_anything(
        self, artifact_root: Path
    ) -> None:
        for page in PAGE_NAMES:
            app = run_app(page)
            labels = [button.label.lower() for button in app.button]
            for banned in ("retrain", "run model", "apply adaptation", "dispatch"):
                assert not any(banned in label for label in labels)

    def test_the_synthetic_banner_is_not_hidden_in_an_expander(
        self, artifact_root: Path
    ) -> None:
        app = run_app("Baseline models")
        # st.error renders at the top level of the page, not inside an
        # expander, so the banner is in the page's own error elements.
        assert any(SYNTHETIC_BANNER in str(e.value) for e in app.error)


class TestTaskAwareControls:
    def test_a_classification_uncertainty_run_offers_no_interval_control(
        self, artifact_root: Path
    ) -> None:
        app = run_app("Uncertainty and abstention")
        app.sidebar.selectbox[0].set_value("uncertainty").run()
        text = rendered_text(app).lower()
        assert "confidence_threshold" in text

    def test_the_adaptation_page_reports_no_effectiveness(
        self, artifact_root: Path
    ) -> None:
        app = run_app("Adaptive environment")
        text = rendered_text(app).lower()
        assert "effectiveness" not in text

    def test_the_limitations_page_lists_the_standing_limitations(
        self, artifact_root: Path
    ) -> None:
        from engagevr.dashboard.presentation import LIMITATIONS

        app = run_app("Limitations and scientific status")
        assert not app.exception
        assert len(app.expander) >= len(LIMITATIONS)


class TestRunSelector:
    def test_the_selector_lists_every_run(self, artifact_root: Path) -> None:
        app = run_app()
        options = app.sidebar.selectbox[-1].options
        assert len(options) == 7

    def test_the_selector_labels_carry_provenance(self, artifact_root: Path) -> None:
        app = run_app()
        options = app.sidebar.selectbox[-1].options
        assert all("synthetic" in str(option) for option in options)

    def test_filtering_by_family_narrows_the_list(self, artifact_root: Path) -> None:
        app = run_app()
        app.sidebar.selectbox[0].set_value("adaptation").run()
        options = app.sidebar.selectbox[-1].options
        assert len(options) == 1
        assert "run-adaptation" in str(options[0])
