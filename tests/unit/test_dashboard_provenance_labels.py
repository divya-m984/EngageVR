"""Synthetic, public, and live data are visually labelled.

``docs/PROJECT_PLAN.md`` §"Milestone 9" accepts the dashboard only when
"synthetic, public, and live data are visually labelled".  No public
dataset and no live participant recording exists on this machine, and
"there is nothing to label" is not a demonstration that the labelling
works — it is a demonstration that it has never been exercised.

So all three cases are built here as **temporary fixtures** and rendered
through the real pages:

* an experiment run whose dataset records ``data_source=synthetic``,
  which must carry the ``SOFTWARE SELF-CHECK — NOT SCIENTIFIC
  EVALUATION`` banner;
* an experiment run whose dataset records ``data_source=public_dataset``
  — the project's own :class:`~engagevr.schemas.session.DataSource`
  member, not an invented string — which must be labelled **PUBLIC**;
* a session recording whose messages record ``data_source=live``, which
  must be labelled **LIVE** under **Mode: LIVE OBSERVATION**.

The second and third cases carry the load.  A public corpus and a live
capture are the two provenances a reader is most likely to promote to
evidence on their own, so each is checked to remain visibly *not*
scientifically eligible, and the standing disclaimer is checked to
survive every one of the three.

None of these fixtures is real data.  They are structures with the right
provenance fields and obviously synthetic contents, written to a
temporary directory and deleted with it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from engagevr.dashboard import presentation
from engagevr.dashboard import views_session as views
from engagevr.dashboard.app import ARTIFACT_ROOT_ENV, SESSION_ROOT_ENV
from engagevr.dashboard.session_catalogue import read_session
from engagevr.schemas.dashboard import DASHBOARD_DISCLAIMER, SYNTHETIC_BANNER
from engagevr.schemas.dashboard_session import (
    SESSION_ELIGIBILITY_NOTE,
    DashboardSessionMode,
)
from engagevr.schemas.session import DataSource
from tests.unit import dashboard_fixtures as fx
from tests.unit import session_fixtures as sfx

#: Generous: the script parses a handful of small local files.
TIMEOUT_SECONDS = 60.0

ARTIFACT_MODE = "Experiment artifacts"
LIVE_MODE = "Live session"
REPLAY_MODE = "Session replay"

#: The exact banner Milestone 5 onwards attaches to a self-check run.
SELF_CHECK = "SOFTWARE SELF-CHECK — NOT SCIENTIFIC EVALUATION"


# --- The vocabulary is the project's own ---------------------------------


class TestDataSourceVocabulary:
    def test_every_project_data_source_has_a_label(self) -> None:
        """No invented provenance strings; the enum is the vocabulary."""
        for member in DataSource:
            assert member.value in presentation.DATA_SOURCE_LABELS
            assert member.value in presentation.DATA_SOURCE_NOTES

    def test_the_banner_constant_is_the_self_check_wording(self) -> None:
        assert SYNTHETIC_BANNER == SELF_CHECK

    def test_the_public_label_says_public(self) -> None:
        label = presentation.data_source_label(DataSource.PUBLIC_DATASET.value)
        assert label.startswith("PUBLIC")
        assert "public_dataset" in label

    def test_the_live_label_says_live(self) -> None:
        label = presentation.data_source_label(DataSource.LIVE.value)
        assert label.startswith("LIVE")
        assert "live" in label

    def test_the_synthetic_label_says_synthetic(self) -> None:
        assert presentation.data_source_label(DataSource.SYNTHETIC.value).startswith(
            "SYNTHETIC"
        )

    @pytest.mark.parametrize(
        "value", [DataSource.PUBLIC_DATASET.value, DataSource.LIVE.value]
    )
    def test_neither_public_nor_live_implies_eligibility(self, value: str) -> None:
        statement = presentation.data_source_statement(value)
        assert "does not make" in statement
        assert "eligible" in statement

    def test_an_absent_data_source_is_stated_not_guessed(self) -> None:
        assert presentation.data_source_label(None).startswith("Unavailable")
        statement = presentation.data_source_statement(None)
        assert "No data source is recorded" in statement
        assert "not scientifically eligible" not in statement
        assert "nothing here is scientifically eligible" in statement

    def test_an_unrecognised_source_is_shown_verbatim(self) -> None:
        label = presentation.data_source_label("some_future_source")
        assert "UNRECOGNISED" in label
        assert "some_future_source" in label
        assert (
            presentation.data_source_statement("some_future_source")
            == presentation.UNKNOWN_DATA_SOURCE_NOTE
        )

    def test_statements_are_deduplicated_in_order(self) -> None:
        statements = presentation.data_source_statements(("live", "synthetic", "live"))
        assert statements == (
            presentation.data_source_statement("live"),
            presentation.data_source_statement("synthetic"),
        )


# --- Rendering the three provenances -------------------------------------


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Temporary artifact and session roots holding all three provenances.

    Three fixtures, one per provenance case the plan names.  They exist
    only for the duration of a test; nothing here is checked in, and
    nothing here is data.
    """
    artifacts = tmp_path / "experiments"
    sessions = tmp_path / "sessions"

    # Synthetic: the case every run in this repository actually is.
    fx.make_baseline_run(artifacts, "fixture-synthetic")

    # Public: a run whose dataset records the public_dataset source and
    # which declares no scientific eligibility. The two are independent
    # fields and this fixture is the proof that they stay independent.
    fx.make_baseline_run(
        artifacts,
        "fixture-public",
        data_source=DataSource.PUBLIC_DATASET.value,
        evaluation_mode="public_dataset_evaluation",
        eligible=False,
    )

    # Live: a recording whose messages record the live source.
    sfx.write_session(
        sessions, "fixture-live", synthetic=False, data_source=DataSource.LIVE.value
    )
    sfx.write_session(sessions, "fixture-synthetic-session")

    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(artifacts))
    monkeypatch.setenv(SESSION_ROOT_ENV, str(sessions))
    yield tmp_path


def run_app(mode: str | None = None, selection: str | None = None) -> AppTest:
    """Render the app, optionally selecting a mode and a run or session."""
    from engagevr.dashboard.launch import app_path

    app = AppTest.from_file(str(app_path()), default_timeout=TIMEOUT_SECONDS)
    app.run()
    if mode is not None:
        app.sidebar.radio[0].set_value(mode).run()
    if selection is None:
        return app
    for box in app.sidebar.selectbox:
        matches = [
            option for option in box.options if str(option).startswith(selection)
        ]
        if matches:
            box.set_value(matches[0]).run()
            return app
    raise AssertionError(f"no selector offered {selection!r}")


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
    for frame in app.dataframe:
        parts.append(frame.value.to_csv(index=False))
    return " ".join(parts)


class TestSyntheticIsVisiblyLabelled:
    def test_the_self_check_banner_is_rendered(self, roots: Path) -> None:
        app = run_app(ARTIFACT_MODE, selection="fixture-synthetic")
        assert not app.exception
        assert SELF_CHECK in rendered_text(app)

    def test_the_run_is_visibly_ineligible(self, roots: Path) -> None:
        text = rendered_text(run_app(ARTIFACT_MODE, selection="fixture-synthetic"))
        assert "scientific_evaluation_eligible = **false**" in text
        assert "Nothing on this page is scientific evidence." in text

    def test_the_synthetic_source_is_labelled(self, roots: Path) -> None:
        text = rendered_text(run_app(ARTIFACT_MODE, selection="fixture-synthetic"))
        assert "SYNTHETIC (recorded as 'synthetic')" in text

    def test_the_standing_disclaimer_is_rendered(self, roots: Path) -> None:
        text = rendered_text(run_app(ARTIFACT_MODE, selection="fixture-synthetic"))
        assert DASHBOARD_DISCLAIMER in text


class TestPublicIsVisiblyLabelled:
    def test_the_public_source_is_labelled_public(self, roots: Path) -> None:
        app = run_app(ARTIFACT_MODE, selection="fixture-public")
        assert not app.exception
        assert "PUBLIC (recorded as 'public_dataset')" in rendered_text(app)

    def test_public_does_not_become_scientifically_eligible(self, roots: Path) -> None:
        text = rendered_text(run_app(ARTIFACT_MODE, selection="fixture-public"))
        assert "scientific_evaluation_eligible = **false**" in text
        assert "Being public does not make anything here scientifically eligible" in (
            text
        )

    def test_public_and_ineligible_stays_visibly_ineligible(self, roots: Path) -> None:
        """The two fields are independent and must render independently."""
        summary = fx.summary_for(roots / "experiments", "fixture-public")
        assert summary.provenance.data_source == DataSource.PUBLIC_DATASET.value
        assert summary.provenance.is_synthetic is False
        assert summary.provenance.scientific_evaluation_eligible is False
        text = rendered_text(run_app(ARTIFACT_MODE, selection="fixture-public"))
        assert "Nothing on this page is scientific evidence." in text

    def test_a_public_run_carries_no_self_check_banner(self, roots: Path) -> None:
        """The banner is about synthetic data, not about ineligibility."""
        text = rendered_text(run_app(ARTIFACT_MODE, selection="fixture-public"))
        assert SELF_CHECK not in text

    def test_the_standing_disclaimer_survives_a_public_source(
        self, roots: Path
    ) -> None:
        text = rendered_text(run_app(ARTIFACT_MODE, selection="fixture-public"))
        assert DASHBOARD_DISCLAIMER in text

    def test_the_dataset_page_labels_the_public_source(self, roots: Path) -> None:
        app = run_app(ARTIFACT_MODE, selection="fixture-public")
        app.sidebar.radio[1].set_value("Dataset and provenance").run()
        assert not app.exception
        text = rendered_text(app)
        assert "PUBLIC (recorded as 'public_dataset')" in text
        assert DASHBOARD_DISCLAIMER in text


class TestLiveIsVisiblyLabelled:
    def test_the_live_mode_is_named_in_words(self, roots: Path) -> None:
        app = run_app(LIVE_MODE, selection="fixture-live")
        assert not app.exception
        assert "Mode: LIVE OBSERVATION" in rendered_text(app)

    def test_the_live_source_is_labelled_live(self, roots: Path) -> None:
        text = rendered_text(run_app(LIVE_MODE, selection="fixture-live"))
        assert "LIVE (recorded as 'live')" in text

    def test_live_does_not_become_scientifically_eligible(self, roots: Path) -> None:
        text = rendered_text(run_app(LIVE_MODE, selection="fixture-live"))
        assert "scientific_evaluation_eligible = **false**" in text
        assert SESSION_ELIGIBILITY_NOTE in text
        assert "Being live does not make it scientifically eligible" in text

    def test_live_and_ineligible_stays_visibly_ineligible(self, roots: Path) -> None:
        read = read_session(
            roots / "sessions" / "fixture-live",
            mode=DashboardSessionMode.LIVE,
            max_records=100,
        )
        provenance = read.summary.provenance
        assert provenance.data_sources == (DataSource.LIVE.value,)
        assert provenance.is_synthetic is False
        assert provenance.scientific_evaluation_eligible is False
        text = rendered_text(run_app(LIVE_MODE, selection="fixture-live"))
        assert "scientific_evaluation_eligible = **false**" in text

    def test_a_live_recording_carries_no_self_check_banner(self, roots: Path) -> None:
        text = rendered_text(run_app(LIVE_MODE, selection="fixture-live"))
        assert SELF_CHECK not in text

    def test_the_standing_disclaimer_survives_a_live_source(self, roots: Path) -> None:
        text = rendered_text(run_app(LIVE_MODE, selection="fixture-live"))
        assert DASHBOARD_DISCLAIMER in text

    def test_a_live_recording_shows_no_estimate(self, roots: Path) -> None:
        """Live observation is not live inference, whatever the source."""
        text = rendered_text(run_app(LIVE_MODE, selection="fixture-live"))
        assert "No estimator was loaded" in text

    def test_replaying_a_live_recording_keeps_the_live_source_label(
        self, roots: Path
    ) -> None:
        app = run_app(REPLAY_MODE, selection="fixture-live")
        assert not app.exception
        text = rendered_text(app)
        assert "LIVE (recorded as 'live')" in text
        assert "Mode: SESSION REPLAY" in text
        assert "scientific_evaluation_eligible = **false**" in text

    def test_a_synthetic_recording_still_carries_its_banner(self, roots: Path) -> None:
        text = rendered_text(run_app(LIVE_MODE, selection="fixture-synthetic-session"))
        assert SELF_CHECK in text
        assert "SYNTHETIC (recorded as 'synthetic')" in text


class TestNoModeHidesTheDisclaimer:
    @pytest.mark.parametrize(
        ("mode", "selection"),
        [
            (ARTIFACT_MODE, "fixture-synthetic"),
            (ARTIFACT_MODE, "fixture-public"),
            (LIVE_MODE, "fixture-live"),
            (LIVE_MODE, "fixture-synthetic-session"),
            (REPLAY_MODE, "fixture-live"),
            (REPLAY_MODE, "fixture-synthetic-session"),
        ],
    )
    def test_the_disclaimer_is_rendered(
        self, roots: Path, mode: str, selection: str
    ) -> None:
        app = run_app(mode, selection=selection)
        assert not app.exception
        assert DASHBOARD_DISCLAIMER in rendered_text(app)

    def test_the_disclaimer_names_every_thing_it_does_not_establish(self) -> None:
        for claim in (
            "engagement",
            "cognitive load",
            "psychological state",
            "health status",
            "safety",
            "adaptation benefit",
        ):
            assert claim in DASHBOARD_DISCLAIMER

    def test_no_mode_is_eligible(self, roots: Path) -> None:
        for mode, selection in (
            (ARTIFACT_MODE, "fixture-synthetic"),
            (ARTIFACT_MODE, "fixture-public"),
            (LIVE_MODE, "fixture-live"),
            (REPLAY_MODE, "fixture-live"),
        ):
            text = rendered_text(run_app(mode, selection=selection))
            assert "scientific_evaluation_eligible = **true**" not in text


class TestSessionViewsCarryTheLabels:
    def test_the_data_source_table_names_the_source_and_denies_eligibility(
        self, roots: Path
    ) -> None:
        read = read_session(
            roots / "sessions" / "fixture-live",
            mode=DashboardSessionMode.LIVE,
            max_records=100,
        )
        table = views.data_source_table(read.summary)
        assert table.columns == (
            "recorded value",
            "label",
            "scientifically eligible",
            "what it establishes",
        )
        assert table.rows[0][0] == DataSource.LIVE.value
        assert table.rows[0][1].startswith("LIVE")
        assert table.rows[0][2] == "no"

    def test_a_recording_with_no_decoded_source_states_that(self, roots: Path) -> None:
        read = read_session(
            roots / "sessions" / "fixture-live",
            mode=DashboardSessionMode.LIVE,
            max_records=100,
        )
        empty = read.summary.model_copy(
            update={
                "provenance": read.summary.provenance.model_copy(
                    update={"data_sources": ()}
                )
            }
        )
        assert views.data_source_table(empty).rows == ()
        assert "not established" in views.NO_DATA_SOURCE_NOTE

    def test_the_catalogue_table_carries_a_label_column(self, roots: Path) -> None:
        from engagevr.dashboard.session_catalogue import build_session_catalogue

        catalogue = build_session_catalogue(
            roots / "sessions", mode=DashboardSessionMode.LIVE
        )
        table = views.catalogue_table(catalogue, max_rows=100)
        assert "data source labels" in table.columns
        joined = " ".join(cell for row in table.rows for cell in row)
        assert "LIVE (recorded as 'live')" in joined
        assert "SYNTHETIC (recorded as 'synthetic')" in joined
