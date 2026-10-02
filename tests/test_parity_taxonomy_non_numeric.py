"""The named list of registered callables that cannot carry a parity grade.

``statspai._parity_taxonomy.NON_NUMERIC_CALLABLES`` moves plots, bundled
datasets, exporters, language-model helpers and catalogue listings that live
in estimator modules out of the estimator denominator of ``docs/parity.md``.
A list like that is an easy place to hide a coverage gap, so each entry is
held to the admission rule written next to it.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import statspai as sp
from statspai import registry
from statspai._parity_taxonomy import INFRASTRUCTURE_CATEGORIES, NON_NUMERIC_CALLABLES

ROOT = Path(__file__).resolve().parent.parent
KINDS = {"plot", "dataset", "export", "llm", "catalog"}


def _doc_head(name: str) -> str:
    doc = inspect.getdoc(getattr(sp, name)) or ""
    return doc.split("\n\n")[0].replace("\n", " ").lower()


def test_every_entry_is_a_registered_function_with_a_known_kind():
    registered = set(sp.list_functions())
    for name, kind in NON_NUMERIC_CALLABLES.items():
        assert kind in KINDS, (name, kind)
        assert name in registered, name
        obj = getattr(sp, name)
        assert callable(obj) and not inspect.isclass(obj), name


def test_no_entry_duplicates_an_infrastructure_category():
    """Entries are for functions the category rule does not already cover."""
    sp.list_functions()
    for name in NON_NUMERIC_CALLABLES:
        spec = registry._REGISTRY[name]
        assert spec.category not in INFRASTRUCTURE_CATEGORIES, (name, spec.category)


def test_no_entry_has_an_evidence_record():
    """A function with numerical evidence is an estimator, not infrastructure."""
    index = json.loads(
        (ROOT / "src" / "statspai" / "_parity_index.json").read_text(encoding="utf-8")
    )
    with_evidence = {r["function"] for r in index["records"]}
    assert not (set(NON_NUMERIC_CALLABLES) & with_evidence)


@pytest.mark.parametrize(
    "name", sorted(n for n, k in NON_NUMERIC_CALLABLES.items() if k == "plot")
)
def test_plot_entries_describe_a_figure(name):
    head = _doc_head(name)
    words = ("plot", "visuali", "diagram", "chart", "figure", "display", "draw")
    assert "plot" in name or any(w in head for w in words), head


@pytest.mark.parametrize(
    "name", sorted(n for n, k in NON_NUMERIC_CALLABLES.items() if k == "dataset")
)
def test_dataset_entries_take_no_data_argument(name):
    params = inspect.signature(getattr(sp, name)).parameters
    assert "data" not in params and "df" not in params, list(params)


@pytest.mark.parametrize(
    "name", sorted(n for n, k in NON_NUMERIC_CALLABLES.items() if k == "export")
)
def test_export_entries_take_a_fitted_result_or_fits(name):
    """An exporter starts from results that exist, not from raw data columns."""
    params = list(inspect.signature(getattr(sp, name)).parameters)
    assert params, name
    assert "y" not in params and "treat" not in params, params


@pytest.mark.parametrize(
    "name", sorted(n for n, k in NON_NUMERIC_CALLABLES.items() if k == "catalog")
)
def test_catalog_entries_report_names_or_metadata(name):
    head = _doc_head(name)
    words = ("list", "report", "return", "identify", "recommend", "headline")
    assert any(w in head for w in words), head


def test_functions_that_compute_are_not_listed():
    """Spot checks on the boundary the rule draws."""
    for name in (
        "lisa_cluster_map",  # classifies observations
        "did_summary",  # fits several estimators
        "rd_compare",
        "synth_compare",
        "llm_dag_validate",  # runs conditional-independence tests on data
        "compare_event_study_conventions",
        "balance_diagnostics",
    ):
        assert name not in NON_NUMERIC_CALLABLES, name


def test_denominators_use_the_list():
    strata = sp.parity_summary()["denominators"]
    sp.list_functions()
    n_infra_by_category = sum(
        1
        for name in sp.list_functions()
        if not inspect.isclass(getattr(sp, name, None))
        and registry._REGISTRY[name].category in INFRASTRUCTURE_CATEGORIES
    )
    assert strata["infrastructure"]["total"] == n_infra_by_category + len(
        NON_NUMERIC_CALLABLES
    )
    assert (
        strata["estimator"]["total"]
        + strata["infrastructure"]["total"]
        + strata["classes"]["total"]
        == strata["all"]["total"]
    )
