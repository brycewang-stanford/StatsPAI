"""``result_card``: a check that was not run is not a check that passed.

``result.violations() == []`` is what an agent sees both when every
diagnostic passed and when none was run (review item A3). The card's
``assumptions`` section lists each diagnostic the estimator family
expects with one of ``passed`` / ``failed`` / ``not_run`` /
``not_applicable``, read from ``sp.audit`` (which never re-runs a test).
"""

from __future__ import annotations

import ast
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

STATUSES = {"passed", "failed", "not_run", "not_applicable"}


@pytest.fixture(scope="module")
def iv_data():
    rng = np.random.default_rng(0)
    n = 400
    z, u = rng.normal(size=n), rng.normal(size=n)
    d = 0.8 * z + u + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": 1 + 2 * d + u + rng.normal(size=n),
            "d": d,
            "z": z,
            "x": rng.normal(size=n),
        }
    )


def _checks(fit):
    return sp.result_card(fit)["assumptions"]


def test_iv_card_separates_passed_not_run_and_not_applicable(iv_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.ivreg("y ~ x + (d ~ z)", data=iv_data)
    section = _checks(fit)
    assert section["checks_available"] is True
    by_name = {c["name"]: c for c in section["checks"]}
    assert by_name["weak_instrument"]["status"] == "passed"
    assert by_name["weak_instrument"]["value"] > 10
    # Just-identified: an over-identification test cannot be run at all.
    assert by_name["overid_test"]["status"] == "not_applicable"
    # Weak-IV-robust interval: expected, and nobody ran it.
    assert by_name["anderson_rubin_ci"]["status"] == "not_run"
    assert by_name["anderson_rubin_ci"]["run_with"] == "sp.anderson_rubin_ci"
    assert section["checks_summary"] == {
        "passed": 1,
        "failed": 0,
        "not_run": 1,
        "not_applicable": 1,
    }


def test_summary_counts_are_the_list(iv_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = [
            sp.ivreg("y ~ x + (d ~ z)", data=iv_data),
            sp.regress("y ~ d + x", data=iv_data),
            sp.callaway_santanna(
                sp.datasets.mpdta(), y="lemp", g="first_treat", t="year", i="countyreal"
            ),
        ]
    for fit in fits:
        section = _checks(fit)
        statuses = [c["status"] for c in section["checks"]]
        assert set(statuses) <= STATUSES
        for status in STATUSES:
            assert section["checks_summary"][status] == statuses.count(status)
        assert all(c["reason"] for c in section["checks"])
        assert sum(section["checks_summary"].values()) == len(statuses) >= 1


def test_no_violations_does_not_read_as_all_checks_passed(iv_data):
    """The case the section exists for."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.callaway_santanna(
            sp.datasets.mpdta(), y="lemp", g="first_treat", t="year", i="countyreal"
        )
    errors = [v for v in fit.violations() if v.get("severity") == "error"]
    section = _checks(fit)
    assert errors == []
    assert section["checks_summary"]["not_run"] >= 1
    not_run = [c["name"] for c in section["checks"] if c["status"] == "not_run"]
    assert "honest_did" in not_run


def test_result_without_a_checklist_says_so():
    """No checklist for the family: say that, never an empty pass."""
    from statspai.result_card import _check_status

    class _Bare:
        model_info: dict = {}

    section = _check_status(_Bare())
    assert section["checks_available"] is False
    assert "not evidence" in section["checks_note"]
    assert "checks" not in section


def test_every_next_steps_method_returns_dicts():
    """One shape for every result class; ``CrossValidationResult`` was the
    exception (a list of strings) until 1.35."""
    root = Path(inspect.getfile(sp)).parent
    offenders = []
    found = 0
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "next_steps":
                found += 1
                returns = ast.unparse(node.returns) if node.returns else ""
                if returns in ("List[str]", "list[str]"):
                    offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert found >= 4
    assert not offenders, offenders


def test_cross_validation_next_steps_have_the_common_keys(iv_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.cross_validate(
            iv_data, "ols", formula="y ~ d + x", engines=["statspai"]
        )
    steps = res.next_steps()
    assert steps and all(
        set(s) == {"action", "reason", "priority", "category"} for s in steps
    )
    assert res.to_dict()["next_steps"] == steps
