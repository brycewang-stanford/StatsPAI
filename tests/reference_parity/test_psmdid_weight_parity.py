"""Stata parity for the PSM-DID weight regimes of ``sp.psmatch2(...).psm_did``.

Stata's ``aweight`` and ``fweight`` share a point estimate but not a
variance.  ``fweight`` says "this row stands for ``w`` identical rows", so
the residual degrees of freedom are ``sum(w) - k``; ``aweight`` keeps
``n_rows - k``.  On the fixture below the DiD standard error is 0.250051
under ``aweight`` and 0.214797 under ``fweight`` — a 14% difference that
comes entirely from the degrees of freedom.

Before v1.22 ``psm_did(weight='fweight')`` passed the weights to
:func:`sp.feols`, which applies ``aweight`` semantics, while the docstring
and the migration guide both advertised ``reg y ... [fweight=_weight]``.
The option now does what its name says; the default moved to ``'aweight'``
so the numbers from a default call are unchanged.

Fixture provenance
------------------
``_fixtures/_generate_psmdid_weights.do``, run under Stata 18 MP with
psmatch2 4.0.12.  It emits

* ``psmdid_baseline.csv`` — 300-unit cross-section plus Stata's own
  ``_pscore`` / ``_support`` / ``_weight`` from
  ``psmatch2 d x1 x2, outcome(y0) neighbor(1) logit common``;
* ``psmdid_panel.csv`` — the matching 2-period panel;
* ``psmdid_weights_stata.json`` — the regression scalars for five regimes.

What is pinned
--------------
* ``sp.psmatch2`` reproduces Stata's ``_weight`` / ``_support`` row for row
  on a dataset the estimator has never seen (an independent check on the
  matched frame, separate from ``test_psmatch2_parity.py``).
* The DiD coefficient and standard error under ``aweight``, ``fweight`` and
  unweighted, with iid and with ``cluster(id)`` standard errors.
* The row-expansion identity that makes the ``fweight`` implementation
  legitimate: Stata's ``expand _weight`` + unweighted ``regress`` equals
  ``regress [fweight=_weight]`` bit-for-bit.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"
_REGEN = (
    "Regenerate with tests/reference_parity/_fixtures/"
    "_generate_psmdid_weights.do under Stata 18 + psmatch2."
)

# Stata prints 17 significant digits; agreement to 1e-10 relative is well
# inside what a different linear-algebra path can be expected to hold.
_RTOL = 1e-10


@pytest.fixture(scope="module")
def baseline() -> pd.DataFrame:
    path = _FIXTURE_DIR / "psmdid_baseline.csv"
    if not path.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing {path.name}. {_REGEN}")
    df = pd.read_csv(path)
    # Stata exports _treated / _support as value labels.
    df["_support_num"] = (df["_support"] == "On support").astype(int)
    return df


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    path = _FIXTURE_DIR / "psmdid_panel.csv"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"missing {path.name}. {_REGEN}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def stata() -> dict:
    path = _FIXTURE_DIR / "psmdid_weights_stata.json"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"missing {path.name}. {_REGEN}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fitted(baseline):
    """``sp.psmatch2`` on the fixture cross-section, matching the .do file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.psmatch2(
            baseline,
            treat="d",
            covariates=["x1", "x2"],
            outcome="y0",
            neighbor=1,
            common_support="minmax",
        )


def _did(fitted, panel, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fitted.psm_did(
            panel, id="id", y="y", time="period", treat_time=1, **kwargs
        )


class TestMatchedFrameReproducesStata:
    """The `_weight` column this fixture's regressions run on is Stata's."""

    def test_att_matches_stata(self, fitted, stata):
        assert fitted.att == pytest.approx(stata["psmatch2"]["att"], rel=1e-9)

    def test_analytic_se_matches_stata(self, fitted, stata):
        assert fitted.se == pytest.approx(stata["psmatch2"]["seatt"], rel=1e-9)

    def test_weight_column_matches_stata_row_for_row(self, fitted, baseline):
        ours = fitted.matched_data["_weight"].to_numpy(dtype=float)
        theirs = baseline["_weight"].to_numpy(dtype=float)
        # The missingness pattern is itself the matched-sample definition.
        assert np.array_equal(np.isfinite(ours), np.isfinite(theirs))
        both = np.isfinite(ours)
        np.testing.assert_allclose(ours[both], theirs[both], rtol=0, atol=0)

    def test_support_column_matches_stata_row_for_row(self, fitted, baseline):
        ours = fitted.matched_data["_support"].to_numpy(dtype=float)
        theirs = baseline["_support_num"].to_numpy(dtype=float)
        np.testing.assert_array_equal(ours, theirs)

    def test_sample_counts_match_stata(self, fitted, stata, panel):
        did = _did(fitted, panel, weight="aweight")
        assert did.model_info["n_matched_rows"] == stata["sample"]["matched_rows"]
        assert did.model_info["weight_sum"] == pytest.approx(
            stata["sample"]["sum_weight"], rel=1e-12
        )


@pytest.mark.parametrize(
    "mode,key",
    [
        ("aweight", "aweight_iid"),
        ("fweight", "fweight_iid"),
        ("none", "unweighted_iid"),
    ],
)
class TestWeightRegimesIID:
    def test_coefficient(self, fitted, panel, stata, mode, key):
        did = _did(fitted, panel, weight=mode)
        assert did.estimate == pytest.approx(stata[key]["b_did"], rel=_RTOL)

    def test_standard_error(self, fitted, panel, stata, mode, key):
        did = _did(fitted, panel, weight=mode)
        assert did.se == pytest.approx(stata[key]["se_did"], rel=_RTOL)

    def test_reported_n_matches_stata(self, fitted, panel, stata, mode, key):
        """Stata reports N = sum(w) under fweight and n_rows otherwise."""
        did = _did(fitted, panel, weight=mode)
        assert did.n_obs == stata[key]["n"]


@pytest.mark.parametrize(
    "mode,key",
    [("aweight", "aweight_cluster"), ("fweight", "fweight_cluster")],
)
class TestWeightRegimesClustered:
    def test_coefficient(self, fitted, panel, stata, mode, key):
        did = _did(fitted, panel, weight=mode, cluster="id")
        assert did.estimate == pytest.approx(stata[key]["b_did"], rel=_RTOL)

    def test_standard_error(self, fitted, panel, stata, mode, key):
        """The cluster sandwich's (N-1)/(N-k) factor uses sum(w) under fweight."""
        did = _did(fitted, panel, weight=mode, cluster="id")
        assert did.se == pytest.approx(stata[key]["se_did"], rel=_RTOL)


class TestRegimesAreActuallyDifferent:
    """Guard against a future refactor collapsing the two regimes."""

    def test_point_estimates_agree_across_weight_regimes(self, fitted, panel):
        a = _did(fitted, panel, weight="aweight").estimate
        f = _did(fitted, panel, weight="fweight").estimate
        assert a == pytest.approx(f, rel=1e-12)

    def test_fweight_se_is_strictly_smaller(self, fitted, panel, stata):
        a = _did(fitted, panel, weight="aweight").se
        f = _did(fitted, panel, weight="fweight").se
        assert f < a
        # The whole difference is sqrt((n_rows-k)/(sum_w-k)); pin the ratio so
        # a change in either df convention shows up here.
        n_rows = stata["aweight_iid"]["df_r"]
        n_expanded = stata["fweight_iid"]["df_r"]
        assert f / a == pytest.approx(np.sqrt(n_rows / n_expanded), rel=1e-9)

    def test_expansion_identity_is_what_stata_reports(self, stata):
        """Stata's own `expand _weight` + OLS == `[fweight=_weight]`.

        This is the fact that licenses implementing fweight by replicating
        rows rather than hand-deriving each VCE's df correction.
        """
        assert stata["expanded_iid"]["b_did"] == pytest.approx(
            stata["fweight_iid"]["b_did"], rel=0, abs=0
        )
        assert stata["expanded_iid"]["se_did"] == pytest.approx(
            stata["fweight_iid"]["se_did"], rel=0, abs=0
        )
        assert stata["expanded_iid"]["df_r"] == stata["fweight_iid"]["df_r"]


class TestFweightGuardrails:
    """Stata refuses non-integer frequency weights; so do we."""

    def test_fractional_weights_rejected(self, baseline, panel):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m2 = sp.psmatch2(
                baseline,
                treat="d",
                covariates=["x1", "x2"],
                outcome="y0",
                neighbor=2,  # 1/2 shares -> fractional _weight
                common_support="minmax",
            )
        with pytest.raises(MethodIncompatibility, match="must be integers"):
            m2.psm_did(
                panel, id="id", y="y", time="period", treat_time=1, weight="fweight"
            )

    def test_fractional_weights_fine_under_aweight(self, baseline, panel):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m2 = sp.psmatch2(
                baseline,
                treat="d",
                covariates=["x1", "x2"],
                outcome="y0",
                neighbor=2,
                common_support="minmax",
            )
            did = m2.psm_did(
                panel, id="id", y="y", time="period", treat_time=1, weight="aweight"
            )
        assert np.isfinite(did.estimate) and np.isfinite(did.se)
        assert did.model_info["weight_is_integer"] is False

    def test_unknown_regime_rejected(self, fitted, panel):
        with pytest.raises(MethodIncompatibility, match="weight must be one of"):
            _did(fitted, panel, weight="pweight")

    def test_model_info_records_the_regime(self, fitted, panel):
        did = _did(fitted, panel, weight="fweight")
        assert did.model_info["weight"] == "fweight"
        assert "sum(w) - k" in did.model_info["weight_semantics"]
        assert did.model_info["weight_is_integer"] is True
