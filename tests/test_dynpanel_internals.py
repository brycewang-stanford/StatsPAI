"""Unit tests for the dynamic-panel GMM internals.

The reference-parity suites prove the estimators reproduce Stata; they say
nothing about what happens when a user passes something malformed. This
file covers the other half: the lag-operator grammar, the panel-layout
validation, and the guards that are supposed to fail loudly.

Every raise in this package exists because the silent alternative was worse
— a mis-parsed lag spec becoming a column name, a duplicated ``(id, time)``
pair making the estimate depend on row order, a cluster variable finer than
the unit quietly invalidating the moment conditions. These tests pin the
message as well as the type, because the message is the feature.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.gmm._dynpanel._data import (
    add_time_dummies,
    build_panel_arrays,
    unit_cluster_codes,
)
from statspai.gmm._dynpanel._estimate import safe_inv
from statspai.gmm._dynpanel._moments import (
    first_difference_H,
    fod_operator,
    fod_weights,
    system_H,
)
from statspai.gmm._dynpanel._spec import (
    GMMBlock,
    IVBlock,
    Term,
    normalize_lag_range,
    parse_terms,
    term_name,
)


@pytest.fixture
def small_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        for t in range(5):
            rows.append(
                {"id": i, "time": t, "y": rng.normal(), "x": rng.normal(), "g": i % 2}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lag-operator grammar
# ---------------------------------------------------------------------------


class TestParseTerms:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("k", [("k", 0)]),
            ("L.k", [("k", 1)]),
            ("l.k", [("k", 1)]),
            ("L2.k", [("k", 2)]),
            ("L(2).k", [("k", 2)]),
            ("L(0/2).k", [("k", 0), ("k", 1), ("k", 2)]),
            ("l(1/3).wage", [("wage", 1), ("wage", 2), ("wage", 3)]),
            ("  L.k  ", [("k", 1)]),
        ],
    )
    def test_accepted_forms(self, spec, expected):
        assert [(t.var, t.lag) for t in parse_terms([spec])] == expected

    def test_order_is_preserved_across_and_within_specs(self):
        """Stata's order: `l(0/2).k` expands shallow-to-deep, in place."""
        terms = parse_terms(["w", "l(0/2).k", "z"])
        assert [term_name(t) for t in terms] == ["w", "k", "L1.k", "L2.k", "z"]

    def test_none_and_empty_list(self):
        assert parse_terms(None) == []
        assert parse_terms([]) == []

    def test_a_plain_name_containing_a_dot_is_not_a_lag(self):
        """Only a leading L/l marks a lag; `my.var` is a column name."""
        assert [(t.var, t.lag) for t in parse_terms(["my.var"])] == [("my.var", 0)]

    @pytest.mark.parametrize(
        "spec,match",
        [
            ("L(3/1).k", "reversed lag range"),
            ("", "empty regressor specification"),
            ("L(a/b).k", "could not parse lag specification"),
        ],
    )
    def test_malformed_specs_raise(self, spec, match):
        with pytest.raises(ValueError, match=match):
            parse_terms([spec])

    def test_non_string_spec_raises(self):
        with pytest.raises(TypeError, match="must be a string"):
            parse_terms([3])

    def test_negative_lag_is_rejected(self):
        with pytest.raises(ValueError, match="negative lag"):
            Term("k", -1)


class TestNormalizeLagRange:
    def test_none_means_class_default_and_full_depth(self):
        assert normalize_lag_range(None, default_min=2, horizon=9) == (2, 9)

    def test_none_elements_fill_individually(self):
        assert normalize_lag_range((None, 4), default_min=2, horizon=9) == (2, 4)
        assert normalize_lag_range((3, None), default_min=2, horizon=9) == (3, 9)

    def test_negative_minimum_rejected(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            normalize_lag_range((-1, 4), default_min=2, horizon=9)

    def test_empty_window_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_lag_range((5, 3), default_min=2, horizon=9)


class TestBlockLabels:
    def test_gmm_block_label_describes_the_moment_set(self):
        block = GMMBlock("n", 2, 9, collapse=True, equation="level")
        assert "n" in block.label and "2/9" in block.label
        assert "collapse" in block.label and "level" in block.label

    def test_iv_block_label_marks_the_equation(self):
        assert IVBlock(Term("w", 0), equation="diff").label == "D.w"
        assert IVBlock(Term("w", 0), equation="level").label == "w"
        assert IVBlock(Term("w", 1), equation="both").label == "D./L.L1.w"

    @pytest.mark.parametrize("cls", [GMMBlock, IVBlock])
    def test_unknown_equation_rejected(self, cls):
        with pytest.raises(ValueError, match="equation"):
            if cls is GMMBlock:
                GMMBlock("n", 2, 9, equation="sideways")
            else:
                IVBlock(Term("w", 0), equation="sideways")


# ---------------------------------------------------------------------------
# Panel layout
# ---------------------------------------------------------------------------


class TestBuildPanelArrays:
    def test_shapes_and_placement(self, small_panel):
        panel = build_panel_arrays(small_panel, "id", "time", ["y", "x"])
        assert panel.n_units == 6 and panel.n_periods == 5
        assert panel.get("y").shape == (6, 5)
        first = small_panel.iloc[0]
        assert panel.get("y")[0, 0] == pytest.approx(first["y"])

    def test_missing_values_stay_nan_per_variable(self, small_panel):
        """The whole point of the layout: availability is per variable."""
        holed = small_panel.copy()
        holed.loc[holed["time"] == 0, "x"] = np.nan
        panel = build_panel_arrays(holed, "id", "time", ["y", "x"])
        assert np.isnan(panel.get("x")[:, 0]).all()
        assert np.isfinite(
            panel.get("y")[:, 0]
        ).all(), "a missing x wiped out y — the listwise-deletion bug is back."

    def test_unknown_variable_names_the_available_columns(self, small_panel):
        with pytest.raises(ValueError, match="not found in the data"):
            build_panel_arrays(small_panel, "id", "time", ["nope"])

    @pytest.mark.parametrize("col,what", [("nope", "id"), ("time", "time")])
    def test_missing_index_columns_rejected(self, small_panel, col, what):
        frame = small_panel.drop(columns=["time"]) if what == "time" else small_panel
        with pytest.raises(ValueError, match="column"):
            build_panel_arrays(frame, "nope" if what == "id" else "id", "time", ["y"])

    def test_duplicate_index_pairs_rejected(self, small_panel):
        """Keeping the last row would make the estimate order-dependent."""
        doubled = pd.concat([small_panel, small_panel.head(1)], ignore_index=True)
        with pytest.raises(ValueError, match="duplicated"):
            build_panel_arrays(doubled, "id", "time", ["y"])

    def test_all_missing_index_rejected(self, small_panel):
        blank = small_panel.copy()
        blank["id"] = np.nan
        with pytest.raises(ValueError, match="no rows left"):
            build_panel_arrays(blank, "id", "time", ["y"])

    def test_a_variable_may_coincide_with_the_index(self, small_panel):
        """Clustering on the time column must reach its own error message."""
        panel = build_panel_arrays(small_panel, "id", "time", ["y", "time"])
        assert "time" in panel.values

    def test_lagged_and_observed_helpers(self, small_panel):
        panel = build_panel_arrays(small_panel, "id", "time", ["y"])
        lagged = panel.lagged("y", 1)
        assert np.isnan(lagged[:, 0]).all()
        np.testing.assert_allclose(lagged[:, 1:], panel.get("y")[:, :-1])
        np.testing.assert_array_equal(panel.lagged("y", 0), panel.get("y"))
        assert panel.lagged("y", 99).shape == panel.get("y").shape
        assert panel.observed("y").all()

    def test_get_unknown_variable_lists_what_exists(self, small_panel):
        panel = build_panel_arrays(small_panel, "id", "time", ["y"])
        with pytest.raises(KeyError, match="available"):
            panel.get("nope")


class TestTimeDummies:
    def test_one_dummy_per_retained_period(self, small_panel):
        panel = build_panel_arrays(small_panel, "id", "time", ["y"])
        names = add_time_dummies(panel, drop_first=1)
        assert len(names) == panel.n_periods - 1
        for offset, name in enumerate(names, start=1):
            column = panel.get(name)
            assert (column[:, offset] == 1).all()
            assert (np.delete(column, offset, axis=1) == 0).all()

    def test_drop_first_zero_keeps_every_period(self, small_panel):
        panel = build_panel_arrays(small_panel, "id", "time", ["y"])
        assert len(add_time_dummies(panel, drop_first=0)) == panel.n_periods

    def test_negative_drop_first_rejected(self, small_panel):
        panel = build_panel_arrays(small_panel, "id", "time", ["y"])
        with pytest.raises(ValueError, match="drop_first"):
            add_time_dummies(panel, drop_first=-1)


class TestUnitClusterCodes:
    def test_codes_are_contiguous_and_unit_aligned(self, small_panel):
        panel = build_panel_arrays(small_panel, "id", "time", ["y", "g"])
        codes = unit_cluster_codes(panel, "g")
        assert codes.shape == (6,)
        assert set(codes.tolist()) == {0, 1}

    def test_within_unit_variation_rejected(self, small_panel):
        panel = build_panel_arrays(small_panel, "id", "time", ["y", "time"])
        with pytest.raises(ValueError, match="varies within unit"):
            unit_cluster_codes(panel, "time")

    def test_a_unit_with_no_value_is_rejected(self, small_panel):
        holed = small_panel.copy()
        holed.loc[holed["id"] == 0, "g"] = np.nan
        panel = build_panel_arrays(holed, "id", "time", ["y", "g"])
        with pytest.raises(ValueError, match="missing for"):
            unit_cluster_codes(panel, "g")


# ---------------------------------------------------------------------------
# Transform operators and weight structure
# ---------------------------------------------------------------------------


class TestTransformOperators:
    def test_first_difference_H_is_the_MA1_band(self):
        H = first_difference_H(np.array([2, 3, 4]))
        np.testing.assert_array_equal(
            H, np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
        )

    def test_first_difference_H_breaks_across_a_gap(self):
        """Non-adjacent periods share no error term, so no off-diagonal."""
        H = first_difference_H(np.array([2, 5]))
        assert H[0, 1] == 0.0 and H[1, 0] == 0.0

    @pytest.mark.parametrize("T", [2, 3, 5, 9, 20])
    def test_fod_operator_is_orthonormal(self, T):
        M = fod_operator(T)
        assert M.shape == (T - 1, T)
        np.testing.assert_allclose(M @ M.T, np.eye(T - 1), atol=1e-12)

    def test_fod_weights_average_the_available_future(self, T=4):
        """c_t (y_t - mean of later y), with c_t = sqrt(T_t/(T_t+1))."""
        M = fod_weights(np.array([0, 1, 2, 3]), T)
        assert M.shape == (3, 4)
        c0 = np.sqrt(3 / 4)
        np.testing.assert_allclose(M[0], [c0, -c0 / 3, -c0 / 3, -c0 / 3])
        np.testing.assert_allclose(M.sum(axis=1), 0.0, atol=1e-12)

    def test_fod_weights_skip_a_gap(self):
        """A hole costs one row, not two — the reason FOD exists."""
        M = fod_weights(np.array([0, 1, 3]), 4)
        assert M.shape == (2, 4)
        assert M[0, 2] == 0.0, "the missing period picked up weight"

    def test_fod_weights_need_two_periods(self):
        assert fod_weights(np.array([2]), 5).shape[0] == 0

    def test_system_H_cross_quadrant(self):
        """Cov(dEps_p, eps_s) = 1[s==p] - 1[s==p-1]."""
        periods = np.array([2, 3, 2, 3])
        eqs = np.array([0, 0, 1, 1])
        H = system_H(periods, eqs)
        np.testing.assert_array_equal(np.diag(H), [2.0, 2.0, 1.0, 1.0])
        assert H[0, 2] == 1.0  # diff(2) x level(2)
        assert H[1, 2] == -1.0  # diff(3) x level(2)
        np.testing.assert_allclose(H, H.T)


class TestSafeInverse:
    def test_exact_inverse_for_a_regular_matrix(self):
        A = np.array([[2.0, 0.0], [0.0, 4.0]])
        np.testing.assert_allclose(safe_inv(A, "test"), np.diag([0.5, 0.25]))

    def test_singular_matrix_warns_and_pseudo_inverts(self):
        """Falling back quietly would return garbage that looks like an answer."""
        singular = np.ones((3, 3))
        with pytest.warns(UserWarning, match="singular"):
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                out = safe_inv(singular, "the weight matrix")
        np.testing.assert_allclose(out, np.linalg.pinv(singular))


# ---------------------------------------------------------------------------
# Orchestration guards
# ---------------------------------------------------------------------------


class TestFitGuards:
    """Every one of these used to be a way to get a wrong answer quietly."""

    @staticmethod
    def _panel(n_units=40, n_periods=6, seed=1):
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n_units):
            a = rng.normal()
            y = a / 0.5 + rng.normal()
            for _ in range(10):
                x = rng.normal()
                y = 0.5 * y + x + a + rng.normal()
            for t in range(n_periods):
                x = rng.normal()
                y = 0.5 * y + x + a + rng.normal()
                rows.append({"id": i, "time": t, "y": y, "x": x, "grp": i % 4})
        return pd.DataFrame(rows)

    def _fit(self, df, **kwargs):
        from statspai.gmm._dynpanel import fit_dynamic_panel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fit_dynamic_panel(df, y="y", x=["x"], id="id", time="time", **kwargs)

    def test_lags_must_be_at_least_one(self):
        with pytest.raises(ValueError, match="lags must be >= 1"):
            self._fit(self._panel(), lags=0)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="method must be"):
            self._fit(self._panel(), method="sideways")

    def test_a_variable_cannot_hold_two_instrument_classes(self):
        """Exogenous and endogenous are mutually exclusive statements."""
        with pytest.raises(ValueError, match="one instrument class"):
            self._fit(self._panel(), endogenous=["x"])

    def test_predetermined_and_endogenous_overlap_rejected(self):
        df = self._panel()
        df["z"] = df["x"]
        from statspai.gmm._dynpanel import fit_dynamic_panel

        with pytest.raises(ValueError, match="both predetermined and"):
            fit_dynamic_panel(
                df,
                y="y",
                id="id",
                time="time",
                predetermined=["z"],
                endogenous=["z"],
            )

    def test_gmm_lags_below_two_rejected(self):
        """Only lags >= 2 are orthogonal to the differenced error."""
        with pytest.raises(ValueError, match="gmm_lags minimum must be >= 2"):
            self._fit(self._panel(), gmm_lags=(1, 4))

    def test_under_identification_is_an_identification_failure(self):
        from statspai.exceptions import IdentificationFailure
        from statspai.gmm._dynpanel import fit_dynamic_panel

        # One collapsed instrument for the lagged y, none usable for the
        # predetermined regressor: 1 moment, 2 parameters.
        with pytest.raises(IdentificationFailure, match="Under-identified"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_dynamic_panel(
                    self._panel(),
                    y="y",
                    id="id",
                    time="time",
                    predetermined=["x"],
                    collapse=True,
                    gmm_lags=(2, 2),
                    predetermined_lags=(20, 20),
                )

    def test_too_few_periods_for_the_lag_structure(self):
        with pytest.raises(ValueError, match="not enough time periods"):
            self._fit(self._panel(n_periods=3), lags=3)

    def test_time_dummies_are_materialised_and_instrumented(self):
        plain = self._fit(self._panel())
        dummied = self._fit(self._panel(), time_dummies=True)
        assert dummied["time_dummies"], "no dummies were created"
        assert dummied["n_params"] > plain["n_params"]
        assert dummied["n_instruments"] > plain["n_instruments"]

    def test_constant_requires_the_level_equation(self):
        with pytest.raises(NotImplementedError, match="method='system'"):
            self._fit(self._panel(), constant=True)

    def test_cluster_is_recorded_on_the_result(self):
        fit = self._fit(self._panel(), cluster="grp")
        assert fit["cluster"] == "grp"
        assert fit["n_clusters"] == 4

    def test_clustered_orthogonal_fit_runs(self):
        """Both auxiliary paths at once: FOD AR basis plus a cluster regroup."""
        fit = self._fit(self._panel(), cluster="grp", transform="fod")
        assert np.isfinite(fit["ar1"]["z"])
        assert fit["n_clusters"] == 4

    def test_two_step_non_robust_warns_about_downward_bias(self):
        from statspai.gmm._dynpanel import fit_dynamic_panel

        with pytest.warns(UserWarning, match="downward biased"):
            fit_dynamic_panel(
                self._panel(),
                y="y",
                x=["x"],
                id="id",
                time="time",
                twostep=True,
                robust=False,
            )


# ---------------------------------------------------------------------------
# `robust=` is a boolean here and an HC-type *string* elsewhere in StatsPAI.
# ---------------------------------------------------------------------------


def _ar1_panel(n_units: int = 60, n_periods: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        y = rng.normal()
        for t in range(n_periods):
            y = 0.5 * y + rng.normal()
            rows.append({"id": i, "year": 1976 + t, "n": y})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("fn", ["xtabond", "xtdpdsys"])
@pytest.mark.parametrize("bad", ["HC1", "HC3", "cluster", "robust", 1, 0, None])
def test_robust_rejects_non_boolean(fn, bad):
    """A string HC-type must fail loudly, not be read as truthy.

    ``robust`` is boolean on the dynamic-panel family and a string selector
    on the regression family (``robust="HC1"``). ``_house_style`` calls that
    split the highest-impact hazard in the signature surface, and here it
    was silent: ``xtabond(..., robust="HC1")`` was accepted, evaluated as
    truthy, and returned the default Windmeijer sandwich — the caller asked
    for HC1 and got something else with no warning.

    ``robust="cluster"`` is the damaging case. Clustering is a separate
    ``cluster=`` argument, so a user who wrote it got *unclustered*
    standard errors and no indication that the request had been ignored.
    """
    df = _ar1_panel()
    with pytest.raises(ValueError, match="must be boolean"):
        getattr(sp, fn)(df, y="n", id="id", time="year", lags=1, robust=bad)


@pytest.mark.parametrize("fn", ["xtabond", "xtdpdsys"])
@pytest.mark.parametrize("good", [True, False, np.True_, np.False_])
def test_robust_accepts_booleans(fn, good):
    """The guard must not reject the values the signature promises."""
    df = _ar1_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = getattr(sp, fn)(df, y="n", id="id", time="year", lags=1, robust=good)
    assert np.isfinite(float(r.detail["se"].iloc[0]))


def test_robust_error_names_the_cluster_alternative():
    """The message has to say where clustered SEs actually live."""
    df = _ar1_panel()
    with pytest.raises(ValueError) as exc:
        sp.xtabond(df, y="n", id="id", time="year", lags=1, robust="cluster")
    assert "cluster=" in str(exc.value)
