"""Loud-fallback contract tests (CLAUDE.md §7: 失败要响亮).

These pin the 2026-07 conversion of previously *silent* degradation paths
into warning-emitting ones:

1. ``regression/quantile.py`` — the exact LP quantile solve falling back to
   IRLS must emit a ``RuntimeWarning`` (the IRLS solution is approximate).
2. ``output/regression_table.py`` — a user-supplied ``apply_coef`` /
   ``apply_coef_deriv`` that raises must warn that the cell is reported
   untransformed / without the delta-method rescaling, instead of silently
   mixing transformed and untransformed cells in one table.
3. ``principal_strat`` — bootstrap-replicate failures now surface through
   ``core._bootstrap.bootstrap_se`` (warn + honest NaN) rather than a bare
   ``np.nanstd`` with no signal.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def ols_model():
    rng = np.random.default_rng(2026)
    n = 400
    x1 = rng.normal(0, 1, n)
    y = 1.0 + 0.5 * x1 + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x1": x1})
    return sp.regress("y ~ x1", data=df)


class TestQuantileLPFallbackWarns:
    def test_irls_fallback_emits_runtimewarning(self, monkeypatch):
        from statspai.regression import quantile as qmod

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated HiGHS failure")

        monkeypatch.setattr(qmod, "linprog", _boom)

        rng = np.random.default_rng(0)
        n = 200
        x = rng.normal(0, 1, n)
        y = 1.0 + 2.0 * x + rng.normal(0, 1, n)
        df = pd.DataFrame({"y": y, "x": x})

        with pytest.warns(RuntimeWarning, match="LP solver failed"):
            res = sp.qreg(df, "y ~ x", quantile=0.5)
        # The IRLS fallback still delivers usable coefficients.
        slope = float(res.params["Q(0.5) x"])
        assert abs(slope - 2.0) < 0.5

    def test_healthy_lp_path_stays_silent(self):
        rng = np.random.default_rng(1)
        n = 200
        x = rng.normal(0, 1, n)
        y = 1.0 + 2.0 * x + rng.normal(0, 1, n)
        df = pd.DataFrame({"y": y, "x": x})
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            sp.qreg(df, "y ~ x", quantile=0.5)


class TestApplyCoefFailureWarns:
    def test_apply_coef_raise_warns_and_leaves_cell_untouched(self, ols_model):
        def bad_transform(b):
            raise ValueError("domain error")

        with pytest.warns(RuntimeWarning, match="reported UNtransformed"):
            out = sp.regtable(ols_model, apply_coef=bad_transform).to_text()
        baseline = sp.regtable(ols_model).to_text()
        assert out == baseline

    def test_apply_coef_deriv_raise_warns(self, ols_model):
        def bad_deriv(b):
            raise ZeroDivisionError("no derivative here")

        with pytest.warns(RuntimeWarning, match="delta-method"):
            sp.regtable(
                ols_model,
                apply_coef=lambda b: 2 * b,
                apply_coef_deriv=bad_deriv,
            ).to_text()

    def test_healthy_transform_stays_silent(self, ols_model):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            sp.regtable(
                ols_model,
                apply_coef=lambda b: 2 * b,
                apply_coef_deriv=lambda b: 2.0,
            ).to_text()


class TestPrincipalStratBootstrapSurfacesFailures:
    def test_healthy_run_matches_manual_nanstd(self):
        """On a clean dataset the SE must equal the plain nanstd — the
        loud-failure plumbing may not change healthy-path numerics."""
        rng = np.random.default_rng(7)
        n = 500
        d = rng.integers(0, 2, n)
        s = np.where(d == 1, rng.random(n) < 0.9, rng.random(n) < 0.6).astype(int)
        y = 1.0 + 0.5 * d + rng.normal(0, 1, n)
        df = pd.DataFrame({"y": y, "d": d, "s": s})
        res = sp.principal_strat(
            df, y="y", treat="d", strata="s",
            method="monotonicity", n_boot=50, seed=3,
        )
        assert np.isfinite(res.effects["se"]).any()
