"""Gate and behaviour tests for the package-wide silent-except ratchet.

``scripts/silent_except_audit.py`` extends the orchestration-only lint in
``tests/test_no_silent_degradation.py`` to the whole package. The first
half of this file pins the audit itself; the second half pins a sample of
the estimator paths that the 2026-10 pass converted from silent
substitution into warning-emitting ones (CLAUDE.md section 3.7).
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core._fallback import warn_dropped, warn_fallback

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "silent_except_audit.py"


def _load_audit():
    spec = importlib.util.spec_from_file_location("silent_except_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


def _scan_source(tmp_path: Path, source: str):
    path = tmp_path / "sample.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return audit.scan_file(path)


# --------------------------------------------------------------------- #
#  The audit
# --------------------------------------------------------------------- #


class TestAuditClassification:
    def test_bare_swallow_is_flagged(self, tmp_path):
        lines = _scan_source(
            tmp_path,
            """
            def f(x):
                try:
                    return fit(x)
                except Exception:
                    return 0.0
            """,
        )
        assert lines == [5]

    def test_bound_but_unused_exception_is_flagged(self, tmp_path):
        lines = _scan_source(
            tmp_path,
            """
            try:
                fit()
            except Exception as exc:  # noqa
                value = float("nan")
            """,
        )
        assert lines == [4]

    def test_bare_except_and_tuple_with_exception_are_flagged(self, tmp_path):
        lines = _scan_source(
            tmp_path,
            """
            try:
                a()
            except:
                pass
            try:
                b()
            except (ValueError, Exception):
                pass
            """,
        )
        assert lines == [4, 8]

    @pytest.mark.parametrize(
        "body",
        [
            "raise",
            "raise RuntimeError('x') from exc",
            "warnings.warn(f'failed: {exc}')",
            "warn_fallback('step', exc, 'using the mean')",
            "record_degradation(None, section='s', exc=exc)",
            "logger.debug('failed', exc_info=True)",
            "errors.append(exc)",
        ],
    )
    def test_handlers_that_leave_a_trace_pass(self, tmp_path, body):
        lines = _scan_source(
            tmp_path,
            f"""
            try:
                fit()
            except Exception as exc:
                {body}
            """,
        )
        assert lines == []

    def test_narrow_handler_is_out_of_scope(self, tmp_path):
        lines = _scan_source(
            tmp_path,
            """
            try:
                solve()
            except (np.linalg.LinAlgError, ValueError):
                pass
            """,
        )
        assert lines == []

    def test_provenance_wrapper_is_exempt(self, tmp_path):
        lines = _scan_source(
            tmp_path,
            """
            try:
                from ..output._lineage import attach_provenance as _attach_prov

                _attach_prov(_result, function="sp.x", params={})
            except Exception:
                pass
            """,
        )
        assert lines == []

    def test_provenance_exemption_does_not_cover_estimation(self, tmp_path):
        lines = _scan_source(
            tmp_path,
            """
            def f():
                try:
                    _attach_prov(r, function="sp.x")
                    return estimate()
                except Exception:
                    return None
            """,
        )
        assert lines == [6]


class TestRatchet:
    def test_no_file_exceeds_its_baseline(self):
        """The gate: a new silent broad handler anywhere in the package fails."""
        sites = audit.scan()
        current = {path: len(lines) for path, lines in sites.items()}
        baseline = audit._load_baseline()
        regressions = audit._regressions(current, baseline)
        detail = "\n".join(
            f"  {path}: {was} -> {now} at lines {sites[path]}"
            for path, was, now in regressions
        )
        assert not regressions, (
            "New silent broad except handlers. Make them loud "
            "(statspai.core._fallback.warn_fallback / record_degradation) "
            "or catch the specific exception:\n" + detail
        )

    def test_baseline_is_not_stale(self):
        """A paid-down site must tighten the baseline in the same change."""
        current = {path: len(lines) for path, lines in audit.scan().items()}
        baseline = audit._load_baseline()
        slack = {
            path: (n, current.get(path, 0))
            for path, n in baseline.items()
            if current.get(path, 0) < n
        }
        assert not slack, (
            "Baseline is looser than the code; run "
            "`python scripts/silent_except_audit.py --write`: " + repr(slack)
        )

    def test_check_flag_fails_on_a_regression(self, monkeypatch):
        baseline = audit._load_baseline()
        victim = next(iter(sorted(baseline)))
        tightened = dict(baseline)
        tightened[victim] -= 1
        monkeypatch.setattr(audit, "_load_baseline", lambda: tightened)
        assert audit.main(["--check"]) == 1


# --------------------------------------------------------------------- #
#  The helpers
# --------------------------------------------------------------------- #


class TestFallbackHelpers:
    def test_warn_fallback_names_step_cause_and_substitute(self):
        with pytest.warns(RuntimeWarning) as rec:
            warn_fallback("propensity model", ValueError("one class"), "using 0.5")
        msg = str(rec[0].message)
        assert "propensity model failed" in msg
        assert "ValueError: one class" in msg
        assert msg.endswith("using 0.5.")

    def test_warn_fallback_respects_category(self):
        with pytest.warns(sp.ConvergenceWarning):
            warn_fallback("R-hat", None, "unverified", category=sp.ConvergenceWarning)

    def test_warn_dropped_is_silent_when_nothing_dropped(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_dropped("cohorts", [], 5, "nothing happens")

    def test_warn_dropped_reports_count_labels_and_first_error(self):
        with pytest.warns(RuntimeWarning) as rec:
            warn_dropped(
                "cohorts",
                [2004, 2006],
                5,
                "the ATT averages over the rest",
                errors=[KeyError("y")],
            )
        msg = str(rec[0].message)
        assert "2 of 5 cohorts dropped [2004, 2006]" in msg
        assert "KeyError" in msg

    def test_warn_dropped_truncates_long_lists(self):
        with pytest.warns(RuntimeWarning, match=r"\(12 more\)"):
            warn_dropped("cells", list(range(20)), 40, "x")


# --------------------------------------------------------------------- #
#  Estimator paths made loud
# --------------------------------------------------------------------- #


def _matched_pairs(n_pairs: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pairs):
        for arm in (0, 1):
            cluster = f"p{p}_c{arm}"
            for _ in range(6):
                rows.append(
                    {
                        "pair": p,
                        "cluster": cluster,
                        "treat": arm,
                        "y": 1.0 * arm + rng.normal(),
                    }
                )
    return pd.DataFrame(rows)


class TestMatchedPairDropsAreLoud:
    def test_clean_design_is_silent(self):
        df = _matched_pairs()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = sp.cluster_matched_pair(
                df, y="y", cluster="cluster", treat="treat", pair="pair"
            )
        assert res.n_pairs == 12

    def test_pair_without_a_control_is_reported_and_excluded(self):
        df = _matched_pairs()
        clean = sp.cluster_matched_pair(
            df[df["pair"] != 3], y="y", cluster="cluster", treat="treat", pair="pair"
        )
        broken = df.copy()
        # Pair 3 loses its control arm: both clusters are now treated.
        broken.loc[broken["pair"] == 3, "treat"] = 1
        with pytest.warns(RuntimeWarning, match=r"1 of 12 matched pairs dropped \[3\]"):
            res = sp.cluster_matched_pair(
                broken, y="y", cluster="cluster", treat="treat", pair="pair"
            )
        # The estimate is exactly the one from the eleven intact pairs.
        assert res.n_pairs == 11
        assert res.estimate == pytest.approx(clean.estimate, rel=0, abs=1e-12)
        assert res.se == pytest.approx(clean.se, rel=0, abs=1e-12)


class TestRdPilotFallbacksAreLoud:
    def _data(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(-1, 1, 400)
        y = 0.5 * x + 0.3 * x**2 + rng.normal(0, 0.2, 400)
        return y, x

    def test_healthy_pilot_fits_are_silent_and_finite(self):
        from statspai.rd import rdrobust as mod

        y, x = self._data()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m2 = mod._estimate_second_deriv(y, x, 0.8, "triangular")
            s2 = mod._local_residual_var(y, x, 0.8, "triangular")
        # m''(0) = 2 * 0.3 and residual variance = 0.2 ** 2 in this DGP.
        assert m2 == pytest.approx(0.6, abs=0.35)
        assert s2 == pytest.approx(0.04, abs=0.02)

    def test_failed_curvature_fit_warns_before_returning_zero(self, monkeypatch):
        from statspai.rd import rdrobust as mod

        def _boom(*args, **kwargs):
            raise np.linalg.LinAlgError("SVD did not converge")

        y, x = self._data()
        monkeypatch.setattr(np.linalg, "lstsq", _boom)
        with pytest.warns(RuntimeWarning, match="pilot curvature"):
            assert mod._estimate_second_deriv(y, x, 0.8, "triangular") == 0.0
        with pytest.warns(RuntimeWarning, match="pilot residual-variance"):
            s2 = mod._local_residual_var(y, x, 0.8, "triangular")
        in_bw = np.abs(x / 0.8) <= 1
        assert s2 == pytest.approx(float(np.var(y[in_bw])))


class TestMsmDensityFallbackIsLoud:
    def test_failed_density_model_warns_and_uses_marginal(self, monkeypatch):
        from scipy import stats

        import statspai.msm.msm  # noqa: F401  (sp.msm.msm is the function)

        mod = sys.modules["statspai.msm.msm"]

        rng = np.random.default_rng(5)
        X = rng.normal(size=(200, 2))
        a = X[:, 0] + rng.normal(size=200)

        def _boom(*args, **kwargs):
            raise np.linalg.LinAlgError("singular")

        monkeypatch.setattr(np.linalg, "lstsq", _boom)
        with pytest.warns(RuntimeWarning, match="drops the confounding adjustment"):
            dens = mod._gauss_density(X, a)
        expected = stats.norm.pdf(a, loc=a.mean(), scale=a.std(ddof=1))
        np.testing.assert_allclose(dens, expected, rtol=1e-12)


class TestRdDiagnosticsRecordsDegradations:
    def test_failed_substep_lands_in_degradations(self, monkeypatch):
        from statspai.rd import diagnostics as mod
        from statspai.workflow._degradation import WorkflowDegradedWarning

        if not hasattr(mod, "rdbwsensitivity"):
            pytest.skip("rdbwsensitivity is not a module-level name here")

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated bandwidth-grid failure")

        monkeypatch.setattr(mod, "rdbwsensitivity", _boom)
        rng = np.random.default_rng(11)
        x = rng.uniform(-1, 1, 600)
        y = 1.0 * (x >= 0) + 0.5 * x + rng.normal(0, 0.3, 600)
        df = pd.DataFrame({"y": y, "x": x})
        with pytest.warns(WorkflowDegradedWarning, match="bandwidth sensitivity"):
            out = sp.rdsummary(df, y="y", x="x", c=0.0, verbose=False)
        assert out["bw_sensitivity"] is None
        entry = out["degradations"][0]
        assert entry["section"] == "rd diagnostics: bandwidth sensitivity"
        assert entry["error_type"] == "RuntimeError"
        # The headline estimate is untouched by the failed side analysis.
        assert abs(float(out["estimate"].estimate) - 1.0) < 0.3
