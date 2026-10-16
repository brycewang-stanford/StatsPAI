"""Tests for sp.cross_validate — cross-engine validation.

Three layers:

1. **Unit** — the agreement / tolerance / verdict logic and the spec parser,
   with synthetic EngineEstimates and no engine dependencies (deterministic).
2. **Integration** — real engines (pyfixest, linearmodels, doubleml, R via
   Rscript) reproducing the same estimand; each guarded by availability so the
   suite degrades gracefully where a backend is absent.
3. **Degradation** — unavailable / unknown engines flow through as statuses and
   are recorded loudly rather than dropped.
"""

from __future__ import annotations

import shutil
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.crossval._agreement import (
    VERDICT_AGREE,
    VERDICT_DISAGREE,
    VERDICT_INSUFFICIENT,
    VERDICT_PARTIAL,
    EngineEstimate,
    default_policy,
    reconcile,
    resolve_policy,
)
from statspai.crossval._spec import EstimandSpec


def _have(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


HAS_PF = _have("pyfixest")
HAS_LM = _have("linearmodels")
HAS_DML = _have("doubleml")
HAS_R = shutil.which("Rscript") is not None


def _r_has(pkg: str) -> bool:
    if shutil.which("Rscript") is None:
        return False
    import subprocess

    try:
        out = subprocess.run(
            ["Rscript", "-e", f'cat(requireNamespace("{pkg}", quietly=TRUE))'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout.strip() == "TRUE"
    except Exception:
        return False


HAS_R_DID = _r_has("did")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def ols_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 0.5 * x1 - 0.4 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.fixture
def iv_data() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 900
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    x = 0.8 * z + 0.6 * u + rng.normal(size=n)
    w = rng.normal(size=n)
    y = 1.0 + 0.7 * x - 0.3 * w + u + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "w": w})


@pytest.fixture
def fe_data() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = 800
    fe = rng.integers(0, 10, n)
    x1 = rng.normal(size=n)
    y = 0.5 * x1 + 1.5 * fe + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "fe": fe})


def _est(engine, coef, se, **kw):
    return EngineEstimate(
        engine=engine, estimand="ols", term="x", coef=coef, se=se, **kw
    )


# --------------------------------------------------------------------------- #
# 1. Unit: agreement + verdict + tolerance
# --------------------------------------------------------------------------- #


class TestReconcile:
    def test_two_engines_agree(self):
        ests = [_est("statspai", 0.5000000, 0.04), _est("pyfixest", 0.5000001, 0.04)]
        rep = reconcile(ests, policy=default_policy("ols"))
        assert rep.verdict == VERDICT_AGREE
        assert rep.n_ok == 2
        assert rep.sign_agree is True

    def test_coefficient_disagreement(self):
        ests = [_est("statspai", 0.50, 0.04), _est("other", 0.80, 0.04)]
        rep = reconcile(ests, policy=default_policy("ols"))
        assert rep.verdict == VERDICT_DISAGREE

    def test_partial_when_only_se_differs(self):
        # Identical coefficients, very different SEs → PARTIAL in exact mode.
        ests = [
            _est("statspai", 0.500000, 0.040),
            _est("pyfixest", 0.500000, 0.080),
        ]
        rep = reconcile(ests, policy=default_policy("ols"))
        assert rep.verdict == VERDICT_PARTIAL

    def test_single_engine_is_insufficient(self):
        ests = [
            _est("statspai", 0.5, 0.04),
            EngineEstimate(
                engine="R::fixest",
                estimand="ols",
                term="x",
                status="unavailable",
                message="Rscript not found",
            ),
        ]
        rep = reconcile(ests, policy=default_policy("ols"))
        assert rep.verdict == VERDICT_INSUFFICIENT
        assert rep.n_ok == 1
        # The reason the other engine failed is surfaced, not hidden.
        assert any("Rscript" in n for n in rep.notes)

    def test_statistical_mode_within_band_agrees(self):
        # |Δcoef| = 0.01, max SE = 0.06 → 0.01 < 0.25*0.06=0.015 → AGREE.
        ests = [_est("statspai", 0.90, 0.06), _est("doubleml", 0.91, 0.06)]
        rep = reconcile(ests, policy=default_policy("dml"))
        assert rep.policy.mode == "statistical"
        assert rep.verdict == VERDICT_AGREE

    def test_statistical_mode_outside_band_is_partial(self):
        # Same sign, |Δcoef|=0.05 > 0.25*0.06 → PARTIAL.
        ests = [_est("statspai", 0.90, 0.06), _est("doubleml", 0.95, 0.06)]
        rep = reconcile(ests, policy=default_policy("dml"))
        assert rep.verdict == VERDICT_PARTIAL

    def test_reference_is_respected(self):
        ests = [_est("a", 0.5, 0.04), _est("statspai", 0.5, 0.04)]
        rep = reconcile(ests, policy=default_policy("ols"), reference="a")
        assert rep.reference == "a"


class TestTolerancePolicy:
    def test_ols_is_exact(self):
        assert default_policy("ols").mode == "exact"
        assert default_policy("iv").mode == "exact"

    def test_dml_is_statistical(self):
        assert default_policy("dml").mode == "statistical"
        assert default_policy("causal_forest").mode == "statistical"

    def test_poisson_is_loosened_exact(self):
        p = default_policy("poisson")
        assert p.mode == "exact"
        assert p.coef_rtol >= 1e-4  # looser than OLS's 1e-6

    def test_every_policy_has_a_rationale(self):
        for est in ["ols", "iv", "poisson", "dml", "unknown_xyz"]:
            assert default_policy(est).rationale  # non-empty

    def test_dict_override_merges(self):
        pol = resolve_policy("ols", {"coef_rtol": 1e-3})
        assert pol.coef_rtol == 1e-3
        assert pol.mode == "exact"  # base preserved

    def test_bad_tol_type_raises(self):
        with pytest.raises(TypeError):
            resolve_policy("ols", tol=3.14)


# --------------------------------------------------------------------------- #
# 2. Unit: EstimandSpec
# --------------------------------------------------------------------------- #


class TestEstimandSpec:
    def test_ols_formula_fills_fields(self, ols_data):
        s = EstimandSpec.from_kwargs(ols_data, "ols", formula="y ~ x1 + x2")
        assert s.y == "y"
        assert s.treatment == "x1"
        assert s.covariates == ["x2"]
        assert s.focal_term() == "x1"

    def test_formula_plus_explicit_treatment(self, ols_data):
        # Regression test: formula + treatment together must still fill y.
        s = EstimandSpec.from_kwargs(
            ols_data, "ols", formula="y ~ x1 + x2", treatment="x1"
        )
        assert s.y == "y"
        assert s.focal_term() == "x1"

    def test_fe_formula_parses_fixed_effects(self, fe_data):
        s = EstimandSpec.from_kwargs(fe_data, "feols", formula="y ~ x1 | fe")
        assert s.fixed_effects == ["fe"]
        assert s.treatment == "x1"

    def test_iv_three_part_formula(self, iv_data):
        s = EstimandSpec.from_kwargs(iv_data, "iv", formula="y ~ w | x ~ z")
        assert s.endog == ["x"]
        assert s.instruments == ["z"]
        assert s.focal_term() == "x"

    def test_missing_column_raises(self, ols_data):
        with pytest.raises(ValueError, match="not found"):
            EstimandSpec.from_kwargs(ols_data, "ols", y="y", treatment="nope")

    def test_iv_without_endog_raises(self, iv_data):
        with pytest.raises(ValueError, match="endogenous"):
            EstimandSpec.from_kwargs(iv_data, "iv", y="y", treatment="x")

    def test_build_formula_roundtrip(self, ols_data):
        s = EstimandSpec.from_kwargs(
            ols_data, "ols", y="y", treatment="x1", covariates=["x2"]
        )
        assert s.build_formula() == "y ~ x1 + x2"


# --------------------------------------------------------------------------- #
# 3. Integration: real engines reproduce the estimate
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not HAS_PF, reason="pyfixest not installed")
class TestIntegrationOLS:
    def test_ols_statspai_vs_pyfixest_agree(self, ols_data):
        cv = sp.cross_validate(
            ols_data,
            "ols",
            formula="y ~ x1 + x2",
            treatment="x1",
            engines=["statspai", "pyfixest"],
        )
        assert cv.verdict == VERDICT_AGREE
        assert cv.agreement.max_rel_coef_diff < 1e-6

    @pytest.mark.skipif(not HAS_LM, reason="linearmodels not installed")
    def test_ols_three_python_engines(self, ols_data):
        cv = sp.cross_validate(
            ols_data,
            "ols",
            formula="y ~ x1 + x2",
            treatment="x1",
            engines=["statspai", "pyfixest", "linearmodels"],
        )
        # Coefficients must match to floating-point; verdict may be PARTIAL if a
        # backend's default dof differs on the SE.
        assert cv.verdict in (VERDICT_AGREE, VERDICT_PARTIAL)
        assert cv.agreement.max_rel_coef_diff < 1e-6
        assert cv.agreement.n_ok == 3


@pytest.mark.skipif(not HAS_PF, reason="pyfixest not installed")
class TestIntegrationIV:
    def test_iv_statspai_vs_pyfixest(self, iv_data):
        cv = sp.cross_validate(
            iv_data,
            "iv",
            y="y",
            endog=["x"],
            instruments=["z"],
            covariates=["w"],
            treatment="x",
            engines=["statspai", "pyfixest"],
        )
        assert cv.verdict == VERDICT_AGREE
        assert cv.agreement.max_rel_coef_diff < 1e-6

    @pytest.mark.skipif(not HAS_LM, reason="linearmodels not installed")
    def test_iv_statspai_vs_linearmodels(self, iv_data):
        cv = sp.cross_validate(
            iv_data,
            "iv",
            y="y",
            endog=["x"],
            instruments=["z"],
            covariates=["w"],
            treatment="x",
            engines=["statspai", "linearmodels"],
        )
        assert cv.agreement.max_rel_coef_diff < 1e-6


@pytest.mark.skipif(not HAS_PF, reason="pyfixest not installed")
class TestIntegrationFE:
    def test_feols_statspai_vs_pyfixest(self, fe_data):
        cv = sp.cross_validate(
            fe_data,
            "feols",
            formula="y ~ x1 | fe",
            treatment="x1",
            engines=["statspai", "pyfixest"],
        )
        assert cv.verdict == VERDICT_AGREE
        assert cv.agreement.max_rel_coef_diff < 1e-6


def _native_mpdta(tmp_path) -> pd.DataFrame:
    """Export R's canonical `did::mpdta` (real, numeric-typed) to a frame."""
    import subprocess

    csv = tmp_path / "mpdta_native.csv"
    subprocess.run(
        [
            "Rscript",
            "-e",
            "suppressMessages(library(did)); data(mpdta); "
            f'write.csv(mpdta, "{csv}", row.names=FALSE)',
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return pd.read_csv(csv)


class TestDidSpec:
    """CS-DID spec wiring — no engines required."""

    def test_focal_term_is_att(self):
        df = pd.DataFrame({"y": [1.0], "g": [0], "t": [1], "i": [1]})
        s = EstimandSpec.from_kwargs(df, "did", y="y", g="g", t="t", i="i")
        assert s.focal_term() == "ATT"

    def test_missing_gti_raises(self):
        df = pd.DataFrame({"y": [1.0], "g": [0], "t": [1], "i": [1]})
        with pytest.raises(ValueError, match="g.*t.*i|cohort"):
            EstimandSpec.from_kwargs(df, "did", y="y")  # no g/t/i

    def test_public_api_requires_gti(self):
        df = pd.DataFrame({"y": [1.0], "g": [0], "t": [1], "i": [1]})
        with pytest.raises(ValueError):
            sp.cross_validate(df, "did", y="y")


class TestDidStatspaiOnly:
    """StatsPAI CS-DID adapter runs even without a second engine."""

    def test_statspai_did_engine_runs(self):
        mp = sp.datasets.mpdta()
        gcol = "first_treat" if "first_treat" in mp.columns else "first.treat"
        cv = sp.cross_validate(
            mp,
            "did",
            y="lemp",
            g=gcol,
            t="year",
            i="countyreal",
            engines=["statspai"],
        )
        sp_est = [e for e in cv.estimates if e.engine == "statspai"][0]
        assert sp_est.status == "ok"
        assert sp_est.term == "ATT"
        # Single engine → cannot cross-validate.
        assert cv.verdict == VERDICT_INSUFFICIENT


@pytest.mark.skipif(not HAS_R_DID, reason="R `did` package not installed")
class TestDidVsR:
    """The Cunningham experiment: StatsPAI CS-DID vs R's `did`."""

    def test_cs_did_agrees_with_r_did(self, tmp_path):
        mp = _native_mpdta(tmp_path)
        cv = sp.cross_validate(
            mp,
            "did",
            y="lemp",
            g="first.treat",
            t="year",
            i="countyreal",
            engines=["statspai", "R::did"],
        )
        assert cv.verdict == VERDICT_AGREE
        # Overall ATT matches to machine precision on the canonical data.
        assert cv.agreement.max_rel_coef_diff < 1e-6

    def test_integer_coded_columns_still_agree(self, tmp_path):
        # Regression guard for the integer-vs-numeric `gname` fragility in R
        # `did`: even with integer cohort/time/id columns, the adapter's
        # numeric coercion keeps StatsPAI and R::did in agreement.
        mp = _native_mpdta(tmp_path)
        for c in ("first.treat", "year", "countyreal"):
            mp[c] = mp[c].astype("int64")
        cv = sp.cross_validate(
            mp,
            "did",
            y="lemp",
            g="first.treat",
            t="year",
            i="countyreal",
            engines=["statspai", "R::did"],
        )
        assert cv.verdict == VERDICT_AGREE
        assert cv.agreement.max_rel_coef_diff < 1e-6


@pytest.mark.skipif(not HAS_DML, reason="doubleml not installed")
class TestIntegrationDML:
    def test_dml_statspai_vs_doubleml_runs(self, ols_data):
        # Binary treatment for a PLR cross-check.
        rng = np.random.default_rng(3)
        df = ols_data.copy()
        df["d"] = (0.5 * df["x1"] + rng.normal(size=len(df)) > 0).astype(float)
        df["y"] = 1.0 + 0.9 * df["d"] + 0.5 * df["x1"] + rng.normal(size=len(df))
        cv = sp.cross_validate(
            df,
            "dml",
            y="y",
            treatment="d",
            covariates=["x1", "x2"],
            engines=["statspai", "doubleml"],
        )
        # Different learners → statistical-scale comparison; both must run.
        assert cv.agreement.n_ok == 2
        assert cv.verdict in (VERDICT_AGREE, VERDICT_PARTIAL, VERDICT_DISAGREE)
        assert cv.agreement.policy.mode == "statistical"


@pytest.mark.skipif(not HAS_R, reason="Rscript not on PATH")
class TestIntegrationR:
    def test_ols_statspai_vs_r_fixest(self, ols_data):
        cv = sp.cross_validate(
            ols_data,
            "ols",
            formula="y ~ x1 + x2",
            treatment="x1",
            engines=["statspai", "R::fixest"],
        )
        if cv.agreement.n_ok < 2:
            pytest.skip("R fixest/jsonlite not installed in this R")
        assert cv.verdict == VERDICT_AGREE
        assert cv.agreement.max_rel_coef_diff < 1e-6
        # Provenance records the R version actually used.
        assert any(k.startswith("R::") for k in cv.provenance)


# --------------------------------------------------------------------------- #
# 4. Degradation: failures stay loud, never silent
# --------------------------------------------------------------------------- #


class TestDegradation:
    def test_unavailable_engine_is_reported_not_dropped(self, ols_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = sp.cross_validate(
                ols_data,
                "ols",
                formula="y ~ x1 + x2",
                treatment="x1",
                engines=["statspai", "Stata"],  # no Stata on PATH
            )
        engines = {e.engine: e.status for e in cv.estimates}
        assert engines.get("Stata") == "unavailable"
        # Recorded as a degradation, not silently skipped.
        assert any("Stata" in d["section"] for d in cv.degradations)

    def test_explicit_unavailable_emits_warning(self, ols_data):
        with pytest.warns(Warning):
            sp.cross_validate(
                ols_data,
                "ols",
                formula="y ~ x1 + x2",
                treatment="x1",
                engines=["statspai", "Stata"],
            )

    def test_unknown_engine_name(self, ols_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = sp.cross_validate(
                ols_data,
                "ols",
                formula="y ~ x1 + x2",
                treatment="x1",
                engines=["statspai", "definitely_not_an_engine"],
            )
        statuses = {e.engine: e.status for e in cv.estimates}
        assert statuses.get("definitely_not_an_engine") == "unavailable"

    def test_one_engine_yields_insufficient(self, ols_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = sp.cross_validate(
                ols_data,
                "ols",
                formula="y ~ x1 + x2",
                treatment="x1",
                engines=["statspai"],
            )
        assert cv.verdict == VERDICT_INSUFFICIENT

    def test_agent_output_marks_insufficient_as_not_cross_engine(self, ols_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = sp.cross_validate(
                ols_data,
                "ols",
                formula="y ~ x1 + x2",
                treatment="x1",
                engines=["statspai", "definitely_not_an_engine"],
            )
        d = cv.to_dict(detail="agent")
        assert d["engine_status_counts"]["ok"] == 1
        assert d["engine_status_counts"]["unavailable"] == 1
        assert d["can_claim_cross_engine_agreement"] is False


# --------------------------------------------------------------------------- #
# 5. Data-MCP provenance: normalized frames carry their source into the result
# --------------------------------------------------------------------------- #


class TestDataMCPProvenance:
    def test_worldbank_ingest_attrs_reach_cross_validate(self):
        rng = np.random.default_rng(44)
        rows = []
        for i in range(30):
            gdp = 10_000 + 250 * i + rng.normal(0, 50)
            life = 55 + 0.001 * gdp + rng.normal(0, 0.5)
            for ind, val in (("gdp", gdp), ("life", life)):
                rows.append(
                    {
                        "indicator": {"id": ind, "value": ind},
                        "country": {"id": f"C{i}", "value": f"Country {i}"},
                        "countryiso3code": f"C{i:02d}",
                        "date": "2020",
                        "value": val,
                    }
                )
        frame = sp.from_worldbank(rows, wide=True)

        cv = sp.cross_validate(
            frame,
            "ols",
            formula="life ~ gdp",
            treatment="gdp",
            engines=["statspai"],
        )

        data_prov = cv.provenance["data"]
        assert data_prov["source"] == "worldbank"
        assert data_prov["normalizer"] == "from_worldbank"
        assert data_prov["shape"] == "wide"
        assert data_prov["n_rows"] == len(frame)
        assert cv.to_dict(detail="agent")["provenance"]["data"] == data_prov


# --------------------------------------------------------------------------- #
# 6. Result object + public API
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not HAS_PF, reason="pyfixest not installed")
class TestResultObject:
    def test_summary_contains_verdict(self, ols_data):
        cv = sp.cross_validate(
            ols_data,
            "ols",
            formula="y ~ x1 + x2",
            treatment="x1",
            engines=["statspai", "pyfixest"],
        )
        text = str(cv.summary())
        assert "VERDICT" in text and cv.verdict in text

    def test_estimates_table_columns(self, ols_data):
        cv = sp.cross_validate(
            ols_data,
            "ols",
            formula="y ~ x1 + x2",
            treatment="x1",
            engines=["statspai", "pyfixest"],
        )
        cols = set(cv.estimates_table.columns)
        assert {"engine", "coef", "se", "status"} <= cols

    def test_to_dict_agent_has_next_steps(self, ols_data):
        cv = sp.cross_validate(
            ols_data,
            "ols",
            formula="y ~ x1 + x2",
            treatment="x1",
            engines=["statspai", "pyfixest"],
        )
        d = cv.to_dict(detail="agent")
        assert "verdict" in d and "next_steps" in d and "provenance" in d

    def test_to_markdown_and_latex(self, ols_data):
        cv = sp.cross_validate(
            ols_data,
            "ols",
            formula="y ~ x1 + x2",
            treatment="x1",
            engines=["statspai", "pyfixest"],
        )
        assert "Cross-engine" in cv.to_markdown()
        assert "tabular" in cv.to_latex().lower()


class TestPublicAPI:
    def test_exported(self):
        assert callable(sp.cross_validate)
        assert hasattr(sp, "CrossValidationResult")

    def test_registered(self):
        assert "cross_validate" in sp.list_functions()

    def test_data_mode_requires_estimand(self, ols_data):
        with pytest.raises(ValueError, match="estimand"):
            sp.cross_validate(ols_data)


class TestIVSpecKeepsEveryExogenousRegressor:
    """An IV formula's exogenous regressors all belong in the model.

    ``_reconcile_formula_and_fields`` used to promote the first exogenous
    term to ``treatment`` and drop it from ``covariates``. The statspai and
    linearmodels IV adapters build the model from ``covariates`` only, so
    ``y ~ w1 + w2 | x ~ z`` was silently fitted without ``w1`` -- and the
    two mis-specified engines agreed with each other, outvoting the one
    correct engine (pyfixest). Every engine is now held to a direct fit.
    """

    @pytest.fixture
    def iv2_data(self) -> pd.DataFrame:
        rng = np.random.default_rng(11)
        n = 900
        z1, z2, u = rng.normal(size=(3, n))
        w1, w2 = rng.normal(size=(2, n))
        x = 0.8 * z1 + 0.4 * z2 + 0.5 * w1 + 0.6 * u + rng.normal(size=n)
        y = 1.0 + 0.7 * x + 0.9 * w1 - 0.3 * w2 + u + rng.normal(size=n)
        return pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2, "w1": w1, "w2": w2})

    @pytest.fixture
    def direct(self, iv2_data):
        return sp.ivreg("y ~ (x ~ z1 + z2) + w1 + w2", data=iv2_data)

    def test_iv_formula_spec_fields(self, iv2_data):
        s = EstimandSpec.from_kwargs(
            iv2_data, "iv", formula="y ~ w1 + w2 | x ~ z1 + z2"
        )
        assert s.treatment is None
        assert s.covariates == ["w1", "w2"]
        assert s.focal_term() == "x"

    def test_aer_formula_equals_fixest_formula(self, iv2_data):
        aer = EstimandSpec.from_kwargs(
            iv2_data, "iv", formula="y ~ (x ~ z1 + z2) + w1 + w2"
        )
        fx = EstimandSpec.from_kwargs(
            iv2_data, "iv", formula="y ~ w1 + w2 | x ~ z1 + z2"
        )
        for field_name in ("y", "treatment", "covariates", "endog", "instruments"):
            assert getattr(aer, field_name) == getattr(fx, field_name)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"formula": "y ~ w1 + w2 | x ~ z1 + z2"},
            {"formula": "y ~ (x ~ z1 + z2) + w1 + w2"},
            # Explicit *exogenous* treatment must stay in the model too.
            {
                "y": "y",
                "treatment": "w1",
                "covariates": ["w2"],
                "endog": ["x"],
                "instruments": ["z1", "z2"],
                "term": "x",
            },
        ],
        ids=["fixest", "aer", "exog_treatment"],
    )
    def test_every_engine_matches_direct_fit(self, iv2_data, direct, kwargs):
        engines = (
            ["statspai"]
            + (["pyfixest"] if HAS_PF else [])
            + (["linearmodels"] if HAS_LM else [])
        )
        cv = sp.cross_validate(iv2_data, "iv", engines=engines, **kwargs)
        ok = [e for e in cv.estimates if e.status == "ok"]
        assert len(ok) == len(engines)
        for est in ok:
            assert est.coef == pytest.approx(direct.params["x"], rel=1e-8), est.engine

    def test_result_mode_error_gives_runnable_call(self, iv2_data, direct):
        with pytest.raises(ValueError) as exc:
            sp.cross_validate(direct)
        msg = str(exc.value)
        call = msg[msg.index("sp.cross_validate(df") :]
        cv = eval(call, {"sp": sp, "df": iv2_data})
        own = next(e for e in cv.estimates if e.engine == "statspai")
        assert own.coef == pytest.approx(direct.params["x"], rel=1e-12)
        assert own.se == pytest.approx(direct.std_errors["x"], rel=1e-12)


class TestEnginesHonourRequestedVariance:
    """``cluster=`` / ``vcov=`` must reach every engine.

    The statspai engine ignored ``cluster`` for OLS / FE / IV (and ``vcov`` for
    IV); the R engine sent fixest the bare string ``"cluster"``, which clusters
    on the first fixed effect or errors without one; linearmodels reported
    large-sample SEs. A clustered cross-check therefore compared different
    variance estimators. Each engine is now held to the direct StatsPAI fit.
    """

    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        rng = np.random.default_rng(21)
        n = 1200
        cid = rng.integers(0, 40, n)
        fe = rng.integers(0, 8, n)
        shock = rng.normal(size=40)[cid]
        z, w, u = rng.normal(size=(3, n))
        x = 0.9 * z + 0.4 * w + 0.5 * u + rng.normal(size=n)
        y = 0.6 * x - 0.2 * w + 0.3 * fe + shock + u + rng.normal(size=n)
        return pd.DataFrame({"y": y, "x": x, "z": z, "w": w, "cid": cid, "fe": fe})

    CASES = {
        "ols": ("y ~ x + w", lambda d, **k: sp.regress("y ~ x + w", data=d, **k)),
        "feols": (
            "y ~ x + w | fe",
            lambda d, **k: sp.feols("y ~ x + w | fe", data=d, **k),
        ),
        "iv": (
            "y ~ (x ~ z) + w",
            lambda d, **k: sp.ivreg("y ~ (x ~ z) + w", data=d, **k),
        ),
    }

    @pytest.mark.parametrize("estimand", list(CASES))
    @pytest.mark.parametrize("variance", ["cluster", "hc1"])
    def test_engines_match_direct_fit(self, panel, estimand, variance):
        formula, direct_fit = self.CASES[estimand]
        if variance == "cluster":
            cv_kw, fit_kw = {"cluster": "cid"}, {"cluster": "cid"}
        elif estimand == "feols":
            cv_kw, fit_kw = {"vcov": "HC1"}, {"vcov": "HC1"}
        else:
            cv_kw, fit_kw = {"vcov": "HC1"}, {"robust": "hc1"}
        direct = direct_fit(panel, **fit_kw)
        want_se = float(direct.std_errors["x"])

        engines = ["statspai"]
        exact = {"statspai"}
        if HAS_PF:
            engines.append("pyfixest")
            exact.add("pyfixest")
        if HAS_LM:
            engines.append("linearmodels")
            if estimand != "feols":  # FE dof counting is a documented convention gap
                exact.add("linearmodels")
        cv = sp.cross_validate(
            panel, estimand, formula=formula, engines=engines, **cv_kw
        )
        by_engine = {e.engine: e for e in cv.estimates}
        for name in engines:
            est = by_engine[name]
            assert est.status == "ok", (name, est)
            assert est.coef == pytest.approx(float(direct.params["x"]), rel=1e-8)
            rel = 1e-8 if name in exact else 5e-3
            assert est.se == pytest.approx(want_se, rel=rel), name
        assert cv.verdict == VERDICT_AGREE

    @pytest.mark.skipif(not HAS_R, reason="Rscript not installed")
    def test_r_fixest_clusters_on_the_requested_variable(self, panel):
        direct = sp.regress("y ~ x + w", data=panel, cluster="cid")
        cv = sp.cross_validate(
            panel, "ols", formula="y ~ x + w", cluster="cid", engines=["R"]
        )
        (est,) = cv.estimates
        if est.status != "ok":
            pytest.skip(f"R fixest unavailable: {est.message}")
        assert est.se == pytest.approx(float(direct.std_errors["x"]), rel=1e-8)
