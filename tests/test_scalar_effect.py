"""Tests for ScalarEffect and the structured CausalForest.ate()/att()."""

import json

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient


def _fit_forest(seed=0, n=400):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    T = rng.binomial(1, 0.5, n)
    Y = (1.0 + X[:, 0]) * T + X[:, 1] + rng.normal(0, 0.5, n)
    cf = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=50, random_state=0)
    return cf, X, T


class TestScalarEffectClass:
    def test_float_semantics(self):
        e = sp.ScalarEffect(1.25, estimand="ATE", se=0.5, ci=(0.27, 2.23), pvalue=0.012)
        assert isinstance(e, float)
        assert float(e) == 1.25
        assert e + 1 == 2.25
        assert f"{e:.2f}" == "1.25"
        assert e.se == 0.5

    def test_repr_shows_inference(self):
        e = sp.ScalarEffect(1.25, estimand="ATE", se=0.5, ci=(0.27, 2.23), pvalue=0.012)
        text = str(e)
        assert "ATE = 1.2500" in text
        assert "SE = 0.5000" in text
        assert "95% CI" in text
        assert "p = 0.0120" in text

    def test_repr_without_inference(self):
        e = sp.ScalarEffect(1.25, estimand="ATT")
        assert str(e) == "ATT = 1.2500"
        e2 = sp.ScalarEffect(1.25, estimand="ATT", inference_error="boom")
        assert "inference unavailable" in str(e2)

    def test_json_serializable(self):
        e = sp.ScalarEffect(1.25, estimand="ATE", se=0.5)
        assert json.loads(json.dumps({"x": e}))["x"] == 1.25
        payload = e.to_dict()
        assert payload["estimate"] == 1.25
        assert payload["se"] == 0.5


class TestForestATE:
    def test_value_is_plugin_mean_cate(self):
        cf, X, _ = _fit_forest()
        a = cf.ate(X)
        # Backward compatibility: float value must equal the historical
        # plug-in mean CATE, bit for bit.
        assert float(a) == float(cf.effect(X).mean())

    def test_inference_attached(self):
        cf, X, _ = _fit_forest()
        a = cf.ate(X)
        assert isinstance(a, sp.ScalarEffect)
        assert a.se is not None and a.se > 0
        assert a.ci is not None and a.ci[0] < a.ci[1]
        assert a.detail["method"] in ("aipw", "plug_in")
        assert "SE = " in str(a)

    def test_att_structured(self):
        cf, X, T = _fit_forest()
        t = cf.att(X, T)
        assert isinstance(t, sp.ScalarEffect)
        assert t.estimand == "ATT"
        assert float(t) == pytest.approx(float(cf.effect(X)[np.asarray(T) == 1].mean()))
        assert t.se is not None

    def test_att_no_treated_raises(self):
        cf, X, _ = _fit_forest()
        with pytest.raises(DataInsufficient):
            cf.att(X, np.zeros(len(X)))

    def test_default_x_uses_training_sample(self):
        cf, X, _ = _fit_forest()
        assert float(cf.ate()) == pytest.approx(float(cf.ate(X)))
