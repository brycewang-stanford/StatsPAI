"""Unit tests for the keyword-alias decorator (grammar convergence plumbing)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from statspai._aliases import accepts_aliases  # noqa: E402


def test_canonical_alias_forwarded() -> None:
    @accepts_aliases(vce="robust")
    def fit(formula, data=None, robust="nonrobust"):
        return robust

    assert fit("y ~ x", vce="hc1") == "hc1"
    assert fit("y ~ x", robust="hc2") == "hc2"
    assert fit("y ~ x") == "nonrobust"


def test_multiple_aliases() -> None:
    @accepts_aliases(vce="robust", outcome="y", treatment="treat")
    def fit(y=None, treat=None, robust=None):
        return (y, treat, robust)

    assert fit(outcome="wage", treatment="union", vce="cluster") == (
        "wage",
        "union",
        "cluster",
    )


def test_conflict_raises() -> None:
    @accepts_aliases(vce="robust")
    def fit(robust=None):
        return robust

    with pytest.raises(TypeError, match="both 'vce' and its canonical target"):
        fit(vce="hc1", robust="hc2")


def test_signature_preserved_for_introspection() -> None:
    @accepts_aliases(vce="robust")
    def fit(formula, data=None, robust="nonrobust"):
        return robust

    import inspect

    params = list(inspect.signature(fit).parameters)
    assert params == ["formula", "data", "robust"]
    assert fit.__statspai_aliases__ == {"vce": "robust"}


def test_warn_off_by_default() -> None:
    @accepts_aliases(vce="robust")
    def fit(robust=None):
        return robust

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        assert fit(vce="hc1") == "hc1"


def test_warn_opt_in() -> None:
    @accepts_aliases(_warn=True, vce="robust")
    def fit(robust=None):
        return robust

    with pytest.warns(DeprecationWarning):
        fit(vce="hc1")


def test_stacked_decorators_merge_alias_record() -> None:
    @accepts_aliases(vce="robust")
    @accepts_aliases(outcome="y")
    def fit(y=None, robust=None):
        return (y, robust)

    assert fit.__statspai_aliases__ == {"vce": "robust", "outcome": "y"}
    assert fit(outcome="wage", vce="hc1") == ("wage", "hc1")


def test_empty_map_rejected() -> None:
    with pytest.raises(ValueError):
        accepts_aliases()


def test_vce_alias_roster_se_theme() -> None:
    """Every SE-bearing estimator/diagnostic in the grammar-convergence roster
    accepts the canonical ``vce=`` spelling (forwarded to its legacy
    ``robust=`` / ``vcov=`` / ``se_type=`` parameter). Additive-only during
    JSS review; the post-review rename flips the map."""
    import statspai as sp

    roster = [
        "auto_iv",
        "bartik",
        "anderson_rubin_test",
        "effective_f_test",
        "weakrobust",
        "iv_diag",
        "panel_logit",
        "panel_probit",
        "panel_compare",
        "ancova",
        "negd",
        "liml",
        "jive",
        "lasso_iv",
        "nbreg",
        "xtnbreg",
        "poisson",
        "ppmlhdfe",
        "fracreg",
        "betareg",
        "glm",
        "logit",
        "probit",
        "cloglog",
        "mlogit",
        "ologit",
        "oprobit",
        "clogit",
        "biprobit",
        "etregress",
        "truncreg",
        "zip_model",
        "zinb",
        "hurdle",
        "subgroup_analysis",
        "spatial_did",
        "cox",
        "survreg",
        # wired natively earlier in the campaign:
        "regress",
        "feols",
        "fepois",
        "feglm",
    ]
    missing = [
        name
        for name in roster
        if "vce" not in getattr(getattr(sp, name), "__statspai_aliases__", {})
        and "vce" not in str(getattr(getattr(sp, name), "__doc__", ""))
        and name not in ("regress",)  # regress takes vce= natively
    ]
    assert not missing, f"functions missing the vce= spelling: {missing}"


def test_vce_alias_forwards_in_estimator() -> None:
    """vce= and the legacy spelling produce identical results end-to-end."""
    import numpy as np
    import pandas as pd

    import statspai as sp

    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)
    df = pd.DataFrame({"y": y, "x": x})
    a = sp.logit("y ~ x", df, vce="hc1")
    b = sp.logit("y ~ x", df, robust="hc1")
    assert float(a.std_errors["x"]) == float(b.std_errors["x"])
    with pytest.raises(TypeError):
        sp.logit("y ~ x", df, vce="hc1", robust="hc1")
