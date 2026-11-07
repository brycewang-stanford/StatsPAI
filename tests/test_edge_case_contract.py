"""Edge-case contract for the core estimators (rOpenSci audit gaps 4 and 16).

One parametrised grid runs the audit's probe inputs over 25 core
estimators. Every cell asserts a true proposition:

* a refusal: a specific taxonomy exception with a non-empty
  ``recovery_hint`` and the diagnostics that name the offending input; or
* a result that equals a known truth or an identity, plus the documented
  warning where the estimator announces what it did. A constant covariate
  must give the fit without that covariate; a covariate that duplicates
  another must give the fit without the duplicate; machine-epsilon noise on
  the outcome must leave the estimate unchanged; an exact fit must recover
  the coefficients that generated the data.

The shared checks (zero rows, an entirely missing column, a text outcome,
``inf``, a single cluster, ``missing="raise"`` and its rule per role) live in
``statspai._input_validation.check_estimator_input``.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
    StatsPAIError,
)

# --------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------- #


def _cross_section(n: int = 150, seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    x1, x2, z = r.normal(size=n), r.normal(size=n), r.normal(size=n)
    u = r.normal(size=n)
    d = 0.6 * z + 0.4 * x1 + 0.5 * u + 0.5 * r.normal(size=n)
    lat = 0.3 + 0.8 * x1 - 0.5 * x2 + r.logistic(size=n)
    s = (0.3 + z + 0.5 * x1 + r.normal(size=n) > 0).astype(float)
    ps = 1 / (1 + np.exp(-(0.5 * x1 - 0.5 * x2)))
    treat = (r.uniform(size=n) < ps).astype(float)
    return pd.DataFrame(
        {
            "y": 1 + 0.5 * x1 - 0.3 * x2 + d + u,
            "x1": x1,
            "x2": x2,
            "z": z,
            "d": d,
            "yb": (lat > 0).astype(float),
            "yc": r.poisson(np.exp(0.2 + 0.4 * x1 - 0.2 * x2)).astype(float),
            "ycens": np.maximum(0.0, 0.2 + x1 - 0.5 * x2 + r.normal(size=n)),
            "s": s,
            "yh": np.where(s == 1, 1 + 0.5 * x1 - 0.3 * x2 + r.normal(size=n), np.nan),
            "treat": treat,
            "ytreat": 1 + 2 * treat + x1 - 0.5 * x2 + r.normal(size=n),
            "cl": np.arange(n) % 15,
        }
    )


def _panel(n_units: int = 30, periods: int = 6, seed: int = 1) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        g = (0, 3, 5)[i % 3]
        a = r.normal()
        for t in range(1, periods + 1):
            treated = int(g > 0 and t >= g)
            x, w = r.normal(), r.normal()
            y = a + 0.2 * t + 0.5 * x - 0.3 * w + 2.0 * treated + 0.5 * r.normal()
            rows.append((i, t, g, x, w, y, i % 10))
    return pd.DataFrame(rows, columns=["id", "t", "g", "x", "w", "y", "cl"])


def _rd(n: int = 500, seed: int = 2) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    xr = r.uniform(-1, 1, n)
    z = r.normal(size=n)
    y = 1 + 0.5 * xr + (xr >= 0) + 0.3 * z + 0.3 * r.normal(size=n)
    return pd.DataFrame({"y": y, "xr": xr, "z": z, "cl": np.arange(n) % 20})


def _synth_panel(n_units: int = 9, periods: int = 18, seed: int = 3) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    f = r.normal(size=periods).cumsum()
    rows = []
    for i in range(n_units):
        lam = r.uniform(0.5, 1.5)
        for t in range(periods):
            effect = 2.0 if (i == 0 and t >= 13) else 0.0
            rows.append((i, t, 5 + lam * f[t] + 0.2 * r.normal() + effect))
    return pd.DataFrame(rows, columns=["unit", "t", "y"])


# --------------------------------------------------------------------- #
#  Estimators
# --------------------------------------------------------------------- #


def _formula(y: str, covs: Sequence[str], tail: str = "") -> str:
    return f"{y} ~ " + (" + ".join(covs) if covs else "1") + tail


def _lin_learners() -> Dict[str, Any]:
    from sklearn.linear_model import LinearRegression

    return {"model_y": LinearRegression(), "model_d": LinearRegression()}


@dataclass
class Spec:
    """How to call one estimator on the grid's data."""

    name: str
    data: Callable[[], pd.DataFrame]
    fit: Callable[..., Any]  # fit(df, covs, **kw)
    y: str
    covs: Sequence[str]
    cluster_kw: Optional[Callable[[str], Dict[str, Any]]] = None
    eps_col: Optional[str] = None  # where eps noise goes (default: y)
    eps_kw: Dict[str, Any] = field(default_factory=dict)
    # constant / collinear: the expected warning, or an exception class
    constant: Any = UserWarning
    collinear: Any = UserWarning
    extra: Dict[str, Any] = field(default_factory=dict)


def _cl(col: str) -> Dict[str, Any]:
    return {"cluster": col}


SPECS = [
    Spec(
        "regress",
        _cross_section,
        lambda df, covs, **k: sp.regress(_formula("y", covs), df, **k),
        "y",
        ("x1", "x2"),
        _cl,
    ),
    Spec(
        "logit",
        _cross_section,
        lambda df, covs, **k: sp.logit(_formula("yb", covs), df, **k),
        "yb",
        ("x1", "x2"),
        _cl,
        eps_col="x1",
    ),
    Spec(
        "probit",
        _cross_section,
        lambda df, covs, **k: sp.probit(_formula("yb", covs), df, **k),
        "yb",
        ("x1", "x2"),
        _cl,
        eps_col="x1",
    ),
    Spec(
        "poisson",
        _cross_section,
        lambda df, covs, **k: sp.poisson(_formula("yc", covs), df, **k),
        "yc",
        ("x1", "x2"),
        _cl,
        eps_col="x1",
    ),
    Spec(
        "glm",
        _cross_section,
        lambda df, covs, **k: sp.glm(_formula("y", covs), df, **k),
        "y",
        ("x1", "x2"),
        _cl,
    ),
    Spec(
        "ivreg",
        _cross_section,
        lambda df, covs, **k: sp.ivreg(_formula("y", ["(d ~ z)", *covs]), df, **k),
        "y",
        ("x1", "x2"),
        _cl,
    ),
    Spec(
        "iv",
        _cross_section,
        lambda df, covs, **k: sp.iv(_formula("y", ["(d ~ z)", *covs]), df, **k),
        "y",
        ("x1", "x2"),
        _cl,
    ),
    Spec(
        "feols",
        _cross_section,
        lambda df, covs, **k: sp.feols(_formula("y", covs), df, **k),
        "y",
        ("x1", "x2"),
        _cl,
    ),
    Spec(
        "fepois",
        _cross_section,
        lambda df, covs, **k: sp.fepois(_formula("yc", covs), df, **k),
        "yc",
        ("x1", "x2"),
        _cl,
        eps_col="x1",
    ),
    Spec(
        "feglm",
        _cross_section,
        lambda df, covs, **k: sp.feglm(_formula("yb", covs), df, family="logit", **k),
        "yb",
        ("x1", "x2"),
        _cl,
        eps_col="x1",
    ),
    Spec(
        "qreg",
        _cross_section,
        lambda df, covs, **k: sp.qreg(df, formula=_formula("y", covs), **k),
        "y",
        ("x1", "x2"),
        _cl,
    ),
    Spec(
        "tobit",
        _cross_section,
        lambda df, covs, **k: sp.tobit(df, y="ycens", x=list(covs), **k),
        "ycens",
        ("x1", "x2"),
        _cl,
    ),
    Spec(
        "heckman",
        _cross_section,
        lambda df, covs, **k: sp.heckman(
            df, y="yh", x=list(covs), select="s", z=["z", *covs], **k
        ),
        "yh",
        ("x1", "x2"),
        lambda col: {"cluster": col, "method": "ml"},
    ),
    Spec(
        "did",
        _panel,
        lambda df, covs, **k: sp.did(
            df, y="y", treat="g", time="t", id="id", covariates=list(covs) or None, **k
        ),
        "y",
        ("x", "w"),
        _cl,
        constant=None,
        collinear=None,
    ),
    Spec(
        "callaway_santanna",
        _panel,
        lambda df, covs, **k: sp.callaway_santanna(
            df, y="y", g="g", t="t", i="id", x=list(covs) or None, **k
        ),
        "y",
        ("x", "w"),
        lambda col: {"clustervars": col},
        constant=None,
        collinear=None,
    ),
    Spec(
        "sun_abraham",
        _panel,
        lambda df, covs, **k: sp.sun_abraham(
            df, y="y", g="g", t="t", i="id", covariates=list(covs) or None, **k
        ),
        "y",
        ("x", "w"),
        _cl,
        constant=None,
        collinear=None,
    ),
    Spec(
        "did_imputation",
        _panel,
        lambda df, covs, **k: sp.did_imputation(
            df,
            y="y",
            group="id",
            time="t",
            first_treat="g",
            controls=list(covs) or None,
            **k,
        ),
        "y",
        ("x", "w"),
        _cl,
        constant=None,
        collinear=None,
    ),
    Spec(
        "event_study",
        _panel,
        lambda df, covs, **k: sp.event_study(
            df,
            y="y",
            treat_time="g",
            time="t",
            unit="id",
            window=(-2, 1),
            covariates=list(covs) or None,
            **k,
        ),
        "y",
        ("x", "w"),
        _cl,
        constant=None,
        collinear=None,
    ),
    Spec(
        "rdrobust",
        _rd,
        lambda df, covs, **k: sp.rdrobust(
            df, y="y", x="xr", covs=list(covs) or None, **k
        ),
        "y",
        ("z",),
        _cl,
        constant=IdentificationFailure,
        collinear="n/a",
    ),
    Spec(
        "synth",
        _synth_panel,
        lambda df, covs, **k: sp.synth(
            df,
            outcome="y",
            unit="unit",
            time="t",
            treated_unit=0,
            treatment_time=13,
            placebo=False,
            **k,
        ),
        "y",
        (),
        None,
        constant="n/a",
        collinear="n/a",
    ),
    Spec(
        "dml",
        _cross_section,
        lambda df, covs, **k: sp.dml(df, y="ytreat", d="treat", X=list(covs), **k),
        "ytreat",
        ("x1", "x2"),
        _cl,
        constant=None,
        collinear=None,
        # Tree learners pick among splits of equal gain by the last bits of
        # y, so 1e-13 noise moves a forest's fit at the 1e-4 level; the
        # contract is checked on the estimator itself, with linear learners.
        eps_kw={"_learners": True},
    ),
    Spec(
        "panel",
        _panel,
        lambda df, covs, **k: sp.panel(
            df, _formula("y", covs), entity="id", time="t", method="fe", **k
        ),
        "y",
        ("x", "w"),
        _cl,
    ),
    Spec(
        "match",
        _cross_section,
        lambda df, covs, **k: sp.match(
            df, y="ytreat", treat="treat", covariates=list(covs), **k
        ),
        "ytreat",
        ("x1", "x2"),
        _cl,
        constant=None,
        collinear=None,
    ),
    Spec(
        "ipw",
        _cross_section,
        lambda df, covs, **k: sp.ipw(
            df,
            y="ytreat",
            treat="treat",
            covariates=list(covs),
            se_method="sandwich",
            **k,
        ),
        "ytreat",
        ("x1", "x2"),
        _cl,
    ),
    Spec(
        "aipw",
        _cross_section,
        lambda df, covs, **k: sp.aipw(
            df, y="ytreat", treat="treat", covariates=list(covs), **k
        ),
        "ytreat",
        ("x1", "x2"),
        _cl,
    ),
]
BY_NAME = {s.name: s for s in SPECS}
IDS = [s.name for s in SPECS]


def _quiet(fn: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _numbers(res: Any) -> pd.Series:
    """The reported point estimates (scalar estimate or coefficient vector)."""
    est = getattr(res, "estimate", None)
    if est is not None and np.ndim(est) == 0:
        return pd.Series({"estimate": float(est)})
    return pd.Series(res.params, dtype=float)


def _same_numbers(a: Any, b: Any, rel: float = 1e-8) -> None:
    na, nb = _numbers(a), _numbers(b)
    common = [k for k in na.index if k in nb.index]
    assert common, (list(na.index), list(nb.index))
    assert list(na.index) == list(nb.index)
    np.testing.assert_allclose(
        na[common].to_numpy(), nb[common].to_numpy(), rtol=rel, atol=rel
    )


def _fit_announcing_omission(spec: Spec, df: pd.DataFrame, col: str) -> Any:
    """Fit and check that a warning of ``spec``'s class names ``col`` as gone."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = spec.fit(df, spec.covs)
    said = [
        str(w.message)
        for w in caught
        if issubclass(w.category, UserWarning)
        and re.search(rf"\b{col}\b", str(w.message))
        and re.search(r"omitted|dropped|absorb", str(w.message))
    ]
    assert said, [str(w.message)[:120] for w in caught]
    return res


def _refusal(excinfo: Any, cls: type) -> StatsPAIError:
    err = excinfo.value
    assert type(err) is cls or isinstance(err, cls), type(err)
    assert isinstance(err, StatsPAIError)
    assert (err.recovery_hint or "").strip(), "taxonomy errors carry a recovery hint"
    return err


# --------------------------------------------------------------------- #
#  The grid
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_zero_rows_is_refused(spec: Spec) -> None:
    df = spec.data().iloc[:0]
    with pytest.raises(DataInsufficient) as ei:
        _quiet(lambda: spec.fit(df, spec.covs))
    err = _refusal(ei, DataInsufficient)
    assert err.diagnostics["n_rows"] == 0


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_entirely_missing_column_is_refused(spec: Spec) -> None:
    df = spec.data()
    col = spec.covs[0] if spec.covs else spec.y
    df[col] = np.nan
    with pytest.raises(DataInsufficient) as ei:
        _quiet(lambda: spec.fit(df, spec.covs))
    err = _refusal(ei, DataInsufficient)
    assert err.diagnostics["all_missing_columns"] == [col]


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_infinite_outcome_is_refused(spec: Spec) -> None:
    df = spec.data()
    df.loc[df.index[df[spec.y].notna()][3], spec.y] = np.inf
    with pytest.raises(DataInsufficient) as ei:
        _quiet(lambda: spec.fit(df, spec.covs))
    err = _refusal(ei, DataInsufficient)
    assert err.diagnostics["infinite_by_column"] == {spec.y: 1}


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_text_outcome_is_refused(spec: Spec) -> None:
    df = spec.data()
    df[spec.y] = np.where(np.arange(len(df)) % 2, "low", "high")
    with pytest.raises(MethodIncompatibility) as ei:
        _quiet(lambda: spec.fit(df, spec.covs))
    err = _refusal(ei, MethodIncompatibility)
    assert err.diagnostics["column"] == spec.y


@pytest.mark.parametrize(
    "spec", [s for s in SPECS if s.cluster_kw], ids=lambda s: s.name
)
def test_single_cluster_is_refused(spec: Spec) -> None:
    df = spec.data()
    df["cl"] = 7
    with pytest.raises(DataInsufficient) as ei:
        _quiet(lambda: spec.fit(df, spec.covs, **spec.cluster_kw("cl")))
    err = _refusal(ei, DataInsufficient)
    assert err.diagnostics == {"cluster": "cl", "n_clusters": 1}


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_eps_noise_on_the_outcome_leaves_the_estimate(spec: Spec) -> None:
    # G5.9a: relative noise of 1e-13 is far below any statistical signal,
    # so every reported estimate must agree with the clean fit to 1e-6.
    df = spec.data()
    col = spec.eps_col or spec.y
    noise = np.random.default_rng(11).standard_normal(len(df))
    noisy = df.copy()
    noisy[col] = df[col] * (1 + 1e-13 * noise)
    kw = _lin_learners() if spec.eps_kw.get("_learners") else {}
    clean = _quiet(lambda: spec.fit(df, spec.covs, **kw))
    jittered = _quiet(lambda: spec.fit(noisy, spec.covs, **kw))
    _same_numbers(jittered, clean, rel=1e-6)


@pytest.mark.parametrize(
    "spec", [s for s in SPECS if s.constant != "n/a"], ids=lambda s: s.name
)
def test_constant_covariate(spec: Spec) -> None:
    df = spec.data()
    col = spec.covs[0]
    df[col] = 1.0
    rest = [c for c in spec.covs if c != col]
    if isinstance(spec.constant, type) and issubclass(spec.constant, StatsPAIError):
        with pytest.raises(spec.constant) as ei:
            _quiet(lambda: spec.fit(df, spec.covs))
        _refusal(ei, spec.constant)
        return
    if spec.constant is None:
        res = _quiet(lambda: spec.fit(df, spec.covs))
    else:
        res = _fit_announcing_omission(spec, df, col)
    # A column that never varies carries no information: the fit must be
    # the fit without it.
    _same_numbers(res, _quiet(lambda: spec.fit(df, rest)))


@pytest.mark.parametrize(
    "spec", [s for s in SPECS if s.collinear != "n/a"], ids=lambda s: s.name
)
def test_duplicated_covariate(spec: Spec) -> None:
    df = spec.data()
    first, second = spec.covs[0], spec.covs[1]
    df[second] = 2.0 * df[first]
    if spec.collinear is None:
        res = _quiet(lambda: spec.fit(df, spec.covs))
    else:
        res = _fit_announcing_omission(spec, df, second)
    # x2 = 2 x1 adds nothing to x1: the fit must be the fit on x1 alone.
    _same_numbers(res, _quiet(lambda: spec.fit(df, [first])))


# --------------------------------------------------------------------- #
#  missing= (audit gap 4)
# --------------------------------------------------------------------- #


def _one_missing(spec: Spec) -> tuple:
    df = spec.data()
    col = spec.covs[0] if spec.covs else spec.y
    row = df.index[df[col].notna() & df[spec.y].notna()][4]
    df.loc[row, col] = np.nan
    return df, col, row


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_missing_raise_names_the_column_and_count(spec: Spec) -> None:
    df, col, _ = _one_missing(spec)
    with pytest.raises(DataInsufficient) as ei:
        _quiet(lambda: spec.fit(df, spec.covs, missing="raise"))
    err = _refusal(ei, DataInsufficient)
    assert err.diagnostics["missing_by_column"] == {col: 1}
    assert err.diagnostics["n_rows_missing"] == 1
    assert err.diagnostics["missing_policy"] == "raise"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_missing_raise_on_complete_data_changes_nothing(spec: Spec) -> None:
    df = spec.data()
    default = _quiet(lambda: spec.fit(df, spec.covs))
    strict = _quiet(lambda: spec.fit(df, spec.covs, missing="raise"))
    _same_numbers(strict, default, rel=1e-12)


@pytest.mark.parametrize(
    "spec",
    [s for s in SPECS if s.name not in ("synth",)],
    ids=lambda s: s.name,
)
def test_missing_drop_is_listwise_deletion(spec: Spec) -> None:
    # The default deletes the incomplete row: same numbers as deleting it
    # by hand before the call.
    df, _, row = _one_missing(spec)
    dropped = _quiet(lambda: spec.fit(df, spec.covs, missing="drop"))
    by_hand = _quiet(lambda: spec.fit(df.drop(index=row), spec.covs))
    _same_numbers(dropped, by_hand, rel=1e-10)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_missing_rejects_unknown_policy(spec: Spec) -> None:
    with pytest.raises(MethodIncompatibility) as ei:
        _quiet(lambda: spec.fit(spec.data(), spec.covs, missing="omit"))
    err = _refusal(ei, MethodIncompatibility)
    assert err.diagnostics["allowed"] == ["drop", "raise"]


@pytest.mark.parametrize(
    "name", ["did", "callaway_santanna", "sun_abraham", "did_imputation", "event_study"]
)
def test_missing_cohort_means_never_treated_not_missing(name: str) -> None:
    spec = BY_NAME[name]
    df = spec.data()
    coded_nan = df.assign(g=df["g"].replace(0, np.nan))
    strict = _quiet(lambda: spec.fit(coded_nan, spec.covs, missing="raise"))
    _same_numbers(strict, _quiet(lambda: spec.fit(df, spec.covs)), rel=1e-10)


def test_heckman_unselected_outcome_is_not_missing() -> None:
    spec = BY_NAME["heckman"]
    df = spec.data()
    assert df["yh"].isna().any()
    strict = _quiet(lambda: spec.fit(df, spec.covs, missing="raise"))
    _same_numbers(strict, _quiet(lambda: spec.fit(df, spec.covs)), rel=1e-12)


@pytest.mark.parametrize(
    "call",
    [
        lambda df, **k: sp.regress("y ~ x1", df, cluster="cl", **k),
        lambda df, **k: sp.feols("y ~ x1", df, vcov={"CRV1": "cl"}, **k),
        lambda df, **k: sp.logit("yb ~ x1", df, robust="cluster cl", **k),
    ],
    ids=["cluster=", "vcov-dict", "vce-string"],
)
def test_missing_raise_counts_the_cluster_column(call: Callable[..., Any]) -> None:
    df = _cross_section()
    df["cl"] = df["cl"].astype(float)
    df.loc[3, "cl"] = np.nan
    with pytest.raises(DataInsufficient) as ei:
        _quiet(lambda: call(df, missing="raise"))
    assert ei.value.diagnostics["missing_by_column"] == {"cl": 1}


def test_missing_raise_counts_the_weights_column() -> None:
    df = _cross_section()
    df["w"] = np.random.default_rng(0).uniform(0.5, 2, len(df))
    df.loc[[3, 9], "w"] = np.nan
    with pytest.raises(DataInsufficient) as ei:
        _quiet(lambda: sp.regress("y ~ x1", df, weights="w", missing="raise"))
    assert ei.value.diagnostics["missing_by_column"] == {"w": 2}
    assert ei.value.diagnostics["n_rows_missing"] == 2


def test_clean_frame_missing_raise() -> None:
    from statspai._input_validation import clean_frame

    df = pd.DataFrame({"y": [1.0, 2.0, np.nan, 4.0], "x": [0.0, 1.0, 2.0, 3.0]})
    assert len(clean_frame(df, ["y", "x"], function="demo", n_params=1)) == 3
    with pytest.raises(DataInsufficient) as ei:
        clean_frame(df, ["y", "x"], function="demo", n_params=1, missing="raise")
    assert ei.value.diagnostics["missing_by_column"] == {"y": 1}


def test_missing_is_registered_with_its_enum() -> None:
    for name in IDS:
        params = {p["name"]: p for p in sp.describe_function(name)["params"]}
        assert params["missing"]["default"] == "drop", name
        assert params["missing"]["enum"] == ["drop", "raise"], name
        # the schema offers the string or a dict of strings per role
        prop = sp.function_schema(name)["parameters"]["properties"]["missing"]
        string, mapping = prop["anyOf"]
        assert string == {"type": "string", "enum": ["drop", "raise"]}, name
        assert set(mapping["properties"]) == {"response", "predictors"}, name
        assert mapping["additionalProperties"] is False, name


# --------------------------------------------------------------------- #
#  missing= per role (rOpenSci RE2.2)
# --------------------------------------------------------------------- #

ROLES = ("response", "predictors")


def _role_column(spec: Spec, role: str) -> str:
    if role == "response":
        return spec.y
    return spec.covs[0] if spec.covs else "t"  # synth: the time column


def _one_missing_in(spec: Spec, role: str) -> tuple:
    df = spec.data()
    col = _role_column(spec, role)
    # a donor row for synth, whose treated unit must be complete
    ok = df[col].notna() & df[spec.y].notna()
    if spec.name == "synth":
        ok &= df["unit"] != 0
    row = df.index[ok][4]
    df.loc[row, col] = np.nan
    return df, col


def _other(role: str) -> str:
    return "predictors" if role == "response" else "response"


ROLE_CELLS = [
    (s, r)
    for s in SPECS
    for r in ROLES
    if not (s.name == "heckman" and r == "response")
]
ROLE_IDS = [f"{s.name}-{r}" for s, r in ROLE_CELLS]


@pytest.mark.parametrize("spec, role", ROLE_CELLS, ids=ROLE_IDS)
def test_missing_raise_for_one_role_names_the_role(spec: Spec, role: str) -> None:
    df, col = _one_missing_in(spec, role)
    rule = {role: "raise", _other(role): "drop"}
    with pytest.raises(DataInsufficient) as ei:
        _quiet(lambda: spec.fit(df, spec.covs, missing=rule))
    err = _refusal(ei, DataInsufficient)
    assert err.diagnostics["missing_policy"] == rule
    assert err.diagnostics["missing_by_role"] == {role: {col: 1}}
    assert err.diagnostics["missing_roles"] == [role]
    assert err.diagnostics["missing_by_column"] == {col: 1}
    assert err.diagnostics["n_rows_missing"] == 1


@pytest.mark.parametrize(
    "spec, role",
    [(s, r) for s, r in ROLE_CELLS if s.name != "synth"],
    ids=[i for (s, _), i in zip(ROLE_CELLS, ROLE_IDS) if s.name != "synth"],
)
def test_missing_drop_for_one_role_while_the_other_raises(
    spec: Spec, role: str
) -> None:
    # The role set to 'drop' is deleted listwise exactly as by the default,
    # and the result records which role's columns held missing values.
    df, col = _one_missing_in(spec, role)
    rule = {_other(role): "raise"}
    res = _quiet(lambda: spec.fit(df, spec.covs, missing=rule))
    default = _quiet(lambda: spec.fit(df, spec.covs))
    np.testing.assert_array_equal(
        _numbers(res).to_numpy(), _numbers(default).to_numpy()
    )
    assert res.diagnostics["missing_by_role"] == {
        "policy": {role: "drop", _other(role): "raise"},
        "missing_by_role": {role: {col: 1}},
        "n_rows_missing": 1,
    }
    assert "missing_by_role" not in default.diagnostics


def test_heckman_response_role_does_not_count_unselected_rows() -> None:
    # A missing outcome in a selection model marks an unselected row, so
    # the response role has nothing to refuse.
    spec = BY_NAME["heckman"]
    df = spec.data()
    assert df["yh"].isna().any()
    res = _quiet(lambda: spec.fit(df, spec.covs, missing={"response": "raise"}))
    _same_numbers(res, _quiet(lambda: spec.fit(df, spec.covs)), rel=1e-12)
    assert res.diagnostics["missing_by_role"]["missing_by_role"] == {"predictors": {}}


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_missing_rejects_an_unknown_role(spec: Spec) -> None:
    with pytest.raises(MethodIncompatibility) as ei:
        _quiet(lambda: spec.fit(spec.data(), spec.covs, missing={"outcome": "raise"}))
    err = _refusal(ei, MethodIncompatibility)
    assert err.diagnostics["unknown_roles"] == ["outcome"]
    assert err.diagnostics["allowed_roles"] == ["response", "predictors"]


@pytest.mark.parametrize(
    "rule, key, value",
    [
        ({}, "unknown_roles", []),
        ({"response": "omit"}, "invalid_values", {"response": "'omit'"}),
        ({"predictors": True}, "invalid_values", {"predictors": "True"}),
        ({"response": "raise", "weights": "drop"}, "unknown_roles", ["weights"]),
    ],
    ids=["empty", "bad-value", "not-a-string", "unknown-role"],
)
def test_missing_rejects_an_invalid_rule_per_role(
    rule: Dict[str, Any], key: str, value: Any
) -> None:
    df = _cross_section()
    with pytest.raises(MethodIncompatibility) as ei:
        sp.regress("y ~ x1", df, missing=rule)
    err = _refusal(ei, MethodIncompatibility)
    assert err.diagnostics[key] == value
    assert err.diagnostics["allowed"] == ["drop", "raise"]


def test_weights_and_cluster_follow_the_predictors_role() -> None:
    df = _cross_section()
    df["w"] = np.random.default_rng(0).uniform(0.5, 2, len(df))
    df["cl"] = df["cl"].astype(float)
    df.loc[[3, 9], "w"] = np.nan
    df.loc[5, "cl"] = np.nan

    def fit(rule: Any) -> Any:
        return _quiet(
            lambda: sp.regress("y ~ x1", df, weights="w", cluster="cl", missing=rule)
        )

    with pytest.raises(DataInsufficient) as ei:
        fit({"predictors": "raise"})
    assert ei.value.diagnostics["missing_by_role"] == {"predictors": {"w": 2, "cl": 1}}
    assert ei.value.diagnostics["n_rows_missing"] == 3
    # with the predictors left to 'drop', the incomplete cluster row goes
    df["w"] = 1.0
    res = fit({"response": "raise"})
    assert res.diagnostics["missing_by_role"]["missing_by_role"] == {
        "predictors": {"cl": 1}
    }
    _same_numbers(res, fit("drop"), rel=0.0)


def test_a_weights_array_follows_the_predictors_role() -> None:
    df = _cross_section()
    w = np.random.default_rng(0).uniform(0.5, 2, len(df))
    w[[4, 7]] = np.nan
    with pytest.raises(DataInsufficient) as ei:
        _quiet(
            lambda: sp.feols("y ~ x1", df, weights=w, missing={"predictors": "raise"})
        )
    assert ei.value.diagnostics["missing_by_role"] == {"predictors": {"weights": 2}}


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_both_roles_raise_on_complete_data_changes_nothing(spec: Spec) -> None:
    df = spec.data()
    rule = {"response": "raise", "predictors": "raise"}
    strict = _quiet(lambda: spec.fit(df, spec.covs, missing=rule))
    default = _quiet(lambda: spec.fit(df, spec.covs))
    np.testing.assert_array_equal(
        _numbers(strict).to_numpy(), _numbers(default).to_numpy()
    )


# --------------------------------------------------------------------- #
#  Known truths: exact fits and ordered outcomes
# --------------------------------------------------------------------- #


def _exact_cross_section() -> pd.DataFrame:
    r = np.random.default_rng(5)
    n = 150
    x1, x2, z = r.normal(size=n), r.normal(size=n), r.normal(size=n)
    d = z + 0.5 * x1 + r.normal(size=n)
    treat = (r.uniform(size=n) < 1 / (1 + np.exp(-x1))).astype(float)
    dc = x1 + r.normal(size=n)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "z": z,
            "d": d,
            "y": 1 + 2 * x1 - 0.5 * x2,
            "yiv": 1 + 0.5 * x1 + 1.5 * d,
            "yc": np.exp(0.5 + 0.3 * x1),
            "treat": treat,
            "yt": 1 + 2 * treat,
            "ytx": 1 + 2 * treat + x1 - 0.5 * x2,
            "dc": dc,
            "yd": 2 * dc + x1,
            "yb": (x1 > 0).astype(float),
        }
    )


EXACT_LINEAR = {
    "regress": lambda df: sp.regress("y ~ x1 + x2", df),
    "glm": lambda df: sp.glm("y ~ x1 + x2", df),
    "feols": lambda df: sp.feols("y ~ x1 + x2", df),
}


@pytest.mark.parametrize("name", list(EXACT_LINEAR))
def test_exact_linear_fit_recovers_coefficients(name: str) -> None:
    res = _quiet(lambda: EXACT_LINEAR[name](_exact_cross_section()))
    np.testing.assert_allclose(
        pd.Series(res.params, dtype=float)[["Intercept", "x1", "x2"]],
        [1.0, 2.0, -0.5],
        atol=1e-10,
    )
    assert float(np.max(np.asarray(res.std_errors, dtype=float))) <= 1e-10 * 2.0


@pytest.mark.parametrize("name", ["ivreg", "iv"])
def test_exact_iv_fit_recovers_coefficients(name: str) -> None:
    df = _exact_cross_section()
    res = _quiet(lambda: getattr(sp, name)("yiv ~ (d ~ z) + x1", df))
    p = pd.Series(res.params, dtype=float)
    np.testing.assert_allclose(p[["Intercept", "x1", "d"]], [1.0, 0.5, 1.5], atol=1e-10)
    assert float(np.max(np.asarray(res.std_errors, dtype=float))) <= 1e-10 * 1.5


def test_exact_quantile_fit_recovers_the_slope() -> None:
    res = _quiet(lambda: sp.qreg(_exact_cross_section(), formula="y ~ x1 + x2"))
    assert res.estimate == pytest.approx(2.0, abs=1e-10)


@pytest.mark.parametrize("name", ["poisson", "fepois"])
def test_deterministic_poisson_mean_recovers_coefficients(name: str) -> None:
    # y = exp(0.5 + 0.3 x1) exactly solves the Poisson score equations.
    res = _quiet(lambda: getattr(sp, name)("yc ~ x1", _exact_cross_section()))
    p = pd.Series(res.params, dtype=float).to_numpy()
    np.testing.assert_allclose(p, [0.5, 0.3], atol=1e-8)


@pytest.mark.parametrize("name", ["logit", "probit", "feglm"])
def test_complete_separation_is_announced(name: str) -> None:
    df = _exact_cross_section()
    kw = {"family": "logit"} if name == "feglm" else {}
    with pytest.warns(sp.ConvergenceWarning, match="separation"):
        getattr(sp, name)("yb ~ x1 + x2", df, **kw)


@pytest.mark.parametrize(
    "name, call, truth",
    [
        (
            "ipw",
            lambda df: sp.ipw(
                df, y="yt", treat="treat", covariates=["x1", "x2"], se_method="sandwich"
            ),
            2.0,
        ),
        (
            "aipw",
            lambda df: sp.aipw(df, y="ytx", treat="treat", covariates=["x1", "x2"]),
            2.0,
        ),
        (
            "match",
            lambda df: sp.match(df, y="yt", treat="treat", covariates=["x1", "x2"]),
            2.0,
        ),
        (
            "dml",
            lambda df: sp.dml(df, y="yd", d="dc", X=["x1", "x2"], **_lin_learners()),
            2.0,
        ),
    ],
)
def test_exact_treatment_effect_is_recovered(
    name: str, call: Any, truth: float
) -> None:
    res = _quiet(lambda: call(_exact_cross_section()))
    assert res.estimate == pytest.approx(truth, abs=1e-10)
    assert res.se <= 1e-10 * truth


def _exact_panel() -> pd.DataFrame:
    rows = []
    for i in range(30):
        g = (0, 3, 5)[i % 3]
        for t in range(1, 7):
            treated = int(g > 0 and t >= g)
            rows.append((i, t, g, 0.5 * i + 0.3 * t + 2.0 * treated))
    df = pd.DataFrame(rows, columns=["id", "t", "g", "y"])
    df["x"] = np.random.default_rng(1).normal(size=len(df))
    df["yx"] = 0.5 * df["id"] + 1.5 * df["x"]
    return df


@pytest.mark.parametrize(
    "call",
    [
        lambda df: sp.did(df, y="y", treat="g", time="t", id="id"),
        lambda df: sp.callaway_santanna(df, y="y", g="g", t="t", i="id"),
        lambda df: sp.sun_abraham(df, y="y", g="g", t="t", i="id"),
        lambda df: sp.did_imputation(df, y="y", group="id", time="t", first_treat="g"),
        lambda df: sp.event_study(
            df, y="y", treat_time="g", time="t", unit="id", window=(-2, 1)
        ),
    ],
    ids=["did", "callaway_santanna", "sun_abraham", "did_imputation", "event_study"],
)
def test_exact_staggered_effect_is_recovered(call: Any) -> None:
    # Unit and period effects plus a constant effect of 2: every DiD
    # estimand is 2 and every residual zero. The iterative ones (SA, BJS)
    # stop at their demeaning tolerance.
    res = _quiet(lambda: call(_exact_panel()))
    assert res.estimate == pytest.approx(2.0, abs=1e-8)
    assert res.se <= 1e-8 * 2.0


def test_exact_within_fit_recovers_the_slope() -> None:
    res = _quiet(lambda: sp.panel(_exact_panel(), "yx ~ x", entity="id", time="t"))
    assert float(res.params["x"]) == pytest.approx(1.5, abs=1e-10)


def test_exact_discontinuity_is_recovered() -> None:
    xr = np.random.default_rng(2).uniform(-1, 1, 400)
    df = pd.DataFrame({"xr": xr, "y": 1 + 0.5 * xr + 1.0 * (xr >= 0)})
    res = _quiet(lambda: sp.rdrobust(df, y="y", x="xr", vce="hc1"))
    assert res.estimate == pytest.approx(1.0, abs=1e-10)
    assert res.se <= 1e-8


def test_exact_synthetic_control_is_recovered() -> None:
    r = np.random.default_rng(3)
    periods, donors = 18, 7
    paths = r.normal(size=(donors, periods)).cumsum(axis=1)
    treated = 0.5 * paths[0] + 0.3 * paths[1] + 0.2 * paths[2]
    rows = [(j + 1, t, paths[j, t]) for j in range(donors) for t in range(periods)]
    rows += [(0, t, treated[t] + (2.0 if t >= 13 else 0.0)) for t in range(periods)]
    df = pd.DataFrame(rows, columns=["unit", "t", "y"])
    res = _quiet(
        lambda: sp.synth(
            df,
            outcome="y",
            unit="unit",
            time="t",
            treated_unit=0,
            treatment_time=13,
            placebo=False,
        )
    )
    assert res.estimate == pytest.approx(2.0, abs=1e-6)


@pytest.mark.parametrize("name", ["ologit", "oprobit"])
def test_unordered_text_outcome_needs_a_declared_order(name: str) -> None:
    r = np.random.default_rng(4)
    x = r.normal(size=150)
    codes = np.digitize(x + r.logistic(size=150), [-0.7, 0.7])
    labels = np.array(["low", "mid", "high"])[codes]
    text = pd.DataFrame({"y": labels, "x": x})
    with pytest.raises(MethodIncompatibility) as ei:
        getattr(sp, name)("y ~ x", text)
    _refusal(ei, MethodIncompatibility)
    ordered = text.assign(
        y=pd.Categorical(labels, categories=["low", "mid", "high"], ordered=True)
    )
    numeric = text.assign(y=codes)
    # The declared order is the numeric coding: same likelihood, same slope.
    a = _quiet(lambda: getattr(sp, name)("y ~ x", ordered))
    b = _quiet(lambda: getattr(sp, name)("y ~ x", numeric))
    assert float(a.params["x"]) == pytest.approx(float(b.params["x"]), rel=1e-8)
