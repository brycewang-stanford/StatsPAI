"""Cross-language parity: ``sp.tobit`` vce / cluster / weights against Stata.

Fixture: ``_fixtures/_generate_ldv_design_stata.do`` (Stata 18 ``tobit``,
left-censored at 0 and two-sided at [0, 1.5], under ``vce(oim)``,
``vce(robust)``, ``vce(cluster g)``, ``[pw=w]`` and ``[pw=w]`` with
clusters). Both sides read the same CSV bytes.

Stata runs with ``tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-12)``.
At its default tolerances Stata's ``ml`` stops ~1e-6 (relative) short of the
optimum; ``sp.tobit`` Newton-polishes on the analytic score and reaches a
log-likelihood ~3e-11 higher, and with the tight options both land on the
same point (coefficients agree to ~1e-15, standard errors to ~1e-10).
"""

from __future__ import annotations

import json
import pathlib
import warnings
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-6  # CLAUDE.md §5.1 default budget
CASES: Dict[str, Tuple[str, Optional[float], Dict[str, Any]]] = {
    "left_oim": ("y", None, {}),
    "left_robust": ("y", None, {"vce": "robust"}),
    "left_cluster": ("y", None, {"cluster": "g"}),
    "left_pw": ("y", None, {"weights": "w"}),
    "left_pw_cluster": ("y", None, {"weights": "w", "cluster": "g"}),
    "two_oim": ("y2", 1.5, {}),
    "two_pw_cluster": ("y2", 1.5, {"weights": "w", "cluster": "g"}),
}
TERMS = (("x1", "x1"), ("x2", "x2"), ("const", "_cons"))


@lru_cache(maxsize=None)
def _ref() -> Dict[str, Any]:
    path = _FIX / "ldv_design_stata.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_ldv_design_stata.do first")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _data() -> pd.DataFrame:
    path = _FIX / "ldv_design_data.csv"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_ldv_design_stata.do first")
    return pd.read_csv(path)


def _fit(case: str):
    dv, ul, kw = CASES[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.tobit(_data(), dv, ["x1", "x2"], ll=0, ul=ul, **kw)


@pytest.mark.parametrize("case", list(CASES))
def test_tobit_matches_stata(case: str) -> None:
    ref = _ref()[case]
    res = _fit(case)
    d = res.detail.set_index("variable")
    for ours, theirs in TERMS:
        np.testing.assert_allclose(
            d.loc[ours, "coefficient"], ref[f"b_{theirs}"], rtol=RTOL
        )
        np.testing.assert_allclose(d.loc[ours, "se"], ref[f"se_{theirs}"], rtol=RTOL)
    np.testing.assert_allclose(d.loc["sigma", "coefficient"], ref["sigma"], rtol=RTOL)
    assert res.n_obs == int(ref["N"])


def test_unweighted_loglik_matches_stata() -> None:
    for case in ("left_oim", "two_oim"):
        np.testing.assert_allclose(
            _fit(case).model_info["log_likelihood"], _ref()[case]["ll"], rtol=1e-10
        )


def test_pweights_imply_robust_and_move_the_estimate() -> None:
    res = _fit("left_pw")
    assert res.model_info["vce"] == "robust"
    # Far outside the parity budget: an ignored weight could not pass.
    assert abs(res.estimate / _fit("left_oim").estimate - 1) > 1e-3


def test_vce_spellings_agree() -> None:
    df = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.tobit(df, "y", ["x1", "x2"], vce="cluster g")
        b = sp.tobit(df, "y", ["x1", "x2"], cluster="g")
        c = sp.tobit(df, "y", ["x1", "x2"], robust="cluster", cluster="g")
    np.testing.assert_allclose(a.detail["se"], b.detail["se"], rtol=1e-12)
    np.testing.assert_allclose(a.detail["se"], c.detail["se"], rtol=1e-12)
    assert a.model_info["n_clusters"] == 48


def test_unsupported_requests_fail_loudly() -> None:
    df = _data()
    with pytest.raises(MethodIncompatibility):
        sp.tobit(df, "y", ["x1", "x2"], vce="hc3")
    with pytest.raises(MethodIncompatibility, match="strictly positive"):
        sp.tobit(df.assign(w0=0.0), "y", ["x1", "x2"], weights="w0")
    with pytest.raises(MethodIncompatibility, match="not found"):
        sp.tobit(df, "y", ["x1", "x2"], cluster="nope")


# --------------------------------------------------------------------------
# heckman, method="ml" (Stata's default heckman) with vce / cluster / weights
# --------------------------------------------------------------------------

HECKMAN_CASES: Dict[str, Dict[str, Any]] = {
    "heckman_oim": {},
    "heckman_robust": {"vce": "robust"},
    "heckman_cluster": {"cluster": "g"},
    "heckman_pw": {"weights": "w"},
    "heckman_pw_cluster": {"weights": "w", "cluster": "g"},
}
HECKMAN_ROWS = (
    ("x1", "b_x1", "se_x1"),
    ("x2", "b_x2", "se_x2"),
    ("const", "b__cons", "se__cons"),
    ("select:x1", "bs_x1", "ses_x1"),
    ("select:x2", "bs_x2", "ses_x2"),
    ("select:z3", "bs_z3", "ses_z3"),
    ("select:const", "bs__cons", "ses__cons"),
    ("athrho", "athrho", "se_athrho"),
    ("lnsigma", "lnsigma", "se_lnsigma"),
    ("lambda", "lambda", "se_lambda"),
)


def _heckman(case: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.heckman(
            _data(),
            y="yh",
            x=["x1", "x2"],
            select="s",
            z=["x1", "x2", "z3"],
            method="ml",
            **HECKMAN_CASES[case],
        )


@pytest.mark.parametrize("case", list(HECKMAN_CASES))
def test_heckman_ml_matches_stata(case: str) -> None:
    ref = _ref()[case]
    res = _heckman(case)
    d = res.detail.set_index("variable")
    for ours, b, s in HECKMAN_ROWS:
        np.testing.assert_allclose(d.loc[ours, "coefficient"], ref[b], rtol=RTOL)
        np.testing.assert_allclose(d.loc[ours, "se"], ref[s], rtol=RTOL)
    np.testing.assert_allclose(res.model_info["rho"], ref["rho"], rtol=RTOL)
    np.testing.assert_allclose(res.model_info["sigma"], ref["sigma"], rtol=RTOL)
    assert res.n_obs == int(ref["N"])
    assert res.model_info["n_selected"] == int(ref["N_selected"])


def test_heckman_ml_loglik_matches_stata() -> None:
    np.testing.assert_allclose(
        _heckman("heckman_oim").model_info["log_likelihood"],
        _ref()["heckman_oim"]["ll"],
        rtol=1e-12,
    )


def test_heckman_twostep_default_refuses_robust_and_weights() -> None:
    kw = dict(y="yh", x=["x1", "x2"], select="s", z=["x1", "x2", "z3"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        two = sp.heckman(_data(), **kw)
    assert two.model_info["method"] == "Heckman Two-Step"
    for bad in ({"vce": "robust"}, {"cluster": "g"}, {"weights": "w"}):
        with pytest.raises(MethodIncompatibility, match="method='ml'"):
            sp.heckman(_data(), **kw, **bad)
    with pytest.raises(MethodIncompatibility, match="twostep' or 'ml'"):
        sp.heckman(_data(), **kw, method="mle")


# --------------------------------------------------------------------------
# qreg: Stata vce(iid) (the default) / vce(robust), and R quantreg se="nid"
# --------------------------------------------------------------------------

QREG_TERMS = (("x1", "x1"), ("x2", "x2"), ("const", "_cons"))


def _qreg(q: int, vce: Optional[str]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.qreg(_data(), y="yq", x=["x1", "x2"], quantile=q / 100, vce=vce)


@pytest.mark.parametrize("q", [25, 50, 75])
@pytest.mark.parametrize("vce", ["iid", "robust"])
def test_qreg_matches_stata(q: int, vce: str) -> None:
    ref = _ref()[f"qreg_{q}_{vce}"]
    res = _qreg(q, vce)
    d = res.detail.set_index("variable")
    for ours, theirs in QREG_TERMS:
        np.testing.assert_allclose(
            d.loc[ours, "coefficient"], ref[f"b_{theirs}"], rtol=RTOL
        )
        np.testing.assert_allclose(d.loc[ours, "se"], ref[f"se_{theirs}"], rtol=RTOL)
        # Stata refers the statistic to t(e(df_r)) = t(N - k), not z.
        np.testing.assert_allclose(
            d.loc[ours, "pvalue"], ref[f"p_{theirs}"], rtol=1e-8, atol=1e-300
        )
    np.testing.assert_allclose(res.model_info["bandwidth"], ref["bwidth"], rtol=1e-12)
    assert res.model_info["df_inference"] == int(ref["df_r"])
    np.testing.assert_allclose(res.ci, (ref["ll_x1"], ref["ul_x1"]), rtol=RTOL)


@pytest.mark.parametrize("q", [25, 50, 75])
def test_qreg_nid_matches_quantreg(q: int) -> None:
    path = _FIX / "qreg_nid_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_qreg_nid_R.R first")
    ref = json.loads(path.read_text(encoding="utf-8"))[f"nid_{q}"]
    res = _qreg(q, "nid")
    np.testing.assert_allclose(res.detail["coefficient"], ref["b"], rtol=RTOL)
    np.testing.assert_allclose(res.detail["se"], ref["se"], rtol=RTOL)
    # summary.rq refers the statistic to t(rdf), rdf = N - k.
    assert res.model_info["df_inference"] == ref["rdf"]
    np.testing.assert_allclose(res.detail["pvalue"], ref["p"], rtol=1e-8, atol=1e-15)


def test_qreg_default_is_stata_iid() -> None:
    a, b = _qreg(50, None), _qreg(50, "iid")
    np.testing.assert_array_equal(a.detail["se"], b.detail["se"])
    assert a.model_info["vce"] == "iid"


def test_qreg_legacy_powell_and_refusals() -> None:
    old = _qreg(50, "powell")
    ref = _ref()["qreg_50_iid"]
    # The pre-1.32 default is kept reproducible, and was several percent off.
    assert (
        abs(old.detail.set_index("variable").loc["x1", "se"] / ref["se_x1"] - 1) > 0.01
    )
    # Cluster-robust SEs exist since 1.32 (qreg2 parity lives in
    # test_qreg_cluster_stata_parity.py) but need a cluster column.
    with pytest.raises(MethodIncompatibility, match="needs a cluster column"):
        _qreg(50, "cluster")
