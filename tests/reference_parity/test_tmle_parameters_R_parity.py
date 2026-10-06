"""Cross-language parity: every parameter ``sp.tmle`` reports, against R ``tmle``.

Fixture: ``_generate_tmle_parameters_R.R`` (``tmle::tmle`` 2.1.1 with the
initial fits supplied, ``gbound = 0.025``). Both sides read
``_fixtures/tmle_design_data.csv`` and the same ``Q`` / ``g1W`` from
``_fixtures/tmle_parameters_nuisance.csv``, so only the targeting step,
the plug-in and the influence-function variance are compared.

What agrees and to what
-----------------------
* Treatment-specific means, the additive effect, the risk ratio and the
  odds ratio: estimate, variance (on the log scale for the ratios),
  interval and p-value to ``1e-8`` (observed ``1e-11``; the residual is the
  Newton tolerance of the two-parameter fluctuation), with and without
  observation weights and clusters. One exception follows.
* Weighted odds ratio, variance only. An influence function has mean zero.
  ``tmle``'s curve for the log odds ratio is not centred: it carries the
  extra term ``w * (1 / (1 - EY1) - 1 / (1 - EY0))``, a constant when
  ``w = 1`` (so the unweighted variance is unaffected) but not when the
  weights vary. Adding that term to ours reproduces ``tmle``'s variance to
  ``1e-9``; we report the centred one.
* Effects among the treated and the controls. Not a strict comparison:
  ``tmle`` reports a different, equally valid TMLE (it also updates ``g``,
  on the rows with ``g >= min(g | A = 1)``, along a small-step path that
  stops on the likelihood; ``test_r2_teffects_parity.py`` ports and pins
  that mechanism for the ATT). Here both are recorded side by side: ours
  uses every row and solves the efficient-influence-function equation
  exactly, R's curve has fewer rows and a mean of ``1e-4`` to ``4e-3``, and
  the estimates are within 0.05 standard errors. The same holds for the
  ATC, which is new and has no other reference.
"""

from __future__ import annotations

import json
import pathlib
import warnings
from functools import lru_cache
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
# Shared initial fits: what is left is Newton tolerance on the fluctuation.
RTOL = 1e-8
COV = ["x1", "x2", "x3"]
CASES: Dict[str, Dict[str, Any]] = {
    "plain": {},
    "cluster": {"cluster": "g"},
    "weights": {"weights": "w"},
    "weights_cluster": {"weights": "w", "cluster": "g"},
}
OUTCOME = {"gaussian": "y", "binomial": "yb"}


@lru_cache(maxsize=None)
def _ref() -> Dict[str, Any]:
    path = _FIX / "tmle_parameters_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_tmle_parameters_R.R first")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "tmle_design_data.csv")


@lru_cache(maxsize=None)
def _nuisance() -> pd.DataFrame:
    return pd.read_csv(_FIX / "tmle_parameters_nuisance.csv")


def _fit(family: str, estimand: str, **extra: Any):
    y = OUTCOME[family]
    nu = _nuisance()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.tmle(
            _data(),
            y=y,
            treat="d",
            covariates=COV,
            Q=nu[[f"Q0_{y}", f"Q1_{y}"]].to_numpy(),
            g1W=nu["g"].to_numpy(),
            estimand=estimand,
            q_bound=5e-4,
            **extra,
        )


def _centred_or_differs(case: str, name: str) -> bool:
    return name == "OR" and "weights" in case


@pytest.mark.parametrize("family", ["gaussian", "binomial"])
@pytest.mark.parametrize("case", list(CASES))
def test_arm_means_and_contrasts_match_r(family: str, case: str) -> None:
    ref = _ref()[f"{family}_{case}"]
    table = _fit(family, "EY1", **CASES[case]).detail.set_index("parameter")
    names = [k for k in ref if k not in ("ATT", "ATC")]
    assert names == list(table.index)
    for name in names:
        r, row = ref[name], table.loc[name]
        np.testing.assert_allclose(row["estimate"], r["psi"], rtol=RTOL)
        if _centred_or_differs(case, name):
            continue
        if "var_log" in r:
            np.testing.assert_allclose(row["se_log"] ** 2, r["var_log"], rtol=RTOL)
        else:
            np.testing.assert_allclose(row["se"] ** 2, r["var"], rtol=RTOL)
        np.testing.assert_allclose(
            [row["ci_lower"], row["ci_upper"]], r["ci"], rtol=RTOL
        )
        np.testing.assert_allclose(row["pvalue"], r["pvalue"], rtol=1e-6)


@pytest.mark.parametrize("estimand", ["EY1", "EY0", "RR", "OR"])
def test_headline_is_the_table_row(estimand: str) -> None:
    res = _fit("binomial", estimand)
    row = res.detail.set_index("parameter").loc[estimand]
    assert res.estimand == estimand
    assert res.estimate == row["estimate"]
    assert res.se == row["se"]
    assert res.ci == (row["ci_lower"], row["ci_upper"])
    assert res.model_info["fluctuation"] == "per_arm"


@pytest.mark.parametrize("case", ["weights", "weights_cluster"])
def test_weighted_odds_ratio_variance_is_the_centred_one(case: str) -> None:
    ref = _ref()[f"binomial_{case}"]["OR"]
    res = _fit("binomial", "OR", **CASES[case])
    table = res.detail.set_index("parameter")
    ours = table.loc["OR", "se_log"] ** 2
    assert abs(ours / ref["var_log"] - 1) > 1e-3

    # Rebuild tmle's number from ours: add the term that centring removes.
    d = _data()
    n = len(d)
    w = d["w"].to_numpy()
    w = w * n / w.sum()
    ey1, ey0 = table.loc["EY1", "estimate"], table.loc["EY0", "estimate"]
    ic = res.model_info["influence_function"]
    assert res.model_info["influence_function_scale"] == "log"
    uncentred = ic + w * (1 / (1 - ey1) - 1 / (1 - ey0))
    assert abs(np.mean(ic)) < 1e-10 < abs(np.mean(uncentred))
    if case == "weights":
        rebuilt = np.var(uncentred, ddof=1) / n
    else:
        S = np.bincount(pd.factorize(d["g"])[0], weights=uncentred)
        G = len(S)
        rebuilt = G / (G - 1) * np.sum((S - S.mean()) ** 2) / n**2
    np.testing.assert_allclose(rebuilt, ref["var_log"], rtol=1e-9)


@pytest.mark.parametrize("family", ["gaussian", "binomial"])
@pytest.mark.parametrize("estimand", ["ATT", "ATC"])
def test_effect_on_the_treated_solves_what_r_leaves_open(
    family: str, estimand: str
) -> None:
    ref = _ref()[f"{family}_plain"][estimand]
    res = _fit(family, estimand)
    ic = res.model_info["influence_function"]
    assert len(ic) == len(_data())
    # Ours solves the efficient-influence-function equation; tmle's curve,
    # on the rows it keeps, has a mean several orders of magnitude larger.
    assert abs(ic.mean()) < 1e-8
    assert ref["n_ic"] < len(ic)
    assert abs(ref["ic_mean"]) > 1e4 * abs(ic.mean())
    # Different TMLEs of the same parameter: a small fraction of a standard
    # error apart.
    assert abs(res.estimate - ref["psi"]) < 0.05 * res.se
    np.testing.assert_allclose(res.se**2, ref["var"], rtol=0.03)
