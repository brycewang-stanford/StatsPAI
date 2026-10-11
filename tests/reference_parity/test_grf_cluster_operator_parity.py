"""Reference parity: clustered GRF inference operators vs ``grf`` / ``sandwich``.

The forest itself cannot be pinned across implementations (independent
random streams), but everything StatsPAI computes *from* a fitted forest --
the calibration test, AIPW averages, the best linear projection and their
cluster-robust covariances -- is a closed-form operator on
``(Y, W, Y_hat, W_hat, tau_oob, clusters, observation weights)``.

The fixture injects StatsPAI's out-of-bag predictions and nuisances into a
``grf`` forest object and records grf 2.6.1's own outputs (see
``_fixtures/_generate_grf_cluster_operator.R``).  Feeding the same vectors to
StatsPAI's operators must reproduce them to the floating-point floor, with
and without ``equalize.cluster.weights``.  The ``sandwich::vcovCL`` rows pin
:func:`cluster_robust_vcov` for HC0-HC3 on a weighted regression, clustered
and unclustered.

Tolerance: ``rtol = 1e-9`` -- linear algebra on identical inputs; any real
formula difference (degrees-of-freedom factor, weight normalisation,
cluster adjustment) shows up at 1e-3 or larger.

Regenerate with::

    PYTHONPATH=src python tests/reference_parity/_fixtures/_generate_grf_cluster_operator_data.py
    Rscript tests/reference_parity/_fixtures/_generate_grf_cluster_operator.R
"""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.forest import _grf_inference as gi
from statspai.forest._grf_fit import observation_weights

_DIR = pathlib.Path(__file__).parent / "_fixtures"
_CSV = _DIR / "grf_cluster_operator_data.csv"
_JSON = _DIR / "grf_cluster_operator_R.json"

pytestmark = pytest.mark.skipif(
    not (_CSV.exists() and _JSON.exists()),
    reason="clustered grf operator fixture is not materialized",
)

RTOL = 1e-9


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_CSV)


@pytest.fixture(scope="module")
def ref():
    return json.loads(_JSON.read_text(encoding="utf-8"))


def _forest(df: pd.DataFrame, equalize: bool) -> SimpleNamespace:
    _, codes = np.unique(df["cluster"].to_numpy(), return_inverse=True)
    n = len(df)
    return SimpleNamespace(
        _engine=object(),
        fe=None,
        _X_original=df[["x1", "x2", "x3"]].to_numpy(),
        _Y_original=df["Y"].to_numpy(),
        _T_original=df["W"].to_numpy(),
        _m_insample=df["Y_hat"].to_numpy(),
        _e_insample=df["W_hat"].to_numpy(),
        _oob_tau=df["tau_oob"].to_numpy(),
        _clusters=codes.astype(np.int64),
        _observation_weight=observation_weights(codes.astype(np.int64), equalize, n),
    )


@pytest.mark.parametrize(
    "key,equalize", [("clusters", False), ("clusters_equalized", True)]
)
def test_calibration_matches_grf(data, ref, key, equalize):
    forest = _forest(data, equalize)
    table = gi.calibration_blp(forest)
    r = ref[key]["calibration"]
    np.testing.assert_allclose(table["coef"].to_numpy(), r["coef"], rtol=RTOL)
    np.testing.assert_allclose(table["se"].to_numpy(), r["se"], rtol=RTOL)
    np.testing.assert_allclose(table["t_vs_zero"].to_numpy(), r["t"], rtol=RTOL)
    # grf's p-values come from a t distribution with n - 2 df; ours use the
    # normal.  They agree to the reference's own resolution only in the
    # tails, so pin the statistic and check the p-values loosely.
    np.testing.assert_allclose(
        table["p_one_sided"].to_numpy(), r["p_one_sided"], atol=1e-3
    )


@pytest.mark.parametrize(
    "key,equalize", [("clusters", False), ("clusters_equalized", True)]
)
def test_grf_calibration_matches_grf(data, ref, key, equalize):
    """The public operator on the same vectors: ``clusters=`` for the
    covariance, ``1 / cluster size`` for equalized cluster weights."""
    size = data["cluster"].map(data["cluster"].value_counts()).to_numpy()
    table = sp.grf_calibration(
        Y=data["Y"],
        W=data["W"],
        Y_hat=data["Y_hat"],
        W_hat=data["W_hat"],
        tau_hat=data["tau_oob"],
        clusters=data["cluster"],
        weights=1.0 / size if equalize else None,
    )
    r = ref[key]["calibration"]
    np.testing.assert_allclose(table["coef"].to_numpy(), r["coef"], rtol=RTOL)
    np.testing.assert_allclose(table["se"].to_numpy(), r["se"], rtol=RTOL)
    np.testing.assert_allclose(table["t"].to_numpy(), r["t"], rtol=RTOL)


@pytest.mark.parametrize(
    "key,equalize", [("clusters", False), ("clusters_equalized", True)]
)
def test_average_effects_match_grf(data, ref, key, equalize):
    forest = _forest(data, equalize)
    for row in ref[key]["ate"]:
        got = gi.average_effect(forest, row["target"], alpha=0.05, clip=0.0)
        np.testing.assert_allclose(got["estimate"], row["estimate"], rtol=RTOL)
        np.testing.assert_allclose(got["se"], row["se"], rtol=RTOL)


@pytest.mark.parametrize(
    "key,equalize", [("clusters", False), ("clusters_equalized", True)]
)
def test_best_linear_projection_matches_grf(data, ref, key, equalize):
    forest = _forest(data, equalize)
    names = ["Intercept", "x1", "x2", "x3"]
    table = gi.best_linear_projection(
        forest, forest._X_original, names, alpha=0.05, clip=0.0
    )
    r = ref[key]["blp"]
    np.testing.assert_allclose(table["coef"].to_numpy(), r["coef"], rtol=RTOL)
    np.testing.assert_allclose(table["se"].to_numpy(), r["se"], rtol=RTOL)


@pytest.mark.parametrize("vtype", ["HC0", "HC1", "HC2", "HC3"])
@pytest.mark.parametrize("clustered", [True, False])
def test_cluster_robust_vcov_matches_sandwich(data, ref, vtype, clustered):
    n = len(data)
    design = np.column_stack(
        [np.ones(n), data["W"].to_numpy(), data["x1"].to_numpy(), data["x2"].to_numpy()]
    )
    w = data["vcov_weight"].to_numpy()
    y = data["Y"].to_numpy()
    beta = np.linalg.solve(design.T @ (design * w[:, None]), design.T @ (w * y))
    np.testing.assert_allclose(beta, ref["vcovCL_coef"], rtol=1e-10)
    V = gi.cluster_robust_vcov(
        design,
        y - design @ beta,
        weights=w,
        clusters=data["cluster"].to_numpy() if clustered else None,
        vcov_type=vtype,
    )
    expected = np.asarray(
        ref["vcovCL"][f"{vtype}_{'cluster' if clustered else 'rows'}"]
    ).reshape(4, 4, order="F")
    np.testing.assert_allclose(V, expected, rtol=RTOL)
