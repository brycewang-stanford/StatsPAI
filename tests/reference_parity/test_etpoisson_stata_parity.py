"""Cross-language parity: ``sp.etpoisson`` against Stata 18 ``etpoisson``.

Fixture: ``_fixtures/_generate_etpoisson_stata.do``. The data are generated
in Stata and exported at ``%21.16e``; ``ml``'s stopping rule is tightened in
the do-file.

Every block is held to 1e-6. Observed: coefficients 3e-15, standard errors
8e-11, log-likelihood to all printed digits, with 24 and with 64
Gauss-Hermite points and under ``vce(oim)``, ``vce(robust)`` and
``vce(cluster)``. The average treatment effect and the potential-outcome
means agree with ``margins r.d`` / ``margins d`` to 1e-9, the standard
error of the ATE included.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
BLOCKS = {
    "default": {},
    "q64": dict(intpoints=64),
    "q64_robust": dict(intpoints=64, vce="robust"),
    "q64_cluster": dict(intpoints=64, cluster="clust"),
}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "etpoisson_data.csv")


@pytest.fixture(scope="module")
def stata() -> dict:
    return json.loads((_FIX / "etpoisson_stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fits(data) -> dict:
    return {
        k: sp.etpoisson(data, y="y", x=["x1", "x2"], treat="d", z=["z1", "x1"], **kw)
        for k, kw in BLOCKS.items()
    }


@pytest.mark.parametrize("block", sorted(BLOCKS))
def test_matches_stata(block, fits, stata):
    res, ref = fits[block], stata[block]
    # Stata lists the base level of the treatment (0b.d) as a zero row.
    keep = [i for i, name in enumerate(ref["names"]) if name != "y:0b.d"]
    names = [ref["names"][i].replace("1.d", "d") for i in keep]
    assert list(res.params.index) == names
    np.testing.assert_allclose(res.params.values, np.array(ref["b"])[keep], rtol=1e-6)
    np.testing.assert_allclose(
        res.std_errors.values, np.array(ref["se"])[keep], rtol=1e-6
    )
    info = res.model_info
    assert info["ll"] == pytest.approx(ref["ll"], abs=1e-8)
    assert info["independence_chi2"] == pytest.approx(ref["chi2_c"], rel=1e-6)
    assert info["intpoints"] == int(ref["n_quad"])
    assert info["converged"]


def test_average_treatment_effect_matches_margins(fits, stata):
    info = fits["q64"].model_info
    assert info["ate"] == pytest.approx(stata["ate_q64"]["b"], rel=1e-9)
    assert info["ate_se"] == pytest.approx(stata["ate_q64"]["se"], rel=1e-6)
    assert info["pomean_0"] == pytest.approx(stata["pomeans_q64"]["b0"], rel=1e-9)
    assert info["pomean_1"] == pytest.approx(stata["pomeans_q64"]["b1"], rel=1e-9)


def test_quadrature_points_matter_in_the_fifth_digit(fits):
    a, b = fits["default"].params.values, fits["q64"].params.values
    gap = np.max(np.abs(a - b) / np.abs(b))
    assert 1e-7 < gap < 1e-3


def test_recovers_the_truth_where_poisson_does_not(data, fits):
    """The do-file's DGP: treatment effect 0.5 on the log mean, rho = 0.6."""
    res = fits["q64"]
    name = "y:d"
    assert abs(res.params[name] - 0.5) < 2.5 * res.std_errors[name]
    assert abs(res.model_info["rho"] - 0.6) < 0.2
    naive = sp.poisson(data=data, y="y", x=["x1", "x2", "d"], robust="robust")
    assert (naive.params["d"] - 0.5) / naive.std_errors["d"] > 5


def test_robust_and_cluster_change_only_the_standard_errors(fits):
    for other in ("q64_robust", "q64_cluster"):
        np.testing.assert_allclose(fits[other].params.values, fits["q64"].params.values)
        assert not np.allclose(
            fits[other].std_errors.values, fits["q64"].std_errors.values
        )
    assert fits["q64_cluster"].model_info["n_clusters"] == 100


def test_refusals(data):
    bad = sp.exceptions.MethodIncompatibility
    with pytest.raises(bad, match="required"):
        sp.etpoisson(data, y="y", x=["x1"], treat="d")
    with pytest.raises(bad, match="leave it out"):
        sp.etpoisson(data, y="y", x=["x1", "d"], treat="d", z=["z1"])
    with pytest.raises(bad, match="0/1"):
        sp.etpoisson(data, y="y", x=["x2"], treat="x1", z=["z1"])
    with pytest.raises(bad, match="non-negative integers"):
        sp.etpoisson(data.assign(y=data["y"] + 0.5), y="y", treatment="d", z=["z1"])


def test_from_stata_round_trip(data, stata):
    out = sp.from_stata("etpoisson y x1 x2, treat(d = z1 x1) intpoints(64) vce(robust)")
    assert out["ok"] and out["untranslated_options"] == [], out
    res = sp.etpoisson(data, **out["arguments"])
    ref = stata["q64_robust"]
    keep = [i for i, name in enumerate(ref["names"]) if name != "y:0b.d"]
    np.testing.assert_allclose(
        res.std_errors.values, np.array(ref["se"])[keep], rtol=1e-6
    )
    assert "terza1998estimating" in res.cite()
