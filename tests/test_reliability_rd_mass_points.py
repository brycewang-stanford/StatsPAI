"""The RD mass-points study, and the failures it turned into clear errors.

``tests/reliability/rd_mass_points.py`` simulates the coverage of the
robust ``sp.rdrobust`` interval as the running variable goes from
continuous to five support points a side. Running it found two ways
``sp.rdrobust`` failed without saying so:

* with ``masspoints='off'`` and a few support points the bandwidth
  selector took a fractional power of a negative ratio and stopped with
  ``TypeError: float() argument must be ... not 'complex'`` (half the
  replications at five support points a side);
* a negative variance came back as a NaN standard error and a NaN
  interval.

Both are now :class:`~statspai.exceptions.NumericalInstability` with a
recovery hint. This file pins that, the warning for clusters that
coincide with the support points, and the statements made about the
stored results.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import AssumptionWarning, NumericalInstability

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "reliability" / "rd_mass_points.py"
RESULTS = ROOT / "reliability" / "rd_mass_points_results.json"


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("rd_mass_points", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _cell(results, n, M, method):
    (cell,) = [
        c
        for c in results["cells"]
        if (c["n"], c["support_per_side"], c["method"]) == (n, M, method)
    ]
    return cell


def test_file_has_the_declared_design(study, results):
    assert results["B"] == 1000 and results["truth"] == 1.0
    keys = {(c["n"], c["support_per_side"], c["method"]) for c in results["cells"]}
    expected = {
        (n, M, m)
        for n in study.N_VALUES
        for M in study.SUPPORT
        for m in study.METHODS
        if not (m == "cluster" and M == 0)
    }
    assert keys == expected and len(keys) == 28
    for c in results["cells"]:
        assert c["n_fitted"] + sum(c["refused"].values()) == c["B"]


def test_a_cell_reproduces_its_stored_prefix(study, results):
    fresh = study.run_cell(1000, 20, "adjust", study.PREFIX)
    stored = _cell(results, 1000, 20, "adjust")
    assert fresh["prefix_hits"] == stored["prefix_hits"]


def test_default_holds_its_coverage_down_to_twenty_support_points(results):
    for n in (1000, 4000):
        for M in (0, 50, 20):
            c = _cell(results, n, M, "adjust")
            assert abs(c["coverage"] - 0.95) < 3 * c["mc_se"] + 0.005, (n, M, c)
            assert not c["refused"]


def test_ten_or_fewer_support_points_break_the_default(results):
    """The bandwidth floor reaches the whole side; the linear fit is biased."""
    for n in (1000, 4000):
        for M in (10, 5):
            c = _cell(results, n, M, "adjust")
            assert c["coverage"] < 0.91, (n, M, c["coverage"])
            assert c["bias_conventional"] > 0.15
    # more data makes it worse, not better: the bias does not shrink
    assert (
        _cell(results, 4000, 10, "adjust")["coverage"]
        < _cell(results, 1000, 10, "adjust")["coverage"]
    )


def test_clustering_on_the_support_points_undercovers_everywhere(results):
    for n in (1000, 4000):
        for M in (50, 20, 10, 5):
            cl = _cell(results, n, M, "cluster")["coverage"]
            plain = _cell(results, n, M, "adjust")["coverage"]
            assert cl < 0.88 and cl < plain - 0.05, (n, M, cl, plain)


def test_masspoints_off_is_refused_rather_than_wrong_with_few_points(results):
    for n in (1000, 4000):
        c = _cell(results, n, 5, "off")
        assert c["refused"].get("NumericalInstability", 0) > 300
        assert all(np.isfinite(v) for v in (c["mean_length"], c["coverage"]))


# ---------------------------------------------------------------------------
# The failures themselves
# ---------------------------------------------------------------------------


def test_degenerate_bandwidth_is_a_named_error_not_a_complex_number(study):
    """Seed 1 of the five-support-point cell used to raise TypeError."""
    refused = 0
    for rep in range(12):
        df = study.draw(1000, 5, study.cell_seed(1000, 5, rep))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sp.rdrobust(df, y="y", x="x", c=0, masspoints="off")
        except NumericalInstability as exc:
            refused += 1
            assert "bandwidth" in str(exc) or "variance" in str(exc)
            assert "rd_discrete" in exc.recovery_hint
            continue
        except ValueError:
            continue  # DataInsufficient: an empty window, also a clear refusal
        assert np.isfinite(res.se) and np.isfinite(res.ci[0])
    assert refused >= 4


def test_no_fit_returns_a_nan_interval(study):
    for M in (5, 10):
        for rep in range(15):
            df = study.draw(1000, M, study.cell_seed(1000, M, rep))
            for mp in ("adjust", "off"):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = sp.rdrobust(df, y="y", x="x", c=0, masspoints=mp)
                except ValueError:
                    continue
                assert np.isfinite(res.ci[0]) and np.isfinite(res.ci[1])
                assert res.model_info["legacy_fallbacks"] in (
                    [],
                    ["bandwidth_selector"],
                )


def test_cluster_on_the_running_variable_warns(study):
    df = study.draw(1000, 20, 1)
    with pytest.warns(AssumptionWarning, match="per cluster") as caught:
        sp.rdrobust(df, y="y", x="x", c=0, cluster="xv", manipulation_test=False)
    (w,) = [c for c in caught if "per cluster" in str(c.message)]
    assert w.message.diagnostics["n_clusters"] == 40
    df["g"] = np.arange(len(df)) % 40
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        sp.rdrobust(df, y="y", x="x", c=0, cluster="g", manipulation_test=False)


def test_constant_outcome_still_returns_a_zero_jump(study):
    df = study.draw(500, 0, 3).assign(y=5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.rdrobust(df, y="y", x="x", c=0)
    assert abs(float(res.detail["estimate"][0])) < 1e-9
