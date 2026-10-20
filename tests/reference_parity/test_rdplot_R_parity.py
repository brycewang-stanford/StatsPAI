"""Reference parity: the numbers behind ``sp.rdplot`` vs R ``rdrobust::rdplot``.

``statspai.rd._rdplot_numbers`` is a clean-room rewrite; R rdrobust 4.0.0
(GPL-3) was used only as a black box (spec: ``docs/dev/cleanroom/rdplot.md``).
Fixture: ``_fixtures/_generate_rdplot_R.R`` writes ``rdplot_cont.csv``,
``rdplot_mass.csv``, ``rdplot_qs.csv`` and ``rdplot_R.json`` (R's ``J``,
``J_IMSE``, ``J_MV``, ``coef``, ``coef_covs``, all ``vars_bins`` columns,
every 9th point of ``vars_poly``, ``h``, ``N``, ``N_h``).

Tolerances: rel 1e-9 on everything computed by closed forms and least squares,
measured against the largest absolute value of the column (a few interval
ends and curve points are within 1e-3 of zero, where an elementwise relative
error is meaningless; observed absolute differences are below 4e-12). The
order-4 global coefficients on the raw ``x - c`` scale are held at 1e-8 (two
least-squares solvers on an ill-conditioned design; observed 6e-12).

Two documented differences (spec, "Partition"):

* when R's last evenly spaced edge rounds one step below ``max(x)``, R puts
  that observation in an extra bin with missing edges; here the last bin is
  closed. The test merges R's extra row into the last bin.
* with empty left-side bins, R's ``rdplot_min_bin`` / ``rdplot_max_bin`` are
  shifted against the other columns; here each bin carries its own edges, so
  those two columns are compared only where R's are consistent with its own
  ``rdplot_mean_bin``.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from statspai.rd._rdplot_numbers import kernel_weights, rdplot_numbers

FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((FIX / "rdplot_R.json").read_text(encoding="utf-8"))["cases"]
DATA = {k: pd.read_csv(FIX / f"rdplot_{k}.csv") for k in ("cont", "mass", "qs")}
OK_CASES = sorted(k for k, v in R.items() if "error" not in v)
COLS = (
    "rdplot_mean_bin",
    "rdplot_mean_x",
    "rdplot_mean_y",
    "rdplot_min_bin",
    "rdplot_max_bin",
    "rdplot_se_y",
    "rdplot_N",
    "rdplot_ci_l",
    "rdplot_ci_r",
)


def _arr(v) -> np.ndarray:
    return np.atleast_1d(np.asarray(v, dtype=float))


def _close(ours, ref, tol=1e-9) -> float:
    ours, ref = _arr(ours), _arr(ref)
    assert ours.shape == ref.shape
    scale = max(float(np.max(np.abs(ref))), 1e-300)
    return float(np.max(np.abs(ours - ref))) / scale


def _call(ref):
    a = dict(ref["args"])
    d = DATA[ref["data"]]
    kw = {
        k: a[k] for k in ("c", "p", "binselect", "kernel", "masspoints", "ci") if k in a
    }
    for k in ("h", "nbins", "scale"):
        if k in a:
            kw[k] = tuple(a[k]) if isinstance(a[k], list) else a[k]
    if "weights" in a:
        kw["weights"] = d[a["weights"]].to_numpy()
    if "covs" in a:
        kw["covs"] = d[list(np.atleast_1d(a["covs"]))].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rdplot_numbers(d["y"].to_numpy(), d["x"].to_numpy(), **kw)


def _r_bins(ref):
    """R's bin table with an overflow row (missing edges) merged into the last bin."""
    vb = {
        k: np.array(
            [np.nan if v is None else v for v in np.atleast_1d(ref["vars_bins"][k])],
            dtype=float,
        )
        for k in COLS
    }
    over = np.isnan(vb["rdplot_mean_bin"])
    if not over.any():
        return vb, None
    assert (
        over.sum() == 1 and over[-1]
    ), "only a last right-hand overflow row is expected"
    return {k: v[:-1] for k, v in vb.items()}, int(vb["rdplot_N"][-1])


@pytest.mark.parametrize("key", OK_CASES)
def test_rdplot_numbers_match_r(key):
    ref = R[key]
    out = _call(ref)
    for k in ("J", "J_IMSE", "J_MV", "N", "N_h"):
        assert list(out[k]) == [int(v) for v in np.atleast_1d(ref[k])], k
    assert _close(out["h"], ref["h"]) < 1e-12

    rb, overflow = _r_bins(ref)
    vb = out["vars_bins"]
    if overflow is not None:
        # identity: the closed last bin holds R's last bin plus its extra row
        assert vb["rdplot_N"][-1] == rb["rdplot_N"][-1] + overflow
        assert vb["rdplot_max_bin"][-1] == pytest.approx(
            DATA[ref["data"]]["x"].max(), rel=1e-15
        )
        vb = {k: v[:-1] for k, v in vb.items()}
        rb = {k: v[:-1] for k, v in rb.items()}
    np.testing.assert_array_equal(vb["rdplot_N"], rb["rdplot_N"].astype(int))
    for col in (
        "rdplot_mean_bin",
        "rdplot_mean_x",
        "rdplot_mean_y",
        "rdplot_se_y",
        "rdplot_ci_l",
        "rdplot_ci_r",
    ):
        assert _close(vb[col], rb[col]) < 1e-9, col
    consistent = np.isclose(
        (rb["rdplot_min_bin"] + rb["rdplot_max_bin"]) / 2,
        rb["rdplot_mean_bin"],
        rtol=0,
        atol=1e-12,
    )
    if not key.startswith("mass_q"):
        assert consistent.all()
    for col in ("rdplot_min_bin", "rdplot_max_bin"):
        assert _close(vb[col][consistent], rb[col][consistent]) < 1e-9, col

    coef = _arr(ref["coef"]).reshape(ref["coef_dim"], order="F")
    assert _close(out["coef"], coef) < 1e-8
    if ref["coef_covs"] is None:
        assert out["coef_covs"] is None
    else:
        assert _close(out["coef_covs"], ref["coef_covs"]) < 1e-8
    idx = np.asarray(ref["poly_idx"], dtype=int) - 1
    for col in ("rdplot_x", "rdplot_y"):
        assert _close(out["vars_poly"][col][idx], ref["vars_poly"][col]) < 1e-9, col


def test_r_left_edge_quirk_is_the_only_min_max_difference():
    """In the mass-point QS cases R's left min/max columns are misaligned."""
    ref = R["mass_qsmv_off"]
    rb, _ = _r_bins(ref)
    bad = ~np.isclose(
        (rb["rdplot_min_bin"] + rb["rdplot_max_bin"]) / 2,
        rb["rdplot_mean_bin"],
        rtol=0,
        atol=1e-12,
    )
    assert bad.any() and np.all(rb["rdplot_mean_bin"][bad] < 0)
    out = _call(ref)["vars_bins"]
    # ours: every bin's edges bracket its own mean of x (up to the rounding
    # of a mean of tied values)
    assert np.all(out["rdplot_min_bin"] <= out["rdplot_mean_x"] + 1e-12)
    assert np.all(out["rdplot_mean_x"] <= out["rdplot_max_bin"] + 1e-12)


def test_r_refusals_are_recorded():
    # R 4.0.0 rejects a two-element scale; here it is one factor per side.
    assert "error" in R["cont_scale_pair"]
    base = _call(R["cont_esmv_p1"])
    d = DATA["cont"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pair = rdplot_numbers(d["y"].to_numpy(), d["x"].to_numpy(), scale=(2, 3))
    assert pair["J"] == (2 * base["J"][0], 3 * base["J"][1])


# --------------------------------------------------------------------------- #
#  analytic checks
# --------------------------------------------------------------------------- #
def test_one_bin_per_support_point_gives_cell_means():
    rng = np.random.default_rng(4)
    support = np.array([-0.9, -0.5, -0.2, 0.1, 0.4, 0.8])
    x = np.repeat(support, 30)
    y = 1.0 + x + 0.5 * (x >= 0) + rng.normal(0, 0.2, x.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rdplot_numbers(y, x, nbins=(3, 3), binselect="qs", p=1)
    vb = out["vars_bins"]
    np.testing.assert_array_equal(vb["rdplot_N"], np.full(6, 30))
    cell = np.array([y[x == s].mean() for s in support])
    np.testing.assert_allclose(vb["rdplot_mean_y"], cell, rtol=1e-14)
    np.testing.assert_allclose(vb["rdplot_mean_x"], support, rtol=1e-14)
    sd = np.array([y[x == s].std(ddof=1) for s in support]) / np.sqrt(30)
    np.testing.assert_allclose(vb["rdplot_se_y"], sd, rtol=1e-13)


def test_polynomial_reproduces_an_exact_polynomial():
    x = np.linspace(-1, 1, 201)
    y = np.where(x < 0, 1 + 2 * x - x**2, 3 - x + 0.5 * x**2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rdplot_numbers(y, x, p=2, nbins=(4, 4), kernel="tri", h=0.7)
    np.testing.assert_allclose(out["coef"][:, 0], [1, 2, -1], atol=1e-10)
    np.testing.assert_allclose(out["coef"][:, 1], [3, -1, 0.5], atol=1e-10)


def test_kernel_weights_are_scaled_by_one_over_h():
    x = np.array([-0.5, 0.0, 0.25, 2.0])
    np.testing.assert_allclose(kernel_weights(x, 0.0, 0.5, "epa"), [0, 1.5, 1.125, 0])
    np.testing.assert_allclose(kernel_weights(x, 0.0, 0.5, "uniform"), [1, 1, 1, 0])
    np.testing.assert_allclose(kernel_weights(x, 0.0, 0.5, "TRI"), [0, 2, 1, 0])
    # each integrates to one over [c - h, c + h]
    g = np.linspace(-0.5, 0.5, 200001)
    for k in ("uni", "tri", "epa"):
        assert np.trapezoid(kernel_weights(g, 0.0, 0.5, k), g) == pytest.approx(
            1, abs=1e-6
        )
