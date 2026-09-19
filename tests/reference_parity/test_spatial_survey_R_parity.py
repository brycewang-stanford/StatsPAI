"""Cross-language parity: StatsPAI spatial weights, GWR / MGWR, SARAR GS2SLS,
spatial IV and spatial panels vs R ``spdep`` / ``GWmodel`` / ``spatialreg`` /
``sphet`` / ``splm``.

Fixture: ``_generate_spatial_survey_R.R`` -> ``_fixtures/spatial_survey_R.json``,
run on CSVs written once by ``_prepare_spatial_survey_data.R`` from public
package datasets (Columbus polygons from ``spData``; Georgia counties from
``GWmodel``; ``plm::Produc`` with ``splm::usaww``). Both sides read those
bytes; nothing below is recomputed from package data.

Conventions each block depends on
---------------------------------
* **Weights.** ``sp.W`` holds the weights as constructed; ``transform``
  ``"R" / "V" / "D" / "B"`` are spdep ``nb2listw`` styles ``"W" / "S" /
  "U" / "B"``. Neighbour lists are compared as sets of 0-based ids.
  ``kernel_weights(fixed=True, kernel="bisquare")`` is spdep's
  double-power distance weight ``(1 - (d/h)^2)^2``
  (``nb2listwdist(type="dpd", alpha=2)``) on the ``dnearneigh(0, h)``
  graph, and GWmodel's ``gw.weight`` bisquare. An adaptive bandwidth of ``k``
  neighbours in ``kernel_weights`` is GWmodel's adaptive ``bw = k + 1``
  (GWmodel counts the point itself). ``kernel_weights`` connects only the
  ``k`` nearest neighbours for every kernel, so its adaptive *Gaussian* is
  GWmodel's adaptive Gaussian restricted to that set (asserted as such).
* **GWR.** ``sp.gwr`` = ``GWmodel::gwr.basic``: same kernels (Gaussian and
  exponential untruncated, bisquare truncated), adaptive scale = distance to
  the ``floor(bw)``-th nearest point counting the point itself, extrapolated
  as ``(bw / n) * max d`` when ``bw > n``; local SE with
  ``sigma^2 = RSS / (n - 2 tr S + tr S'S)``; AIC / AICc / BIC as GWmodel.
  **Local R^2 is a documented convention difference:** GWmodel centres the
  weighted TSS on the *global* mean of y, PySAL mgwr (and StatsPAI) on the
  *local* weighted mean. Both definitions are asserted from the fitted
  pieces, so the gap is pinned, not tolerated. Under an *adaptive* kernel
  GWmodel's value is additionally computed with the transposed weight
  matrix (``gw.weight(dMat, ...) %*% dybar2`` sums row i of a matrix whose
  columns are the regression points), i.e. point i's R^2 uses the weights
  point i receives in the other points' kernels rather than its own
  kernel. Fixed kernels are symmetric, so there the two coincide; the
  transposition is asserted below (reference-side defect, T4).
* **Bandwidth selection.** ``sp.gwr_bandwidth`` reproduces
  ``GWmodel:::gold`` (bounds, floor / round integer probes, ``1e-4`` stop)
  because the criterion is not unimodal on the neighbour lattice: the
  selected bandwidth is a property of the search path. Criterion values
  (``gwr.aic`` = AICc, ``gwr.cv`` = LOO sum of squares) are compared
  directly.
* **MGWR** is compared at *fixed* covariate bandwidths (``bws=``): the
  back-fitting fixed point, which is unique. GWmodel converged to
  ``dCVR < 1e-12``; its bandwidth search path is not reproduced.
* **SARAR.** ``sp.sarar_gmm`` = ``spatialreg::gstsls``: instruments
  ``[X, WX, W^2 X]``, Kelejian-Prucha (1999) moments on the 2SLS residuals,
  second-stage instruments ``[(I - lambda W) X, WX, W^2 X]``; SE from the
  second-stage 2SLS with ``e'e / n`` (``sig2n_k=False``, gstsls default) or
  HC0 (``robust``). StatsPAI minimises the GM objective exactly (profiled
  quartic, admissible root); gstsls's ``nlminb`` stops ~7e-9 short on this
  flat objective, hence ``lambda`` at 1e-7 and everything else at 1e-9.
* **Spatial IV** = ``sphet::spreg(model="lag", het=TRUE)`` (excluded
  instrument not lagged, White/HC0 sandwich).
* **Spatial panel** = ``splm::spml(model="within")``: concentrated ML, SE
  from the analytic information matrix of (sigma^2, rho, beta). splm's
  ``stats::optimize`` cannot resolve the maximiser beyond ~sqrt(eps)
  relative (its internal tolerance floor; ``control = list(tol.opt=)``
  does not move it), StatsPAI polishes the analytic score root, so the
  spatial parameter and everything downstream are compared at 1e-7.
  splm labels the lag coefficient ``lambda`` and the error coefficient
  ``rho``.
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
gpd = pytest.importorskip("geopandas")
shapely_wkt = pytest.importorskip("shapely.wkt")


def _rel(a, b, floor=1e-300):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), floor)))


def _close(a, b, rtol, atol=0.0):
    np.testing.assert_allclose(
        np.asarray(a, float), np.asarray(b, float), rtol=rtol, atol=atol
    )


@pytest.fixture(scope="module")
def R():
    path = _FIX / "spatial_survey_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_spatial_survey_R.R first")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def columbus():
    df = pd.read_csv(_FIX / "spatial_survey_columbus.csv")
    gdf = gpd.GeoDataFrame(df, geometry=[shapely_wkt.loads(s) for s in df["wkt"]])
    return df, gdf


@pytest.fixture(scope="module")
def georgia():
    df = pd.read_csv(_FIX / "spatial_survey_georgia.csv")
    coords = df[["X", "Y"]].to_numpy()
    y = df["PctBach"].to_numpy()
    X = df[["PctRural", "PctPov", "PctBlack"]].to_numpy()
    return coords, y, X


def _nb_sets(w):
    return [sorted(int(j) for j in w.neighbors[i]) for i in range(w.n)]


def _row_weights(w):
    """Row weights in sorted-neighbour order (the fixture's order)."""
    out = []
    M = w.sparse.tocsr()
    for i in range(w.n):
        nb = sorted(int(j) for j in w.neighbors[i])
        out.append([M[i, j] for j in nb])
    return out


def _cmp_rows(py_rows, r_rows, rtol):
    assert len(py_rows) == len(r_rows)
    for a, b in zip(py_rows, r_rows):
        assert len(a) == len(b)
        if len(b):
            _close(a, b, rtol)


# ====================================================================== #
#  Weights
# ====================================================================== #


@pytest.mark.parametrize("kind", ["queen", "rook"])
def test_contiguity_neighbours_match_poly2nb(R, columbus, kind):
    _, gdf = columbus
    w = sp.queen_weights(gdf) if kind == "queen" else sp.rook_weights(gdf)
    assert _nb_sets(w) == [sorted(v) for v in R["contiguity"][kind]]


def test_queen_and_rook_differ_on_columbus(R):
    """Guard that the fixture discriminates the two criteria at all."""
    q = sum(len(v) for v in R["contiguity"]["queen"])
    r = sum(len(v) for v in R["contiguity"]["rook"])
    assert (q, r) == (236, 200)


@pytest.mark.parametrize(
    "style,key", [("R", "queen_W"), ("V", "queen_S"), ("D", "queen_U")]
)
def test_contiguity_transforms_match_nb2listw(R, columbus, style, key):
    _, gdf = columbus
    w = sp.queen_weights(gdf)
    w.transform = style
    _cmp_rows(_row_weights(w), R["contiguity"][key], rtol=1e-12)


def test_rook_row_standardised_matches(R, columbus):
    _, gdf = columbus
    w = sp.rook_weights(gdf)
    w.transform = "R"
    _cmp_rows(_row_weights(w), R["contiguity"]["rook_W"], rtol=1e-12)


def test_block_weights_match_nb2blocknb(R, columbus):
    df, _ = columbus
    w = sp.block_weights(df["regime"].to_numpy())
    assert _nb_sets(w) == [sorted(v) for v in R["block"]["neighbours"]]
    w.transform = "R"
    _cmp_rows(_row_weights(w), R["block"]["W"], rtol=1e-12)


def test_kernel_bisquare_fixed_matches_spdep_dpd(R, georgia):
    coords, _, _ = georgia
    K = R["kernel"]
    w = sp.kernel_weights(coords, bandwidth=K["h"], kernel="bisquare", fixed=True)
    assert _nb_sets(w) == [sorted(v) for v in K["dpd_neighbours"]]
    _cmp_rows(_row_weights(w), K["dpd_raw"], rtol=1e-12)


def test_kernel_row_standardise_keeps_kernel_values(R, georgia):
    """Regression: ``transform = "R"`` used to rebuild the weights from 1.0,
    so any kernel / inverse-distance W became a binary row-standardised W.
    Reference: ``nb2listwdist(type="dpd", style="W")``."""
    coords, _, _ = georgia
    K = R["kernel"]
    w = sp.kernel_weights(coords, bandwidth=K["h"], kernel="bisquare", fixed=True)
    w.transform = "R"
    _cmp_rows(_row_weights(w), K["dpd_W"], rtol=1e-12)
    w.transform = "O"  # back to the constructed weights, not to binary
    _cmp_rows(_row_weights(w), K["dpd_raw"], rtol=1e-12)


def _dense_offdiag(w):
    M = w.full()
    np.fill_diagonal(M, 0.0)
    return M


@pytest.mark.parametrize(
    "kernel,fixed,key",
    [
        ("gaussian", True, "gw_gaussian_fixed"),
        ("bisquare", True, "gw_bisquare_fixed"),
        ("bisquare", False, "gw_bisquare_adaptive"),
    ],
)
def test_kernel_weights_match_gw_weight(R, georgia, kernel, fixed, key):
    coords, _, _ = georgia
    K = R["kernel"]
    bw = K["h"] if fixed else K["k"]
    w = sp.kernel_weights(coords, bandwidth=bw, kernel=kernel, fixed=fixed)
    ref = np.asarray(K[key], dtype=float).T  # R column i = weights around i
    np.fill_diagonal(ref, 0.0)
    # Weights live in [0, 1]; near u = 1 the bisquare (1 - u^2)^2 cancels,
    # so an absolute 1e-15 floor (not relative) is the meaningful bound.
    _close(_dense_offdiag(w), ref, rtol=1e-12, atol=1e-15)


def test_kernel_adaptive_gaussian_is_knn_restricted(R, georgia):
    """Documented convention: the adaptive Gaussian of ``kernel_weights``
    keeps only the k nearest neighbours (a spatial-weights graph), where
    GWmodel's adaptive Gaussian weights every point."""
    coords, _, _ = georgia
    K = R["kernel"]
    w = sp.kernel_weights(coords, bandwidth=K["k"], kernel="gaussian", fixed=False)
    ref = np.asarray(K["gw_gaussian_adaptive"], dtype=float).T
    np.fill_diagonal(ref, 0.0)
    mine = _dense_offdiag(w)
    mask = mine > 0
    assert (mask.sum(axis=1) == K["k"]).all()
    _close(mine[mask], ref[mask], rtol=1e-12)
    assert np.all(ref[~mask] >= 0)
    assert ref[~mask].max() > 0.1  # GWmodel does keep weight outside the kNN


def test_variance_stabilising_identity(columbus):
    """Reference-free: style S rescales so all weights sum to n."""
    _, gdf = columbus
    w = sp.queen_weights(gdf)
    w.transform = "V"
    assert abs(w.full().sum() - w.n) < 1e-10


# ====================================================================== #
#  GWR
# ====================================================================== #

_GWR_IDS = [
    "bisquare_adaptive",
    "gaussian_adaptive",
    "exponential_adaptive",
    "bisquare_fixed",
    "gaussian_fixed",
    "exponential_fixed",
    "bisquare_adaptive_frac",
    "bisquare_adaptive_over_n",
]


def _fit(georgia, cfg):
    coords, y, X = georgia
    return sp.gwr(
        coords, y, X, bw=cfg["bw"], kernel=cfg["kernel"], fixed=not cfg["adaptive"]
    )


@pytest.mark.parametrize("cid", _GWR_IDS)
def test_gwr_matches_gwr_basic(R, georgia, cid):
    cfg = R["gwr"]["fits"][cid]
    res = _fit(georgia, cfg)
    _close(res.params, cfg["betas"], rtol=1e-9)
    _close(res.se, cfg["se"], rtol=1e-9)
    _close(res.predicted, cfg["yhat"], rtol=1e-11)
    d = cfg["diag"]
    _close(res.resid_ss, d["RSS.gw"], rtol=1e-11)
    _close([res.aic, res.aicc, res.bic], [d["AIC"], d["AICc"], d["BIC"]], rtol=1e-11)
    enp = 2 * res.tr_S - res.tr_StS
    _close([enp, res.n - enp], [d["enp"], d["edf"]], rtol=1e-10)
    _close(res.R2, d["gw.R2"], rtol=1e-11)


@pytest.mark.parametrize("cid", ["bisquare_adaptive", "gaussian_fixed"])
def test_gwr_local_r2_convention(R, georgia, cid):
    """GWmodel: TSS_w around the global mean; StatsPAI / mgwr: around the
    local weighted mean. Both reconstructed from the same fit."""
    from statspai.spatial.gwr.gwr import _weights_row

    coords, y, _ = georgia
    cfg = R["gwr"]["fits"][cid]
    res = _fit(georgia, cfg)
    e2 = res.residuals**2
    Wm = np.array(
        [
            _weights_row(coords, i, cfg["bw"], cfg["kernel"], not cfg["adaptive"])
            for i in range(len(y))
        ]
    )
    # GWmodel: global-mean TSS, and weights TRANSPOSED (column-kernel) --
    # immaterial for a symmetric fixed kernel, decisive for an adaptive one.
    Wg = Wm.T
    glob = 1 - (Wg @ e2) / (Wg @ (y - y.mean()) ** 2)
    _close(glob, cfg["local_R2"], rtol=1e-11)
    if cfg["adaptive"]:
        own = 1 - (Wm @ e2) / (Wm @ (y - y.mean()) ** 2)
        assert _rel(own, cfg["local_R2"]) > 1e-2
    ybar = (Wm @ y) / Wm.sum(axis=1)
    loc = 1 - (Wm @ e2) / ((Wm * (y[None, :] - ybar[:, None]) ** 2).sum(axis=1))
    _close(res.local_R2, loc, rtol=1e-12)
    assert _rel(res.local_R2, cfg["local_R2"]) > 1e-3


def test_gwr_criteria_match_gwr_aic_and_cv(R, georgia):
    coords, y, X = georgia
    for c in R["gwr"]["criteria"]:
        res = sp.gwr(
            coords, y, X, bw=c["bw"], kernel=c["kernel"], fixed=not c["adaptive"]
        )
        _close(res.aicc, c["aicc"], rtol=1e-11)
        _close(res.cv, c["cv"], rtol=1e-11)


def test_gwr_cv_is_leave_one_out(georgia):
    """Reference-free: the hat-diagonal CV equals explicit LOO refits."""
    from statspai.spatial.gwr.gwr import _weights_row

    coords, y, X = georgia
    Xc = np.column_stack([np.ones(len(y)), X])
    res = sp.gwr(coords, y, X, bw=50, kernel="bisquare")
    loo = []
    for i in range(len(y)):
        w = _weights_row(coords, i, 50, "bisquare", False)
        w[i] = 0.0
        b = np.linalg.solve(Xc.T @ (Xc * w[:, None]), Xc.T @ (w * y))
        loo.append(y[i] - Xc[i] @ b)
    _close(res.cv, float(np.dot(loo, loo)), rtol=1e-10)


def test_gwr_bandwidth_matches_bw_gwr(R, georgia):
    coords, y, X = georgia
    for s in R["gwr"]["bw_select"]:
        bw = sp.gwr_bandwidth(
            coords,
            y,
            X,
            kernel=s["kernel"],
            fixed=not s["adaptive"],
            criterion=s["approach"],
        )
        _close(bw, s["bw"], rtol=1e-10)


# ====================================================================== #
#  MGWR at fixed bandwidths
# ====================================================================== #


def test_mgwr_fixed_bandwidth_fixed_point(R, georgia):
    coords, y, X = georgia
    m = R["mgwr"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = sp.mgwr(coords, y, X, bws=m["bws"], tol=1e-13, max_iter=20000)
    _close(res.params, m["betas"], rtol=1e-8)
    _close(res.predicted, m["yhat"], rtol=1e-10)
    assert res.bws == [float(b) for b in m["bws"]]


def test_mgwr_fixed_point_identity(georgia):
    """Reference-free: each additive term is the GWR smooth of its partial
    residual at its own bandwidth."""
    coords, y, X = georgia
    bws = [40, 80, 120, 60]
    res = sp.mgwr(coords, y, X, bws=bws, tol=1e-12, max_iter=20000)
    Xc = np.column_stack([np.ones(len(y)), X])
    f = res.params * Xc
    for j in range(4):
        part = y - (f.sum(axis=1) - f[:, j])
        rj = sp.gwr(coords, part, Xc[:, [j]], bw=bws[j], add_constant=False)
        assert np.max(np.abs(rj.params.ravel() - res.params[:, j])) < 1e-9


# ====================================================================== #
#  SARAR GS2SLS
# ====================================================================== #


@pytest.fixture(scope="module")
def queen_W(columbus):
    _, gdf = columbus
    return sp.queen_weights(gdf)


@pytest.mark.parametrize(
    "cid,kw",
    [("default", {}), ("robust", {"robust": "het"}), ("sig2n_k", {"sig2n_k": True})],
)
def test_sarar_gmm_matches_gstsls(R, columbus, queen_W, cid, kw):
    df, _ = columbus
    ref = R["sarar"][cid]
    res = sp.sarar_gmm(queen_W, df, "CRIME ~ INC + HOVAL", **kw)
    name = {"(Intercept)": "const", "INC": "INC", "HOVAL": "HOVAL", "Rho_Wy": "rho"}
    for rn, pn in name.items():
        _close(res.params[pn], ref["coef"][rn], rtol=1e-9)
        _close(res.std_errors[pn], ref["se"][rn], rtol=1e-9)
    _close(res.params["lambda"], ref["lambda"], rtol=1e-7)
    assert np.isnan(res.std_errors["lambda"])  # gstsls: lambda.se = NULL


def test_sarar_lambda_is_admissible_gm_minimiser(columbus, queen_W):
    """Reference-free: lambda minimises the KP moment objective inside
    |lambda| < 1 (the quartic's other minimum, at ~4.07 here, is lower but
    inadmissible), and gstsls's nlminb value is worse."""
    from statspai.spatial.models.gmm import _kp_gm_lambda, _tsls
    from statspai.spatial.models.ml import _coerce_W, _parse_formula

    df, _ = columbus
    y, X, _, _ = _parse_formula("CRIME ~ INC + HOVAL", df)
    M = _coerce_W(queen_W, len(y), True)
    WX = M @ X[:, 1:]
    _, _, u = _tsls(
        y, np.column_stack([M @ y, X]), np.column_stack([X, WX, M @ WX]), False, False
    )
    lam, s2, obj = _kp_gm_lambda(u, M)
    assert abs(lam) < 1
    n = len(u)
    wu, wwu = M @ u, M @ (M @ u)
    tr = float(M.multiply(M).sum())
    G = (
        np.array(
            [
                [2 * u @ wu, -(wu @ wu), n],
                [2 * wwu @ wu, -(wwu @ wwu), tr],
                [u @ wwu + wu @ wu, -(wwu @ wu), 0.0],
            ]
        )
        / n
    )
    g = np.array([u @ u, wu @ wu, u @ wu]) / n

    def f(lm, sg):
        r = G @ np.array([lm, lm * lm, sg]) - g
        return float(r @ r)

    assert abs(f(lam, s2) - obj) < 1e-12 * max(1.0, obj)
    for d in (1e-6, -1e-6):
        assert f(lam + d, s2) >= obj


# ====================================================================== #
#  Spatial IV
# ====================================================================== #


def test_spatial_iv_matches_sphet_lag_het(R, columbus):
    df, _ = columbus
    ref = R["spatial_iv"]
    # spatial_iv takes W as given; sphet row-standardised it.
    wr = sp.queen_weights(columbus[1])
    wr.transform = "R"
    res = sp.spatial_iv(
        df, y="CRIME", endog=["HOVAL"], exog=["INC"], W=wr, instruments=["DISCBD"]
    )
    tab = res.coefficients.set_index("variable")
    name = {
        "(Intercept)": "(Intercept)",
        "INC": "INC",
        "HOVAL": "HOVAL",
        "lambda": "rho (WY)",
    }
    for rn, pn in name.items():
        _close(tab.loc[pn, "coef"], ref["coef"][rn], rtol=1e-10)
        _close(tab.loc[pn, "se"], ref["se"][rn], rtol=1e-10)


def test_spatial_iv_alpha_sets_interval_width(columbus):
    """Regression: ``alpha`` was accepted and ignored."""
    df, gdf = columbus
    w = sp.queen_weights(gdf)
    w.transform = "R"
    kw = dict(y="CRIME", endog=["HOVAL"], exog=["INC"], W=w, instruments=["DISCBD"])
    a = sp.spatial_iv(df, alpha=0.05, **kw).coefficients
    b = sp.spatial_iv(df, alpha=0.10, **kw).coefficients
    from scipy.stats import norm

    _close(a["ci_upper"] - a["coef"], norm.isf(0.025) * a["se"], rtol=1e-12)
    _close(b["ci_upper"] - b["coef"], norm.isf(0.05) * b["se"], rtol=1e-12)


# ====================================================================== #
#  Spatial panel
# ====================================================================== #


@pytest.fixture(scope="module")
def produc():
    d = pd.read_csv(_FIX / "spatial_survey_produc.csv")
    W = pd.read_csv(_FIX / "spatial_survey_usaww.csv", index_col=0).to_numpy()
    return d, W


_PANEL = [(m, e) for e in ("individual", "twoways") for m in ("sar", "sem", "sdm")]


@pytest.mark.parametrize("model,eff", _PANEL)
def test_spatial_panel_matches_spml(R, produc, model, eff):
    d, W = produc
    ref = R["panel"][f"{model}_{eff}"]
    res = sp.spatial_panel(
        d,
        "lgsp ~ lpcap + lpc + lemp + unemp",
        entity="state",
        time="year",
        W=W,
        model=model,
        effects="fe" if eff == "individual" else "twoways",
    )
    sp_name = "lambda" if model in ("sar", "sdm") else "rho"  # splm labels
    mine = "rho" if model in ("sar", "sdm") else "lambda"
    _close(res.params[mine], ref["coef"][sp_name], rtol=1e-7)
    _close(res.std_errors[mine], ref["se"][sp_name], rtol=1e-7)
    for k, v in ref["coef"].items():
        if k == sp_name:
            continue
        _close(res.params[k], v, rtol=1e-7, atol=1e-12)
        _close(res.std_errors[k], ref["se"][k], rtol=1e-7)
    _close(res.sigma2, ref["sigma2"], rtol=1e-7)
    if ref.get("logLik") is not None:
        _close(res.log_likelihood, ref["logLik"], rtol=1e-11)


def test_spatial_panel_rho_is_score_root(produc):
    """Reference-free: the concentrated score vanishes at rho-hat."""
    d, W = produc
    res = sp.spatial_panel(
        d,
        "lgsp ~ lpcap + lpc + lemp + unemp",
        entity="state",
        time="year",
        W=W,
        model="sar",
    )
    Wr = W / W.sum(axis=1, keepdims=True)
    ll = []
    h = 1e-6
    for r in (res.params["rho"] - h, res.params["rho"] + h):
        # log-likelihood is maximal at rho-hat: both neighbours lower
        Y = d.pivot(index="state", columns="year", values="lgsp").to_numpy()
        Xs = [
            d.pivot(index="state", columns="year", values=v).to_numpy()
            for v in ["lpcap", "lpc", "lemp", "unemp"]
        ]
        dm = lambda a: a - a.mean(axis=1, keepdims=True)  # noqa: E731
        X = np.column_stack([dm(x).ravel(order="F") for x in Xs])
        ys = (dm(Y) - r * Wr @ dm(Y)).ravel(order="F")
        e = ys - X @ np.linalg.lstsq(X, ys, rcond=None)[0]
        N, T = Y.shape
        ev = np.real(np.linalg.eigvals(Wr))
        s2 = e @ e / (N * T)
        ll.append(
            -N * T / 2 * np.log(2 * np.pi * s2)
            + T * np.sum(np.log(1 - r * ev))
            - e @ e / (2 * s2)
        )
    assert max(ll) <= res.log_likelihood + 1e-9


# ---------------------------------------------------------------------- #
#  Stata xsmle (vce(oim), entity fixed effects)
# ---------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def S():
    path = _FIX / "spatial_survey_stata.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _fixtures/_generate_spatial_survey_stata.do first")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("model", ["sar", "sem", "sdm"])
def test_spatial_panel_oim_matches_xsmle(S, produc, model):
    """``vce="oim"`` = xsmle's observed-information SEs; with -ml-'s
    tolerances tightened xsmle also lands on the exact maximiser. Points and
    the beta SEs are compared at 1e-9 (observed <= 1.1e-12). The SE of the
    spatial parameter is 1e-8 (observed 3.0e-9): StatsPAI's Hessian is
    analytic, xsmle's comes from -ml-'s numerical second derivatives, whose
    truncation error in the curved spatial direction is the gap."""
    d, W = produc
    ref = S[f"{model}_ind"]
    assert ref["converged"] == 1
    res = sp.spatial_panel(
        d,
        "lgsp ~ lpcap + lpc + lemp + unemp",
        entity="state",
        time="year",
        W=W,
        model=model,
        vce="oim",
    )
    sname = "lambda" if model == "sem" else "rho"
    for k, v in ref["coef"].items():
        eq, var = k.split(":")
        if eq == "Variance":
            _close(res.sigma2, v, rtol=1e-9)
            continue
        mine = {"Main": var, "Wx": f"W_{var}", "Spatial": sname}[eq]
        _close(res.params[mine], v, rtol=1e-9)
        _close(
            res.std_errors[mine], ref["se"][k], rtol=1e-8 if eq == "Spatial" else 1e-9
        )
    _close(res.log_likelihood, ref["ll"], rtol=1e-12)


def test_spatial_panel_twoways_xsmle_is_a_different_estimator(S, R):
    """Documented reference disagreement (not a parity claim): for two-way
    effects xsmle's SAR / SDM lag regressor is the within transform of
    ``W y`` while splm (and StatsPAI) lag the demeaned ``y``; xsmle also
    reports non-convergence there even with tightened tolerances. SEM
    agrees to the optimizer level."""
    assert S["sar_both"]["converged"] == 0 and S["sdm_both"]["converged"] == 0
    gap = abs(
        S["sar_both"]["coef"]["Spatial:rho"]
        - R["panel"]["sar_twoways"]["coef"]["lambda"]
    )
    assert gap > 1e-4
    _close(
        S["sem_both"]["coef"]["Spatial:lambda"],
        R["panel"]["sem_twoways"]["coef"]["rho"],
        rtol=1e-7,
    )
