"""
Design-based effects of spatial treatments (Pollmann 2026).

A spatial treatment (a grocery store, a school, a power plant) sits at a
location and affects nearby outcome units. The ideal experiment
randomises the treatment among a set of candidate locations, so units
near *unrealised* candidate locations are the control group. For a
distance ring ``d`` the estimator compares the weighted mean outcome of
units near realised locations with the inverse-odds weighted mean of
units near unrealised candidates::

    tau_hat(d) = sum_s W_s sum_i w_i(s, d) Y_i / sum_s W_s sum_i w_i(s, d)
               - sum_s (1 - W_s) r_s sum_i w_i(s, d) Y_i
                 / sum_s (1 - W_s) r_s sum_i w_i(s, d),     r_s = p_s / (1 - p_s)

where ``W_s`` marks realised candidates and ``p_s`` is the design
probability that candidate ``s`` is realised. Under independent
assignment and additively separable effects it targets the average
effect on the treated at distance ``d`` (the paper's Equation 1); without
additive separability it targets the average marginal effect (Eq. 2).

The standard error is the paper's design-based one: the only randomness
is which candidates are realised. Units are exposed to the candidates
within ``max_effect_distance`` of them (Assumption 3(a)); the estimator
is the observable part of the variance in Theorem 1(iv),
``(V_t + V_c + V_x) / |S|``, which is conservative because the terms
that involve outcomes never observed together are bounded by squares.
Every probability in it has a closed form under independent assignment,
so no exposure is enumerated.

References
----------
[@pollmann2026causal]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import AssumptionWarning, DataInsufficient, MethodIncompatibility

_EARTH_RADIUS = {"km": 6371.0088, "mi": 3958.7613}


# --------------------------------------------------------------------
# Result
# --------------------------------------------------------------------


@dataclass
class SpatialTreatmentResult(ResultProtocolMixin):
    """Result of :func:`spatial_treatment`: one estimate per distance ring.

    ``table`` holds one row per ring (bounds, estimate, design-based
    standard error, interval, normal and randomization p-values, counts).
    ``estimate`` / ``se`` / ``ci`` / ``pvalue`` repeat the innermost ring
    that could be estimated, so the result also reads as a single number.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> g = np.arange(6) / 5
    >>> locs = pd.DataFrame({"x": np.repeat(g, 6), "y": np.tile(g, 6)})
    >>> locs["realized"] = rng.binomial(1, 0.5, len(locs))
    >>> units = pd.DataFrame(rng.uniform(0, 1, (400, 2)), columns=["x", "y"])
    >>> d = np.hypot(units.x.values[:, None] - locs.x.values,
    ...              units.y.values[:, None] - locs.y.values)
    >>> near = (d[:, locs.realized.values == 1] <= 0.05).sum(1)
    >>> units["visits"] = 10 + 2 * near + rng.normal(size=400)
    >>> r = sp.spatial_treatment(units, "visits", ["x", "y"], locs,
    ...                          bins=[0, 0.05, 0.1], n_perm=0)
    >>> type(r).__name__
    'SpatialTreatmentResult'
    >>> list(r.table.columns[:4])
    ['ring', 'lower', 'upper', 'estimate']
    """

    table: pd.DataFrame
    alpha: float = 0.05
    n_obs: int = 0
    n_locations: int = 0
    n_realized: int = 0
    method: str = "Spatial treatment (design-based, Pollmann 2026)"
    estimand: str = "ATT(d)"
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)

    _citation_keys: ClassVar[Tuple[str, ...]] = ("pollmann2026causal",)

    def _headline(self) -> Optional[pd.Series]:
        ok = self.table[np.isfinite(self.table["estimate"].to_numpy(dtype=float))]
        return None if ok.empty else ok.iloc[0]

    @property
    def estimate(self) -> float:
        row = self._headline()
        return float("nan") if row is None else float(row["estimate"])

    @property
    def se(self) -> float:
        row = self._headline()
        return float("nan") if row is None else float(row["se"])

    @property
    def pvalue(self) -> float:
        row = self._headline()
        return float("nan") if row is None else float(row["pvalue"])

    @property
    def ci(self) -> Tuple[float, float]:
        row = self._headline()
        if row is None:
            return (float("nan"), float("nan"))
        return (float(row["ci_lower"]), float(row["ci_upper"]))

    def tidy(self, conf_level: Optional[float] = None) -> pd.DataFrame:
        """The per-ring table (a copy).

        The intervals are at the level the estimator was fitted with;
        ``conf_level`` is accepted for the common result interface and must
        equal ``1 - alpha`` of the fit.
        """
        if conf_level is not None and abs(conf_level - (1 - self.alpha)) > 1e-12:
            raise MethodIncompatibility(
                f"conf_level={conf_level} differs from the fitted level "
                f"{1 - self.alpha}.",
                recovery_hint="Refit sp.spatial_treatment with "
                "alpha = 1 - conf_level.",
            )
        return self.table.copy()

    def summary(self) -> str:
        """Plain-text summary of the per-ring estimates."""
        cols = [
            "lower",
            "upper",
            "estimate",
            "se",
            "ci_lower",
            "ci_upper",
            "pvalue",
            "ri_pvalue",
            "n_units",
        ]
        lines = [
            self.method,
            f"Estimand: {self.estimand} by distance ring; "
            f"{self.n_realized} realised of {self.n_locations} candidate "
            f"locations; {self.n_obs} outcome units.",
            f"Design-based SE, exposure radius "
            f"{self.model_info.get('max_effect_distance')}; "
            f"{int(round(100 * (1 - self.alpha)))}% normal intervals.",
            self.table[cols].to_string(index=False, float_format="%.4g"),
        ]
        out = "\n".join(lines)
        print(out)
        return out

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Estimates against ring midpoints with interval bars."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 4)))
        else:
            fig = ax.figure
        t = self.table
        mid = t["midpoint"].to_numpy(dtype=float)
        est = t["estimate"].to_numpy(dtype=float)
        lo = est - t["ci_lower"].to_numpy(dtype=float)
        hi = t["ci_upper"].to_numpy(dtype=float) - est
        ax.axhline(0.0, color="#666666", lw=1)
        ax.errorbar(mid, est, yerr=[lo, hi], fmt="o-", capsize=3, **kwargs)
        ax.set_xlabel("Distance from treatment location")
        ax.set_ylabel("Effect")
        return fig, ax


# --------------------------------------------------------------------
# Distances and ring weights
# --------------------------------------------------------------------


def _numeric_coords(frame: pd.DataFrame, cols: Sequence[str], name: str) -> np.ndarray:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise DataInsufficient(
            f"{name} has no column(s) {missing}.",
            recovery_hint=f"Pass the coordinate column names present in {name}.",
            diagnostics={"missing": missing},
        )
    try:
        xy = frame[list(cols)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"Coordinates of {name} must be numeric.",
            recovery_hint="Convert the coordinate columns to floats.",
        ) from exc
    if not np.all(np.isfinite(xy)):
        raise DataInsufficient(
            f"Coordinates of {name} contain missing or infinite values.",
            recovery_hint="Drop or geocode the rows with missing coordinates.",
            diagnostics={"n_bad": int((~np.isfinite(xy)).any(axis=1).sum())},
        )
    return np.asarray(xy)


def _distances(a: np.ndarray, b: np.ndarray, metric: str, unit: str) -> np.ndarray:
    if metric == "euclidean":
        diff = a[:, None, :] - b[None, :, :]
        return np.asarray(np.sqrt((diff**2).sum(axis=2)))
    # haversine on (latitude, longitude) in degrees
    lat1, lon1 = np.radians(a[:, 0])[:, None], np.radians(a[:, 1])[:, None]
    lat2, lon2 = np.radians(b[:, 0])[None, :], np.radians(b[:, 1])[None, :]
    h = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return np.asarray(
        2 * _EARTH_RADIUS[unit] * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
    )


def _ring_weights(
    D: np.ndarray, lo: float, hi: float, kernel: str, first: bool
) -> np.ndarray:
    if kernel == "bin":
        w = (D > lo) & (D <= hi)
        if first:
            w |= D == lo
        return w.astype(float)
    mid, h = 0.5 * (lo + hi), hi - lo
    return np.clip(1.0 - np.abs(D - mid) / h, 0.0, None)


# --------------------------------------------------------------------
# Point estimate and design-based variance for one ring
# --------------------------------------------------------------------


def _ring_means(
    colY: np.ndarray, colw: np.ndarray, W: np.ndarray, r: np.ndarray
) -> Tuple[Any, Any]:
    """Treated and inverse-odds control means; ``W`` may be (B, S)."""
    den_t = W @ colw
    den_c = ((1.0 - W) * r) @ colw
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_t = (W @ colY) / den_t
        mu_c = (((1.0 - W) * r) @ colY) / den_c
    return mu_t, mu_c


def _ring_variance(
    w: sparse.csr_matrix,
    E: sparse.csr_matrix,
    W: np.ndarray,
    p: np.ndarray,
    Y: np.ndarray,
    mu_t: float,
    mu_c: float,
    nbar: float,
    n_locations: int,
    Ypair: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Design-based variance of the ring estimator (Theorem 1(iv)).

    ``w`` (units x candidates) holds the ring weights, ``E`` marks the
    candidates within the exposure radius of each unit. Returns the
    observable components ``V_t``, ``V_c``, ``V_x`` (each already divided
    by ``|S|``) and ``var = (V_t + V_c + V_x) / |S|``.

    ``Ypair`` (units x candidates, dense) replaces the unit outcome ``Y``
    by an outcome per (unit, candidate) term, as in the double machine
    learning estimator where the outcome of term ``(i, s)`` is
    ``Y_i - m_{s,i}``. ``mu_t`` / ``mu_c`` are then the values the treated
    and control terms are centred at.
    """
    S = n_locations
    Wb = W.astype(bool)
    q = np.where(Wb, p, 1.0 - p)  # probability of the realised state
    f = np.where(Wb, 1.0, p / (1.0 - p))  # |weight| of the realised state
    logq = np.log(q)
    E = E.tocsr().astype(float)
    w = w.tocsr()
    logpm = np.asarray(E @ logq).ravel()
    pm = np.exp(logpm)

    # ---- V_t, V_c: one term per observed (unit, candidate) pair --------
    # Pairs of units whose exposure sets overlap, with the log probability
    # of the overlap's realised states.
    EE = (E @ E.T).tocoo()
    ElogQ = (E.multiply(logq[None, :])).tocsr() @ E.T
    logPO = np.asarray(ElogQ[EE.row, EE.col]).ravel()
    PO = np.exp(logPO)
    A = np.asarray(w @ (2.0 * p)).ravel() / nbar  # sum_s' w 2 p_s'
    Bmat = E @ (E.multiply(w).multiply((2.0 * p - f)[None, :])).T
    B = np.asarray(Bmat[EE.row, EE.col]).ravel() / nbar
    conflict = np.bincount(
        EE.row, weights=A[EE.col] * (1.0 - PO) + PO * B, minlength=w.shape[0]
    )

    wc = w.tocoo()
    ti, ts, tw = wc.row, wc.col, wc.data
    keep = tw != 0
    ti, ts, tw = ti[keep], ts[keep], tw[keep]
    in_E = np.asarray(E[ti, ts]).ravel() > 0
    pi_ms = pm[ti] * np.where(in_E, 1.0, q[ts])
    v = f[ts] * ((1.0 - pi_ms) * f[ts] * tw / nbar + conflict[ti])
    mu_real = np.where(Wb[ts], mu_t, mu_c)
    Yt = Y[ti] if Ypair is None else Ypair[ti, ts]
    contrib = (tw / nbar) * v * (Yt - mu_real) ** 2
    Vt = float(contrib[Wb[ts]].sum()) / S
    Vc = float(contrib[~Wb[ts]].sum()) / S

    # ---- V_x: cross products of distinct terms -------------------------
    # A term is (unit, candidate, state). A candidate inside the unit's
    # exposure set has its realised state; one outside it takes either
    # state because the outcome does not depend on it.
    t_i = np.concatenate([ti[in_E], ti[~in_E], ti[~in_E]])
    t_s = np.concatenate([ts[in_E], ts[~in_E], ts[~in_E]])
    t_w = np.concatenate([tw[in_E], tw[~in_E], tw[~in_E]])
    n_out = int((~in_E).sum())
    t_a = np.concatenate(
        [W[ts[in_E]].astype(int), np.ones(n_out, int), np.zeros(n_out, int)]
    )
    order = np.argsort(t_i, kind="stable")
    t_i, t_s, t_w, t_a = t_i[order], t_s[order], t_w[order], t_a[order]
    t_y = None
    if Ypair is not None:
        t_y = np.concatenate([Yt[in_E], Yt[~in_E], Yt[~in_E]])[order]
    n_units = w.shape[0]
    counts = np.bincount(t_i, minlength=n_units)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    # Linked units: exposure set or ring support overlap.
    L = (E + (w != 0).astype(float)).tocsr()
    L.data[:] = 1.0
    LL = (L @ L.T).tocoo()
    i1, i2 = LL.row, LL.col
    n1, n2 = counts[i1], counts[i2]
    has = (n1 > 0) & (n2 > 0)
    i1, i2, n1, n2 = i1[has], i2[has], n1[has], n2[has]
    if i1.size == 0:
        Vx = 0.0
    else:
        npair = n1 * n2
        pair_id = np.repeat(np.arange(i1.size), npair)
        offs = np.arange(pair_id.size) - np.repeat(
            np.concatenate([[0], np.cumsum(npair)[:-1]]), npair
        )
        k1 = starts[i1[pair_id]] + offs // n2[pair_id]
        k2 = starts[i2[pair_id]] + offs % n2[pair_id]
        distinct = k1 != k2
        k1, k2, pair_id = k1[distinct], k2[distinct], pair_id[distinct]
        u1, u2 = i1[pair_id], i2[pair_id]
        s1, s2 = t_s[k1], t_s[k2]
        a1, a2 = t_a[k1], t_a[k2]
        e11 = np.asarray(E[u1, s1]).ravel() > 0
        e21 = np.asarray(E[u2, s1]).ravel() > 0
        e12 = np.asarray(E[u1, s2]).ravel() > 0
        e22 = np.asarray(E[u2, s2]).ravel() > 0
        logov = np.asarray(ElogQ[u1, u2]).ravel()
        pmm = pm[u1] * pm[u2] / np.exp(logov)
        q1 = np.where(a1 == 1, p[s1], 1.0 - p[s1])
        q2 = np.where(a2 == 1, p[s2], 1.0 - p[s2])
        consistent = (
            ((a1 == W[s1]) | (~e11 & ~e21))
            & ((a2 == W[s2]) | (~e12 & ~e22))
            & ((a1 == a2) | (s1 != s2))
        )
        joint = np.where(
            consistent,
            pmm
            * np.where(~e11 & ~e21, q1, 1.0)
            * np.where(~e12 & ~e22 & (s1 != s2), q2, 1.0),
            0.0,
        )
        marg = pm[u1] * np.where(e11, 1.0, q1) * pm[u2] * np.where(e22, 1.0, q2)
        c1 = np.where(a1 == 1, 1.0, -p[s1] / (1.0 - p[s1]))
        c2 = np.where(a2 == 1, 1.0, -p[s2] / (1.0 - p[s2]))
        yv1, yv2 = (Y[u1], Y[u2]) if t_y is None else (t_y[k1], t_y[k2])
        y1 = yv1 - np.where(a1 == 1, mu_t, mu_c)
        y2 = yv2 - np.where(a2 == 1, mu_t, mu_c)
        Vx = (
            float(
                np.sum((joint - marg) / pmm * c1 * c2 * t_w[k1] * t_w[k2] * y1 * y2)
                / nbar**2
            )
            / S
        )
    var = (Vt + Vc + Vx) / S
    return {"var": var, "Vt": Vt, "Vc": Vc, "Vx": Vx}


# --------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------


def spatial_treatment(
    data: pd.DataFrame,
    y: str,
    coords: Optional[Sequence[str]] = None,
    locations: Optional[pd.DataFrame] = None,
    treat: str = "realized",
    bins: Optional[Sequence[float]] = None,
    location_coords: Optional[Sequence[str]] = None,
    propensity: Optional[str] = None,
    distances: Optional[np.ndarray] = None,
    max_effect_distance: Optional[float] = None,
    kernel: Literal["bin", "triangular"] = "bin",
    location_weights: bool = False,
    metric: Literal["euclidean", "haversine"] = "euclidean",
    distance_unit: Literal["km", "mi"] = "km",
    n_perm: int = 999,
    seed: Optional[int] = 0,
    alpha: float = 0.05,
    method: Literal["ipw", "nearest", "dml"] = "ipw",
    covariates: Optional[Sequence[str]] = None,
    location_covariates: Optional[Sequence[str]] = None,
    n_folds: int = 10,
    learner: Literal["ols", "post_lasso"] = "ols",
) -> SpatialTreatmentResult:
    """
    Effects of spatial treatments by distance, with design-based inference.

    Compares outcome units near realised treatment locations with units
    near counterfactual (unrealised) candidate locations, ring by ring,
    following the ideal experiment that randomises the treatment among
    the candidates (Pollmann 2026).

    Parameters
    ----------
    data : pd.DataFrame
        Outcome units, one row each (for example restaurants).
    y : str
        Outcome column of ``data``. For panel data, pass the change from
        the pre-treatment period (the paper's results carry over).
    coords : list of two str, optional
        Coordinate columns of ``data``. With ``metric="haversine"`` they
        are (latitude, longitude) in degrees. Not needed when
        ``distances`` is given.
    locations : pd.DataFrame
        Candidate treatment locations, one row each: the realised ones
        and the counterfactual candidates. Finding plausible
        counterfactual candidates is the user's step (the paper uses a
        convolutional network for it); it is not part of this function.
    treat : str, default "realized"
        0/1 column of ``locations`` marking realised locations.
    bins : sequence of float
        Ring edges, strictly increasing and non-negative, e.g.
        ``[0, 0.025, 0.05, 0.075]``. Ring ``k`` holds units with distance
        in ``(bins[k], bins[k+1]]``; the first ring also includes its
        lower edge.
    location_coords : list of two str, optional
        Coordinate columns of ``locations``; defaults to ``coords``.
    propensity : str, optional
        Column of ``locations`` with each candidate's design probability
        of being realised, strictly between 0 and 1. When omitted every
        candidate gets the realised share, the conditional-inference
        reading of a design with a fixed number of realised locations
        (Section 4.1.3 of the paper).
    distances : ndarray of shape (n_units, n_locations), optional
        Precomputed distances (any metric, for example travel time),
        rows in the order of ``data`` and columns in the order of
        ``locations``. Overrides ``coords`` and ``metric``.
    max_effect_distance : float, optional
        Distance beyond which a treatment is assumed to have no effect
        (``d0`` in the paper). It defines which candidates a unit is
        exposed to and enters the standard error only. Defaults to the
        last ring edge.
    kernel : {"bin", "triangular"}, default "bin"
        Ring weights. ``"triangular"`` puts weight ``1 - |d - m| / h`` on
        units within ``h`` (the ring width) of the ring midpoint ``m``.
    location_weights : bool, default False
        Give each candidate location equal total weight in a ring rather
        than each unit (the paper's Remark 7). Changes the estimand.
    metric : {"euclidean", "haversine"}, default "euclidean"
        Distance between coordinates.
    distance_unit : {"km", "mi"}, default "km"
        Unit of haversine distances (ignored for euclidean).
    n_perm : int, default 999
        Randomization draws for the sharp null of no effect of any
        location on any unit. Draws permute the realised labels among
        candidates when ``propensity`` is omitted and draw them
        independently with the given probabilities otherwise. 0 skips it.
    seed : int, optional
        Seed for the randomization draws.
    alpha : float, default 0.05
        Level of the normal intervals.
    method : {"ipw", "nearest", "dml"}, default "ipw"
        ``"ipw"`` is the ring estimator of Equation 3, which assumes the
        effects of different locations add up. ``"nearest"`` assumes only
        the nearest realised location matters to a unit (Assumption 4)
        and uses the estimator of Theorem 2 / Online Appendix 6: treated
        units are those whose nearest realised location is ``s``, controls
        those with no realised location within ``max_effect_distance``,
        each reweighted by the closed-form probability of that event.
        ``"dml"`` is the cross-fitted double machine learning estimator of
        Section 4.1.1 (latent experiment perspective).
    covariates : list of str, optional
        ``method="dml"`` only: unit covariates (columns of ``data``) in the
        linear outcome model, next to the unit's exposure to realised
        locations in each ring.
    location_covariates : list of str, optional
        ``method="dml"`` only: location characteristics (columns of
        ``locations``) for a cross-fitted logit of the design
        probabilities. Without them the probabilities are ``propensity``
        or the realised share.
    n_folds : int, default 10
        ``method="dml"`` only: folds of candidate locations for
        cross-fitting. Each fold's nuisances are fitted on units and
        locations far enough from the fold to be design-independent of it,
        so dense candidate sets need more folds.
    learner : {"ols", "post_lasso"}, default "ols"
        ``method="dml"`` only: first-step learners. ``"ols"`` fits the
        logit and the linear outcome model on every regressor.
        ``"post_lasso"`` is the paper's choice: ``sp.rlassologit(post=True)``
        for the design probabilities and, ring by ring, a post-LASSO
        outcome model (hdm's data-driven penalty) that always keeps the
        intercept and that ring's exposure.

    Returns
    -------
    SpatialTreatmentResult
        ``table`` has one row per ring: ``ring``, ``lower``, ``upper``,
        ``estimate``, ``se``, ``ci_lower``, ``ci_upper``, ``pvalue``,
        ``ri_pvalue``, ``midpoint``, ``n_units``, ``n_pairs_treated``,
        ``n_pairs_control``, ``mu_treated``, ``mu_control`` and the
        variance components ``V_t``, ``V_c``, ``V_x``. With
        ``method="nearest"`` the components are ``V_cov`` (pairs of terms
        that can be observed together) and ``V_bound`` (the bound for
        pairs that cannot) and ``V_t`` / ``V_c`` / ``V_x`` are missing.
        With ``method="dml"``, ``mu_treated`` and ``mu_control`` are the
        treated and inverse-odds control sums of ``Y - m`` divided by the
        treated weight, so that ``estimate = mu_treated - mu_control``.

    Notes
    -----
    The estimand is the average effect on the treated at distance ``d``
    when effects of different locations add up (Assumption 2), and the
    average marginal effect otherwise. The standard error assumes
    independent assignment across candidates (Assumption 1) and no effect
    beyond ``max_effect_distance``; it drops the three variance terms
    that cannot be estimated, so it is conservative, as in the paper.
    With ``propensity`` omitted the Bernoulli-design variance is used for
    a fixed-count design, which is conservative in expectation and
    equivalent asymptotically.

    The standard error divides by the total number of candidate
    locations ``|S|`` throughout, as in Theorem 1. The author's example
    code divides the own-term part by the number of candidates relevant
    to the ring instead; the two agree when every candidate has a unit
    within reach.

    ``method="nearest"``: the paper says the variance "can be derived
    analogously" without writing it out. The standard error here is the
    conservative Aronow-Samii estimator of the variance of the linearised
    estimator, with every inclusion probability in closed form under
    independent assignment; its expectation equals the true variance plus
    the bound for pairs of terms that can never be observed together.
    The control group is units with no realised location within
    ``max_effect_distance`` (inclusive, as for the exposure sets of the
    ring estimator), so the estimator uses few units when locations are
    dense, as the paper warns.

    ``method="dml"``: nuisances follow the paper's application, a logit
    for the design probabilities and a linear outcome model on unit
    covariates and ring exposure counts whose prediction with location
    ``s`` removed is ``m_{s,i}``; ``learner="post_lasso"`` uses the
    paper's post-LASSO for both.
    Estimated probabilities are clipped to ``[1e-6, 1 - 1e-6]``. The
    standard error is the design-based variance of Theorem 1(iv) applied
    to the linearised estimator (treated terms centred at the estimate,
    control terms at zero); by Theorem 3 the first steps do not enter it.
    Randomization p-values are not computed, since the nuisances would
    have to be refit under every draw.

    Not implemented: the convolutional network that proposes
    counterfactual candidates, reweighting the cross-fitting samples by
    their inclusion probabilities, and penalised first steps. The
    design-based perspective is shared with
    :func:`sp.recentered_iv` (Borusyak and Hull), which recentres an
    exposure by its average over counterfactual assignments.

    References
    ----------
    [@pollmann2026causal]
    [@aronow2017estimating] (the variance bound of ``method="nearest"``)

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> g = np.arange(8) / 7
    >>> locs = pd.DataFrame({"x": np.repeat(g, 8), "y": np.tile(g, 8)})
    >>> locs["realized"] = rng.binomial(1, 0.5, len(locs))
    >>> units = pd.DataFrame(rng.uniform(0, 1, (600, 2)), columns=["x", "y"])
    >>> d = np.hypot(units.x.values[:, None] - locs.x.values,
    ...              units.y.values[:, None] - locs.y.values)
    >>> near = (d[:, locs.realized.values == 1] <= 0.04).sum(1)
    >>> units["visits"] = 10 + 2 * near + rng.normal(size=600)
    >>> r = sp.spatial_treatment(units, "visits", ["x", "y"], locs,
    ...                          bins=[0, 0.04, 0.08], n_perm=99)
    >>> bool(abs(r.table.estimate.iloc[0] - 2) < 1)
    True
    >>> rn = sp.spatial_treatment(units, "visits", ["x", "y"], locs,
    ...                           bins=[0, 0.04, 0.08], n_perm=0,
    ...                           method="nearest")
    >>> rn.model_info["estimator"]
    'nearest'
    """
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            "alpha must be in (0, 1).", recovery_hint="Use e.g. alpha=0.05."
        )
    if method not in ("ipw", "nearest", "dml"):
        raise MethodIncompatibility(
            f"method must be 'ipw', 'nearest' or 'dml', got {method!r}.",
            recovery_hint="Use method='ipw' (additive effects), 'nearest' "
            "(only the nearest realised location matters) or 'dml'.",
        )
    if method != "dml" and (covariates or location_covariates):
        raise MethodIncompatibility(
            "covariates= and location_covariates= are used by method='dml' only.",
            recovery_hint="Pass method='dml', or drop the covariates.",
        )
    if method == "dml":
        if location_covariates and propensity is not None:
            raise MethodIncompatibility(
                "Pass either known design probabilities (propensity=) or "
                "location covariates to estimate them (location_covariates=), "
                "not both.",
                recovery_hint="Drop one of propensity= and location_covariates=.",
            )
        if not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
            raise MethodIncompatibility(
                f"n_folds must be an integer of at least 2, got {n_folds!r}.",
                recovery_hint="Use n_folds=10.",
            )
    if learner not in ("ols", "post_lasso"):
        raise MethodIncompatibility(
            f"learner must be 'ols' or 'post_lasso', got {learner!r}.",
            recovery_hint="Use learner='ols' or learner='post_lasso'.",
        )
    if learner != "ols" and method != "dml":
        raise MethodIncompatibility(
            "learner= is used by method='dml' only.",
            recovery_hint="Pass method='dml', or drop learner=.",
        )
    if kernel not in ("bin", "triangular"):
        raise MethodIncompatibility(
            f"kernel must be 'bin' or 'triangular', got {kernel!r}.",
            recovery_hint="Use kernel='bin'.",
        )
    if metric not in ("euclidean", "haversine"):
        raise MethodIncompatibility(
            f"metric must be 'euclidean' or 'haversine', got {metric!r}.",
            recovery_hint="Use metric='euclidean' or pass distances=.",
        )
    if distance_unit not in _EARTH_RADIUS:
        raise MethodIncompatibility(
            f"distance_unit must be 'km' or 'mi', got {distance_unit!r}.",
            recovery_hint="Use distance_unit='km'.",
        )
    if not isinstance(locations, pd.DataFrame):
        raise MethodIncompatibility(
            "locations must be a DataFrame of candidate treatment locations.",
            recovery_hint="Pass realised and counterfactual candidates as rows, "
            "with a 0/1 column named by treat=.",
        )
    if bins is None:
        raise MethodIncompatibility(
            "bins (ring edges) are required.",
            recovery_hint="Pass e.g. bins=[0, 0.025, 0.05, 0.075].",
        )
    edges = np.asarray(bins, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0) or edges[0] < 0:
        raise MethodIncompatibility(
            "bins must be at least two non-negative, strictly increasing edges.",
            recovery_hint="Pass e.g. bins=[0, 0.025, 0.05].",
            diagnostics={"bins": list(np.ravel(edges))},
        )
    if y not in data.columns:
        raise DataInsufficient(
            f"Outcome column {y!r} not in data.",
            recovery_hint="Pass the name of the outcome column.",
        )
    Y = pd.to_numeric(data[y], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(Y)):
        raise DataInsufficient(
            f"Outcome {y!r} has missing or non-numeric values.",
            recovery_hint="Drop units with a missing outcome before the call.",
            diagnostics={"n_missing": int((~np.isfinite(Y)).sum())},
        )
    if treat not in locations.columns:
        raise DataInsufficient(
            f"locations has no column {treat!r}.",
            recovery_hint="Name the 0/1 realised-location column with treat=.",
        )
    Wraw = pd.to_numeric(locations[treat], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isin(Wraw, (0.0, 1.0))):
        raise MethodIncompatibility(
            f"locations[{treat!r}] must be 0/1.",
            recovery_hint="Code realised locations 1 and counterfactual candidates 0.",
        )
    W = Wraw.astype(float)
    n_loc, n_real = W.size, int(W.sum())
    if n_real == 0:
        raise DataInsufficient(
            "No realised treatment location.",
            recovery_hint="Mark at least one location as realised.",
        )
    if n_real == n_loc:
        raise DataInsufficient(
            "Every candidate location is realised, so there is no "
            "counterfactual candidate to compare with.",
            recovery_hint="Add counterfactual (unrealised) candidate locations.",
        )
    if propensity is None:
        p = np.full(n_loc, n_real / n_loc)
        p_source = "realised share"
    else:
        if propensity not in locations.columns:
            raise DataInsufficient(
                f"locations has no column {propensity!r}.",
                recovery_hint="Name the design-probability column with propensity=.",
            )
        p = pd.to_numeric(locations[propensity], errors="coerce").to_numpy(dtype=float)
        if not np.all((p > 0) & (p < 1)):
            raise MethodIncompatibility(
                "Design probabilities must lie strictly between 0 and 1.",
                recovery_hint="Drop candidates that can never (or always) be "
                "realised; they carry no design information.",
                diagnostics={"min": float(np.nanmin(p)), "max": float(np.nanmax(p))},
            )
        p_source = propensity

    if distances is not None:
        D = np.asarray(distances, dtype=float)
        if D.shape != (len(data), n_loc):
            raise MethodIncompatibility(
                f"distances must have shape (n_units, n_locations) = "
                f"({len(data)}, {n_loc}), got {D.shape}.",
                recovery_hint="Rows follow data, columns follow locations.",
            )
        if not np.all(np.isfinite(D)) or np.any(D < 0):
            raise DataInsufficient(
                "distances must be finite and non-negative.",
                recovery_hint="Replace missing distances by a large value.",
            )
    else:
        if coords is None or len(coords) != 2:
            raise MethodIncompatibility(
                "coords must name two coordinate columns (or pass distances=).",
                recovery_hint="Use coords=['x', 'y'] or ['lat', 'lon'].",
            )
        lcols = list(location_coords) if location_coords is not None else list(coords)
        a = _numeric_coords(data, list(coords), "data")
        b = _numeric_coords(locations, lcols, "locations")
        D = _distances(a, b, metric, distance_unit)

    d0 = float(edges[-1] if max_effect_distance is None else max_effect_distance)
    if not d0 > 0:
        raise MethodIncompatibility(
            "max_effect_distance must be positive.",
            recovery_hint="Use the distance beyond which effects are negligible.",
        )
    E = sparse.csr_matrix(D <= d0)
    r = p / (1.0 - p)
    rng = np.random.default_rng(seed)
    draws = None
    if n_perm and n_perm > 0 and method != "dml":
        if propensity is None:
            draws = np.array([rng.permutation(W) for _ in range(int(n_perm))])
        else:
            draws = (rng.uniform(size=(int(n_perm), n_loc)) < p).astype(float)

    z = stats.norm.ppf(1 - alpha / 2)
    rows: List[Dict[str, Any]] = []
    empty: List[int] = []
    dml_info: Optional[Dict[str, Any]] = None
    if method == "ipw":
        for k in range(edges.size - 1):
            lo, hi = float(edges[k]), float(edges[k + 1])
            wk = _ring_weights(D, lo, hi, kernel, first=(k == 0))
            if location_weights:
                tot = wk.sum(axis=0)
                wk = np.divide(wk, tot, out=np.zeros_like(wk), where=tot > 0)
            colw = wk.sum(axis=0)
            colY = Y @ wk
            mu_t, mu_c = _ring_means(colY, colw, W, r)
            n_t = int(((wk > 0) & (W[None, :] == 1)).sum())
            n_c = int(((wk > 0) & (W[None, :] == 0)).sum())
            row: Dict[str, Any] = {
                "ring": k,
                "lower": lo,
                "upper": hi,
                "midpoint": 0.5 * (lo + hi),
                "n_units": int((wk > 0).any(axis=1).sum()),
                "n_pairs_treated": n_t,
                "n_pairs_control": n_c,
            }
            if n_t == 0 or n_c == 0:
                empty.append(k)
                row.update(
                    estimate=np.nan,
                    se=np.nan,
                    ci_lower=np.nan,
                    ci_upper=np.nan,
                    pvalue=np.nan,
                    ri_pvalue=np.nan,
                    mu_treated=np.nan,
                    mu_control=np.nan,
                    V_t=np.nan,
                    V_c=np.nan,
                    V_x=np.nan,
                )
                rows.append(row)
                continue
            est = float(mu_t - mu_c)
            nbar = float(colw @ p) / n_loc
            comp = _ring_variance(
                sparse.csr_matrix(wk), E, W, p, Y, float(mu_t), float(mu_c), nbar, n_loc
            )
            var = comp["var"]
            if var < 0:
                warnings.warn(
                    AssumptionWarning(
                        f"Ring {k}: the variance estimate is negative ({var:.3g}); "
                        "the standard error is set to 0.",
                        recovery_hint="Use more candidate locations or a wider ring.",
                    ),
                    stacklevel=2,
                )
            se = float(np.sqrt(max(var, 0.0)))
            pval = float(2 * stats.norm.sf(abs(est) / se)) if se > 0 else np.nan
            ri_p = np.nan
            if draws is not None:
                mt, mc = _ring_means(colY, colw, draws, r)
                tb = mt - mc
                ok = np.isfinite(tb)
                if ok.any():
                    # Relative slack so the observed draw counts against itself.
                    tol = 1e-12 * max(1.0, abs(est))
                    hits = np.sum(np.abs(tb[ok]) >= abs(est) - tol)
                    ri_p = float((1 + hits) / (1 + ok.sum()))
            row.update(
                estimate=est,
                se=se,
                ci_lower=est - z * se,
                ci_upper=est + z * se,
                pvalue=pval,
                ri_pvalue=ri_p,
                mu_treated=float(mu_t),
                mu_control=float(mu_c),
                V_t=comp["Vt"],
                V_c=comp["Vc"],
                V_x=comp["Vx"],
            )
            rows.append(row)
    else:
        from ._treatments_common import alternative_rows

        rows, empty, dml_info = alternative_rows(
            method,
            D,
            W,
            p,
            Y,
            edges,
            kernel,
            bool(location_weights),
            d0,
            draws,
            z,
            data,
            locations,
            covariates,
            location_covariates,
            propensity,
            int(n_folds),
            rng,
            learner,
        )
        if dml_info is not None:
            p = dml_info["p_hat"]

    if len(empty) == edges.size - 1:
        raise DataInsufficient(
            "No ring has units near both realised and counterfactual locations.",
            recovery_hint="Widen the rings (bins=) or check the coordinate units.",
            diagnostics={"bins": edges.tolist(), "min_distance": float(D.min())},
        )
    if empty:
        warnings.warn(
            AssumptionWarning(
                f"Ring(s) {empty} have no units near realised or near "
                "counterfactual locations; their estimates are missing.",
                recovery_hint="Widen those rings or drop them from bins=.",
                diagnostics={"empty_rings": empty},
            ),
            stacklevel=2,
        )
    cols = [
        "ring",
        "lower",
        "upper",
        "estimate",
        "se",
        "ci_lower",
        "ci_upper",
        "pvalue",
        "ri_pvalue",
        "midpoint",
        "n_units",
        "n_pairs_treated",
        "n_pairs_control",
        "mu_treated",
        "mu_control",
        "V_t",
        "V_c",
        "V_x",
    ]
    if method == "nearest":
        cols += ["V_cov", "V_bound"]
    table = pd.DataFrame(rows)[cols]
    diagnostics: Dict[str, Any] = {
        "empty_rings": empty,
        "propensity_source": p_source,
        "randomization": (
            None
            if draws is None
            else ("permutation" if propensity is None else "bernoulli")
        ),
        "n_perm": int(n_perm or 0),
        "max_exposure_set": int(np.asarray(E.sum(axis=1)).max()),
    }
    model_info = {
        "bins": edges.tolist(),
        "kernel": kernel,
        "location_weights": bool(location_weights),
        "max_effect_distance": d0,
        "metric": "precomputed" if distances is not None else metric,
        "estimator": method,
    }
    labels = {
        "ipw": ("Spatial treatment (design-based, Pollmann 2026)", "ATT(d)"),
        "nearest": (
            "Spatial treatment, nearest realised location (Pollmann 2026)",
            "ATT(d), only the nearest realised location matters",
        ),
        "dml": (
            "Spatial treatment, cross-fitted DML (Pollmann 2026)",
            "ATT(d), latent experiment",
        ),
    }
    if method == "nearest":
        diagnostics["variance"] = (
            "Aronow-Samii conservative bound; V_bound is the part from pairs "
            "of terms that cannot occur together"
        )
    if dml_info is not None:
        from ._treatments_common import dml_report

        dml_report(
            dml_info,
            model_info,
            diagnostics,
            covariates,
            location_covariates,
            int(n_folds),
            p_source,
            bool(n_perm),
        )
    return SpatialTreatmentResult(
        table=table,
        alpha=float(alpha),
        n_obs=int(len(data)),
        n_locations=int(n_loc),
        n_realized=n_real,
        method=labels[method][0],
        estimand=labels[method][1],
        diagnostics=diagnostics,
        model_info=model_info,
    )


__all__ = ["spatial_treatment", "SpatialTreatmentResult"]
