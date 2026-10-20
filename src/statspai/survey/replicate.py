"""
Replicate-weight survey designs (BRR, Fay, jackknife, bootstrap,
successive difference) -- R ``survey::svrepdesign`` / Stata ``svyset,
brrweight() / jkrweight() / bsrweight() / sdrweight()``.

A replicate design carries the full-sample weights and ``R`` columns of
replicate (analysis) weights. A statistic ``theta`` is computed once with
the full-sample weights and once with every replicate's weights; its
variance is

    V = scale * sum_r rscales_r (theta_r - c)(theta_r - c)'

with ``c`` the mean of the replicates, or the full-sample estimate when
``mse=True`` -- the formula of R ``svrepdesign`` (its documentation; R
``survey`` is GPL and was used as a black box, documentation and outputs
only). The defaults of ``scale`` / ``rscales`` / ``mse`` per ``type`` and
the design degrees of freedom (rank of the replicate weights minus one)
are those R reports for the same input.

Estimators that accept ``design=`` take a :class:`ReplicateDesign` as
well when they have a ``weights=`` argument to refit with (see
:mod:`statspai.survey._replicate_refit`).
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

#: replicate types and their R names
_TYPES = {
    "brr": "BRR",
    "fay": "Fay",
    "jk1": "JK1",
    "jkn": "JKn",
    "jk2": "JK2",
    "bootstrap": "bootstrap",
    "successive-difference": "successive-difference",
    "sdr": "successive-difference",
    "acs": "ACS",
    "other": "other",
}


class ReplicateDesign:
    """
    A survey design described by replicate weights.

    Build it with :func:`sp.svrepdesign` (replicate weights shipped with
    the data, e.g. ACS, MEPS, NHANES-style files) or
    :func:`sp.as_svrepdesign` (jackknife / BRR weights built from a
    :class:`SurveyDesign`).

    Attributes
    ----------
    data : pd.DataFrame
    weights : ndarray, shape (n,)
        Full-sample weights.
    repweights : ndarray, shape (n, R)
        Replicate analysis weights (sampling weights included).
    type : str
        R's name of the replicate type.
    scale : float
    rscales : ndarray, shape (R,)
    mse : bool
        Centre the squared deviations at the full-sample estimate (True)
        or at the mean of the replicates (False).
    df : float
        Design degrees of freedom.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"psu": np.repeat(np.arange(10), 20),
    ...                    "w": rng.uniform(1, 3, 200),
    ...                    "y": rng.normal(size=200)})
    >>> rep = sp.as_svrepdesign(sp.svydesign(df, weights="w", cluster="psu"))
    >>> rep.n_replicates
    10
    >>> round(float(rep.mean("y").estimate["y"]), 6) == round(
    ...     float(np.average(df["y"], weights=df["w"])), 6)
    True
    """

    is_replicate_design = True

    def __init__(
        self,
        data: pd.DataFrame,
        repweights: Any,
        weights: Union[str, np.ndarray, None] = None,
        type: str = "JK1",
        rho: Optional[float] = None,
        scale: Optional[float] = None,
        rscales: Optional[Sequence[float]] = None,
        mse: Optional[bool] = None,
        combined_weights: bool = True,
        df: Optional[float] = None,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise MethodIncompatibility(
                "svrepdesign: data must be a pandas DataFrame.",
                recovery_hint="Pass the survey microdata as a DataFrame.",
            )
        key = str(type).lower()
        if key not in _TYPES:
            raise MethodIncompatibility(
                f"svrepdesign: unknown type={type!r}.",
                recovery_hint="Use one of "
                + ", ".join(sorted(set(_TYPES.values())))
                + ".",
                diagnostics={"type": type},
            )
        self.type = _TYPES[key]
        self.data = data.copy()
        n = len(data)
        self.n = n

        rep, rep_cols = _resolve_repweights(data, repweights)
        R = rep.shape[1]
        if R < 2:
            raise DataInsufficient(
                "svrepdesign: at least two replicate weights are needed.",
                recovery_hint="Pass all replicate-weight columns.",
                diagnostics={"n_replicates": R},
            )
        self.repweight_cols = rep_cols

        # full-sample weights
        if weights is None:
            if not combined_weights:
                raise MethodIncompatibility(
                    "svrepdesign: combined_weights=False needs weights= (the "
                    "replicate columns are then multipliers of them).",
                    recovery_hint="Pass weights=<sampling-weight column>.",
                )
            w = rep.mean(axis=1)
            self._weight_col: Optional[str] = None
            warnings.warn(
                "svrepdesign: no weights= given; the full-sample weights are "
                "taken as the mean of the replicate weights.",
                UserWarning,
                stacklevel=3,
            )
        elif isinstance(weights, str):
            if weights not in data.columns:
                raise MethodIncompatibility(
                    f"svrepdesign: weights column {weights!r} not found.",
                    recovery_hint="Pass the name of the full-sample weight column.",
                )
            w = data[weights].to_numpy(dtype=float)
            self._weight_col = weights
        else:
            w = np.asarray(weights, dtype=float)
            self._weight_col = "__weight__"
            if w.shape[0] != n:
                raise MethodIncompatibility(
                    f"svrepdesign: weights has {w.shape[0]} entries for {n} rows."
                )
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise MethodIncompatibility(
                "svrepdesign: full-sample weights must be finite and positive.",
                recovery_hint="Drop rows with zero or missing weight.",
            )
        if not combined_weights:
            rep = rep * w[:, None]
        if not np.all(np.isfinite(rep)) or np.any(rep < 0):
            raise MethodIncompatibility(
                "svrepdesign: replicate weights must be finite and non-negative.",
                recovery_hint="Check the replicate-weight columns for missing "
                "or negative values.",
            )
        self.weights = w
        self.repweights = rep
        self.combined_weights = bool(combined_weights)
        self.rho = None if rho is None else float(rho)

        # scale / rscales / mse defaults, as svrepdesign reports them
        sc, rs, mse_default = _defaults(self.type, R, self.rho, scale, rscales)
        self.scale = float(sc)
        self.rscales = np.asarray(rs, dtype=float)
        if self.rscales.shape != (R,):
            raise MethodIncompatibility(
                f"svrepdesign: rscales has {self.rscales.shape[0]} entries for "
                f"{R} replicates.",
                recovery_hint="Give one rscale per replicate-weight column.",
            )
        self.mse = mse_default if mse is None else bool(mse)
        if df is None:
            df = float(np.linalg.matrix_rank(rep, tol=_rank_tol(rep)) - 1)
        self.df = float(df)

    # ------------------------------------------------------------------ #
    @property
    def n_replicates(self) -> int:
        """Number of replicate-weight columns."""
        return int(self.repweights.shape[1])

    def all_weights(self) -> np.ndarray:
        """``(n, 1 + R)``: the full-sample weights, then the replicates."""
        return np.column_stack([self.weights, self.repweights])

    def variance(self, theta_full: Any, theta_reps: Any) -> np.ndarray:
        """Replicate covariance of a statistic.

        Parameters
        ----------
        theta_full : (p,) array
        theta_reps : (R, p) array
            The statistic recomputed with each replicate's weights.
        """
        full = np.atleast_1d(np.asarray(theta_full, dtype=float))
        reps = np.asarray(theta_reps, dtype=float).reshape(self.n_replicates, -1)
        if not np.all(np.isfinite(reps)):
            bad = int((~np.isfinite(reps).all(axis=1)).sum())
            raise MethodIncompatibility(
                f"replicate variance: {bad} replicate estimates are not finite "
                "(the statistic could not be computed on those replicates, "
                "e.g. an empty arm or a separated logit).",
                recovery_hint="Check that every replicate keeps both arms / "
                "every category; coarsen the model if needed.",
                diagnostics={"n_bad_replicates": bad},
            )
        centre = full if self.mse else reps.mean(axis=0)
        dev = reps - centre[None, :]
        V: np.ndarray = self.scale * (dev * self.rscales[:, None]).T @ dev
        return V

    def info(self) -> Dict[str, Any]:
        """Summary recorded in a result's ``model_info['survey_design']``."""
        return {
            "n": int(self.n),
            "replicate_type": self.type,
            "n_replicates": self.n_replicates,
            "scale": self.scale,
            "mse": self.mse,
            "rho": self.rho,
            "df": self.df,
            "weights": self._weight_col,
            "repweights": (
                list(self.repweight_cols) if self.repweight_cols is not None else None
            ),
        }

    # -- convenience methods (as SurveyDesign) ----------------------------
    def mean(self, variables: Union[str, List[str]], **kwargs: Any) -> Any:
        """Replicate-weight :func:`sp.svymean`."""
        from .estimators import svymean

        return svymean(variables, self, **kwargs)

    def total(self, variables: Union[str, List[str]], **kwargs: Any) -> Any:
        """Replicate-weight :func:`sp.svytotal`."""
        from .estimators import svytotal

        return svytotal(variables, self, **kwargs)

    def glm(self, formula: str, family: str = "gaussian", **kwargs: Any) -> Any:
        """Replicate-weight :func:`sp.svyglm`."""
        from .estimators import svyglm

        return svyglm(formula, self, family=family, **kwargs)

    def __repr__(self) -> str:
        return (
            f"ReplicateDesign(n={self.n}, type={self.type!r}, "
            f"replicates={self.n_replicates}, scale={self.scale:.6g}, "
            f"mse={self.mse}, df={self.df:g})"
        )


def is_replicate_design(design: Any) -> bool:
    """True for a :class:`ReplicateDesign`."""
    return bool(getattr(design, "is_replicate_design", False))


def _rank_tol(rep: np.ndarray) -> float:
    # QR-rank analogue: singular values below 1e-7 of the largest count as 0
    s = np.linalg.svd(rep, compute_uv=False)
    return float(s.max() * 1e-7) if s.size else 0.0


def _resolve_repweights(data: pd.DataFrame, repweights: Any) -> Any:
    if isinstance(repweights, str):
        if repweights in data.columns:
            cols = [repweights]
        else:
            pat = re.compile(repweights)
            cols = [c for c in data.columns if pat.search(str(c))]
            if not cols:
                raise MethodIncompatibility(
                    f"svrepdesign: no column matches repweights={repweights!r}.",
                    recovery_hint="Pass the list of replicate-weight columns or "
                    "a regular expression that matches them (e.g. '^repw').",
                )
        return data[cols].to_numpy(dtype=float), cols
    if isinstance(repweights, (list, tuple, pd.Index)) and all(
        isinstance(c, str) for c in repweights
    ):
        missing = [c for c in repweights if c not in data.columns]
        if missing:
            raise MethodIncompatibility(
                f"svrepdesign: replicate-weight columns not found: {missing}.",
                recovery_hint="Check the column names.",
            )
        cols = list(repweights)
        return data[cols].to_numpy(dtype=float), cols
    arr = np.asarray(repweights, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != len(data):
        raise MethodIncompatibility(
            "svrepdesign: repweights must be column names, a regex, or an "
            f"(n, R) array with n = {len(data)} rows.",
            recovery_hint="Pass the replicate-weight columns.",
        )
    return arr, None


def _defaults(
    type_: str,
    R: int,
    rho: Optional[float],
    scale: Optional[float],
    rscales: Optional[Sequence[float]],
) -> Any:
    """``(scale, rscales, mse)`` of svrepdesign for a replicate type."""
    ones = np.ones(R)
    fixed = {"BRR", "Fay", "ACS", "successive-difference", "JK2"}
    if type_ in fixed and (scale is not None or rscales is not None):
        warnings.warn(
            f"svrepdesign: scale / rscales are fixed for type={type_!r} and "
            "were ignored (as R svrepdesign does).",
            UserWarning,
            stacklevel=4,
        )
        scale = rscales = None
    if type_ == "BRR":
        return 1.0 / R, ones, False
    if type_ == "Fay":
        if rho is None or not (0.0 <= rho < 1.0):
            raise MethodIncompatibility(
                "svrepdesign: type='Fay' needs rho in [0, 1).",
                recovery_hint="Pass rho= (the Fay coefficient, e.g. 0.5).",
            )
        return 1.0 / (R * (1.0 - rho) ** 2), ones, False
    if type_ in ("ACS", "successive-difference"):
        return 4.0 / R, ones, True
    if type_ == "JK2":
        return 1.0, ones, False
    rs = ones if rscales is None else np.asarray(rscales, dtype=float)
    if type_ == "JKn" and rscales is None:
        raise MethodIncompatibility(
            "svrepdesign: type='JKn' needs rscales= ((n_h - 1) / n_h for the "
            "stratum each replicate deletes from), which the replicate "
            "weights alone do not reveal.",
            recovery_hint="Pass rscales=, or build the weights with "
            "sp.as_svrepdesign(design, type='JKn').",
        )
    if scale is not None:
        return float(scale), rs, False
    if type_ == "JK1":
        return (R - 1.0) / R, rs, False
    if type_ == "bootstrap":
        return 1.0 / (R - 1.0), rs, False
    return 1.0, rs, False  # JKn, other


def svrepdesign(
    data: pd.DataFrame,
    repweights: Any,
    weights: Union[str, np.ndarray, None] = None,
    type: str = "JK1",
    rho: Optional[float] = None,
    scale: Optional[float] = None,
    rscales: Optional[Sequence[float]] = None,
    mse: Optional[bool] = None,
    combined_weights: bool = True,
    df: Optional[float] = None,
) -> ReplicateDesign:
    """
    Declare a survey design by its replicate weights.

    R ``survey::svrepdesign`` / Stata ``svyset [pw=w], brrweight(...)``,
    ``jkrweight(...)``, ``bsrweight(...)``, ``sdrweight(...)``.

    Parameters
    ----------
    data : pd.DataFrame
        Survey microdata.
    repweights : list of str, str or (n, R) array
        The replicate-weight columns, a regular expression matching their
        names (``'^repwt'``), or the matrix itself.
    weights : str or array, optional
        Full-sample sampling weights. Without it they are the mean of the
        replicate weights (with a warning).
    type : {'BRR', 'Fay', 'JK1', 'JKn', 'JK2', 'bootstrap', 'ACS', \
'successive-difference', 'other'}
        Replicate method; sets ``scale`` / ``rscales`` / ``mse`` as R
        ``svrepdesign`` does: BRR ``1/R``; Fay ``1/(R (1 - rho)^2)``; JK1
        ``(R - 1)/R`` unless given; JKn ``1`` with ``rscales`` required;
        JK2 ``1``; bootstrap ``1/(R - 1)``; ACS / successive-difference
        ``4/R`` with ``mse=True``; other ``1``.
    rho : float, optional
        Fay coefficient (``type='Fay'``).
    scale, rscales : optional
        Overall and per-replicate multipliers of the squared deviations
        (ignored, with a warning, for the types that fix them).
    mse : bool, optional
        True: deviations from the full-sample estimate (Stata ``mse``);
        False: from the mean of the replicates. Default False, True for
        ACS / successive difference.
    combined_weights : bool, default True
        The replicate columns already include the sampling weights (the
        usual case). False: they are multipliers of ``weights``.
    df : float, optional
        Design degrees of freedom; default the numerical rank of the
        replicate weights minus one (R ``degf``).

    Returns
    -------
    ReplicateDesign
        Usable wherever ``design=`` is accepted by an estimator with a
        ``weights=`` argument (the statistic is recomputed with every
        replicate's weights), and by :func:`sp.svymean`,
        :func:`sp.svytotal`, :func:`sp.svyglm`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"y": rng.normal(size=100), "w": 1.0})
    >>> for r in range(10):
    ...     df[f"rw{r}"] = np.where(np.arange(100) // 10 == r, 0.0, 10 / 9)
    >>> rep = sp.svrepdesign(df, repweights="^rw", weights="w", type="JK1")
    >>> rep.n_replicates, rep.df
    (10, 9.0)
    """
    return ReplicateDesign(
        data,
        repweights,
        weights=weights,
        type=type,
        rho=rho,
        scale=scale,
        rscales=rscales,
        mse=mse,
        combined_weights=combined_weights,
        df=df,
    )


def _hadamard(order: int) -> np.ndarray:
    """Sylvester Hadamard matrix of ``order`` (a power of two)."""
    H = np.array([[1.0]])
    while H.shape[0] < order:
        H = np.block([[H, H], [H, -H]])
    return H


def as_svrepdesign(
    design: Any,
    type: str = "auto",
    rho: float = 0.0,
    mse: Optional[bool] = None,
) -> ReplicateDesign:
    """
    Replicate weights from a strata / PSU design (R ``as.svrepdesign``).

    Parameters
    ----------
    design : SurveyDesign
        Built with :func:`sp.svydesign`; not calibrated.
    type : {'auto', 'JK1', 'JKn', 'BRR', 'Fay'}
        ``'auto'``: JKn with strata, JK1 without.

        * ``JK1`` (unstratified): one replicate per PSU, the PSU deleted
          and the others scaled by ``R/(R-1)``; ``scale = (R-1)/R (1-f)``.
        * ``JKn``: one replicate per PSU, deleting it and scaling the rest
          of its stratum by ``n_h/(n_h-1)``; ``rscales = (n_h-1)/n_h
          (1-f_h)``. Every stratum needs two or more PSUs.
        * ``BRR`` / ``Fay``: exactly two PSUs per stratum; half-samples
          from the columns of a Sylvester Hadamard matrix (order the
          smallest power of two above the number of strata), weights
          ``2 / 0`` (BRR) or ``2 - rho / rho`` (Fay). The half-samples are
          fully balanced but not R's: for linear statistics the variance is
          the same; for others it can differ at the order of the
          replication noise. To reproduce R exactly, export R's replicate
          weights and use :func:`sp.svrepdesign`.

        Bootstrap weights are random and are not built here: pass weights
        produced elsewhere to :func:`sp.svrepdesign` with
        ``type='bootstrap'``.
    rho : float, default 0
        Fay coefficient for ``type='Fay'``.
    mse : bool, optional
        As :func:`sp.svrepdesign`.

    Returns
    -------
    ReplicateDesign

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> df = pd.DataFrame({"h": np.repeat(np.arange(4), 30),
    ...                    "psu": np.repeat(np.arange(12), 10),
    ...                    "w": rng.uniform(1, 2, 120),
    ...                    "y": rng.normal(size=120)})
    >>> des = sp.svydesign(df, weights="w", strata="h", cluster="psu")
    >>> sp.as_svrepdesign(des, type="JKn").n_replicates
    12
    """
    from .design import SurveyDesign

    if not isinstance(design, SurveyDesign):
        raise MethodIncompatibility(
            "as_svrepdesign: design must be a SurveyDesign from sp.svydesign.",
            recovery_hint="Build the design with sp.svydesign(...) first.",
        )
    if design._calibration is not None:
        raise MethodIncompatibility(
            "as_svrepdesign: a calibrated design cannot be converted (the "
            "calibration would have to be repeated on every replicate).",
            recovery_hint="Convert the uncalibrated design.",
        )
    t = str(type).lower()
    strata = design._strata_codes
    psu = design._psu_codes
    H = int(strata.max()) + 1
    if t == "auto":
        t = "jkn" if H > 1 else "jk1"
    w = design.weights
    f = (
        np.zeros(design.n)
        if design.fpc_values is None
        else np.asarray(design.fpc_values, dtype=float)
    )
    psu_ids = np.unique(psu)
    # stratum and first-row fpc of each PSU
    first = np.array([np.flatnonzero(psu == p)[0] for p in psu_ids])
    psu_stratum = strata[first]
    psu_f = f[first]

    if t == "jk1":
        if H > 1:
            raise MethodIncompatibility(
                "as_svrepdesign: JK1 is for unstratified designs; this one "
                f"has {H} strata.",
                recovery_hint="Use type='JKn'.",
            )
        R = len(psu_ids)
        if R < 2:
            raise DataInsufficient("as_svrepdesign: JK1 needs two or more PSUs.")
        rep = np.repeat((w * R / (R - 1.0))[:, None], R, axis=1)
        for r, p in enumerate(psu_ids):
            rep[psu == p, r] = 0.0
        scale = (R - 1.0) / R * (1.0 - float(psu_f[0]))
        return ReplicateDesign(
            design.data,
            rep,
            weights=w,
            type="JK1",
            scale=scale,
            mse=mse,
        )
    if t == "jkn":
        n_h = np.bincount(psu_stratum, minlength=H)
        if (n_h < 2).any():
            raise DataInsufficient(
                "as_svrepdesign: JKn needs two or more PSUs in every stratum; "
                f"{int((n_h < 2).sum())} strata have one.",
                recovery_hint="Collapse the single-PSU strata first.",
            )
        R = len(psu_ids)
        rep = np.repeat(w[:, None], R, axis=1)
        rscales = np.empty(R)
        for r, p in enumerate(psu_ids):
            h = psu_stratum[r]
            in_h = strata == h
            rep[in_h, r] = w[in_h] * n_h[h] / (n_h[h] - 1.0)
            rep[psu == p, r] = 0.0
            rscales[r] = (n_h[h] - 1.0) / n_h[h] * (1.0 - float(psu_f[r]))
        return ReplicateDesign(
            design.data,
            rep,
            weights=w,
            type="JKn",
            rscales=rscales,
            mse=mse,
        )
    if t in ("brr", "fay"):
        n_h = np.bincount(psu_stratum, minlength=H)
        if not (n_h == 2).all():
            raise MethodIncompatibility(
                "as_svrepdesign: BRR / Fay need exactly two PSUs per stratum.",
                recovery_hint="Use type='JKn' for other designs.",
                diagnostics={"psus_per_stratum": sorted(set(n_h.tolist()))},
            )
        rho_v = 0.0 if t == "brr" else float(rho)
        if not (0.0 <= rho_v < 1.0):
            raise MethodIncompatibility("as_svrepdesign: rho must be in [0, 1).")
        order = 1
        while order < H + 1:
            order *= 2
        hd = _hadamard(order)[:, 1 : H + 1]  # (R, H), drop the ones column
        R = order
        rep = np.empty((design.n, R))
        # the first PSU of each stratum (in PSU-code order) is "half one"
        first_psu = np.array([psu_ids[psu_stratum == h][0] for h in range(H)])
        is_first = psu == first_psu[strata]
        for r in range(R):
            s = hd[r, strata]  # +1 / -1 per row
            up = np.where(is_first, s > 0, s < 0)
            rep[:, r] = w * np.where(up, 2.0 - rho_v, rho_v)
        return ReplicateDesign(
            design.data,
            rep,
            weights=w,
            type="Fay" if t == "fay" else "BRR",
            rho=rho_v if t == "fay" else None,
            mse=mse,
        )
    raise MethodIncompatibility(
        f"as_svrepdesign: type={type!r} is not built here.",
        recovery_hint="Use 'JK1', 'JKn', 'BRR' or 'Fay'; for bootstrap "
        "weights pass them to sp.svrepdesign(type='bootstrap').",
    )
