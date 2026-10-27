"""Exact Shapley and Owen decomposition of the OLS R-squared.

The R-squared of a linear regression is split across its regressors (or
across groups of regressors) by treating it as a cooperative game: the
players are the regressors, and the value of a coalition ``S`` is the
R-squared of the regression on ``S``.  The Shapley value of a player is
its marginal contribution averaged over every order in which the players
can enter [@shorrocks2013decomposition; @lipovetsky2001analysis].  The
contributions add up to the full-model R-squared exactly.

With ``groups=`` the players are the groups, as in Stata ``shapley2``'s
``group()`` option and R ``domir``'s ``.set``.  Each regressor inside a
group then also gets its Owen value: the group's Shapley value is shared
among its members by a second Shapley game played inside the group, with
the other groups entering as blocks [@huettner2012axiomatic].  The Owen
values of a group sum to the group's Shapley value.

The coalition values are computed exactly over all ``2**k`` subsets.  To
keep that cheap the regressors are centred (and weighted) once and
reduced by a QR factorisation to a ``k x k`` triangle; the R-squared of a
subset is then the squared norm of a projection inside that triangle,
computed by a batched QR rather than by normal equations.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..exceptions import (
    ColumnNotFound,
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)
from ._results import DecompResultMixin

__all__ = ["shapley_r2", "ShapleyR2Result"]

#: Largest number of regressors whose ``2**k`` subsets are enumerated.  It
#: is also Stata ``shapley2``'s limit without its ``force`` option.
MAX_PLAYERS = 20

#: Subsets of one size handled per batched QR call (memory bound).
_CHUNK = 16384


# ════════════════════════════════════════════════════════════════════════
# Result
# ════════════════════════════════════════════════════════════════════════


@dataclass
class ShapleyR2Result(DecompResultMixin):
    """Output of :func:`shapley_r2`.

    Attributes
    ----------
    r2 : float
        R-squared of the full model (fixed controls plus every regressor).
    r2_fixed : float
        R-squared of the fixed controls alone (0 when there are none).
    decomposed : float
        ``r2 - r2_fixed``: the part split across the players.  The
        ``contribution`` column of :attr:`detailed` sums to it exactly.
    detailed : pandas.DataFrame
        One row per player (regressor, or group when ``groups=`` is
        given): ``player``, ``members``, ``contribution`` (R-squared
        units), ``share`` (contribution / decomposed) and
        ``pct_of_r2`` (100 x contribution / r2).
    owen : pandas.DataFrame or None
        With ``groups=``: one row per regressor with its ``group``, its
        Owen value ``contribution`` and ``share`` of the decomposed
        R-squared.  ``None`` without groups (each regressor is its own
        player, so the Owen value equals the Shapley value).
    n_obs : int
        Rows used after listwise deletion.
    n_regressions : int
        Number of coalition values computed (subsets enumerated).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    >>> df["y"] = df.a + 0.5 * df.b + rng.normal(size=200)
    >>> res = sp.shapley_r2(df, y="y", x=["a", "b", "c"])
    >>> bool(abs(res.detailed.contribution.sum() - res.r2) < 1e-12)
    True
    """

    method_name: ClassVar[str] = "Shapley decomposition of R-squared"
    bib_keys: ClassVar[Tuple[str, ...]] = (
        "shorrocks2013decomposition",
        "lipovetsky2001analysis",
        "huettner2012axiomatic",
    )
    _citation_keys: ClassVar[Tuple[str, ...]] = bib_keys

    r2: float
    r2_fixed: float
    decomposed: float
    detailed: pd.DataFrame
    owen: Optional[pd.DataFrame]
    n_obs: int
    n_regressions: int
    y: str = ""
    covariates_fixed: Tuple[str, ...] = ()
    weights: Optional[str] = None
    groups: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # -- views ---------------------------------------------------------- #
    @property
    def shapley(self) -> pd.DataFrame:
        """Alias of :attr:`detailed` (the per-player table)."""
        return self.detailed

    def to_frame(self) -> pd.DataFrame:
        """The per-player table (a copy)."""
        return self.detailed.copy()

    def tidy(self) -> pd.DataFrame:
        """Per-player table with broom-style column names."""
        out = self.detailed.rename(
            columns={"player": "term", "contribution": "estimate"}
        )
        return out[["term", "estimate", "share", "pct_of_r2"]].copy()

    # -- display -------------------------------------------------------- #
    def summary(self) -> str:
        lines = [
            "━" * 64,
            "  Shapley decomposition of R-squared",
            "━" * 64,
            f"  Dependent variable : {self.y}",
            f"  Observations       : {self.n_obs}",
            f"  Weights            : {self.weights or 'none'}",
            f"  R-squared (full)   : {self.r2:.6f}",
        ]
        if self.covariates_fixed:
            lines.append(
                f"  R-squared (fixed)  : {self.r2_fixed:.6f}  "
                f"[{', '.join(self.covariates_fixed)}]"
            )
            lines.append(f"  Decomposed         : {self.decomposed:.6f}")
        lines.append(f"  Subsets enumerated : {self.n_regressions}")
        lines.append("")
        tab = self.detailed[["player", "contribution", "share"]].copy()
        lines.append(tab.round(6).to_string(index=False))
        if self.owen is not None:
            lines.extend(["", "  Owen values (within groups):"])
            lines.append(
                self.owen[["variable", "group", "contribution", "share"]]
                .round(6)
                .to_string(index=False)
            )
        lines.append("━" * 64)
        text = "\n".join(lines)
        print(text)
        return text

    def plot(self, **kwargs: Any) -> Any:
        from .plots import detailed_waterfall

        return detailed_waterfall(
            self.detailed, value_col="contribution", label_col="player", **kwargs
        )

    def to_latex(self) -> str:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Shapley decomposition of $R^2$}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Regressor & Contribution & Share \\",
            r"\midrule",
        ]
        for _, row in self.detailed.iterrows():
            name = str(row["player"]).replace("_", r"\_")
            lines.append(
                f"{name} & {row['contribution']:.4f} & {100 * row['share']:.1f}\\% \\\\"
            )
        lines.extend(
            [
                r"\midrule",
                f"Total & {self.decomposed:.4f} & 100.0\\% \\\\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        html: str = self.detailed.round(6).to_html(index=False)
        return (
            "<div style='font-family:monospace;'>"
            "<h3>Shapley decomposition of R-squared</h3>"
            f"<p>R2 = {self.r2:.6f}</p>" + html + "</div>"
        )

    def __repr__(self) -> str:
        return (
            f"ShapleyR2Result(r2={self.r2:.6f}, decomposed={self.decomposed:.6f}, "
            f"players={len(self.detailed)}, n_obs={self.n_obs})"
        )


# ════════════════════════════════════════════════════════════════════════
# Internals
# ════════════════════════════════════════════════════════════════════════


def _popcount(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.int64)
    out = np.zeros_like(a)
    while np.any(a):
        out += a & 1
        a = a >> 1
    return out


def _shapley_weights(p: int) -> np.ndarray:
    """``w[s] = s! (p-s-1)! / p!`` for s = 0..p-1."""
    return np.array([1.0 / (p * math.comb(p - 1, s)) for s in range(p)])


class _SubsetR2:
    """Coalition values ``v(S) = R2(fixed + S) - R2(fixed)``.

    ``R`` is the ``k x k`` triangle of the weighted, centred, fixed-
    partialled regressors and ``b = Q'y``; the explained sum of squares
    of a subset ``S`` is ``|| proj_{R[:, S]} b ||^2``.
    """

    def __init__(self, R: np.ndarray, b: np.ndarray, tss: float) -> None:
        self.R = R
        self.b = b
        self.tss = tss
        self.k = R.shape[1]

    def values(self, masks: np.ndarray) -> np.ndarray:
        masks = np.asarray(masks, dtype=np.int64)
        out = np.zeros(len(masks))
        sizes = _popcount(masks)
        shifts = np.arange(self.k)[None, :]
        for m in np.unique(sizes):
            if m == 0:
                continue
            idx = np.flatnonzero(sizes == m)
            for start in range(0, len(idx), _CHUNK):
                sub = idx[start : start + _CHUNK]
                bits = (masks[sub, None] >> shifts) & 1
                # column indices of each subset, in increasing order
                cols = np.nonzero(bits)[1].reshape(len(sub), m)
                A = self.R[:, cols].transpose(1, 0, 2)  # (N, k, m)
                Q = np.linalg.qr(A, mode="reduced")[0]  # (N, k, m)
                proj = np.einsum("nkm,k->nm", Q, self.b)
                out[sub] = np.einsum("nm,nm->n", proj, proj)
        return out / self.tss


def _prepare(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    fixed: List[str],
    weights: Optional[Union[str, np.ndarray, pd.Series]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Optional[str]]:
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "shapley_r2 needs a pandas DataFrame.",
            recovery_hint="Pass data as a DataFrame with the y / x columns.",
            diagnostics={"type": type(data).__name__},
        )
    wname: Optional[str] = None
    cols = [y] + x + fixed
    if isinstance(weights, str):
        wname = weights
        cols = cols + [weights]
    missing = [c for c in dict.fromkeys(cols) if c not in data.columns]
    if missing:
        raise ColumnNotFound.from_missing(missing, list(data.columns))
    frame = data[list(dict.fromkeys(cols))].copy()
    if weights is not None and not isinstance(weights, str):
        w_arr = np.asarray(weights, dtype=float).ravel()
        if len(w_arr) != len(data):
            raise MethodIncompatibility(
                "weights array length does not match data.",
                recovery_hint="Pass one weight per row of data, or a column name.",
                diagnostics={"n_weights": len(w_arr), "n_rows": len(data)},
            )
        frame["__w__"] = w_arr
        wname = None
        wcol = "__w__"
    else:
        wcol = wname
    try:
        num = frame.apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise MethodIncompatibility(
            f"shapley_r2 needs numeric columns ({exc}).",
            recovery_hint=(
                "Encode categorical regressors as dummies and pass the dummy "
                "set as one entry of groups= so they enter together."
            ),
        ) from exc
    keep = np.isfinite(num.to_numpy()).all(axis=1)
    n_dropped = int((~keep).sum())
    num = num.loc[keep]
    yv = num[y].to_numpy()
    Xv = num[x].to_numpy()
    Fv = num[fixed].to_numpy() if fixed else np.empty((len(num), 0))
    if wcol is not None:
        wv = num[wcol].to_numpy()
        if np.any(wv <= 0):
            raise MethodIncompatibility(
                "weights must be strictly positive.",
                recovery_hint="Drop rows with zero or negative weight before the call.",
                diagnostics={"n_nonpositive": int((wv <= 0).sum())},
            )
    else:
        wv = np.ones(len(num))
    return yv, Xv, Fv, wv, n_dropped, wname


def _resolve_groups(
    x: List[str], groups: Optional[Mapping[str, Sequence[str]]]
) -> Tuple[List[str], List[List[int]], bool]:
    """Return player names and the regressor indices of each player."""
    if groups is None:
        return list(x), [[j] for j in range(len(x))], False
    if not isinstance(groups, Mapping) or not groups:
        raise MethodIncompatibility(
            "groups must be a non-empty dict {group_name: [regressors]}.",
            recovery_hint="e.g. groups={'income': ['gdp', 'gdp2'], 'region': [...]}",
        )
    pos = {v: j for j, v in enumerate(x)}
    seen: Dict[str, str] = {}
    names: List[str] = []
    members: List[List[int]] = []
    for gname, gvars in groups.items():
        gvars = [gvars] if isinstance(gvars, str) else list(gvars)
        if not gvars:
            raise MethodIncompatibility(
                f"group {gname!r} is empty.",
                recovery_hint="Every group needs at least one regressor.",
            )
        idx = []
        for v in gvars:
            if v not in pos:
                raise MethodIncompatibility(
                    f"group {gname!r} lists {v!r}, which is not in x.",
                    recovery_hint="List every grouped regressor in x as well.",
                    diagnostics={"group": gname, "variable": v},
                )
            if v in seen:
                raise MethodIncompatibility(
                    f"{v!r} is in two groups ({seen[v]!r} and {gname!r}).",
                    recovery_hint="Groups must not overlap.",
                )
            seen[v] = str(gname)
            idx.append(pos[v])
        if str(gname) in x and str(gname) not in gvars:
            raise MethodIncompatibility(
                f"group name {gname!r} clashes with a regressor name.",
                recovery_hint="Rename the group.",
            )
        names.append(str(gname))
        members.append(idx)
    for v in x:  # regressors outside every group play alone
        if v not in seen:
            names.append(v)
            members.append([pos[v]])
    return names, members, True


# ════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════


def shapley_r2(
    data: pd.DataFrame,
    y: str,
    x: Sequence[str],
    *,
    groups: Optional[Mapping[str, Sequence[str]]] = None,
    weights: Optional[Union[str, np.ndarray, pd.Series]] = None,
    covariates_fixed: Optional[Sequence[str]] = None,
) -> ShapleyR2Result:
    """Exact Shapley (and Owen) decomposition of an OLS R-squared.

    Every subset of the regressors (or of the groups) is fitted, and the
    R-squared is split into per-regressor contributions that are the
    regressors' Shapley values in the R-squared game: the increase in
    R-squared a regressor brings, averaged over every order in which the
    regressors can be added [@shorrocks2013decomposition;
    @lipovetsky2001analysis].  This is the same quantity as R
    ``relaimpo``'s LMG measure and ``domir``'s general dominance, and
    Stata ``shapley2``'s ``stat(r2)``.  Contributions sum to the
    full-model R-squared exactly.

    Parameters
    ----------
    data : pandas.DataFrame
        Analysis data.  Rows with a missing value in any used column are
        dropped (listwise).
    y : str
        Dependent variable.
    x : sequence of str
        Regressors whose contributions are computed.  At most 20 (the
        ``2**k`` subsets are enumerated exactly); with ``groups=`` the
        limit applies to the number of groups, and the within-group Owen
        values are computed when the total number of regressors is also
        at most 20.
    groups : dict, optional
        ``{group_name: [regressors]}``.  Each group is one player (Stata
        ``shapley2 , group()``, R ``domir(.set=)``); regressors of ``x``
        not in any group play alone.  Groups must not overlap.  The
        result then also carries the Owen value of each regressor inside
        its group [@huettner2012axiomatic].
    weights : str or array-like, optional
        Analytic weights (column name or one value per row of ``data``),
        strictly positive.  R-squared is the weighted one,
        ``1 - sum w e^2 / sum w (y - ybar_w)^2``, as in ``lm(weights=)``
        and Stata ``regress [aweight=]``.
    covariates_fixed : sequence of str, optional
        Controls included in every regression (R ``domir(.all=)``,
        ``relaimpo``'s ``always=``).  Their own R-squared ``r2_fixed`` is
        not decomposed; the players share ``r2 - r2_fixed``.  Dividing a
        contribution by ``1 - r2_fixed`` gives its share of the partial
        R-squared after the controls are partialled out.

    Returns
    -------
    ShapleyR2Result
        ``r2``, ``r2_fixed``, ``decomposed`` and the per-player table
        ``detailed`` (``contribution``, ``share``, ``pct_of_r2``); with
        groups also the per-regressor Owen table ``owen``.

    Raises
    ------
    MethodIncompatibility
        More than 20 players, overlapping or unknown group members,
        non-numeric columns, non-positive weights.
    DataInsufficient
        Fewer observations than parameters, or a constant dependent
        variable.
    NumericalInstability
        Collinear regressors (after the fixed controls are partialled
        out): the R-squared game is then not well defined per regressor.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> df = pd.DataFrame(rng.normal(size=(300, 4)), columns=["a", "b", "c", "d"])
    >>> df["y"] = df.a + df.b + 0.5 * df.c + rng.normal(size=300)
    >>> res = sp.shapley_r2(df, y="y", x=["a", "b", "c", "d"])
    >>> round(float(res.detailed.contribution.sum() - res.r2), 12)
    0.0
    >>> g = sp.shapley_r2(df, y="y", x=["a", "b", "c", "d"],
    ...                   groups={"ab": ["a", "b"]})
    >>> list(g.detailed.player)
    ['ab', 'c', 'd']
    >>> bool(np.isclose(g.owen.query("group == 'ab'").contribution.sum(),
    ...                 g.detailed.contribution.iloc[0]))
    True

    References
    ----------
    [@shorrocks2013decomposition; @lipovetsky2001analysis]
    [@huettner2012axiomatic]
    """
    if isinstance(x, str):
        x = [x]
    x = list(x)
    fixed = (
        []
        if covariates_fixed is None
        else list(
            [covariates_fixed]
            if isinstance(covariates_fixed, str)
            else covariates_fixed
        )
    )
    if not x:
        raise MethodIncompatibility(
            "x must name at least one regressor.",
            recovery_hint="Pass the regressors to decompose as x=[...].",
        )
    dup = sorted({v for v in x if x.count(v) > 1})
    if dup:
        raise MethodIncompatibility(
            f"x lists {dup} more than once.",
            recovery_hint="Pass each regressor once.",
        )
    clash = sorted((set(x) | set(fixed)) & {y}) + sorted(set(x) & set(fixed))
    if clash:
        raise MethodIncompatibility(
            f"{clash} appear in more than one of y / x / covariates_fixed.",
            recovery_hint="Each column can play one role only.",
        )

    names, members, grouped = _resolve_groups(x, groups)
    p = len(names)
    k = len(x)
    if p > MAX_PLAYERS:
        raise MethodIncompatibility(
            f"{p} players: exact enumeration needs 2**{p} regressions, more "
            f"than the 2**{MAX_PLAYERS} this function allows.",
            recovery_hint=(
                "Combine regressors into groups= (each group is one player), "
                "move nuisance controls to covariates_fixed=, or drop regressors."
            ),
            diagnostics={"n_players": p, "max_players": MAX_PLAYERS},
            alternative_functions=["sp.shapley_inequality"],
        )

    yv, Xv, Fv, wv, n_dropped, wname = _prepare(data, y, x, fixed, weights)
    n = len(yv)
    n_par = 1 + k + len(fixed)
    if n <= n_par:
        raise DataInsufficient(
            f"{n} complete observations for {n_par} parameters.",
            recovery_hint="Use more data or fewer regressors.",
            diagnostics={"n_obs": n, "n_params": n_par, "n_dropped": n_dropped},
        )

    # weighted centring (the intercept), then sqrt-weight scaling
    sw = np.sqrt(wv / wv.mean())
    mu = lambda a: (wv @ a) / wv.sum()  # noqa: E731
    yc = (yv - mu(yv)) * sw
    Xc = (Xv - mu(Xv)) * sw[:, None]
    tss = float(yc @ yc)
    if not tss > 1e-24 * max(1.0, float(np.max(np.abs(yv))) ** 2) * n:
        raise DataInsufficient(
            f"{y!r} is constant in the estimation sample; R-squared is undefined.",
            recovery_hint="Check the dependent variable and the sample.",
            diagnostics={"n_obs": n},
        )
    r2_fixed = 0.0
    if fixed:
        Fc = (Fv - mu(Fv)) * sw[:, None]
        Qf, Rf = np.linalg.qr(Fc)
        dF = np.abs(np.diag(Rf))
        if dF.min() <= 1e-10 * max(dF.max(), 1e-300):
            raise NumericalInstability(
                "covariates_fixed are collinear (with each other or the constant).",
                recovery_hint="Drop the redundant fixed controls.",
                diagnostics={"covariates_fixed": fixed},
            )
        yc = yc - Qf @ (Qf.T @ yc)
        Xc = Xc - Qf @ (Qf.T @ Xc)
        r2_fixed = 1.0 - float(yc @ yc) / tss

    Qx, Rx = np.linalg.qr(Xc)
    dX = np.abs(np.diag(Rx))
    scale = np.sqrt(np.sum(((Xv - mu(Xv)) * sw[:, None]) ** 2, axis=0))
    rel = dX / np.where(scale > 0, scale, 1.0)
    if np.any(scale == 0) or rel.min() <= 1e-10:
        bad = [x[j] for j in np.flatnonzero((scale == 0) | (rel <= 1e-10))]
        raise NumericalInstability(
            f"regressors are collinear{' given covariates_fixed' if fixed else ''}: "
            f"{bad} add (numerically) nothing to the column space.",
            recovery_hint=(
                "Drop the redundant regressor(s), or put collinear regressors "
                "(e.g. a full dummy set) into one entry of groups= after "
                "dropping a reference category."
            ),
            diagnostics={"collinear": bad, "relative_pivots": rel.tolist()},
        )
    game = _SubsetR2(Rx, Qx.T @ yc, tss)

    # -- coalition values over players --------------------------------- #
    pmask = np.array([sum(1 << j for j in mem) for mem in members], dtype=np.int64)
    all_p = np.arange(1 << p, dtype=np.int64)
    var_mask_of_p = np.zeros(1 << p, dtype=np.int64)
    for j in range(p):
        var_mask_of_p |= np.where((all_p >> j) & 1, pmask[j], 0)
    need_owen = grouped and k <= MAX_PLAYERS
    if need_owen:
        v_var = game.values(np.arange(1 << k, dtype=np.int64))
        v_p = v_var[var_mask_of_p]
        n_regs = 1 << k
    else:
        v_p = game.values(var_mask_of_p)
        v_var = None
        n_regs = 1 << p
    full_value = float(v_p[-1])
    r2 = r2_fixed + full_value

    wts = _shapley_weights(p)
    sizes = _popcount(all_p)
    phi = np.zeros(p)
    for j in range(p):
        without = all_p[((all_p >> j) & 1) == 0]
        phi[j] = float(wts[sizes[without]] @ (v_p[without | (1 << j)] - v_p[without]))

    denom = full_value if full_value != 0 else np.nan
    detailed = pd.DataFrame(
        {
            "player": names,
            "members": [", ".join(x[i] for i in mem) for mem in members],
            "contribution": phi,
            "share": phi / denom,
            "pct_of_r2": 100.0 * phi / r2 if r2 != 0 else np.nan,
        }
    )

    owen_df: Optional[pd.DataFrame] = None
    diagnostics: Dict[str, Any] = {
        "n_dropped": n_dropped,
        "efficiency_gap": float(phi.sum() - full_value),
    }
    if grouped and v_var is not None:
        rows = []
        for g, mem in enumerate(members):
            others = [h for h in range(p) if h != g]
            m = len(others)
            wt_top = _shapley_weights(p)
            # coalitions T of the other groups, as variable masks
            tmask = np.arange(1 << m, dtype=np.int64)
            tb = (tmask[:, None] >> np.arange(m)[None, :]) & 1
            q_t = tb @ pmask[others] if m else np.zeros(1, dtype=np.int64)
            w_t = wt_top[_popcount(tmask)] if m else np.ones(1)
            ng = len(mem)
            w_in = _shapley_weights(ng)
            for i in mem:
                rest = [r for r in mem if r != i]
                smask = np.arange(1 << len(rest), dtype=np.int64)
                sb = (smask[:, None] >> np.arange(len(rest))[None, :]) & 1
                s_var = (
                    sb @ np.array([1 << r for r in rest], dtype=np.int64)
                    if rest
                    else np.zeros(1, dtype=np.int64)
                )
                w_s = w_in[_popcount(smask)] if rest else np.ones(1)
                base = q_t[:, None] | s_var[None, :]
                diff = v_var[base | (1 << i)] - v_var[base]
                val = float(w_t @ diff @ w_s)
                rows.append({"variable": x[i], "group": names[g], "contribution": val})
        owen_df = pd.DataFrame(rows)
        owen_df["share"] = owen_df["contribution"] / denom
        gap = (
            owen_df.groupby("group", sort=False)["contribution"]
            .sum()
            .reindex(names)
            .to_numpy()
            - phi
        )
        diagnostics["owen_group_gap"] = float(np.max(np.abs(gap)))
    elif grouped:
        warnings.warn(
            f"{k} regressors in {p} groups: the group Shapley values are exact, "
            f"but the within-group Owen values need 2**{k} regressions and were "
            f"not computed (limit 2**{MAX_PLAYERS}).",
            UserWarning,
            stacklevel=2,
        )
        diagnostics["owen_skipped"] = f"{k} regressors > {MAX_PLAYERS}"

    return ShapleyR2Result(
        r2=float(r2),
        r2_fixed=float(r2_fixed),
        decomposed=full_value,
        detailed=detailed,
        owen=owen_df,
        n_obs=int(n),
        n_regressions=int(n_regs),
        y=y,
        covariates_fixed=tuple(fixed),
        weights=wname if wname is not None else (None if weights is None else "array"),
        groups=(
            {nm: tuple(x[i] for i in mem) for nm, mem in zip(names, members)}
            if grouped
            else {}
        ),
        diagnostics=diagnostics,
    )
