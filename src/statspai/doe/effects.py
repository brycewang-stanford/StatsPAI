"""Analysis of two-level factorial experiments: ``sp.factorial_effects``.

With every factor coded -1 / +1 the least-squares coefficient of a column
is half the difference between the mean response at its high and low
levels, the *effect*. A design without replication leaves no degrees of
freedom for the error variance, so significance is judged from the effects
themselves: most of them are assumed to be noise (effect sparsity) and the
few that stand out from the rest are declared active.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import ColumnNotFound, DataInsufficient, MethodIncompatibility


def lenth_pse(effects: np.ndarray) -> float:
    """Lenth's pseudo standard error of a set of effect estimates."""
    a = np.abs(np.asarray(effects, dtype=float))
    s0 = 1.5 * float(np.median(a))
    keep = a[a < 2.5 * s0]
    return 1.5 * float(np.median(keep)) if keep.size else float("nan")


@lru_cache(maxsize=32)
def _lenth_null(m: int, n_sim: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sorted null draws of ``|effect| / PSE``: all effects, and the row maxima."""
    rng = np.random.default_rng([seed, m])
    a = np.abs(rng.standard_normal((n_sim, m)))
    s0 = 1.5 * np.median(a, axis=1)
    kept = np.where(a < 2.5 * s0[:, None], a, np.nan)
    pse = 1.5 * np.nanmedian(kept, axis=1)
    t = a / pse[:, None]
    return np.sort(t.ravel()), np.sort(t.max(axis=1))


def _tail(sorted_null: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Share of null draws at least as large, with the usual +1 correction."""
    n = sorted_null.size
    above = n - np.searchsorted(sorted_null, value, side="left")
    return np.asarray((above + 1.0) / (n + 1.0))


@dataclass
class FactorialEffectsResult(ResultProtocolMixin):
    """Effects of a two-level factorial experiment.

    Attributes
    ----------
    effects : DataFrame
        Indexed by term. ``effect`` (high minus low), ``coef`` (half of
        it, the regression coefficient on the -1 / +1 column),
        ``half_normal_score``, ``lenth_t``, ``lenth_p``,
        ``lenth_p_simultaneous``, ``active`` (beyond the margin of
        error); and ``se``, ``t``, ``p`` when the design leaves residual
        degrees of freedom.
    intercept : float
    pse, me, sme : float
        Lenth's pseudo standard error, margin of error and simultaneous
        margin of error.
    df_resid : int
    aliases : dict
        Terms that were dropped because their column repeats that of an
        earlier term.
    model_info : dict

    Examples
    --------
    >>> import statspai as sp
    >>> d = sp.factorial_design(3).design
    >>> d["y"] = [504, 984, 928, 808, 992, 784, 464, 976]
    >>> fit = sp.factorial_effects(d, "y")
    >>> fit.effects.shape[0]
    7
    """

    effects: pd.DataFrame
    intercept: float
    pse: float
    me: float
    sme: float
    df_resid: int
    aliases: Dict[str, List[str]] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> List[str]:
        """Terms beyond the margin of error."""
        return [str(i) for i in self.effects.index[self.effects["active"]]]

    def summary(self) -> str:
        info = self.model_info
        lines = [
            "Two-level factorial: effects",
            "=" * 60,
            f"Runs: {info['n_obs']}    Terms: {self.effects.shape[0]}    "
            f"Residual df: {self.df_resid}",
        ]
        cols = ["effect", "coef", "lenth_t", "lenth_p"]
        if self.df_resid > 0:
            cols += ["se", "t", "p"]
        tab = self.effects[cols + ["active"]]
        lines.append(tab.to_string(float_format=lambda v: f"{v:.4g}"))
        lines.append(
            f"Lenth: PSE = {self.pse:.4g}, margin of error = {self.me:.4g}, "
            f"simultaneous = {self.sme:.4g} (level {info['alpha']:g})"
        )
        sep = ":" if any(":" in str(t) for t in self.effects.index) else ""
        size = (lambda t: t.count(":") + 1) if sep else len
        hidden = 0
        for k, v in self.aliases.items():
            low = [a for a in v if size(a) <= 2]
            hidden += len(v) - len(low)
            if low and k != "(Intercept)":
                lines.append(f"Aliased: {k} = " + " = ".join(low))
        if hidden:
            lines.append(
                f"{hidden} aliases with interactions of three or more factors "
                "are listed in .aliases."
            )
        for note in info.get("notes", []):
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def plot(self, ax: Any = None, label: Optional[int] = None, **kwargs: Any) -> Any:
        """Half-normal plot of the absolute effects (Daniel 1959).

        Noise effects fall on a line through the origin; active effects
        lie to the right of it. ``label`` is the number of largest
        effects to name (default: the active ones, at least three).
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 4.5))
        tab = self.effects.assign(abs_effect=self.effects["effect"].abs())
        tab = tab.sort_values("abs_effect")
        ax.scatter(tab["abs_effect"], tab["half_normal_score"], **{"s": 28, **kwargs})
        if np.isfinite(self.pse) and self.pse > 0:
            top = float(tab["half_normal_score"].max())
            ax.plot([0, top * self.pse], [0, top], lw=1, ls="--", color="0.4")
            ax.axvline(self.me, lw=0.8, ls=":", color="0.4")
        k = max(int(tab["active"].sum()), 3) if label is None else int(label)
        for term, row in tab.tail(min(k, tab.shape[0])).iterrows():
            ax.annotate(
                str(term),
                (row["abs_effect"], row["half_normal_score"]),
                textcoords="offset points",
                xytext=(-4, 4),
                ha="right",
            )
        ax.set_xlabel("absolute effect")
        ax.set_ylabel("half-normal quantile")
        return ax


def factorial_effects(
    data: pd.DataFrame,
    y: str,
    factors: Optional[Sequence[str]] = None,
    order: Optional[int] = None,
    alpha: float = 0.05,
    reference: str = "simulated",
    n_sim: int = 20000,
    seed: Optional[int] = 0,
) -> FactorialEffectsResult:
    """Main effects and interactions of a two-level factorial experiment.

    Estimates every effect up to a chosen order and tells which of them
    stand out, including in a design with no replication, where a
    regression has no residual degrees of freedom and reports no
    standard errors.

    Parameters
    ----------
    data : DataFrame
        One row per run.
    y : str
        The response.
    factors : list of str, optional
        The factor columns. Each must take exactly two values; the
        smaller (in sort order) is coded -1 and the larger +1. Default:
        every other column with exactly two values.
    order : int, optional
        Largest interaction fitted. Default: all orders, so that a full
        factorial is fitted with a saturated model.
    alpha : float, default 0.05
        Level of the margins of error.
    reference : {'simulated', 't'}, default 'simulated'
        Null distribution of ``effect / PSE``. ``'simulated'`` draws it
        (all effects null, normal errors), as Lenth's own R package
        ``unrepx`` does; ``'t'`` is the t distribution with ``m / 3``
        degrees of freedom of the 1989 paper, which is conservative: with
        seven effects its margin of error is 1.6 times the simulated one
        and flags 2.1% of null effects at a nominal 5%, where the
        simulated margin flags 5.1%.
    n_sim : int, default 20000
        Simulated sets of effects.
    seed : int, default 0
        For the simulation. Fixed by default so that repeated calls agree;
        ``None`` is read as 0.

    Returns
    -------
    FactorialEffectsResult
        ``effects`` (one row per term), ``pse``, ``me``, ``sme``,
        ``active``, ``aliases``, ``summary()``, ``plot()`` (half-normal).

    Notes
    -----
    ``effect`` is the mean response at the high level minus that at the
    low level, twice the regression coefficient ``coef``.

    Lenth (1989): with ``m`` effects, ``s0 = 1.5 median |effect|`` and
    the pseudo standard error is ``1.5`` times the median of the absolute
    effects below ``2.5 s0``. The margin of error (ME) is the PSE times
    the ``1 - alpha`` quantile of ``|effect| / PSE`` under the null, the
    simultaneous margin (SME) uses the largest of the ``m`` ratios
    instead. ``lenth_p`` and ``lenth_p_simultaneous`` are the matching
    tail probabilities. With ``reference='t'`` the quantiles are those of
    ``t(m / 3)`` at ``1 - alpha / 2`` and at ``gamma = (1 + (1 -
    alpha)^(1/m)) / 2``. The method assumes that only a minority of the
    effects are real; when half of them or more are, the PSE is inflated
    and nothing is found.

    A term whose column equals (up to sign) that of an earlier term is
    not estimable separately: it is dropped and listed in ``aliases``
    under the term that stays. Terms are ordered by interaction order, so
    in a fraction the lower-order member of each alias set is kept.

    When there are residual degrees of freedom (replication, or ``order``
    below the number of factors) the usual ``se``, ``t`` and ``p`` are
    reported next to Lenth's. Centre points are not used: drop those runs
    before calling.

    Examples
    --------
    An unreplicated 2^3 experiment:

    >>> import statspai as sp
    >>> d = sp.factorial_design(3).design
    >>> d["y"] = [60, 72, 54, 68, 52, 83, 45, 80]
    >>> fit = sp.factorial_effects(d, "y")
    >>> round(float(fit.effects.loc["C", "effect"]), 2)
    23.0
    >>> fit.df_resid
    0

    References
    ----------
    lenth1989quick; daniel1959use; box2005statistics
    """
    if y not in data.columns:
        raise ColumnNotFound(f"Response column {y!r} is not in the data.")
    if not 0.0 < alpha < 1.0:
        raise MethodIncompatibility(f"alpha must be in (0, 1); got {alpha}.")
    ref = str(reference).lower()
    if ref not in ("simulated", "t"):
        raise MethodIncompatibility(
            f"reference must be 'simulated' or 't'; got {reference!r}."
        )
    if n_sim < 1000:
        raise MethodIncompatibility("n_sim must be at least 1000.")
    seed = 0 if seed is None else int(seed)
    if factors is None:
        cols = [c for c in data.columns if c != y and data[c].nunique() == 2]
        if not cols:
            raise MethodIncompatibility(
                "No column other than the response takes exactly two values. "
                "If the design has centre points, drop those runs first: the "
                "effects are defined by the two-level runs."
            )
    else:
        cols = list(factors)
        miss = [str(f) for f in cols if f not in data.columns]
        if miss:
            raise ColumnNotFound(f"Not in the data: {', '.join(miss)}.")
    names = [str(c) for c in cols]
    use = data[[y] + cols].dropna()
    yy = use[y].to_numpy(dtype=float)
    n, k = yy.size, len(names)
    coded = np.zeros((n, k))
    level_map: Dict[str, List[Any]] = {}
    for j, (col, nm) in enumerate(zip(cols, names)):
        lv = sorted(pd.unique(use[col]))
        if len(lv) != 2:
            raise MethodIncompatibility(
                f"Factor {nm!r} takes {len(lv)} values; a two-level analysis "
                "needs exactly two. Drop centre points, or fit a regression "
                "with sp.regress."
            )
        coded[:, j] = np.where(use[col].to_numpy() == lv[1], 1.0, -1.0)
        level_map[nm] = [lv[0], lv[1]]
    top = k if order is None else int(order)
    if not 1 <= top <= k:
        raise MethodIncompatibility(f"order must be between 1 and {k}.")
    sep = "" if all(len(nm) == 1 for nm in names) else ":"
    terms: List[str] = []
    columns: List[np.ndarray] = []
    aliases: Dict[str, List[str]] = {}
    notes: List[str] = []
    for o in range(1, top + 1):
        for combo in itertools.combinations(range(k), o):
            col = np.prod(coded[:, list(combo)], axis=1)
            label = sep.join(names[j] for j in combo)
            if np.all(col == col[0]):
                aliases.setdefault("(Intercept)", []).append(label)
                continue
            twin = next(
                (
                    t
                    for t, c in zip(terms, columns)
                    if np.array_equal(c, col) or np.array_equal(c, -col)
                ),
                None,
            )
            if twin is not None:
                aliases.setdefault(twin, []).append(label)
                continue
            terms.append(label)
            columns.append(col)
    X = np.column_stack([np.ones(n)] + columns)
    rank = int(np.linalg.matrix_rank(X))
    if rank < X.shape[1]:
        raise MethodIncompatibility(
            f"The {X.shape[1] - 1} terms are not all estimable from these "
            f"{n} runs (rank {rank}); the design is not a regular two-level "
            "fraction. Lower order=, or fit the model you want with sp.regress."
        )
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    df = n - X.shape[1]
    eff = 2.0 * beta[1:]
    m = eff.size
    if m < 1:
        raise DataInsufficient("No effect is estimable.")
    tab = pd.DataFrame({"effect": eff, "coef": beta[1:]}, index=terms)
    rank_abs = stats.rankdata(np.abs(eff), method="ordinal")
    tab["half_normal_score"] = stats.norm.ppf(0.5 + 0.5 * (rank_abs - 0.5) / m)
    pse = lenth_pse(eff) if m >= 3 else float("nan")
    dfl = m / 3.0
    if np.isfinite(pse) and pse > 0:
        ratio = np.abs(eff) / pse
        tab["lenth_t"] = eff / pse
        if ref == "t":
            gamma = 0.5 * (1.0 + (1.0 - alpha) ** (1.0 / m))
            me = float(stats.t.ppf(1 - alpha / 2, dfl) * pse)
            sme = float(stats.t.ppf(gamma, dfl) * pse)
            p_one = 2.0 * stats.t.sf(ratio, dfl)
            tab["lenth_p"] = p_one
            tab["lenth_p_simultaneous"] = 1.0 - (1.0 - p_one) ** m
        else:
            null_all, null_max = _lenth_null(m, int(n_sim), int(seed))
            me = float(np.quantile(null_all, 1 - alpha) * pse)
            sme = float(np.quantile(null_max, 1 - alpha) * pse)
            tab["lenth_p"] = _tail(null_all, ratio)
            tab["lenth_p_simultaneous"] = _tail(null_max, ratio)
        tab["active"] = np.abs(eff) > me
    else:
        me = sme = float("nan")
        tab["lenth_t"] = np.nan
        tab["lenth_p"] = np.nan
        tab["lenth_p_simultaneous"] = np.nan
        tab["active"] = False
        notes.append(
            "Lenth's method needs at least three effects and a positive "
            "pseudo standard error."
        )
    if df > 0:
        s2 = float(resid @ resid) / df
        cov = s2 * np.linalg.inv(X.T @ X)
        se_coef = np.sqrt(np.diag(cov))[1:]
        tab["se"] = 2.0 * se_coef
        tab["t"] = beta[1:] / se_coef
        tab["p"] = 2.0 * stats.t.sf(np.abs(tab["t"]), df)
    else:
        notes.append(
            "No residual degrees of freedom: significance is judged by "
            "Lenth's method and the half-normal plot."
        )
    return FactorialEffectsResult(
        effects=tab,
        intercept=float(beta[0]),
        pse=float(pse),
        me=me,
        sme=sme,
        df_resid=int(df),
        aliases=aliases,
        model_info={
            "n_obs": int(n),
            "alpha": float(alpha),
            "factors": names,
            "levels": level_map,
            "order": top,
            "reference": ref,
            "lenth_df": dfl,
            "resid_var": float(resid @ resid) / df if df > 0 else None,
            "notes": notes,
        },
    )
