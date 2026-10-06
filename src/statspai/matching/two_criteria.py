"""Optimal matching according to two criteria, and tightening of blocks.

:func:`two_criteria_match` builds matched sets that are close on one
distance while the selected control group as a whole is balanced according
to another [@zhang2023matching]. The first distance decides who is paired
with whom; the second only decides which controls are chosen. Fine balance
[@rosenbaum2007minimum], near-fine balance [@yang2012optimal], near-exact
matching, calipers and directional penalties [@yu2019directional] are all
ways of filling in one of the two distances.

:func:`tighten_blocks` applies the same machinery inside an existing block
design: it keeps the treated individual of each block and the controls that
make the retained sample balanced on further covariates, optionally
dropping blocks that cannot be balanced [@rosenbaum2012optimal].

Outcomes play no part in either function. A matched design is built and
judged on covariates alone and analysed afterwards, for instance with
:func:`statspai.weighted_rank`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import require_columns
from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

Term = Mapping[str, Any]


def solve_two_criteria(
    pair: np.ndarray, balance: np.ndarray, use: np.ndarray, skip: np.ndarray, ratio: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """The network solver, imported on first use so that ``import statspai``
    does not load numba."""
    from ._mcf import solve_two_criteria as _solve

    return _solve(pair, balance, use, skip, ratio)


#: Above this many treated-by-control cells the dense cost matrices stop
#: being a sensible thing to hold in memory.
_MAX_CELLS = 60_000_000


# --------------------------------------------------------------------
# Cost terms
# --------------------------------------------------------------------


def rank_mahalanobis(x: np.ndarray, treated: np.ndarray) -> np.ndarray:
    """Rank-based Mahalanobis distance between treated units and controls.

    Each covariate is replaced by its ranks, so that an outlying value
    cannot dominate the distance, and the variances of the ranks are set to
    the variance of untied ranks, so that a rare binary covariate is not
    given outsized weight merely because its ranks are heavily tied
    [@rosenbaum2010design]. Returns squared distances, treated by control.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    ranks = stats.rankdata(x, method="average", axis=0)
    cov = np.atleast_2d(np.cov(ranks, rowvar=False))
    untied = n * (n + 1) / 12.0
    sd = np.sqrt(np.diag(cov))
    if np.any(sd == 0):
        raise MethodIncompatibility(
            "a covariate in a Mahalanobis term is constant; drop it from the term"
        )
    scale = np.sqrt(untied) / sd
    cov = cov * np.outer(scale, scale)
    inv = np.linalg.pinv(cov)
    rt, rc = ranks[treated], ranks[~treated]
    diff = rt[:, None, :] - rc[None, :, :]
    return np.asarray(
        np.einsum("tck,kl,tcl->tc", diff, inv, diff, optimize=True), dtype=float
    )


def _column(frame: pd.DataFrame, name: Any, what: str) -> np.ndarray:
    if not isinstance(name, str):
        raise MethodIncompatibility(
            f"two_criteria_match: {what} needs a column name, got {name!r}"
        )
    require_columns(frame, [name], function=f"two_criteria_match ({what})")
    col = frame[name]
    if col.isna().any():
        raise MethodIncompatibility(
            f"two_criteria_match: column {name!r} has missing values"
        )
    return np.asarray(col.to_numpy())


def _numeric(frame: pd.DataFrame, name: Any, what: str) -> np.ndarray:
    values = _column(frame, name, what)
    try:
        return values.astype(float)
    except (TypeError, ValueError):
        raise MethodIncompatibility(
            f"two_criteria_match: {what} needs a numeric column, and {name!r} "
            "is not; use a near_exact term for a nominal covariate"
        ) from None


def _term_cost(term: Term, frame: pd.DataFrame, treated: np.ndarray) -> np.ndarray:
    if not isinstance(term, Mapping) or "type" not in term:
        raise MethodIncompatibility(
            "each cost term must be a dict with a 'type' key, for instance "
            "{'type': 'near_exact', 'on': 'female', 'penalty': 1000}"
        )
    kind = str(term["type"]).lower()
    known = {
        "mahalanobis": {"type", "on"},
        "near_exact": {"type", "on", "penalty"},
        "integer": {"type", "on", "penalty"},
        "caliper": {"type", "on", "width", "penalty", "two_step"},
        "quantile": {"type", "on", "probs", "penalty"},
    }
    if kind not in known:
        raise MethodIncompatibility(
            f"unknown cost term type {kind!r}; expected one of {sorted(known)}"
        )
    extra = set(term) - known[kind]
    if extra or "on" not in term:
        raise MethodIncompatibility(
            f"a {kind!r} term takes the keys {sorted(known[kind])}; got {sorted(term)}"
        )
    what = f"the {kind!r} term"
    penalty = float(term.get("penalty", 1000.0))
    if kind != "mahalanobis" and not penalty > 0:
        raise MethodIncompatibility(f"{what} needs a positive penalty")
    on = term["on"]

    if kind == "mahalanobis":
        cols = [on] if isinstance(on, str) else list(on)
        x = np.column_stack([_numeric(frame, c, what) for c in cols])
        return rank_mahalanobis(x, treated)
    if kind == "near_exact":
        v = _column(frame, on, what)
        return np.asarray(v[treated][:, None] != v[~treated][None, :]) * penalty
    if kind == "integer":
        v = _numeric(frame, on, what)
        return np.asarray(np.abs(v[treated][:, None] - v[~treated][None, :])) * penalty
    if kind == "quantile":
        v = _numeric(frame, on, what)
        probs = np.asarray(term.get("probs", (0.2, 0.4, 0.6, 0.8)), dtype=float)
        if probs.size == 0 or np.any((probs <= 0) | (probs >= 1)):
            raise MethodIncompatibility(f"{what} needs probs strictly between 0 and 1")
        cuts = np.quantile(v, np.sort(probs))
        # right-closed intervals with the minimum in the first one
        level = np.searchsorted(cuts, v, side="left").astype(float)
        gap = np.asarray(np.abs(level[treated][:, None] - level[~treated][None, :]))
        return gap * penalty
    # caliper
    v = _numeric(frame, on, what)
    width = term.get("width")
    if width is None:
        half = 0.2 * float(np.std(v, ddof=1))
        low, high = -half, half
    elif isinstance(width, (int, float)):
        low, high = -abs(float(width)), abs(float(width))
    else:
        ends = sorted(float(w) for w in width)
        if len(ends) != 2 or ends[0] > 0 or ends[1] < 0:
            raise MethodIncompatibility(
                f"{what}: width must be a number or a pair (low, high) with "
                "low <= 0 <= high"
            )
        low, high = ends
    diff = v[treated][:, None] - v[~treated][None, :]
    cost = ((diff > high).astype(float) + (diff < low)) * penalty
    if term.get("two_step", True):
        cost += ((diff > 2 * high).astype(float) + (diff < 2 * low)) * penalty
    return np.asarray(cost, dtype=float)


def _sum_terms(
    terms: Union[Sequence[Term], np.ndarray, None],
    frame: pd.DataFrame,
    treated: np.ndarray,
    name: str,
) -> np.ndarray:
    shape = (int(treated.sum()), int((~treated).sum()))
    if terms is None:
        return np.zeros(shape)
    if isinstance(terms, np.ndarray):
        if terms.shape != shape:
            raise MethodIncompatibility(
                f"{name} given as a matrix must have shape {shape} "
                "(treated by control, in data order)"
            )
        cost = terms.astype(float)
    else:
        if isinstance(terms, Mapping):
            terms = [terms]
        cost = np.zeros(shape)
        for term in terms:
            cost = cost + _term_cost(term, frame, treated)
    if not np.all(np.isfinite(cost)) or np.any(cost < 0):
        raise MethodIncompatibility(f"{name} must be finite and non-negative")
    return cost


def _term_columns(*groups: Any) -> List[str]:
    out: List[str] = []
    for terms in groups:
        if terms is None or isinstance(terms, np.ndarray):
            continue
        if isinstance(terms, Mapping):
            terms = [terms]
        for term in terms:
            on = term.get("on") if isinstance(term, Mapping) else None
            for c in [on] if isinstance(on, str) else list(on or []):
                if c not in out:
                    out.append(c)
    return out


# --------------------------------------------------------------------
# Result
# --------------------------------------------------------------------


@dataclass
class TwoCriteriaMatchResult(ResultProtocolMixin):
    """Result of :func:`two_criteria_match` and :func:`tighten_blocks`.

    Attributes
    ----------
    matched : pandas.DataFrame
        The rows of the input that were matched, treated unit first within
        each set, with a column ``mset`` numbering the sets (and ``pscore``
        when a propensity score was fitted).
    pairs : pandas.DataFrame
        One row per treated-control pair: index labels of the two units,
        ``mset`` and the pairing cost.
    balance : pandas.DataFrame
        Covariate means of the treated, the matched controls and all
        controls, with standardized differences before and after.
    pair_cost, balance_cost, control_cost, total_cost : float
        The parts of the minimised objective and their sum.
    n_treated, n_sets, n_unmatched : int
        Treated units, matched sets formed, treated units left out.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> df = pd.DataFrame({"age": rng.normal(50, 10, n),
    ...                    "female": rng.integers(0, 2, n)})
    >>> df["z"] = rng.binomial(1, 1 / (1 + np.exp(-(df.age - 62) / 6)))
    >>> m = sp.two_criteria_match(
    ...     df, "z",
    ...     pair=[{"type": "mahalanobis", "on": ["age", "female"]}],
    ...     balance=[{"type": "near_exact", "on": "female"}])
    >>> type(m).__name__
    'TwoCriteriaMatchResult'
    >>> m.n_sets == int(df.z.sum())
    True
    """

    _citation_keys = ("zhang2023matching",)

    matched: pd.DataFrame
    pairs: pd.DataFrame
    balance: pd.DataFrame
    pair_cost: float
    balance_cost: float
    control_cost: float
    total_cost: float
    ratio: int
    n_treated: int
    n_control: int
    n_sets: int
    n_unmatched: int
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "Two-criteria optimal matching",
            "=============================",
            f"Treated / controls available : {self.n_treated} / {self.n_control}",
            f"Matched sets (1:{self.ratio})            : {self.n_sets}"
            + (f"   ({self.n_unmatched} treated left out)" if self.n_unmatched else ""),
            f"Pairing cost                 : {self.pair_cost:.4f}",
            f"Balance cost                 : {self.balance_cost:.4f}",
            f"Control-use cost             : {self.control_cost:.4f}",
        ]
        if len(self.balance):
            lines += [
                "",
                self.balance.to_string(float_format=lambda v: f"{v:.3f}"),
            ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TwoCriteriaMatchResult(n_sets={self.n_sets}, ratio={self.ratio}, "
            f"total_cost={self.total_cost:.4g})"
        )


def _balance_table(
    frame: pd.DataFrame,
    treated: np.ndarray,
    in_match: np.ndarray,
    columns: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for c in columns:
        try:
            v = frame[c].to_numpy(dtype=float)
        except (TypeError, ValueError):
            continue
        t_all, c_all = v[treated], v[~treated]
        t_m, c_m = v[treated & in_match], v[~treated & in_match]
        pooled = np.sqrt((np.var(t_all, ddof=1) + np.var(c_all, ddof=1)) / 2.0)
        scale = pooled if pooled > 0 else np.nan
        rows.append(
            {
                "covariate": c,
                "treated": float(np.mean(t_m)) if t_m.size else np.nan,
                "matched_control": float(np.mean(c_m)) if c_m.size else np.nan,
                "all_control": float(np.mean(c_all)),
                "smd_before": float((np.mean(t_all) - np.mean(c_all)) / scale),
                "smd_after": (
                    float((np.mean(t_m) - np.mean(c_m)) / scale)
                    if t_m.size and c_m.size
                    else np.nan
                ),
            }
        )
    table = pd.DataFrame(
        rows,
        columns=[
            "covariate",
            "treated",
            "matched_control",
            "all_control",
            "smd_before",
            "smd_after",
        ],
    )
    return table.set_index("covariate")


def _fit_pscore(
    frame: pd.DataFrame, treated: np.ndarray, covariates: Sequence[str]
) -> np.ndarray:
    import statsmodels.api as sm

    x = np.column_stack(
        [np.ones(len(frame))]
        + [_numeric(frame, c, "the propensity score") for c in covariates]
    )
    fit = sm.GLM(treated.astype(float), x, family=sm.families.Binomial()).fit(
        tol=1e-12, maxiter=200
    )
    return np.asarray(fit.fittedvalues, dtype=float)


def _assemble(
    frame: pd.DataFrame,
    treated: np.ndarray,
    pair: np.ndarray,
    balance: np.ndarray,
    use: np.ndarray,
    skip: np.ndarray,
    ratio: int,
    balance_columns: Sequence[str],
    diagnostics: dict,
) -> TwoCriteriaMatchResult:
    n_t, n_c = pair.shape
    if n_t * n_c > _MAX_CELLS:
        raise MethodIncompatibility(
            f"two_criteria_match: {n_t} treated by {n_c} controls is too many "
            "pairs for dense cost matrices.",
            recovery_hint=(
                "Match within exact strata of a coarse covariate (one call "
                "per stratum), or reduce the control reservoir first."
            ),
            diagnostics={"n_treated": n_t, "n_control": n_c},
            alternative_functions=["sp.match", "sp.cardinality_match"],
        )
    owner, owner2, skipped, feasible = solve_two_criteria(
        pair, balance, use, skip, ratio
    )
    if not feasible:
        raise DataInsufficient(
            f"two_criteria_match: cannot give each of {n_t} treated units "
            f"{ratio} distinct controls from {n_c}.",
            recovery_hint="Lower ratio, or pass subset_cost to let some "
            "treated units go unmatched.",
            diagnostics={"n_treated": n_t, "n_control": n_c, "ratio": ratio},
            alternative_functions=[],
        )

    t_pos = np.flatnonzero(treated)
    c_pos = np.flatnonzero(~treated)
    chosen = np.flatnonzero(owner >= 0)
    order = np.lexsort((chosen, owner[chosen]))
    chosen = chosen[order]
    kept_t = np.flatnonzero(~skipped)
    set_of = np.full(n_t, -1, dtype=int)
    set_of[kept_t] = np.arange(1, len(kept_t) + 1)

    pairs = pd.DataFrame(
        {
            "treated": frame.index[t_pos[owner[chosen]]],
            "control": frame.index[c_pos[chosen]],
            "mset": set_of[owner[chosen]],
            "pair_cost": pair[owner[chosen], chosen],
        }
    )
    rows: List[int] = []
    msets: List[int] = []
    for t in kept_t:
        rows.append(int(t_pos[t]))
        mine = chosen[owner[chosen] == t]
        rows.extend(int(c_pos[c]) for c in mine)
        msets.extend([int(set_of[t])] * (1 + len(mine)))
    matched = frame.iloc[rows].copy()
    matched["mset"] = msets

    in_match = np.zeros(len(frame), dtype=bool)
    in_match[rows] = True
    pair_cost = float(pair[owner[chosen], chosen].sum())
    balance_cost = float(balance[owner2[chosen], chosen].sum())
    control_cost = float(use[chosen].sum())
    skip_cost = float(skip[skipped].sum()) if skipped.any() else 0.0
    diagnostics = dict(diagnostics)
    diagnostics["skip_cost"] = skip_cost
    return TwoCriteriaMatchResult(
        matched=matched,
        pairs=pairs,
        balance=_balance_table(frame, treated, in_match, balance_columns),
        pair_cost=pair_cost,
        balance_cost=balance_cost,
        control_cost=control_cost,
        total_cost=pair_cost + balance_cost + control_cost + skip_cost,
        ratio=int(ratio),
        n_treated=int(n_t),
        n_control=int(n_c),
        n_sets=int(len(kept_t)),
        n_unmatched=int(skipped.sum()),
        diagnostics=diagnostics,
    )


def _prepare(
    data: pd.DataFrame, treat: str, caller: str
) -> Tuple[pd.DataFrame, np.ndarray]:
    require_columns(data, [treat], function=caller)
    frame = data[data[treat].notna()]
    z = frame[treat].to_numpy(dtype=float)
    if not np.all((z == 0) | (z == 1)):
        raise MethodIncompatibility(f"{caller}: the treat column must be coded 0/1")
    treated = z == 1
    if treated.sum() == 0 or (~treated).sum() == 0:
        raise DataInsufficient(
            f"{caller}: need both treated units and controls.",
            recovery_hint="Check the treat column and any sample filter.",
            diagnostics={
                "n_treated": int(treated.sum()),
                "n_control": int((~treated).sum()),
            },
            alternative_functions=[],
        )
    return frame, treated


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------


def two_criteria_match(
    data: pd.DataFrame,
    treat: str,
    *,
    pair: Union[Sequence[Term], np.ndarray, None] = None,
    balance: Union[Sequence[Term], np.ndarray, None] = None,
    ratio: int = 1,
    ps: Union[str, Sequence[str], None] = None,
    control_cost: Union[str, Sequence[float], None] = None,
    subset_cost: Optional[float] = None,
) -> TwoCriteriaMatchResult:
    """Optimal matching that pairs on one distance and balances on another.

    Chooses ``ratio`` distinct controls for each treated unit so as to
    minimise the sum of two costs [@zhang2023matching]. The *pair* cost is
    the total distance between each treated unit and its own controls. The
    *balance* cost is the smallest total distance at which the chosen
    controls can be assigned to the treated units when that assignment is
    free to ignore the pairing, so it depends only on which controls were
    chosen. A covariate placed in ``pair`` is matched closely within
    sets; a covariate placed in ``balance`` has its distribution equalised
    between the groups without constraining who is paired with whom. A
    large ``near_exact`` or ``integer`` penalty in ``balance`` is fine
    balance when it can be met and near-fine balance when it cannot.

    Parameters
    ----------
    data : DataFrame
        One row per unit. Rows with a missing ``treat`` are ignored.
    treat : str
        0/1 treatment indicator.
    pair, balance : list of dict, or ndarray, optional
        The two distances, each the sum of its terms. A term is a dict
        with a ``"type"``, the column(s) it is ``"on"``, and for all but
        the first type a ``"penalty"`` (default 1000):

        * ``{"type": "mahalanobis", "on": [...]}``: squared rank-based
          Mahalanobis distance on the listed covariates.
        * ``{"type": "near_exact", "on": col, "penalty": p}``: ``p`` when
          the two units differ on a nominal covariate.
        * ``{"type": "integer", "on": col, "penalty": p}``: ``p`` times the
          absolute difference of an integer-coded covariate.
        * ``{"type": "caliper", "on": col, "width": w, "penalty": p,
          "two_step": True}``: ``p`` when treated minus control falls
          outside the caliper, and ``p`` more when it falls outside twice
          the caliper. ``w`` is a half-width, a pair ``(low, high)`` for an
          asymmetric caliper that tolerates mismatches in one direction,
          or omitted for 0.2 standard deviations.
        * ``{"type": "quantile", "on": col, "probs": [...], "penalty":
          p}``: an ``integer`` term on the covariate cut at its quantiles.

        The column name ``"pscore"`` refers to the propensity score
        requested through ``ps``. A treated-by-control matrix (rows and
        columns in data order) may be passed instead of a list.
    ratio : int, default 1
        Controls per treated unit.
    ps : str or list of str, optional
        A column holding a propensity score, or the covariates of a
        logistic regression that is fitted to produce one.
    control_cost : str or array-like, optional
        Non-negative cost of using each control at all (a column name, or
        one value per control in data order): a way to steer the match
        away from controls that are unlike any treated unit.
    subset_cost : float, optional
        Allow a treated unit to be left out at this cost, so that the
        match keeps the subset of treated units that can be matched well
        [@rosenbaum2012optimal]. Requires ``ratio=1``. By default every
        treated unit is matched.

    Returns
    -------
    TwoCriteriaMatchResult
        ``.matched`` is the matched sample in long format, ready for an
        outcome analysis by matched set.

    Notes
    -----
    The costs are used as given. R's ``iTOS::makematch`` passes them
    through ``rcbalance::callrelax``, which truncates every cost to an
    integer before solving, so a Mahalanobis term enters that solution
    only through its integer part.

    Examples
    --------
    Pair on age and sex, keep the sexes balanced, and stay within a
    propensity-score caliper:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> df = pd.DataFrame({"age": rng.normal(50, 10, n),
    ...                    "female": rng.integers(0, 2, n)})
    >>> df["z"] = rng.binomial(1, 1 / (1 + np.exp(-(df.age - 62) / 6)))
    >>> m = sp.two_criteria_match(
    ...     df, "z", ps=["age", "female"],
    ...     pair=[{"type": "mahalanobis", "on": ["age", "female"]},
    ...           {"type": "caliper", "on": "pscore", "penalty": 100}],
    ...     balance=[{"type": "near_exact", "on": "female"}])
    >>> m.matched["mset"].value_counts().unique().tolist()
    [2]
    >>> bool(abs(m.balance.loc["age", "smd_after"])
    ...      < abs(m.balance.loc["age", "smd_before"]))
    True

    References
    ----------
    [@zhang2023matching], [@rosenbaum1989optimal],
    [@rosenbaum2007minimum], [@yang2012optimal], [@yu2019directional],
    [@rosenbaum2012optimal]
    """
    caller = "two_criteria_match"
    frame, treated = _prepare(data, treat, caller)
    if int(ratio) != ratio or ratio < 1:
        raise MethodIncompatibility("ratio must be a positive integer")
    if pair is None and balance is None:
        raise MethodIncompatibility(
            f"{caller}: give at least one of pair= and balance="
        )
    if subset_cost is not None:
        if ratio != 1:
            raise MethodIncompatibility(
                f"{caller}: subset_cost needs ratio=1.",
                recovery_hint="Use ratio=1, or drop subset_cost.",
                diagnostics={"ratio": int(ratio)},
                alternative_functions=[],
            )
        if not subset_cost >= 0:
            raise MethodIncompatibility("subset_cost must be non-negative")

    covariates: List[str] = []
    if ps is not None:
        frame = frame.copy()
        if isinstance(ps, str):
            frame["pscore"] = _numeric(frame, ps, "ps")
        else:
            covariates = list(ps)
            frame["pscore"] = _fit_pscore(frame, treated, covariates)
    pair_cost = _sum_terms(pair, frame, treated, "pair")
    balance_cost = _sum_terms(balance, frame, treated, "balance")

    n_t, n_c = pair_cost.shape
    if control_cost is None:
        use = np.zeros(n_c)
    elif isinstance(control_cost, str):
        use = _numeric(frame, control_cost, "control_cost")[~treated]
    else:
        use = np.asarray(control_cost, dtype=float)
    if use.shape != (n_c,) or np.any(use < 0) or not np.all(np.isfinite(use)):
        raise MethodIncompatibility(
            "control_cost must hold one finite non-negative value per control"
        )
    skip = np.full(n_t, np.inf if subset_cost is None else float(subset_cost))

    columns = covariates + [
        c for c in _term_columns(pair, balance) if c not in covariates
    ]
    if ps is not None and "pscore" not in columns:
        columns.append("pscore")
    return _assemble(
        frame,
        treated,
        pair_cost,
        balance_cost,
        use,
        skip,
        int(ratio),
        columns,
        {"subset_cost": subset_cost},
    )


def tighten_blocks(
    data: pd.DataFrame,
    treat: str,
    block: str,
    *,
    covariates: Optional[Sequence[str]] = None,
    fine_balance: Optional[Sequence[str]] = None,
    ratio: int = 1,
    subset_cost: Optional[float] = None,
    penalty_scale: float = 10.0,
) -> TwoCriteriaMatchResult:
    """Tighten a block design into smaller or fewer blocks.

    Starts from blocks that each hold one treated individual and several
    controls, already matched for some covariates, and keeps ``ratio`` of
    the controls in each block (and, with ``subset_cost``, only some of
    the blocks) so that the retained sample is also comparable on
    covariates the original blocks did not control. Controls are always
    taken from the treated individual's own block. Among such choices the
    function first balances the ``fine_balance`` covariates between the
    retained treated and control groups, and then minimises the
    within-block rank-based Mahalanobis distance on ``covariates``.

    Typical uses are a secondary analysis that adjusts for a covariate
    possibly affected by the treatment, which the primary design
    deliberately left out, and a differential comparison within the
    blocks that have a particular pattern of a second exposure.

    Parameters
    ----------
    data : DataFrame
        One row per individual of the block design.
    treat, block : str
        0/1 treatment indicator and block identifier. Each block must
        hold exactly one treated individual and at least one control.
    covariates : list of str, optional
        Covariates for the within-block Mahalanobis distance.
    fine_balance : list of str, optional
        Nominal covariates whose distributions should agree between the
        retained treated and control groups. Each contributes a penalty
        for every treated-control mismatch that cannot be avoided.
    ratio : int, default 1
        Controls kept per block.
    subset_cost : float, optional
        Price of dropping a block. Smaller values drop more blocks in
        exchange for better balance; ``None`` keeps every block. Requires
        ``ratio=1``.
    penalty_scale : float, default 10.0
        Factor separating the priorities: staying within the block
        outweighs fine balance, which outweighs covariate distance.

    Returns
    -------
    TwoCriteriaMatchResult
        ``.matched`` holds the tightened design; ``mset`` numbers its
        blocks.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> blocks = np.repeat(np.arange(80), 4)
    >>> df = pd.DataFrame({"block": blocks,
    ...                    "z": np.tile([1, 0, 0, 0], 80),
    ...                    "bmi": rng.normal(27, 4, 320),
    ...                    "smoker": rng.integers(0, 2, 320)})
    >>> tight = sp.tighten_blocks(df, "z", "block", covariates=["bmi"],
    ...                           fine_balance=["smoker"], ratio=2)
    >>> tight.matched.groupby("mset").size().unique().tolist()
    [3]
    >>> bool(tight.matched.groupby("mset")["block"].nunique().eq(1).all())
    True

    References
    ----------
    [@rosenbaum2012optimal], [@zhang2023matching], [@rosenbaum2010design]
    """
    caller = "tighten_blocks"
    frame, treated = _prepare(data, treat, caller)
    if covariates is None and fine_balance is None:
        raise MethodIncompatibility(
            f"{caller}: give covariates=, fine_balance= or both"
        )
    if int(ratio) != ratio or ratio < 1:
        raise MethodIncompatibility("ratio must be a positive integer")
    if not penalty_scale > 0:
        raise MethodIncompatibility("penalty_scale must be positive")
    if subset_cost is not None and ratio != 1:
        raise MethodIncompatibility(
            f"{caller}: subset_cost needs ratio=1.",
            recovery_hint="Use ratio=1, or drop subset_cost.",
            diagnostics={"ratio": int(ratio)},
            alternative_functions=[],
        )
    blocks = _column(frame, block, "block")
    per_block = pd.Series(treated).groupby(blocks).agg(["sum", "size"])
    if not (per_block["sum"] == 1).all() or (per_block["size"] < 2).any():
        raise MethodIncompatibility(
            f"{caller}: every block must hold exactly one treated individual "
            "and at least one control.",
            recovery_hint="Build the blocks first, for example with "
            "sp.two_criteria_match.",
            diagnostics={
                "n_blocks": int(len(per_block)),
                "n_blocks_invalid": int(
                    ((per_block["sum"] != 1) | (per_block["size"] < 2)).sum()
                ),
            },
            alternative_functions=["sp.two_criteria_match"],
        )
    if (per_block["size"] - 1 < ratio).any():
        raise DataInsufficient(
            f"{caller}: some blocks have fewer than {ratio} controls.",
            recovery_hint="Lower ratio.",
            diagnostics={"min_controls": int(per_block["size"].min() - 1)},
            alternative_functions=[],
        )

    cov = list(covariates or [])
    fine = list(fine_balance or [])
    left = _sum_terms(
        [{"type": "mahalanobis", "on": cov}] if cov else None, frame, treated, "pair"
    )
    penalty = (10.0 + np.ceil(left.max())) * penalty_scale
    right = _sum_terms(
        [{"type": "near_exact", "on": c, "penalty": penalty} for c in fine] or None,
        frame,
        treated,
        "balance",
    )
    if fine:
        penalty = (penalty + np.ceil(right.max())) * penalty_scale
    same_block = blocks[treated][:, None] == blocks[~treated][None, :]
    left = left + (~same_block) * penalty

    n_t = left.shape[0]
    skip = np.full(n_t, np.inf if subset_cost is None else float(subset_cost))
    result = _assemble(
        frame,
        treated,
        left,
        right,
        np.zeros(left.shape[1]),
        skip,
        int(ratio),
        cov + [c for c in fine if c not in cov],
        {"subset_cost": subset_cost, "block_penalty": float(penalty)},
    )
    if result.matched.groupby("mset")[block].nunique().gt(1).any():
        raise MethodIncompatibility(  # pragma: no cover - the penalty dominates
            f"{caller}: a control was taken from another block; raise " "penalty_scale"
        )
    return result


__all__ = [
    "two_criteria_match",
    "tighten_blocks",
    "TwoCriteriaMatchResult",
    "rank_mahalanobis",
]
