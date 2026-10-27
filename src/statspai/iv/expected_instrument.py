"""Expected value of a formula instrument over its known shock design.

``expected_instrument`` simulates ``mu_i = E[z_i | exposure]`` for an
instrument built from as-good-as-random shocks and non-random exposure, the
quantity Borusyak and Hull (2023) recentre by. Its ``draws`` (one column per
redraw of the shocks) are what :func:`statspai.recentered_iv` takes as
``counterfactual=``; the module docstring of :mod:`statspai.iv.recentered`
describes the estimator and the randomization inference.

References
----------
[@borusyak2023nonrandom]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability

__all__ = ["expected_instrument", "ExpectedInstrumentResult"]

#: Above this many bytes the draws matrix is not kept by ``keep_draws='auto'``.
_AUTO_KEEP_BYTES = 256 * 2**20

ShockSampler = Union[Callable[[np.random.Generator], Any], Mapping[str, Any]]
InstrumentFn = Callable[[pd.DataFrame, Any], Any]


# ----------------------------------------------------------------------
# Shock samplers
# ----------------------------------------------------------------------


def _bernoulli_sampler(spec: Mapping[str, Any]) -> Callable[[np.random.Generator], Any]:
    p = np.asarray(spec.get("p", 0.5), dtype=float)
    size = spec.get("size")
    if p.ndim == 0:
        if size is None:
            raise MethodIncompatibility(
                "A Bernoulli shock sampler with a scalar p needs 'size'.",
                recovery_hint=(
                    "Pass {'kind': 'bernoulli', 'p': 0.5, 'size': n_shocks} or "
                    "give one probability per shock as an array."
                ),
                diagnostics={"spec": dict(spec)},
            )
        p = np.full(int(size), float(p))
    elif size is not None and int(size) != p.size:
        raise MethodIncompatibility(
            "Bernoulli sampler: 'size' disagrees with the length of 'p'.",
            recovery_hint="Drop 'size' when 'p' already has one entry per shock.",
            diagnostics={"size": int(size), "len_p": int(p.size)},
        )
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise MethodIncompatibility(
            "Bernoulli sampler: 'p' must be a probability or a 1-D array of them.",
            recovery_hint="Give success probabilities in [0, 1].",
        )

    def draw(rng: np.random.Generator) -> np.ndarray:
        return (rng.random(p.size) < p).astype(float)

    return draw


def _permute_sampler(spec: Mapping[str, Any]) -> Callable[[np.random.Generator], Any]:
    if "shocks" not in spec:
        raise MethodIncompatibility(
            "A permutation shock sampler needs the observed 'shocks'.",
            recovery_hint="Pass {'kind': 'permute', 'shocks': g, 'strata': s}.",
        )
    shocks = np.asarray(spec["shocks"])
    if shocks.ndim != 1:
        raise MethodIncompatibility(
            "Permutation sampler: 'shocks' must be one-dimensional.",
            recovery_hint="Pass one shock per shock unit.",
        )
    strata = spec.get("strata")
    if strata is None:
        groups: List[np.ndarray] = [np.arange(shocks.size)]
    else:
        codes, _ = pd.factorize(pd.Series(np.asarray(strata)), sort=True)
        if codes.size != shocks.size or np.any(codes < 0):
            raise MethodIncompatibility(
                "Permutation sampler: 'strata' must have one non-missing "
                "label per shock.",
                recovery_hint="Give a stratum for every shock, with no NaN.",
                diagnostics={"n_shocks": int(shocks.size), "n_strata": codes.size},
            )
        order = np.argsort(codes, kind="stable")
        bounds = np.flatnonzero(np.diff(codes[order])) + 1
        groups = np.split(order, bounds)

    def draw(rng: np.random.Generator) -> np.ndarray:
        out = shocks.copy()
        for g in groups:
            out[g] = shocks[g[rng.permutation(g.size)]]
        return out

    return draw


def _resolve_sampler(
    shock_sampler: ShockSampler,
) -> Callable[[np.random.Generator], Any]:
    if callable(shock_sampler):
        return shock_sampler  # type: ignore[return-value]
    if isinstance(shock_sampler, Mapping):
        kind = str(shock_sampler.get("kind", "")).lower()
        if kind == "bernoulli":
            return _bernoulli_sampler(shock_sampler)
        if kind == "permute":
            return _permute_sampler(shock_sampler)
        raise MethodIncompatibility(
            f"Unknown shock sampler kind {kind!r}.",
            recovery_hint=(
                "Use {'kind': 'bernoulli', ...}, {'kind': 'permute', ...} or "
                "pass a callable rng -> shocks."
            ),
            diagnostics={"kind": kind},
        )
    raise MethodIncompatibility(
        "shock_sampler must be a callable rng -> shocks or a spec dict.",
        recovery_hint=(
            "Pass e.g. lambda rng: rng.random(m) < 0.5, or "
            "{'kind': 'bernoulli', 'p': 0.5, 'size': m}."
        ),
        diagnostics={"type": type(shock_sampler).__name__},
    )


def _evaluate_instrument(
    instrument_fn: InstrumentFn, data: pd.DataFrame, shocks: Any, n: int
) -> np.ndarray:
    z = instrument_fn(data, shocks)
    try:
        arr = np.asarray(z, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "instrument_fn must return one number per row of data.",
            recovery_hint=(
                "Return a numeric array or Series of length len(data) from "
                "instrument_fn(data, shocks)."
            ),
            diagnostics={"returned_type": type(z).__name__},
        ) from exc
    if arr.size != n:
        raise MethodIncompatibility(
            f"instrument_fn returned {arr.size} values for {n} rows.",
            recovery_hint=(
                "instrument_fn(data, shocks) must return one value per row of "
                "data, in the row order of data."
            ),
            diagnostics={"returned": int(arr.size), "n_rows": int(n)},
        )
    if np.any(np.isinf(arr)):
        raise NumericalInstability(
            "instrument_fn returned infinite values.",
            recovery_hint="Check divisions by zero exposure inside instrument_fn.",
        )
    return arr


def _seed_sequence(seed: Any) -> Tuple[np.random.SeedSequence, Any]:
    if isinstance(seed, np.random.SeedSequence):
        return seed, seed.entropy
    if isinstance(seed, np.random.Generator):
        child = seed.bit_generator.seed_seq.spawn(1)[0]  # type: ignore[attr-defined]
        return child, None
    ss = np.random.SeedSequence(seed)
    return ss, ss.entropy


# ----------------------------------------------------------------------
# Expected instrument
# ----------------------------------------------------------------------


@dataclass
class ExpectedInstrumentResult(ResultProtocolMixin):
    """Simulated expectation of a formula instrument over its shock design.

    Attributes
    ----------
    mu : ndarray, shape (n,)
        Simulation mean of the instrument for each row of ``data``
        (``NaN`` where the instrument is undefined in every draw).
    mc_se : ndarray, shape (n,)
        Monte Carlo standard error of ``mu``: ``sd / sqrt(n_draws)``.
    sd : ndarray, shape (n,)
        Standard deviation of the instrument across draws (its design
        standard deviation, up to simulation error).
    n_draws : int
        Number of shock vectors drawn.
    draws : ndarray of shape (n, n_draws) or None
        Every simulated instrument, kept when ``keep_draws`` allows it.
        Pass it as ``sp.recentered_iv(..., counterfactual=draws)``; rows
        where the instrument is undefined are ``NaN`` and are dropped there.
    observed : ndarray or None
        The instrument at ``observed_shocks`` when those were given.
    index : pandas.Index
        Row labels of ``data``, aligned with ``mu``.
    seed_entropy : int or None
        Entropy of the ``SeedSequence`` used, so an unseeded call can be
        repeated exactly.
    diagnostics : dict
        ``max_mc_se``, ``mu_range``, ``mu_constant`` (``mu`` has no spread
        beyond simulation error, so recentring is a level shift),
        ``n_undefined`` and ``draws_kept``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"weight": [1.0, 2.0, 3.0]})
    >>> res = sp.expected_instrument(
    ...     df,
    ...     instrument_fn=lambda d, g: d["weight"].to_numpy() * g.sum(),
    ...     shock_sampler={"kind": "bernoulli", "p": 0.5, "size": 4},
    ...     n_draws=200, seed=1)
    >>> res.mu.shape
    (3,)
    >>> res.counterfactual().shape
    (3, 200)
    """

    mu: np.ndarray
    mc_se: np.ndarray
    sd: np.ndarray
    n_draws: int
    draws: Optional[np.ndarray]
    observed: Optional[np.ndarray]
    index: pd.Index
    seed_entropy: Optional[Any] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    _citation_keys = ("borusyak2023nonrandom",)

    def to_frame(self) -> pd.DataFrame:
        """``mu``, ``mc_se`` and ``sd`` as a DataFrame indexed like ``data``."""
        out = pd.DataFrame(
            {"mu": self.mu, "mc_se": self.mc_se, "sd": self.sd}, index=self.index
        )
        if self.observed is not None:
            out.insert(0, "observed", self.observed)
            out["recentered"] = self.observed - self.mu
        return out

    def counterfactual(self) -> pd.DataFrame:
        """The kept draws as an ``(n, n_draws)`` DataFrame indexed like ``data``.

        Columns ``draw_0`` .. ``draw_{n_draws-1}``; the input
        ``sp.recentered_iv(counterfactual=...)`` expects.

        Raises
        ------
        MethodIncompatibility
            If the draws were not kept (``keep_draws=False``, or
            ``'auto'`` above 256 MiB).
        """
        if self.draws is None:
            raise MethodIncompatibility(
                "The draws were not kept, so there are no counterfactual "
                "instruments to return.",
                recovery_hint=(
                    "Rerun sp.expected_instrument with keep_draws=True (it "
                    "needs 8 * n * n_draws bytes)."
                ),
                diagnostics={
                    "n_rows": int(self.mu.size),
                    "n_draws": int(self.n_draws),
                },
            )
        return pd.DataFrame(
            self.draws,
            index=self.index,
            columns=[f"draw_{s}" for s in range(self.draws.shape[1])],
        )

    def summary(self) -> str:
        """Plain-text summary of the simulated expectation."""
        d = self.diagnostics
        finite = np.isfinite(self.mu)
        lines = [
            "Expected instrument (simulated over the shock design)",
            "=" * 62,
            f"  rows / draws         : {self.mu.size:,} / {self.n_draws:,}",
            f"  mean of mu           : {np.mean(self.mu[finite]): .6f}",
            f"  range of mu          : [{d['mu_range'][0]: .6f},"
            f" {d['mu_range'][1]: .6f}]",
            f"  largest MC SE of mu  : {d['max_mc_se']:.3g}",
            f"  undefined rows       : {d['n_undefined']:,}",
        ]
        if d.get("mu_constant"):
            lines.append(
                "  mu does not vary beyond simulation error: recentring only "
                "shifts the level and leaves the IV slope unchanged."
            )
        return "\n".join(lines)


def _check_n_draws(n_draws: Any, name: str, minimum: int) -> int:
    if isinstance(n_draws, bool) or not isinstance(n_draws, (int, np.integer)):
        raise MethodIncompatibility(
            f"{name} must be an integer.",
            recovery_hint=f"Pass {name} as an int (e.g. 999).",
            diagnostics={name: repr(n_draws)},
        )
    if int(n_draws) < minimum:
        raise DataInsufficient(
            f"{name} must be at least {minimum}; got {int(n_draws)}.",
            recovery_hint=f"Increase {name}; a few hundred is a sensible floor.",
            diagnostics={name: int(n_draws)},
        )
    return int(n_draws)


def _simulate_expected(
    data: pd.DataFrame,
    instrument_fn: InstrumentFn,
    sampler: Callable[[np.random.Generator], Any],
    rng: np.random.Generator,
    n_draws: int,
    keep: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Welford mean / variance over draws; returns mu, var, draws, undefined."""
    n = len(data)
    mean = np.zeros(n)
    m2 = np.zeros(n)
    draws = np.empty((n, n_draws)) if keep else None
    undefined: Optional[np.ndarray] = None
    for s in range(n_draws):
        z = _evaluate_instrument(instrument_fn, data, sampler(rng), n)
        nan = np.isnan(z)
        if undefined is None:
            undefined = nan
        elif not np.array_equal(nan, undefined):
            raise NumericalInstability(
                "The instrument is undefined (NaN) for some rows in some shock "
                "draws but not in others.",
                recovery_hint=(
                    "Make instrument_fn return NaN only where the instrument is "
                    "undefined whatever the shocks (e.g. no exposure), or define "
                    "it for every draw."
                ),
                diagnostics={"draw": s, "rows_changed": int(np.sum(nan != undefined))},
            )
        z = np.where(nan, 0.0, z)
        delta = z - mean
        mean += delta / (s + 1)
        m2 += delta * (z - mean)
        if draws is not None:
            draws[:, s] = z
    assert undefined is not None
    var = m2 / (n_draws - 1)
    mean[undefined] = np.nan
    var[undefined] = np.nan
    if draws is not None:
        draws[undefined, :] = np.nan
    return mean, var, draws, undefined


@accepts_aliases(random_state="seed")
def expected_instrument(
    data: pd.DataFrame,
    instrument_fn: InstrumentFn,
    shock_sampler: ShockSampler,
    *,
    n_draws: int = 1000,
    seed: Optional[Any] = None,
    keep_draws: Union[bool, str] = "auto",
    observed_shocks: Optional[Any] = None,
) -> ExpectedInstrumentResult:
    """Expected value of a formula instrument over its known shock design.

    For an instrument ``z_i = f(shocks, exposure_i)`` whose shocks are as
    good as randomly assigned by a known process, draws ``n_draws``
    counterfactual shock vectors, rebuilds the instrument for each and
    returns the simulation mean ``mu_i`` -- the expected instrument that
    Borusyak and Hull (2023) recentre by -- with its Monte Carlo standard
    error.

    Parameters
    ----------
    data : pandas.DataFrame
        Unit-level data; passed unchanged to ``instrument_fn``.
    instrument_fn : callable
        ``instrument_fn(data, shocks) -> array`` returning one instrument
        value per row of ``data``, in row order. Return ``NaN`` for rows
        where the instrument is undefined whatever the shocks (e.g. no
        exposure); a row that is undefined in some draws only is an error.
    shock_sampler : callable or dict
        ``shock_sampler(rng) -> shocks`` drawing one counterfactual shock
        vector from a ``numpy.random.Generator``. Two specs are built in:
        ``{'kind': 'bernoulli', 'p': p, 'size': m}`` (independent coin
        flips; ``p`` scalar or one per shock) and
        ``{'kind': 'permute', 'shocks': g, 'strata': s}`` (the observed
        shocks permuted within strata).
    n_draws : int, default 1000
        Number of shock vectors drawn (at least 2).
    seed : int, SeedSequence or Generator, optional
        Seed for the draws. ``None`` draws fresh entropy, which is
        recorded in ``seed_entropy`` so the call can be repeated.
    keep_draws : bool or 'auto', default 'auto'
        Keep the ``(n, n_draws)`` matrix of simulated instruments.
        ``'auto'`` keeps it when it needs at most 256 MiB; ``False``
        streams the mean and variance and keeps nothing.
    observed_shocks : optional
        The realised shocks; when given, the observed instrument
        ``instrument_fn(data, observed_shocks)`` is returned in
        ``observed``.

    Returns
    -------
    ExpectedInstrumentResult
        ``mu``, ``mc_se``, ``sd``, optionally ``draws`` and ``observed``;
        ``to_frame()`` lines them up with ``data.index``, and
        ``counterfactual()`` returns the draws as a DataFrame.

    Notes
    -----
    This function only simulates; the estimator is
    :func:`statspai.recentered_iv`. The usual hand-off is
    ``sp.recentered_iv(df, y, treat, exposure, counterfactual=ei.draws)``,
    which takes the row mean of the draws as ``mu`` and uses the same draws,
    together with the realised instrument, for randomization inference.
    Do not also pass ``expected=ei.mu`` with those draws: ``mu`` would then
    depend on the draws it is compared with. For a fixed ``mu`` from
    simulation, make two calls with different seeds and pass one's ``mu``
    as ``expected=`` and the other's ``draws`` as ``counterfactual=``.

    Raises
    ------
    MethodIncompatibility
        If ``instrument_fn`` does not return one numeric value per row, or
        the sampler spec is malformed.
    NumericalInstability
        If the instrument is infinite, or undefined in some draws only.
    DataInsufficient
        If ``n_draws < 2``.

    Examples
    --------
    Share of opposite-sex children among ``m`` neighbours whose sexes are
    fair coin flips: the expectation is 0.5 for everyone.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n_children = 300
    >>> hood = rng.integers(0, 30, n_children)
    >>> df = pd.DataFrame({"hood": np.arange(30)})
    >>> def share_girls(data, sex):
    ...     return pd.Series(sex).groupby(hood).mean().reindex(
    ...         data["hood"]).to_numpy()
    >>> res = sp.expected_instrument(
    ...     df, share_girls, {"kind": "bernoulli", "p": 0.5,
    ...                       "size": n_children},
    ...     n_draws=500, seed=0)
    >>> bool(np.all(np.abs(res.mu - 0.5) < 6 * res.mc_se))
    True

    Two steps: simulate the counterfactual instruments, then hand them to
    ``sp.recentered_iv``. Here units are exposed to the shocks with
    non-random intensity ``expo``, and the exposure also raises ``y``:

    >>> n = 300
    >>> expo = rng.uniform(0, 2, n)
    >>> units = pd.DataFrame({"expo": expo})
    >>> def instrument(data, g):
    ...     return data["expo"].to_numpy() * g
    >>> g_obs = rng.binomial(1, 0.5, n)
    >>> ei = sp.expected_instrument(
    ...     units, instrument, {"kind": "bernoulli", "p": 0.5, "size": n},
    ...     n_draws=200, seed=1, observed_shocks=g_obs)
    >>> df = pd.DataFrame({"z": ei.observed})
    >>> df["y"] = 1.0 * df["z"] + 2.0 * expo + rng.normal(size=n)
    >>> fit = sp.recentered_iv(df, "y", "z", "z", counterfactual=ei.draws)
    >>> abs(fit.estimate - 1.0) < 0.5
    True

    References
    ----------
    [@borusyak2023nonrandom]
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "data must be a pandas DataFrame.",
            recovery_hint="Pass the unit-level data as a DataFrame.",
            diagnostics={"type": type(data).__name__},
        )
    if not callable(instrument_fn):
        raise MethodIncompatibility(
            "instrument_fn must be callable as instrument_fn(data, shocks).",
            recovery_hint="Pass a function building the instrument from shocks.",
        )
    if len(data) == 0:
        raise DataInsufficient(
            "data has no rows.", recovery_hint="Pass a non-empty DataFrame."
        )
    n_draws = _check_n_draws(n_draws, "n_draws", 2)
    sampler = _resolve_sampler(shock_sampler)
    ss, entropy = _seed_sequence(seed)
    rng = np.random.default_rng(ss)
    n = len(data)
    if keep_draws == "auto":
        keep = n * n_draws * 8 <= _AUTO_KEEP_BYTES
    elif isinstance(keep_draws, (bool, np.bool_)):
        keep = bool(keep_draws)
    else:
        raise MethodIncompatibility(
            "keep_draws must be True, False or 'auto'.",
            recovery_hint="Use keep_draws=False for large n * n_draws.",
            diagnostics={"keep_draws": repr(keep_draws)},
        )
    mu, var, draws, undefined = _simulate_expected(
        data, instrument_fn, sampler, rng, n_draws, keep
    )
    sd = np.sqrt(np.maximum(var, 0.0))
    mc_se = sd / np.sqrt(n_draws)
    observed = None
    if observed_shocks is not None:
        observed = _evaluate_instrument(instrument_fn, data, observed_shocks, n)
    finite = ~undefined
    if np.any(finite):
        mu_f = mu[finite]
        spread = float(np.max(mu_f) - np.min(mu_f))
        max_se = float(np.max(mc_se[finite]))
        # Constant up to simulation error: the spread of mu is within what
        # MC noise alone produces (about 2 * 4 MC SEs across the rows).
        mu_range = (float(np.min(mu_f)), float(np.max(mu_f)))
        mu_constant = spread <= 8.0 * max_se + 1e-12 * max(
            1.0, float(np.max(np.abs(mu_f)))
        )
    else:
        max_se, mu_range, mu_constant = float("nan"), (np.nan, np.nan), False
    diagnostics = {
        "max_mc_se": max_se,
        "mu_range": mu_range,
        "mu_constant": bool(mu_constant),
        "n_undefined": int(np.sum(undefined)),
        "draws_kept": bool(keep),
    }
    return ExpectedInstrumentResult(
        mu=mu,
        mc_se=mc_se,
        sd=sd,
        n_draws=n_draws,
        draws=draws,
        observed=observed,
        index=data.index.copy(),
        seed_entropy=entropy,
        diagnostics=diagnostics,
    )
