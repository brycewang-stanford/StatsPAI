"""Starting values, continuation and progress output for MCMC chains.

Used by :func:`statspai.bayes_regress` and, through :class:`ChainRun`, by
every other sampler of :mod:`statspai.mcmc`. A chain is described by its
generator and its sampler state after the last iteration; a later run
restores both (``continue_from=``) and goes on from there.

Continuation is exact for samplers that draw their random numbers
iteration by iteration (the Gibbs and data-augmentation samplers): a run of
``m`` draws continued by ``m'`` equals one run of ``m + m'`` draws with the
same seed, bit for bit. The Metropolis samplers built on
:func:`statspai.mcmc._core.metropolis_mixture` draw all the proposals of a
run in one block, so the stream a continued run reads is not the one a
longer run would have read. Their continuation is still the same Markov
chain from the same state, and is reproducible, but its draws differ from
a single longer run.
"""

from __future__ import annotations

import hashlib
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility


def resolve_inits(
    inits: Any, names: Sequence[str], chains: int, model: Any
) -> List[np.ndarray]:
    """Starting values for each chain on the reported scale.

    ``inits`` is a mapping ``{parameter: value}`` (or a Series) used for
    every chain, a sequence of such mappings (one per chain), or an array
    of shape ``(n_par,)`` or ``(chains, n_par)`` in the order of ``names``.
    Every parameter needs a value, and the values must lie in the support
    of the posterior (finite log kernel).
    """

    def support(row: np.ndarray) -> bool:
        try:
            lk = float(model.log_kernel(model.to_u(row)))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return False
        return bool(np.isfinite(lk))

    return resolve_start_values(inits, names, chains, support=support)


_SUPPORT_TEXT = "check positive variances and scales and increasing cutpoints"


def resolve_start_values(
    inits: Any,
    names: Sequence[str],
    chains: int,
    *,
    support: Optional[Callable[[np.ndarray], bool]] = None,
    drawn_first: Sequence[str] = (),
    support_text: str = _SUPPORT_TEXT,
) -> List[np.ndarray]:
    """Starting values for each chain, one row per chain in the order of
    ``names``.

    ``inits`` is a mapping ``{parameter: value}`` (or a Series) used for
    every chain, a sequence of such mappings (one per chain), or an array
    of shape ``(len(names),)`` or ``(chains, len(names))``. Every name in
    ``names`` needs a value. A mapping may also name the parameters in
    ``drawn_first``: the sampler draws those from their full conditional
    before it uses them, so their values are accepted (``result.params``
    can be passed as it is) and not used. ``support(row)`` is False for a
    row outside the support of the posterior.
    """
    names = [str(n) for n in names]
    optional = [str(n) for n in drawn_first]
    k = len(names)

    def from_mapping(m: Mapping[str, Any], where: str) -> np.ndarray:
        keys = [str(x) for x in m.keys()]
        unknown = [x for x in keys if x not in names and x not in optional]
        missing = [x for x in names if x not in keys]
        if unknown or missing:
            raise MethodIncompatibility(
                f"inits{where}: "
                + (f"unknown parameters {unknown}; " if unknown else "")
                + (f"no value for {missing}; " if missing else "")
                + f"the parameters are {names}"
                + (
                    f" ({optional} are drawn before they are used, so a "
                    "value for them is not needed)."
                    if optional
                    else "."
                ),
                recovery_hint="Give one value per parameter, named as in "
                "result.params.",
            )
        lookup = {str(a): b for a, b in m.items()}
        return np.array([float(lookup[x]) for x in names], dtype=float)

    rows: List[np.ndarray]
    if isinstance(inits, (Mapping, pd.Series)):
        rows = [from_mapping(dict(inits), "")] * chains
    elif (
        isinstance(inits, Sequence)
        and inits
        and isinstance(inits[0], (Mapping, pd.Series))
    ):
        if len(inits) != chains:
            raise MethodIncompatibility(
                f"inits: {len(inits)} sets of starting values for {chains} " "chains.",
                recovery_hint="Give one mapping per chain, or one for all.",
            )
        rows = [from_mapping(dict(m), f"[{i}]") for i, m in enumerate(inits)]
    else:
        arr = np.asarray(inits, dtype=float)
        if arr.shape == (k,):
            rows = [arr.copy() for _ in range(chains)]
        elif arr.shape == (chains, k):
            rows = [arr[c].copy() for c in range(chains)]
        else:
            raise MethodIncompatibility(
                f"inits: an array must have shape ({k},) or ({chains}, {k}); "
                f"got {arr.shape}. The parameters are {names}.",
                recovery_hint="Or pass a dict {parameter: value}.",
            )
    for c, row in enumerate(rows):
        bad = ~np.isfinite(row)
        if bad.any():
            raise MethodIncompatibility(
                f"inits for chain {c}: {[names[j] for j in np.flatnonzero(bad)]} "
                "is not a finite number.",
                recovery_hint="Give finite starting values.",
            )
        if support is not None and not support(row):
            raise MethodIncompatibility(
                f"inits for chain {c} lie outside the support of the "
                f"posterior (zero density there): {support_text}.",
                recovery_hint="Start the chain where the posterior is "
                "positive, e.g. from result.params of an earlier fit.",
                diagnostics={"inits": dict(zip(names, row.tolist()))},
            )
    return rows


def model_fingerprint(model: Any) -> str:
    """Digest of the data, prior and settings a sampler runs on.

    Two models with the same digest have the same posterior and the same
    sampler, so a chain of one may be continued by the other.
    """
    h = hashlib.sha256(type(model).__name__.encode())
    for key in sorted(vars(model)):
        if key.startswith("_"):
            continue  # caches (the mode) and private scratch
        val = getattr(model, key)
        if isinstance(val, (np.ndarray, float, int, np.floating, np.integer)):
            arr = np.ascontiguousarray(np.asarray(val, dtype=float))
            h.update(key.encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        elif isinstance(val, (str, bool, type(None))):
            h.update(f"{key}={val!r}".encode())
        elif isinstance(val, (list, tuple)) and all(
            isinstance(v, (str, int, float)) for v in val
        ):
            h.update(f"{key}={list(val)!r}".encode())
    return h.hexdigest()[:16]


def restore_generator(state: Mapping[str, Any]) -> np.random.Generator:
    """A generator positioned where a stored one stopped."""
    name = str(state.get("bit_generator", "PCG64"))
    bit_gen_cls = getattr(np.random, name, None)
    if bit_gen_cls is None:  # pragma: no cover - numpy always has PCG64
        raise MethodIncompatibility(f"continue_from: unknown bit generator {name!r}.")
    bit_gen = bit_gen_cls()
    bit_gen.state = dict(state)
    return np.random.Generator(bit_gen)


def check_continuation(
    prev: Any,
    model: Any,
    model_key: str,
    *,
    chains: int,
    thin: int,
    burnin: int,
    seed: Any,
    inits: Any,
) -> List[Dict[str, Any]]:
    """The stored chain states of ``prev``, after checking that the new
    call describes the same chains."""
    states = getattr(prev, "_chain_state", None)
    if states is None or getattr(prev, "model", None) is None:
        raise MethodIncompatibility(
            "continue_from must be a bayes_regress result fitted by MCMC "
            "(this one keeps no chain state).",
            recovery_hint="Pass the result of an earlier sp.bayes_regress "
            "call with inference='mcmc'.",
        )
    if prev.model != model_key:
        raise MethodIncompatibility(
            f"continue_from: the previous fit is model={prev.model!r}, "
            f"this call model={model_key!r}.",
            recovery_hint="Use the same model, data and prior as the run "
            "being continued.",
        )
    if inits is not None:
        raise MethodIncompatibility(
            "continue_from starts each chain where the previous run stopped; "
            "it cannot be combined with inits=.",
            recovery_hint="Drop inits=.",
        )
    if getattr(prev, "_chain_key", None) != model_fingerprint(model):
        raise MethodIncompatibility(
            "continue_from: the previous run used different data, prior or "
            "sampler settings, so its chains are not chains of this "
            "posterior.",
            recovery_hint="Pass the same formula, data, prior and settings "
            "as the run being continued.",
        )
    for name, given, default, stored in (
        ("chains", chains, 1, prev.chains),
        ("thin", thin, 1, prev.thin),
    ):
        if given != default and given != stored:
            raise MethodIncompatibility(
                f"continue_from: the previous run has {name}={stored}; got "
                f"{name}={given}.",
                recovery_hint=f"Leave {name} out; it is taken from the "
                "previous run.",
            )
    if burnin not in (0, 2000):
        raise MethodIncompatibility(
            "continue_from: a continued chain is already past its burn-in; "
            f"got burnin={burnin}.",
            recovery_hint="Leave burnin out (or set 0).",
        )
    prev_seed = getattr(prev, "_seed", None)
    if seed is not None and seed != prev_seed:
        raise MethodIncompatibility(
            "continue_from: the chains continue from the previous run's "
            "generators, so seed= has no effect; pass the same seed or none.",
            recovery_hint="Drop seed=.",
        )
    return list(states)


class Progress:
    """Progress lines on stdout when ``verbose`` (BS2.12)."""

    def __init__(self, verbose: bool, label: str) -> None:
        if not isinstance(verbose, (bool, np.bool_)):
            raise MethodIncompatibility(
                f"verbose must be True or False; got {verbose!r}."
            )
        self.on = bool(verbose)
        self.label = label

    def say(self, text: str) -> None:
        if self.on:
            print(f"{self.label}: {text}", file=sys.stdout, flush=True)


# --------------------------------------------------------------------------
# The shared path of the other samplers
# --------------------------------------------------------------------------


def _feed(h: Any, val: Any) -> None:
    """Add ``val`` to the digest ``h``: arrays and numbers by their bytes,
    containers item by item, functions by identity."""
    if val is None or isinstance(val, (str, bool)):
        h.update(f"{type(val).__name__}={val!r};".encode())
    elif isinstance(val, (int, float, np.integer, np.floating, np.ndarray)):
        arr = np.ascontiguousarray(np.asarray(val))
        if arr.dtype == object:
            for item in arr.ravel():
                _feed(h, item)
            return
        h.update(f"{arr.dtype}{arr.shape}".encode())
        h.update(arr.tobytes())
    elif isinstance(val, (pd.Series, pd.DataFrame)):
        _feed(h, [str(c) for c in np.atleast_1d(getattr(val, "columns", val.name))])
        _feed(h, val.to_numpy())
    elif isinstance(val, Mapping):
        for key in sorted(val, key=str):
            h.update(f"{key}:".encode())
            _feed(h, val[key])
    elif isinstance(val, (list, tuple)):
        h.update(f"[{len(val)}".encode())
        for item in val:
            _feed(h, item)
        h.update(b"]")
    elif hasattr(val, "dist") and hasattr(val, "args"):
        # a frozen scipy.stats distribution
        _feed(h, [val.dist.name, list(val.args), dict(getattr(val, "kwds", {}))])
    else:
        # a user function or another object: the same object only
        h.update(f"{type(val).__qualname__}@{id(val)};".encode())


def digest(fn: str, parts: Mapping[str, Any]) -> str:
    """Key of the data, prior and settings of a sampler call: two calls
    with the same key sample the same posterior with the same sampler."""
    h = hashlib.sha256(fn.encode())
    _feed(h, parts)
    return f"{fn}:{h.hexdigest()[:16]}"


def as_array(value: Any, dtype: Any = float) -> np.ndarray:
    """A stored piece of chain state as a fresh array."""
    return np.array(value, dtype=dtype)


class ChainRun:
    """``inits=``, ``continue_from=``, ``verbose=`` and ``keep_burnin=``
    for one call of a :mod:`statspai.mcmc` sampler.

    The sampler creates it before reading the data (the arguments are
    checked first), calls :meth:`setup` once the posterior is fixed, then
    :meth:`begin` and :meth:`end` around each chain, :meth:`stack` to put
    the draws of an earlier run in front of the new ones, and
    :meth:`attach` on the result. ``end`` records each chain's generator
    and sampler state, from which ``continue_from=`` goes on.

    ``refuse`` (a reason) makes the three chain arguments an error, for a
    sampler without a Markov chain; ``no_inits`` (a reason) does so for
    ``inits=`` alone.
    """

    def __init__(
        self,
        fn: str,
        *,
        inits: Any = None,
        continue_from: Any = None,
        verbose: Any = False,
        keep_burnin: Any = False,
        refuse: Optional[str] = None,
        no_inits: Optional[str] = None,
    ) -> None:
        self.fn = fn
        self.progress = Progress(verbose, fn)
        if not isinstance(keep_burnin, (bool, np.bool_)):
            raise MethodIncompatibility(
                f"keep_burnin must be True or False; got {keep_burnin!r}."
            )
        if refuse is not None and (
            inits is not None or continue_from is not None or keep_burnin
        ):
            raise MethodIncompatibility(
                f"sp.{fn}: inits=, continue_from= and keep_burnin= are for "
                f"Markov chains; {refuse}",
                recovery_hint="Drop them, or raise draws= for more draws.",
            )
        if no_inits is not None and inits is not None:
            raise MethodIncompatibility(
                f"sp.{fn}: inits= is not available; {no_inits}",
                recovery_hint="Drop inits=. continue_from= starts a run "
                "where an earlier one stopped.",
            )
        self.prev: Any = None
        self.prev_states: Optional[List[Dict[str, Any]]] = None
        if continue_from is not None:
            states = getattr(continue_from, "_chain_state", None)
            key = getattr(continue_from, "_chain_key", None)
            if not states or not str(key).startswith(fn + ":"):
                raise MethodIncompatibility(
                    f"continue_from must be a result of sp.{fn} (this one "
                    "keeps no chain state of it).",
                    recovery_hint=f"Pass the result of an earlier sp.{fn} "
                    "call on the same data.",
                )
            if inits is not None:
                raise MethodIncompatibility(
                    "continue_from starts each chain where the previous run "
                    "stopped; it cannot be combined with inits=.",
                    recovery_hint="Drop inits=.",
                )
            self.prev = continue_from
            self.prev_states = list(states)
        self.inits = inits
        self.keep_burnin = bool(keep_burnin)
        self.key = ""
        self.exact = True
        self.chains = 1
        self.thin = 1
        self.burnin = 0
        self.draws = 0
        self.n_iter = 0
        self.seed: Any = None
        self.rngs: List[np.random.Generator] = []
        self.starts: List[Any] = []
        self._end: List[Dict[str, Any]] = []
        self._first: List[Optional[np.ndarray]] = []
        self._burn: List[np.ndarray] = []
        self._t0 = 0.0
        self._meta: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    @property
    def continuing(self) -> bool:
        return self.prev is not None

    def setup(
        self,
        parts: Mapping[str, Any],
        *,
        chains: int,
        thin: int,
        burnin: int,
        draws: int,
        seed: Any,
        default_burnin: int,
        make_rngs: Callable[[int], List[np.random.Generator]],
        exact: bool,
        default_chains: int = 1,
        init_names: Sequence[str] = (),
        drawn_first: Sequence[str] = (),
        support: Optional[Callable[[np.ndarray], bool]] = None,
        support_text: str = _SUPPORT_TEXT,
        to_state: Optional[Callable[[np.ndarray], Any]] = None,
    ) -> None:
        """Fix the run: generators, starting states and burn-in.

        ``parts`` holds everything the posterior and the sampler depend on
        (data, prior, settings); ``to_state`` maps a row of starting
        values (in the order of ``init_names``) to the state the sampler
        starts from.
        """
        self.key = digest(self.fn, parts)
        self.exact = bool(exact)
        self.draws = int(draws)
        if self.prev is not None:
            meta = dict(self.prev_states[0].get("meta", {}))  # type: ignore[index]
            if self.prev._chain_key != self.key:
                raise MethodIncompatibility(
                    "continue_from: the previous run used different data, "
                    "prior or sampler settings, so its chains are not chains "
                    "of this posterior.",
                    recovery_hint="Pass the same data, prior and settings "
                    "as the run being continued.",
                )
            for name, given, default in (
                ("chains", chains, default_chains),
                ("thin", thin, 1),
            ):
                stored = meta[name]
                if given != default and given != stored:
                    raise MethodIncompatibility(
                        f"continue_from: the previous run has {name}={stored}; "
                        f"got {name}={given}.",
                        recovery_hint=f"Leave {name} out; it is taken from "
                        "the previous run.",
                    )
            if burnin not in (0, default_burnin):
                raise MethodIncompatibility(
                    "continue_from: a continued chain is already past its "
                    f"burn-in; got burnin={burnin}.",
                    recovery_hint="Leave burnin out (or set 0).",
                )
            if seed is not None and seed != meta["seed"]:
                raise MethodIncompatibility(
                    "continue_from: the chains continue from the previous "
                    "run's generators, so seed= has no effect; pass the same "
                    "seed or none.",
                    recovery_hint="Drop seed=.",
                )
            self.chains, self.thin, self.seed = (
                int(meta["chains"]),
                int(meta["thin"]),
                meta["seed"],
            )
            stored = list(self.prev_states or [])
            self.rngs = [restore_generator(st["rng"]) for st in stored]
            self.starts = [st["state"] for st in stored]
            self.burnin = 0
            self._meta = meta
        else:
            self.chains, self.thin, self.seed = int(chains), int(thin), seed
            self.rngs = list(make_rngs(self.chains))
            self.starts = [None] * self.chains
            if self.inits is not None:
                rows = resolve_start_values(
                    self.inits,
                    init_names,
                    self.chains,
                    support=support,
                    drawn_first=drawn_first,
                    support_text=support_text,
                )
                self.starts = [
                    row if to_state is None else to_state(row) for row in rows
                ]
            self.burnin = int(burnin)
            self._meta = {
                "chains": self.chains,
                "thin": self.thin,
                "seed": seed,
                "burnin": int(burnin),
                "draws": 0,
                "iterations": 0,
            }
        self.n_iter = self.burnin + self.draws * self.thin

    # ------------------------------------------------------------------
    def begin(self, c: int) -> None:
        """Progress line at the start of chain ``c``."""
        self.progress.say(
            f"chain {c + 1}/{self.chains}: {self.n_iter} iterations"
            + (f" ({self.burnin} burn-in)" if self.burnin else "")
            + (
                f", continuing from draw {self._meta['draws']}"
                if self.prev is not None
                else ""
            )
        )
        self._t0 = time.perf_counter()

    def end(
        self,
        c: int,
        state: Any,
        *,
        start: Optional[np.ndarray] = None,
        burn: Optional[np.ndarray] = None,
        accept: Optional[float] = None,
    ) -> None:
        """Record chain ``c``: its generator and ``state`` after the last
        iteration, its starting values on the reported scale (NaN for a
        parameter the first step draws before using it) and, with
        ``keep_burnin``, its burn-in rows."""
        self._end.append({"rng": self.rngs[c].bit_generator.state, "state": state})
        self._first.append(None if start is None else np.asarray(start, dtype=float))
        if self.keep_burnin and self.burnin and burn is not None:
            self._burn.append(np.asarray(burn))
        self.progress.say(
            f"chain {c + 1}/{self.chains} done in "
            f"{time.perf_counter() - self._t0:.2f} s"
            + (f", acceptance {accept:.2f}" if accept is not None else "")
        )

    def map_rows(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        """Map the recorded starting values and burn-in rows, given on the
        sampler's scale, to the reported scale (``fn`` maps a 2-d array
        row by row)."""
        self._first = [
            None if row is None else np.asarray(fn(row[None, :]))[0]
            for row in self._first
        ]
        self._burn = [np.asarray(fn(b)) for b in self._burn]

    def prev_state(self, c: int) -> Any:
        """The stored state of chain ``c`` of the run being continued."""
        return None if self.prev_states is None else self.prev_states[c]["state"]

    # ------------------------------------------------------------------
    def stack(
        self, pieces: Sequence[np.ndarray], before: Any = None, chain: Any = None
    ) -> np.ndarray:
        """The new rows of every chain after the earlier run's rows.

        ``before`` holds the earlier run's rows, chains stacked and
        labelled by ``chain`` (omitted for one chain).
        """
        if self.prev is None or before is None:
            return np.concatenate([np.asarray(p) for p in pieces], axis=0)
        before = np.asarray(before)
        lab = np.zeros(len(before), dtype=int) if chain is None else np.asarray(chain)
        return np.concatenate(
            [
                np.concatenate([before[lab == c], np.asarray(pieces[c])], axis=0)
                for c in range(self.chains)
            ],
            axis=0,
        )

    @property
    def draws_per_chain(self) -> int:
        return int(self._meta["draws"]) + self.draws if self.prev else self.draws

    def chain_index(self) -> np.ndarray:
        return np.repeat(np.arange(self.chains), self.draws_per_chain)

    def rate(self, new: Optional[float], old: Optional[float]) -> Optional[float]:
        """An acceptance rate over all iterations of both runs."""
        if new is None:
            return None
        if self.prev is None or old is None:
            return float(new)
        n_old = float(self._meta["iterations"])
        w_old = n_old / (n_old + self.n_iter)
        return w_old * float(old) + (1.0 - w_old) * float(new)

    # ------------------------------------------------------------------
    def attach(
        self,
        res: Any,
        names: Sequence[str],
        info: Optional[Dict[str, Any]] = None,
        frame_fields: bool = True,
    ) -> None:
        """Store the chain record on ``res``.

        ``frame_fields`` sets ``inits``, ``burnin_draws`` and
        ``burnin_chain`` (the fields of ``BayesRegressResult``), and the
        burn-in of a continued run is that of the first run.
        """
        meta = dict(self._meta)
        meta["draws"] = self.draws_per_chain
        meta["iterations"] = int(meta["iterations"]) + self.n_iter
        for st in self._end:
            st["meta"] = meta
        prev = self.prev
        if frame_fields:
            if prev is not None:
                res.inits = prev.inits
                res.burnin = prev.burnin
            elif all(row is not None for row in self._first):
                rows = [row for row in self._first if row is not None]
                res.inits = pd.DataFrame(np.vstack(rows), columns=list(names))
            if self._burn:
                res.burnin_draws = pd.DataFrame(
                    np.vstack(self._burn), columns=list(names)
                )
                res.burnin_chain = np.repeat(np.arange(self.chains), self.burnin)
            elif prev is not None:
                res.burnin_draws = prev.burnin_draws
                res.burnin_chain = prev.burnin_chain
        res._chain_state = self._end
        res._chain_key = self.key
        res._seed = self.seed
        if prev is not None and info is not None:
            info["continued"] = {
                "draws_before": int(self._meta["draws"]),
                "draws_added": self.draws,
                "exact": self.exact,
            }

    def done(self, min_ess: float, max_rhat: float) -> None:
        """Last progress line."""
        self.progress.say(
            f"{self.chains} chain(s) x {self.draws_per_chain} draws; min ESS "
            f"{min_ess:.0f}, max split R-hat {max_rhat:.3f}"
        )
