"""Starting values, continuation and progress output for MCMC chains.

Used by :func:`statspai.bayes_regress`. A chain is described by its
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
from typing import Any, Dict, List, Mapping, Sequence

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
    names = [str(n) for n in names]
    k = len(names)

    def from_mapping(m: Mapping[str, Any], where: str) -> np.ndarray:
        keys = [str(x) for x in m.keys()]
        unknown = [x for x in keys if x not in names]
        missing = [x for x in names if x not in keys]
        if unknown or missing:
            raise MethodIncompatibility(
                f"inits{where}: "
                + (f"unknown parameters {unknown}; " if unknown else "")
                + (f"no value for {missing}; " if missing else "")
                + f"the parameters are {names}.",
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
        try:
            lk = float(model.log_kernel(model.to_u(row)))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            lk = -np.inf
        if not np.isfinite(lk):
            raise MethodIncompatibility(
                f"inits for chain {c} lie outside the support of the "
                "posterior (zero density there): check positive variances "
                "and scales and increasing cutpoints.",
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
