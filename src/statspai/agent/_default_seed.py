"""Default seed for tool calls that leave a seed argument unset.

About half of the seed-taking estimators default to ``seed=None``, i.e. fresh
entropy on every call. For a person at a prompt that is the usual convention;
for an agent it means that repeating a tool call returns different numbers
and nothing in the first response lets a later reader reproduce it.

The library defaults are not changed here (that would move published seeded
numbers and belongs to a per-estimator migration). The tool dispatcher instead
fills the argument when the caller did not pass it and says so in the
response: ``seed_injected: {param, value, ...}``. The filled value is part of
the recorded arguments, so the ``replay`` snippet and the result card carry
it too.

``STATSPAI_MCP_DEFAULT_SEED`` sets the value (an integer, default 42);
``off`` / ``none`` disables the injection.
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Dict, Optional

from .._result_contract import _SEED_PARAMS

DEFAULT_SEED_ENV = "STATSPAI_MCP_DEFAULT_SEED"
DEFAULT_SEED = 42
_OFF = frozenset({"off", "none", "false", "no", ""})
#: ``rng`` takes a Generator on most functions, not an integer.
_INJECTABLE = tuple(p for p in _SEED_PARAMS if p != "rng")


# --------------------------------------------------------------------------- #
#  Was the filled seed consumed?
# --------------------------------------------------------------------------- #
#
# ``sp.did`` takes ``seed=`` for its bootstrap; a 2x2 fit with analytic
# standard errors never reads it. Reporting ``seed_injected`` there told the
# agent a stochastic step had been pinned when the call has none, and put a
# ``seed=42`` in the replay line that the result card (which records what
# the estimator used) did not know about.
#
# The filled value is therefore an ``int`` that notices being used as a
# seed: seeding a NumPy ``Generator`` / ``SeedSequence`` / bit generator
# converts it through ``__index__``-style slots, the legacy
# ``RandomState`` / ``np.random.seed`` path asks for ``.squeeze``, and
# derived seeds (``seed + fold``) go through arithmetic. Comparisons,
# ``isinstance`` checks, ``repr`` and JSON encoding do not count.

_USES = [0]  # process-wide count of consumptions; compared before / after


def _noting(name: str) -> Any:
    base = getattr(int, name)

    def method(self: Any, *args: Any) -> Any:
        _USES[0] += 1
        return base(self, *args)

    method.__name__ = name
    return method


class WatchedSeed(int):
    """An ``int`` seed that records when something seeds a generator with it."""

    __slots__ = ()

    def squeeze(self) -> int:
        # numpy's legacy seeding (``RandomState(seed)``, ``np.random.seed``)
        # calls ``seed.squeeze()`` when the attribute exists.
        _USES[0] += 1
        return int.__int__(self)

    def __reduce__(self) -> Any:
        return (WatchedSeed, (int.__int__(self),))


for _name in (
    "__index__",
    "__int__",
    "__float__",
    "__trunc__",
    "__hash__",
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__mod__",
    "__rmod__",
    "__floordiv__",
    "__divmod__",
    "__pow__",
    "__and__",
    "__rand__",
    "__or__",
    "__ror__",
    "__xor__",
    "__rxor__",
    "__lshift__",
    "__rshift__",
    "__neg__",
    "__abs__",
):
    setattr(WatchedSeed, _name, _noting(_name))
del _name


def seed_uses() -> int:
    """Marker to take before a call; pass it to :func:`settle_injected_seed`."""
    return _USES[0]


def settle_injected_seed(
    note: Optional[Dict[str, Any]],
    marker: int,
    *argument_dicts: Dict[str, Any],
    result: Any = None,
) -> Optional[Dict[str, Any]]:
    """After the call: keep the note only if the filled seed was consumed.

    When a generator was seeded with it, the recorded arguments keep the
    value as a plain ``int`` (the replay line and the cached record carry
    it) and the note is returned. When nothing read it, the call had no
    stochastic step that the seed controls: the argument is removed from
    the recorded dicts, the result's own call record goes back to the
    library default (``None``), and ``None`` is returned, so the response,
    the replay and the result card describe the same call.
    """
    if not note:
        return None
    name = note["param"]
    used = _USES[0] != marker
    for args in argument_dicts:
        if not isinstance(args.get(name), WatchedSeed):
            continue
        if used:
            args[name] = int.__int__(args[name])
        else:
            del args[name]
    records = [
        getattr(getattr(result, "_provenance", None), "params", None),
        getattr(result, "model_info", None),
    ]
    for record in records:
        if isinstance(record, dict) and isinstance(record.get(name), WatchedSeed):
            record[name] = int.__int__(record[name]) if used else None
    return note if used else None


def reconcile_card_seed(card: Any, note: Optional[Dict[str, Any]]) -> None:
    """Let the result card state a seed the tool layer filled and the call used.

    An estimator that attaches a curated call record (or none) leaves the
    card saying "the seed used is unknown" beside a ``seed_injected`` note
    and a replay line that both name it.
    """
    if not note or not isinstance(card, dict):
        return
    prov = card.get("provenance")
    if not isinstance(prov, dict) or prov.get("seed_source") != "not_recorded":
        return
    prov["seed"] = note["value"]
    prov["reproducible"] = True
    prov["seed_source"] = "tool_layer"
    prov["seed_note"] = (
        f"{note['param']}={note['value']} was set by the tool layer "
        "(see seed_injected) and seeded a generator in this call; the "
        "estimator's own call record does not include it."
    )


def default_seed() -> Optional[int]:
    """The configured default seed, or ``None`` when injection is off."""
    raw = os.environ.get(DEFAULT_SEED_ENV)
    if raw is None:
        return DEFAULT_SEED
    text = raw.strip().lower()
    if text in _OFF:
        return None
    try:
        return int(text)
    except ValueError:
        # An unreadable setting must not silently change what a call does.
        return None


def inject_default_seed(
    fn: Callable[..., Any], arguments: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Fill the first unset seed argument of ``fn`` in ``arguments``.

    Touches only a parameter that is named like a seed, defaults to ``None``
    and was not passed (an explicit ``null`` from the caller is respected).
    Returns the note to attach to the response, or ``None`` when nothing was
    filled. The filled value is a :class:`WatchedSeed`; after the call,
    :func:`settle_injected_seed` decides whether the note is reported.
    """
    seed = default_seed()
    if seed is None:
        return None
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    if any(name in arguments for name in _SEED_PARAMS):
        return None
    for name in _INJECTABLE:
        param = params.get(name)
        if param is None or param.default is not None:
            continue
        arguments[name] = WatchedSeed(seed)
        return {
            "param": name,
            "value": seed,
            "note": (
                f"{name}= was not passed and defaults to None (fresh "
                "entropy for any stochastic step); the tool layer set "
                f"{name}={seed} so the call can be repeated. Pass {name}= "
                "yourself to choose another "
                "value; a direct Python call keeps the library default."
            ),
        }
    return None


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_SEED_ENV",
    "WatchedSeed",
    "default_seed",
    "inject_default_seed",
    "reconcile_card_seed",
    "seed_uses",
    "settle_injected_seed",
]
