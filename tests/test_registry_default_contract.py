"""Contract: hand-written spec defaults must not contradict the signature.

Item #1 locked param *names*; this locks *default values*. When a hand-written
``ParamSpec`` pins a concrete ``default`` for a parameter that exists in the
signature, that default must match what the function actually uses — otherwise
``describe_function`` tells an agent the wrong default and it reasons from a false
premise (e.g. "n_trees defaults to 200" when the estimator uses 50).

Scope, kept deliberately narrow to avoid false positives:

* only hand-written specs (auto specs derive defaults from the signature),
* only params present in the signature (dispatcher logical params routed through
  ``**kwargs`` are skipped),
* only when the spec pins a *concrete* default (``default is not None``) and the
  signature param is *optional* (has a default to compare against),
* ``list``/``tuple`` are treated as equivalent (JSON-identical), and floats are
  compared with tolerance — so benign representational differences don't trip it.
"""

import inspect

import pytest

import statspai as sp
from statspai import registry as R


def _norm(v):
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    return v


def _matches(spec_default, sig_default) -> bool:
    a, b = _norm(spec_default), _norm(sig_default)
    if a == b:
        return True
    if (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    ):
        return abs(a - b) < 1e-9
    return False


def _hand_written_callables():
    R._ensure_full_registry()
    out = []
    for name, spec in R._REGISTRY.items():
        if getattr(spec, "_auto", False):
            continue
        obj = getattr(sp, name, None)
        if obj is None or not callable(obj):
            continue
        try:
            inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        out.append(name)
    return sorted(out)


HAND = _hand_written_callables()


@pytest.mark.parametrize("name", HAND)
def test_spec_defaults_match_signature(name):
    obj = getattr(sp, name)
    sig = inspect.signature(obj)
    sig_defaults = {
        p.name: p.default
        for p in sig.parameters.values()
        if p.default is not inspect.Parameter.empty
    }
    bad = []
    for ps in R._REGISTRY[name].params:
        if ps.default is None:
            continue  # spec didn't pin a default — incompleteness, not a lie
        if ps.name not in sig_defaults:
            continue  # required sig param, or dispatcher logical param
        if not _matches(ps.default, sig_defaults[ps.name]):
            bad.append((ps.name, ps.default, sig_defaults[ps.name]))
    assert not bad, (
        f"{name}: spec advertises default(s) that contradict the signature "
        f"(param, spec_default, real_default): {bad}. Fix the ParamSpec default "
        f"in registry.py so describe_function('{name}') stops misleading agents."
    )


@pytest.mark.parametrize("name", HAND)
def test_spec_default_is_in_its_own_enum(name):
    """A pinned default that is not among the advertised enum choices is a lie.

    Catches the compound drift where both the default *and* the choice list went
    stale (e.g. ``drdid`` defaulting to ``'imp'`` while the enum listed only
    ``['dr','or','ipw',...]``): an agent that trusts the enum never tries the
    real default, and validation against the enum would reject the true default.
    """
    bad = []
    for ps in R._REGISTRY[name].params:
        if ps.enum and ps.default is not None and ps.default not in ps.enum:
            bad.append((ps.name, ps.default, ps.enum))
    assert not bad, (
        f"{name}: ParamSpec default(s) absent from their own enum "
        f"(param, default, enum): {bad}. Either the default or the enum is stale."
    )
