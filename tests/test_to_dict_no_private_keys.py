"""``to_dict()`` never exports ``_``-prefixed internals.

Agent probe 2026-11-13, #45: ``MEGLMResult.to_dict()`` carried
``_cov_full``, ``_G``, ``_x_fixed`` ... (the MCP serializer scrubbed them,
a direct Python call did not). The class walk below found the same in
``MixedResult`` (its data blocks, ``_blocks``), ``CTreeResult``,
``FMMResult``, ``ETSResult``, ``SimpleForecastResult`` and
``DecompositionResult``: every dataclass result with private fields.

Rule (``_result_serialize.drop_private``): no key starting with ``_`` at
the top level of the export, nor inside its ``diagnostics`` /
``model_info`` mappings, at any ``detail``. Deeper, a leading underscore
can be data (a coefficient named ``_cons``) and is kept.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai._result_serialize import drop_private, result_to_dict
from statspai._schema_enrich import _init_assigned_names

DETAILS = ("minimal", "standard", "agent")


def _private_keys(payload: Dict[str, Any]) -> List[str]:
    bad = [str(k) for k in payload if str(k).startswith("_")]
    for container in ("diagnostics", "model_info"):
        inner = payload.get(container)
        if isinstance(inner, dict):
            bad += [f"{container}/{k}" for k in inner if str(k).startswith("_")]
    return bad


def _result_classes() -> Dict[str, type]:
    classes: Dict[str, type] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for info in pkgutil.walk_packages(sp.__path__, "statspai."):
            try:
                mod = importlib.import_module(info.name)
            except Exception:  # optional dependency missing
                continue
            for name, obj in vars(mod).items():
                if (
                    inspect.isclass(obj)
                    and obj.__module__ == mod.__name__
                    and (name.endswith("Result") or name.endswith("Results"))
                ):
                    classes[f"{obj.__module__}.{name}"] = obj
    return classes


def _sentinel(annotation: Any) -> Any:
    """A value of roughly the annotated kind (enough for ``to_dict``)."""
    text = str(annotation)
    if "DataFrame" in text:
        return pd.DataFrame({"a": [1.0]})
    if "Series" in text:
        return pd.Series([1.0], index=["a"])
    if "ndarray" in text:
        return np.array([1.0])
    if any(t in text for t in ("Dict", "dict", "Mapping")):
        return {}
    if any(t in text for t in ("List", "list", "Sequence", "Tuple", "tuple")):
        return []
    if "str" in text:
        return "x"
    if "bool" in text:
        return True
    if "int" in text:
        return 1
    return 1.0


def _construct(cls: type) -> Any:
    """An instance with every declared field set, private ones included.

    No estimator runs: the test is about which fields the export walks,
    not about their values.
    """
    obj = object.__new__(cls)
    if dataclasses.is_dataclass(cls):
        values = {f.name: _sentinel(f.type) for f in dataclasses.fields(cls)}
    else:
        values = {name: 1.0 for name in _init_assigned_names(cls)}
        values["_working_state"] = 1.0
    for name, value in values.items():
        object.__setattr__(obj, name, value)
    return obj


def test_no_result_class_exports_a_private_field():
    classes = _result_classes()
    assert len(classes) >= 380
    constructed = 0
    with_private_fields = 0
    leaks: List[str] = []
    for qual, cls in sorted(classes.items()):
        if not callable(getattr(cls, "to_dict", None)):
            continue
        try:
            obj = _construct(cls)
        except (TypeError, AttributeError):
            continue  # no bare instance (a NamedTuple, __slots__)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                payloads = {d: obj.to_dict(detail=d) for d in ("standard", "agent")}
        except Exception:
            # A hand-written to_dict that needs real values; the core
            # result trees among these are covered by the fits below.
            continue
        constructed += 1
        if dataclasses.is_dataclass(cls) and any(
            f.name.startswith("_") for f in dataclasses.fields(cls)
        ):
            with_private_fields += 1
        for detail, payload in payloads.items():
            bad = _private_keys(payload)
            if bad:
                leaks.append(f"{qual} [{detail}]: {bad}")
    # the walk is not vacuous: most classes are exercised, and some of
    # them do declare private fields
    assert constructed >= 330
    assert with_private_fields >= 15
    assert not leaks, "\n".join(leaks)


def _multilevel_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n, groups = 300, 30
    gid = np.repeat(np.arange(groups), n // groups)
    u = rng.normal(size=groups)[gid]
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "gid": gid,
            "x": x,
            "yb": (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.5 * x + u)))).astype(int),
            "yc": x + u + rng.normal(size=n),
        }
    )


def _fits():
    ml = _multilevel_frame()
    rng = np.random.default_rng(1)
    ts = pd.DataFrame({"y": np.cumsum(rng.normal(size=200))})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp = sp.datasets.mpdta()
        nsw = sp.datasets.nsw_dw()
    covs = ["age", "education", "re74", "re75"]
    return {
        "melogit": lambda: sp.melogit(ml, y="yb", x_fixed=["x"], group="gid"),
        "mixed": lambda: sp.mixed(ml, y="yc", x_fixed=["x"], group="gid"),
        "arima": lambda: sp.arima("y", order=(1, 1, 0), data=ts),
        "regress": lambda: sp.regress("re78 ~ treat + age", data=nsw),
        # stores _cluster_ids / _cell_bases in model_info for aggte
        "callaway_santanna": lambda: sp.callaway_santanna(
            mp, y="lemp", g="first_treat", t="year", i="countyreal"
        ),
        # stores _y_resid / _d_resid in model_info for the sensitivity tools
        "dml": lambda: sp.dml(nsw, y="re78", treat="treat", covariates=covs),
        "match": lambda: sp.match(nsw, y="re78", treat="treat", covariates=covs),
    }


_FITS = _fits()


@pytest.mark.parametrize("name", sorted(_FITS))
def test_fitted_result_exports_no_private_key(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _FITS[name]()
        payloads = {d: result.to_dict(detail=d) for d in DETAILS}
    for detail, payload in payloads.items():
        assert _private_keys(payload) == [], f"{name} [{detail}]"


def test_private_dataclass_fields_go_and_public_ones_stay():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = _FITS["mixed"]()
    declared = [f.name for f in dataclasses.fields(fit)]
    private = [n for n in declared if n.startswith("_")]
    assert "_blocks" in private  # the premise: the class keeps data there
    exported = fit.to_dict()
    assert set(exported) == {n for n in declared if not n.startswith("_")}
    # the estimator still has its working state
    assert all(hasattr(fit, n) for n in private)


def test_drop_private_keeps_underscore_names_that_are_data():
    payload = {
        "estimate": 1.0,
        "_cache": [1, 2],
        "coefficients": {"_cons": 0.5, "x": 1.5},
        "diagnostics": {"_y_resid": [0.1], "r2": 0.3},
        "model_info": {"_estimator": "obj", "n_folds": 5},
    }
    assert drop_private(payload) == {
        "estimate": 1.0,
        "coefficients": {"_cons": 0.5, "x": 1.5},
        "diagnostics": {"r2": 0.3},
        "model_info": {"n_folds": 5},
    }
    clean = {"estimate": 1.0, "diagnostics": {"r2": 0.3}}
    assert drop_private(clean) is clean  # untouched payloads are not copied


def test_result_to_dict_skips_private_dataclass_fields():
    @dataclasses.dataclass
    class _Toy:
        estimate: float
        _design: Any = None

    assert result_to_dict(_Toy(1.5, _design=np.ones(3))) == {"estimate": 1.5}
