"""Every argument a registered function accepts reaches its agent schema.

Hand-written ``params=[...]`` lists in ``registry.py`` were written once
and then drifted: 116 entries hid 400+ parameters from
``sp.function_schema()`` (``synth`` had no ``covariates`` / ``placebo``,
``regtable`` no ``keep`` / ``drop``, ``rdrobust`` no ``vce``).
``registry._apply_signature_param_completion`` now appends what is missing,
read off the code: name, required-ness and default from the signature,
description from the function's ``Parameters`` section, or from the function
it forwards the argument to (``registry.PARAM_DOC_FORWARDS``).
"""

from __future__ import annotations

import inspect

import pytest

import statspai as sp
from statspai import registry as R

IMPORTANT = (
    "synth",
    "dml",
    "ipw",
    "regtable",
    "rdrobust",
    "causal_forest",
    "did",
    "esttab",
    "modelsummary",
    "metalearner",
    "tmle",
    "ivreg",
    "regress",
)

_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _schema_props(name):
    return sp.function_schema(name)["parameters"]["properties"]


@pytest.mark.parametrize("name", IMPORTANT)
def test_every_signature_parameter_is_in_the_schema(name):
    hidden = set(R.SIGNATURE_PARAMS_NOT_IN_SCHEMA.get(name, ()))
    props = _schema_props(name)
    sig = inspect.signature(getattr(sp, name))
    for pname, p in sig.parameters.items():
        if p.kind in _VARIADIC or pname.startswith("_") or pname in hidden:
            continue
        assert pname in props, f"{name}.{pname} missing from function_schema"
        desc = props[pname].get("description", "")
        assert desc.strip(), f"{name}.{pname} has an empty description"
        assert not desc.startswith(f"{pname} parameter"), (
            f"{name}.{pname} fell back to the generic placeholder; document "
            "it in the function's Parameters section"
        )


def test_derived_description_is_the_docstring_entry():
    # ``ipw(normalize=)`` was missing from the hand-written entry.
    props = _schema_props("ipw")
    desc = props["normalize"]["description"]
    assert desc.startswith("If True, use Hajek (normalised) weights.")
    assert "Horvitz-Thompson" in desc
    assert props["normalize"]["type"] == "boolean"
    assert props["normalize"]["default"] is True
    assert props["normalize"]["x-statspai-source"] == "signature"
    assert "normalize" not in sp.function_schema("ipw")["parameters"]["required"]


def test_describe_function_marks_derived_parameters():
    rows = {r["name"]: r for r in sp.describe_function("synth")["params"]}
    assert rows["placebo"]["source"] == "signature"
    assert rows["placebo"]["default"] is True
    assert "source" not in rows["outcome"]  # hand-written


def test_hand_written_entries_win():
    # ``did`` writes its own ``method`` entry; the pass must not replace or
    # duplicate it.
    names = [p.name for p in R._REGISTRY["did"].params]
    assert len(names) == len(set(names))
    method = next(p for p in R._REGISTRY["did"].params if p.name == "method")
    assert not getattr(method, "_derived", False)


def test_forwarded_descriptions_come_from_the_target():
    from statspai._schema_export import ASCII_TRANSLITERATION as ascii_
    from statspai.dml.double_ml import dml as dml_impl
    from statspai.synth.scm import _dispatch_synth_impl

    # The docstring text, in the ASCII form the committed schemas carry.
    props = _schema_props("dml")
    assert props["model_y"]["description"] == (
        R._doc_param_table(dml_impl)["ml_g"]["description"].translate(ascii_)
    )
    synth = _schema_props("synth")
    assert synth["penalization"]["description"] == (
        R._doc_param_table(_dispatch_synth_impl)["penalization"][
            "description"
        ].translate(ascii_)
    )


def test_every_forward_target_documents_what_it_is_credited_with():
    for name, table in R.PARAM_DOC_FORWARDS.items():
        live = inspect.signature(getattr(sp, name)).parameters
        for pname, target in table.items():
            path, _, other = target.partition(":")
            obj = R._resolve_dotted(path)
            assert obj is not None, (name, target)
            docs = R._doc_param_table(obj)
            if pname == "*":
                continue
            assert pname in live, (name, pname)
            assert docs.get(other or pname, {}).get("description"), (name, target)


def test_hidden_parameters_stay_out_of_the_schema():
    for name, hidden in R.SIGNATURE_PARAMS_NOT_IN_SCHEMA.items():
        props = _schema_props(name)
        for pname in hidden:
            assert pname not in props, (name, pname)


def test_typeless_numpy_header_is_parsed():
    doc = (
        "Summary.\n\nParameters\n----------\n"
        "rope : (float, float), optional\n"
        "    Region of practical equivalence.\n"
        "hdi_prob, draws :\n"
        "    Sampler controls.\n"
    )
    out = R._parse_docstring_params(doc)
    assert out["rope"]["description"] == "Region of practical equivalence."
    assert out["hdi_prob"]["description"] == "Sampler controls."
    assert out["draws"]["description"] == "Sampler controls."
