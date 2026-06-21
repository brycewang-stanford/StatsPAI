"""Contract for the machine-readable schema bundle (Day-4 agent-native).

Locks three things:

1. The agent-facing *result* payload that ``execute_tool`` / the MCP
   server return — ``result.to_dict(detail='agent')`` — actually
   validates against the published ``RESULT_AGENT_SCHEMA``, for BOTH the
   causal-effect shape (``sp.did`` -> CausalResult) and the regression
   shape (``sp.regress`` -> EconometricResults).  If a refactor changes
   the envelope an agent reasons over, this fails.
2. Every advertised tool entry carries the ``{name, description,
   input_schema}`` triple an agent needs to call it.
3. The committed ``schemas/`` bundle is in sync with the live surface
   (delegates to ``scripts/dump_schemas.py --check``), so the offline
   artifact never silently drifts from the package.

No network / R / Stata.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai._schema_export import (
    RESULT_AGENT_SCHEMA,
    build_schemas,
    render_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = REPO_ROOT / "schemas"


def _jsonschema():
    return pytest.importorskip("jsonschema")


# --------------------------------------------------------------------------- #
#  Real fitted-result payloads validate against the published result schema
# --------------------------------------------------------------------------- #


def _did_dataset() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    n = 400
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n // 2), 2),
            "time": np.tile([0, 1], n // 2),
        }
    )
    df["treat"] = (df["id"] % 2 == 0).astype(int)
    df["post"] = df["time"]
    df["y"] = 1.0 + 0.5 * df["treat"] * df["post"] + rng.normal(0, 1, len(df))
    return df


def _ols_dataset() -> pd.DataFrame:
    rng = np.random.RandomState(1)
    n = 300
    x = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    y = 0.3 + 1.2 * x - 0.5 * z + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_causal_result_agent_payload_matches_schema():
    jsonschema = _jsonschema()
    import statspai as sp

    r = sp.did(_did_dataset(), y="y", treat="treat", time="post")
    payload = r.to_dict(detail="agent")
    # Round-trips through JSON (agents receive it as JSON) ...
    payload = json.loads(json.dumps(payload))
    # ... and conforms to the published contract.
    jsonschema.validate(payload, RESULT_AGENT_SCHEMA)
    # Envelope an agent reasons over is well-formed when present.
    assert isinstance(payload.get("warnings", []), list)
    assert isinstance(payload.get("suggested_functions", []), list)


def test_econometric_result_agent_payload_matches_schema():
    jsonschema = _jsonschema()
    import statspai as sp

    r = sp.regress("y ~ x + z", data=_ols_dataset())
    payload = json.loads(json.dumps(r.to_dict(detail="agent")))
    jsonschema.validate(payload, RESULT_AGENT_SCHEMA)
    # Regression payloads carry a coefficient table.
    assert "coefficients" in payload


def test_minimal_and_standard_payloads_also_validate():
    """The schema is permissive enough to accept the leaner detail levels."""
    jsonschema = _jsonschema()
    import statspai as sp

    r = sp.did(_did_dataset(), y="y", treat="treat", time="post")
    for detail in ("minimal", "standard", "agent"):
        payload = json.loads(json.dumps(r.to_dict(detail=detail)))
        jsonschema.validate(payload, RESULT_AGENT_SCHEMA)


def test_result_schema_is_itself_a_valid_jsonschema():
    jsonschema = _jsonschema()
    # Will raise SchemaError if the schema document is malformed.
    jsonschema.Draft202012Validator.check_schema(RESULT_AGENT_SCHEMA)


# --------------------------------------------------------------------------- #
#  Tool manifest entries are complete
# --------------------------------------------------------------------------- #


def test_every_tool_entry_has_the_call_triple():
    bundle = build_schemas()
    bad = []
    for t in bundle["tools"]:
        if not t.get("name"):
            bad.append(repr(t)[:60])
            continue
        if not isinstance(t.get("description"), str):
            bad.append(f"{t['name']}: missing description")
        if not isinstance(t.get("input_schema"), dict):
            bad.append(f"{t['name']}: missing input_schema")
    assert not bad, (
        "Tool entries missing the {name, description, input_schema} triple:\n  "
        + "\n  ".join(bad)
    )


def test_bundle_renders_to_valid_json():
    files = render_files(build_schemas())
    assert {
        "index.json",
        "tools.json",
        "functions.json",
        "agent_cards.json",
        "result.schema.json",
    } <= set(files)
    for fname, text in files.items():
        json.loads(text)  # raises on malformed JSON


def _walk_strings(obj, path="$"):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from _walk_strings(item, f"{path}[{i}]")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _walk_strings(str(key), f"{path}.{key!s}<key>")
            yield from _walk_strings(value, f"{path}.{key!s}")


def test_rendered_bundle_strings_are_ascii_normalized():
    """No hidden Unicode escapes should reappear after JSON decoding.

    JSS requires submitted source files to be ASCII.  The archive packager
    therefore normalizes source files, and the schema bundle must be stable
    before and after that normalization.  Checking decoded strings catches
    JSON-safe but semantically non-ASCII escapes such as ``\u2014``.
    """
    files = render_files(build_schemas())
    bad = []
    double_spaced = []
    for fname, text in files.items():
        payload = json.loads(text)
        for path, value in _walk_strings(payload):
            if any(ord(ch) > 127 for ch in value):
                bad.append(f"{fname}:{path}: {value!r}")
            if "  " in value:
                double_spaced.append(f"{fname}:{path}: {value!r}")
    assert not bad, "Non-ASCII decoded schema strings:\n  " + "\n  ".join(bad[:20])
    assert (
        not double_spaced
    ), "Schema strings with repeated spaces after normalization:\n  " + "\n  ".join(
        double_spaced[:20]
    )


def test_tool_descriptions_do_not_contain_join_artifacts():
    """Agent-facing descriptions should not expose sloppy sentence joins."""
    files = render_files(build_schemas())
    bad = []
    sentence_double_period = re.compile(r"(?<!\.)\.\.(?:\s|$)")
    for fname in ("tools.json", "functions.json", "agent_cards.json"):
        payload = json.loads(files[fname])
        entries = payload.values() if isinstance(payload, dict) else payload
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            candidates = [entry.get("description")]
            signature = entry.get("signature")
            if isinstance(signature, dict):
                candidates.append(signature.get("description"))
            for value in candidates:
                if isinstance(value, str) and sentence_double_period.search(value):
                    bad.append(f"{fname}[{i}]: {value}")
    assert (
        not bad
    ), "Schema descriptions contain double-period sentence joins:\n  " + "\n  ".join(
        bad[:20]
    )


def test_scoped_certified_limitations_do_not_claim_unqualified_parity():
    """Certified-but-limited tools must not overstate exact parity in schemas."""
    import statspai as sp

    schema = sp.function_schema("rddensity")
    desc = schema["description"]

    assert "Validation: certified evidence with scoped limitations." in desc
    assert "Validation: certified parity evidence." not in desc
    assert "backend='r'" in desc
    assert "not a reference-parity guarantee" in desc


def test_limited_certified_agent_cards_use_scoped_wording():
    """Agent cards should not make limited certified evidence look absolute."""
    import statspai as sp

    for card in sp.agent_cards(validation_status="certified"):
        if not card.get("limitations"):
            continue
        for key in ("description", "signature"):
            desc = card[key]["description"] if key == "signature" else card[key]
            assert "Validation: certified evidence with scoped limitations." in desc, (
                card["name"],
                key,
            )
            assert "Validation: certified parity evidence." not in desc, (
                card["name"],
                key,
            )


def test_validated_schema_wording_is_evidence_tier_not_blanket_claim():
    """Validated functions should read as scoped evidence, not marketing."""
    import statspai as sp

    desc = sp.function_schema("aipw")["description"]

    assert "Validation: validated evidence tier" in desc
    assert "Validation: validated" "." not in desc
    assert "external-parity" in desc or "Monte Carlo" in desc


def test_tool_manifest_validation_wording_matches_registry_tiers():
    """MCP tool descriptions must not drop or overstate validation tiers."""
    import statspai as sp

    registered = set(sp.list_functions())
    for tool in sp.agent.tools.tool_manifest():
        name = tool["name"]
        if name not in registered:
            continue
        desc = tool["description"]
        status = sp.describe_function(name)["validation_status"]
        has_certified = "Validation: certified" in desc
        has_validated = "Validation: validated evidence tier" in desc
        if status == "certified":
            assert has_certified, name
            assert not has_validated, name
        elif status == "validated":
            assert has_validated, name
            assert not has_certified, name
        else:
            assert not has_certified, name
            assert not has_validated, name


def test_experimental_schema_prefix_is_not_duplicated():
    """Descriptions that already include [experimental] should not repeat it."""
    import statspai as sp

    desc = sp.function_schema("text_treatment_effect")["description"]

    assert desc.startswith("[experimental] ")
    assert "[experimental] " "[experimental]" not in desc
    assert "Validation status: experimental." in desc


# --------------------------------------------------------------------------- #
#  Committed bundle stays in sync with the live surface
# --------------------------------------------------------------------------- #


def test_schema_bundle_uses_posix_path_separators():
    """Validation-evidence notes must be OS-independent (forward slashes).

    A Windows runner that recorded ``tests\\reference_parity\\...`` notes both
    failed the JSS evidence-grade marker check and drifted ``agent_cards.json``
    away from the POSIX-generated committed bundle. This invariant catches the
    regression on *any* OS, not only on the Windows shard whose regeneration
    diverged — the `--check` guard alone fires only where the artifact differs.
    """
    files = render_files(build_schemas())
    bad = []
    for fname in ("functions.json", "agent_cards.json", "tools.json"):
        for _path, value in _walk_strings(json.loads(files[fname])):
            if re.search(r"tests\\", value) or re.search(
                r"\\(reference_parity|external_parity|r_parity)", value
            ):
                bad.append(f"{fname}:{_path}: {value!r}")
    assert not bad, (
        "OS-native (backslash) path separators leaked into the schema bundle "
        "— use Path.as_posix():\n  " + "\n  ".join(bad[:20])
    )


def test_schema_bundle_has_no_internal_pandas_paths():
    """Annotations must use the public pandas path, stable across versions.

    pandas >= 3.0 renders ``pandas.DataFrame`` while older pandas renders the
    internal ``pandas.core.frame.DataFrame``; pinning to the public path keeps
    ``functions.json`` byte-identical regardless of which pandas a runner
    resolves (pandas 3.0 needs Python >= 3.11, so this used to split the CI
    matrix). Guards :func:`statspai.registry._canonicalize_annotation_path`.
    """
    files = render_files(build_schemas())
    bad = []
    for fname in ("functions.json", "agent_cards.json", "tools.json"):
        for _path, value in _walk_strings(json.loads(files[fname])):
            if "pandas.core." in value:
                bad.append(f"{fname}:{_path}: {value!r}")
    assert not bad, (
        "Internal pandas module paths leaked into the schema bundle "
        "(version-fragile):\n  " + "\n  ".join(bad[:20])
    )


def test_committed_schemas_dir_is_in_sync():
    """`scripts/dump_schemas.py --check` must pass against committed schemas/.

    If this fails, run `python scripts/dump_schemas.py` and commit the
    refreshed bundle — the offline artifact has drifted from the package.
    """
    if not SCHEMAS_DIR.exists():
        pytest.skip("schemas/ not generated in this checkout")
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "dump_schemas.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert (
        res.returncode == 0
    ), f"schemas/ is stale:\nstdout={res.stdout!r}\nstderr={res.stderr!r}"
