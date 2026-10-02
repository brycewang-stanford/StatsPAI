"""Track C timings are bound to the code they timed (scripts/trace_perf_path.py).

The record says which source files a timing depends on; the timings stay
valid while those files are unchanged, whatever the version number. This test
holds the record's shape and the staleness logic. It does not require the
timings to be current on ``main``: that is a question asked at a release or
when a paper is built, by ``python scripts/trace_perf_path.py --check``.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "trace_perf_path", SCRIPTS / "trace_perf_path.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tpp = _load()
RECORD = json.loads(tpp.RECORD.read_text(encoding="utf-8"))


def test_every_track_c_module_is_traced_without_error():
    assert set(RECORD) == set(tpp.MODULES)
    for stem, rec in RECORD.items():
        assert rec["error"] is None, stem
        assert rec["exercised_sources"], stem
        assert rec["timings_measured_with"], stem
        assert set(rec["harness_sha256"]) == set(tpp.HARNESS)


def test_the_timed_path_contains_the_estimator_it_times():
    expected = {
        "01_hdfe": "src/statspai/fast/",
        "02_csdid": "src/statspai/did/callaway_santanna.py",
        "03_scm": "src/statspai/synth/scm.py",
        "04_dml": "src/statspai/dml/",
    }
    for stem, needle in expected.items():
        assert any(needle in rel for rel in RECORD[stem]["exercised_sources"]), stem


def test_the_record_matches_the_committed_timing_files():
    for stem, rec in RECORD.items():
        assert rec["timings_measured_with"] == tpp._measured_version(stem), (
            f"{stem} was re-measured without re-tracing: "
            "run python scripts/trace_perf_path.py on the measured tree"
        )


def test_a_changed_timed_path_file_makes_the_timings_stale():
    stem = "02_csdid"
    rec = copy.deepcopy(RECORD[stem])
    baseline = tpp.stale_reasons(stem, rec)
    victim = "src/statspai/did/callaway_santanna.py"
    rec["exercised_sources"][victim] = "0" * 64
    reasons = tpp.stale_reasons(stem, rec)
    assert f"{victim} changed" in reasons
    assert len(reasons) == len(baseline) + (f"{victim} changed" not in baseline)


def test_a_version_bump_alone_does_not_stale_the_timings(tmp_path):
    # Hashes mask the __version__ line, so a documentation-only release keeps
    # every timing valid (1.34.1 and 1.34.2 after 1.34.0).
    init = REPO_ROOT / "src" / "statspai" / "__init__.py"
    text = init.read_text(encoding="utf-8")
    bumped = tmp_path / "statspai" / "__init__.py"
    bumped.parent.mkdir()
    import re

    bumped.write_text(
        re.sub(r'__version__ = "[^"]+"', '__version__ = "99.0.0"', text),
        encoding="utf-8",
    )
    assert tpp._sha256(bumped) == tpp._sha256(init)
