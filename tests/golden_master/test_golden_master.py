"""Version golden-master: flagship headline outputs must not drift silently.

This suite pins the headline numbers of flagship estimators (``golden_values.json``)
and fails if a future StatsPAI version changes them. It is StatsPAI's
**reproducibility guarantee made mechanical**: a researcher who reruns an
analysis after upgrading must get the same number, unless the change is a
*declared* correctness fix.

When a change here fails:

* If the change is **unintended**, you introduced a numerical regression — find
  and fix it.
* If the change is an **intended correctness fix**, document it in
  ``CHANGELOG.md`` under ``⚠️ Correctness`` and ``MIGRATION.md`` (per CLAUDE.md
  §12), then re-pin with ``STATSPAI_UPDATE_GOLDEN=1 pytest
  tests/golden_master/`` so the new value becomes the baseline.

Never re-pin to make a red build green without understanding *why* the number
moved — that is exactly the silent regression this gate exists to catch.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from _cases import compute_all  # noqa: E402  (tests/golden_master on sys.path)

_GOLDEN = Path(__file__).resolve().parent / "golden_values.json"

# Headline estimates are pinned tightly; these are deterministic computations on
# fixed seeds, so the only legitimate source of drift across machines is
# last-bit floating point. A genuine algorithm change moves digits well above
# this tolerance.
_RTOL = 1e-6
_ATOL = 1e-8


def _load_golden() -> dict:
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


def test_golden_master_no_silent_drift():
    current = compute_all()

    if os.environ.get("STATSPAI_UPDATE_GOLDEN") == "1":
        _GOLDEN.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pytest.skip("golden values re-pinned (STATSPAI_UPDATE_GOLDEN=1)")

    if not _GOLDEN.exists():  # pragma: no cover - first-run bootstrap
        pytest.fail("golden_values.json missing; run STATSPAI_UPDATE_GOLDEN=1 first")

    golden = _load_golden()

    # New cases must be pinned deliberately, not auto-accepted.
    new_cases = set(current) - set(golden)
    assert not new_cases, (
        f"unpinned golden cases {sorted(new_cases)} — run STATSPAI_UPDATE_GOLDEN=1 "
        "to pin them after confirming the values are correct."
    )

    drifted = []
    for case, metrics in golden.items():
        assert case in current, f"golden case '{case}' no longer produced"
        for metric, want in metrics.items():
            got = current[case].get(metric)
            assert got is not None, f"{case}.{metric} no longer produced"
            if abs(got - want) > _ATOL + _RTOL * abs(want):
                drifted.append((f"{case}.{metric}", want, got))

    assert not drifted, (
        "Flagship headline outputs drifted vs the pinned golden master:\n"
        + "\n".join(f"  {name}: pinned {w!r} → now {g!r}" for name, w, g in drifted)
        + "\n\nIf this is an intended correctness fix, document it in CHANGELOG "
        "(⚠️ Correctness) + MIGRATION, then re-pin with STATSPAI_UPDATE_GOLDEN=1."
    )
