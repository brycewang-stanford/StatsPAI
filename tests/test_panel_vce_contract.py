"""``sp.panel(vce=...)`` refuses what it does not implement.

Until 2026-10 any ``vce=`` outside the extended menu fell through and
returned the classical standard errors: ``vce='robust'`` gave the same
numbers as no option, and so did ``vce='nonsense'``. Found while checking
whether Stata's ``xtreg, re vce(robust)`` could be translated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(40):
        a = rng.normal()
        for t in range(1, 7):
            x = rng.normal() + 0.3 * a
            rows.append((i, t, 1 + 0.9 * x + a + rng.normal(), x))
    return pd.DataFrame(rows, columns=["id", "t", "y", "x"])


def _se(frame, **kw):
    return float(sp.panel(frame, "y ~ x", entity="id", time="t", **kw).std_errors["x"])


@pytest.mark.parametrize("method", ["fe", "re"])
@pytest.mark.parametrize("vce", ["robust", "cluster", "hc1", "nonsense"])
def test_unknown_vce_is_refused_not_ignored(frame, method, vce):
    with pytest.raises(MethodIncompatibility, match="not a variance option"):
        sp.panel(frame, "y ~ x", entity="id", time="t", method=method, vce=vce)


def test_the_refusal_names_the_working_spelling(frame):
    with pytest.raises(MethodIncompatibility) as exc:
        sp.panel(frame, "y ~ x", entity="id", time="t", vce="robust")
    hint = str(exc.value)
    assert "robust='robust'" in hint and "cluster=" in hint


def test_supported_spellings_still_change_the_standard_error(frame):
    classical = _se(frame, method="fe")
    robust = _se(frame, method="fe", robust="robust")
    clustered = _se(frame, method="fe", cluster="id")
    cr2 = _se(frame, method="fe", vce="CR2", cluster="id")
    assert len({round(v, 10) for v in (classical, robust, clustered, cr2)}) == 4
    # Omitting vce= is still the classical SE.
    assert _se(frame, method="fe", vce=None) == classical


# ---------------------------------------------------------------------------
# The default small-sample convention is named in the result
# ---------------------------------------------------------------------------


def test_default_convention_is_recorded_and_differs_by_the_cluster_factor(frame):
    """``ssc=None`` is linearmodels' scaling: no G/(G-1), tests on N - K.

    On an entity-clustered within fit xtreg scales the sandwich by
    ``G/(G-1) * (N-1)/(N-K)`` with K counting the slope and the constant
    (the absorbed effects are nested in the clusters); linearmodels scales
    it by ``N/(N-k)`` with k the slopes only and has no cluster factor.
    Asserted so the recorded description stays true.
    """
    default = sp.panel(frame, "y ~ x", entity="id", time="t", cluster="id")
    stata = sp.panel(frame, "y ~ x", entity="id", time="t", cluster="id", ssc="stata")
    assert default.model_info["ssc"] == "linearmodels"
    assert "G/(G-1)" in default.model_info["ssc_reference"]
    assert stata.model_info["ssc"] == "stata"

    g, n = 40, len(frame)
    ratio = float(stata.std_errors["x"] / default.std_errors["x"]) ** 2
    xtreg = g / (g - 1) * (n - 1) / (n - 2)
    linearmodels = n / (n - 1)
    assert ratio == pytest.approx(xtreg / linearmodels, rel=1e-10)
    # Same coefficient, different reference distribution.
    assert float(default.params["x"]) == pytest.approx(
        float(stata.params["x"]), rel=1e-12
    )
    assert float(stata.pvalues["x"]) > float(default.pvalues["x"])


def test_few_cluster_warning_points_at_the_convention_only_under_the_default(frame):
    from statspai.exceptions import AssumptionWarning

    small = frame[frame["id"] < 12]
    with pytest.warns(AssumptionWarning) as caught:
        sp.panel(small, "y ~ x", entity="id", time="t", cluster="id")
    hints = [getattr(w.message, "recovery_hint", "") for w in caught]
    assert any("ssc='stata'" in h for h in hints)
    assert int(caught[0].message.diagnostics["n_clusters"]) == 12

    with pytest.warns(AssumptionWarning) as caught:
        sp.panel(small, "y ~ x", entity="id", time="t", cluster="id", ssc="stata")
    hints = [getattr(w.message, "recovery_hint", "") for w in caught]
    assert not any("ssc='stata'" in h for h in hints)
