"""The few-cluster size study: stored results are reproducible and say what we say.

``tests/reliability/few_clusters.py`` simulates the rejection rate of a
true null under four inference methods across 16 designs (2,000
replications each) and writes ``few_clusters_results.json``. This file
checks three things: the stored file has the declared shape; one cell
recomputed on its first 60 replications reproduces the stored counts
exactly (same seeds, deterministic); and the statements the package makes
about few clusters (``FEW_CLUSTERS_HINT``) are the ones the numbers
support, each with its Monte Carlo error taken into account.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from statspai.core._agent_summary import FEW_CLUSTERS_HINT

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "reliability" / "few_clusters.py"
RESULTS = ROOT / "reliability" / "few_clusters_results.json"


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("few_clusters", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _cell(results, G, treated, sizes):
    (cell,) = [
        c
        for c in results["cells"]
        if (c["G"], c["treated"], c["sizes"]) == (G, treated, sizes)
    ]
    return cell


def _rate(results, G, treated, sizes, method):
    c = _cell(results, G, treated, sizes)[method]
    return c["rejection_rate"], c["mc_se"]


def test_file_has_the_declared_design(study, results):
    assert results["B"] == 2000 and results["alpha"] == 0.05
    keys = {(c["G"], c["treated"], c["sizes"]) for c in results["cells"]}
    assert keys == {
        (g, t, s) for g in study.G_VALUES for t in study.TREATED for s in study.SIZES
    }
    assert len(results["cells"]) == 16
    for c in results["cells"]:
        for m in study.METHODS:
            assert 0.0 <= c[m]["rejection_rate"] <= 1.0


def test_a_cell_reproduces_its_stored_prefix(study, results):
    """Same seeds, same counts: the stored file came from this script."""
    fresh = study.run_cell(6, "two", "unbalanced", study.PREFIX)
    stored = _cell(results, 6, "two", "unbalanced")
    for m in study.METHODS:
        assert (
            fresh[m]["prefix_rejections"] == stored[m]["prefix_rejections"]
        ), f"{m}: recomputed prefix differs from the stored one"


def test_balanced_half_treated_is_where_every_method_settles(results):
    for method in ("cr1", "cr3", "wild"):
        rate, se = _rate(results, 40, "half", "balanced", method)
        assert abs(rate - 0.05) < 3 * se + 0.005, (method, rate)
    # ... and CR1 with t(G - 1) is already mild at six clusters
    rate, _ = _rate(results, 6, "half", "balanced", "cr1")
    assert 0.06 < rate < 0.11


def test_cr2_and_cr3_on_the_t_reference_hold_their_size_when_balanced(results):
    """On a normal reference CR2 rejected 12% at six clusters; on t(G-1), 6%."""
    for G in (6, 10, 20, 40):
        cr2, se2 = _rate(results, G, "half", "balanced", "cr2")
        cr3, _ = _rate(results, G, "half", "balanced", "cr3")
        assert cr2 < 0.06 + 2 * se2, (G, cr2)
        assert 0.02 < cr3 < 0.055, (G, cr3)


def test_wild_bootstrap_is_near_nominal_with_similar_clusters(results):
    for G in (6, 10, 20, 40):
        rate, se = _rate(results, G, "half", "balanced", "wild")
        assert 0.04 - 2 * se < rate < 0.08 + 2 * se, (G, rate)


def test_wild_bootstrap_almost_never_rejects_with_two_treated_clusters(results):
    for G in (10, 20, 40):
        rate, _ = _rate(results, G, "two", "balanced", "wild")
        assert rate < 0.02, (G, rate)
    # while the analytic variances over-reject several-fold
    rate, _ = _rate(results, 40, "two", "balanced", "cr1")
    assert rate > 0.25


def test_one_dominant_cluster_breaks_cr1_and_the_bootstrap_but_not_cr3(results):
    for G in (20, 40):
        cr1, _ = _rate(results, G, "half", "unbalanced", "cr1")
        wild, se_w = _rate(results, G, "half", "unbalanced", "wild")
        cr3, se_3 = _rate(results, G, "half", "unbalanced", "cr3")
        assert cr1 > 0.20, (G, cr1)
        assert wild - 2 * se_w > 0.08, (G, wild)
        assert 0.03 < cr3 < 0.06 + 2 * se_3, (G, cr3)
    # more clusters do not help CR1 here: the large one still holds half
    assert (
        _rate(results, 40, "half", "unbalanced", "cr1")[0]
        > _rate(results, 6, "half", "unbalanced", "cr1")[0]
    )


def test_size_dispersion_block_supports_the_effective_cluster_threshold(results):
    """Below about 30 effective clusters CR1 over-rejects; above, it does not."""
    cells = results["size_dispersion"]
    assert len(cells) == 6
    for c in cells:
        rate, se = c["cr1"]["rejection_rate"], c["cr1"]["mc_se"]
        if c["effective_clusters_mean"] < 30:
            assert rate - 2 * se > 0.06, c
        else:
            assert rate < 0.065 + 2 * se, c
        # CR3 stays within two points of nominal throughout
        assert c["cr3"]["rejection_rate"] < 0.07, c
    by = {(c["G"], c["sizes"]): c["effective_clusters_mean"] for c in cells}
    assert by[(40, "lognormal_1")] < 20 < 30 < by[(40, "lognormal_0.5")]


def test_the_hint_says_what_the_study_found():
    text = FEW_CLUSTERS_HINT
    assert "one or two" in text and "almost never rejects" in text
    assert "one cluster" in text and "cr3" in text
    assert "few_clusters_results.json" in text
    assert "keeps correct size" not in text


# ---------------------------------------------------------------------------
# The diagnostic the study motivates
# ---------------------------------------------------------------------------


def _frame(sizes, seed=0):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(len(sizes)), sizes)
    return pd.DataFrame({"y": rng.normal(size=g.size), "d": g % 2, "g": g})


def test_effective_clusters_is_the_inverse_herfindahl_of_the_sizes():
    import numpy as np

    from statspai.core._agent_summary import effective_n_clusters

    assert effective_n_clusters(np.repeat(np.arange(40), 30)) == pytest.approx(40.0)
    sizes = np.array([600] + [15] * 39)
    keys = np.repeat(np.arange(40), sizes)
    expected = sizes.sum() ** 2 / (sizes**2).sum()
    assert effective_n_clusters(keys) == pytest.approx(expected)
    assert 3.5 < expected < 4.5
    assert effective_n_clusters([]) == 0.0


def test_regress_warns_when_many_clusters_are_few_in_effect():
    """Forty clusters, one holding half the rows: the case the count misses."""
    import warnings

    import statspai as sp
    from statspai.exceptions import AssumptionWarning

    df = _frame([600] + [15] * 39)
    with pytest.warns(AssumptionWarning, match="effective number") as caught:
        res = sp.regress("y ~ d", df, cluster="g")
    diag = caught[0].message.diagnostics
    assert diag["n_clusters"] == 40
    assert diag["n_clusters_effective"] == pytest.approx(3.81, abs=0.01)
    assert diag["largest_cluster_share"] == pytest.approx(600 / 1185)
    assert res.model_info["n_clusters_effective"] == pytest.approx(3.81, abs=0.01)

    balanced = _frame([30] * 40)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        res = sp.regress("y ~ d", balanced, cluster="g")
    assert res.model_info["n_clusters_effective"] == pytest.approx(40.0)


def test_few_clusters_by_count_keeps_its_own_warning():
    import statspai as sp
    from statspai.exceptions import AssumptionWarning

    with pytest.warns(AssumptionWarning, match="Only 10 clusters") as caught:
        sp.regress("y ~ d", _frame([30] * 10), cluster="g")
    assert len([w for w in caught if "effective number" in str(w.message)]) == 0


# ---------------------------------------------------------------------------
# Few treated clusters
# ---------------------------------------------------------------------------


def _treated_frame(n_treated, n_clusters=40, seed=0):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_clusters), 30)
    return pd.DataFrame(
        {
            "y": rng.normal(size=g.size),
            "d": np.isin(g, np.arange(n_treated)).astype(int),
            "x": rng.normal(size=g.size),
            "unit_level": (rng.uniform(size=g.size) < 0.5).astype(int),
            "g": g,
        }
    )


def test_regress_warns_about_two_treated_clusters_out_of_forty():
    """The case where CR1 rejects 31% and the bootstrap 0% (see the table)."""
    import statspai as sp
    from statspai.exceptions import AssumptionWarning

    with pytest.warns(AssumptionWarning, match="cluster-level 0/1") as caught:
        res = sp.regress("y ~ d + x + unit_level", _treated_frame(2), cluster="g")
    (warning,) = [w for w in caught if "cluster-level" in str(w.message)]
    diag = warning.message.diagnostics
    assert diag["variable"] == "d"
    assert (diag["clusters_at_one"], diag["clusters_at_zero"]) == (2, 38)
    assert res.model_info["few_treated_clusters"][0]["clusters_at_one"] == 2


def test_no_warning_for_a_balanced_treatment_or_a_unit_level_dummy(results):
    import warnings

    import statspai as sp
    from statspai.exceptions import AssumptionWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        res = sp.regress("y ~ d + x + unit_level", _treated_frame(20), cluster="g")
    assert "few_treated_clusters" not in res.model_info
    # and the balanced, half-treated design is where CR1 is fine
    assert _rate(results, 40, "half", "balanced", "cr1")[0] < 0.06


def test_five_treated_of_ten_is_not_the_few_treated_case(results):
    """Balanced and few: the count warning speaks, the few-treated one does not."""
    import statspai as sp
    from statspai.exceptions import AssumptionWarning

    with pytest.warns(AssumptionWarning) as caught:
        sp.regress("y ~ d + x", _treated_frame(5, n_clusters=10), cluster="g")
    texts = [str(w.message) for w in caught]
    assert any("Only 10 clusters" in t for t in texts)
    assert not any("cluster-level 0/1" in t for t in texts)
    # ... and there the bootstrap is near nominal
    assert _rate(results, 10, "half", "balanced", "wild")[0] < 0.08


def test_cluster_dummies_are_not_mistaken_for_treatments():
    import warnings

    import statspai as sp
    from statspai.exceptions import AssumptionWarning

    df = _treated_frame(20)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        sp.regress("y ~ x + C(g)", df, cluster="g")
        # hand-made indicators for five clusters: recorded, not warned
        for k in range(5):
            df[f"c{k}"] = (df["g"] == k).astype(int)
        res = sp.regress("y ~ x + c0 + c1 + c2 + c3 + c4", df, cluster="g")
    assert len(res.model_info["few_treated_clusters"]) == 5


# ---------------------------------------------------------------------------
# The same diagnostic on the fixed-effects entry points
# ---------------------------------------------------------------------------


def _unit_panel(sizes, seed=0):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "id": i,
                    "t": np.arange(T),
                    "x": rng.normal(size=T),
                    "y": rng.normal(size=T),
                }
            )
            for i, T in enumerate(sizes)
        ],
        ignore_index=True,
    )


_FE_CALLS = {
    "panel": lambda sp, d: sp.panel(d, "y ~ x", entity="id", time="t", cluster="id"),
    "hdfe_ols": lambda sp, d: sp.hdfe_ols("y ~ x | id", d, cluster="id"),
    "feols": lambda sp, d: sp.feols("y ~ x | id", d, cluster="id"),
}


@pytest.mark.parametrize("entry", sorted(_FE_CALLS))
def test_fixed_effects_entry_points_warn_on_a_dominant_unit(entry):
    import warnings

    import statspai as sp
    from statspai.exceptions import AssumptionWarning

    dominant = _unit_panel([8 * 39] + [8] * 39)
    with pytest.warns(AssumptionWarning, match="effective number") as caught:
        _FE_CALLS[entry](sp, dominant)
    (w,) = [c for c in caught if "effective number" in str(c.message)]
    assert w.message.diagnostics["n_clusters"] == 40
    assert w.message.diagnostics["n_clusters_effective"] == pytest.approx(3.9)
    assert w.message.diagnostics["largest_cluster_share"] == pytest.approx(0.5)

    balanced = _unit_panel([8] * 40)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        _FE_CALLS[entry](sp, balanced)


def test_panel_block_shows_the_distortion_the_warning_is_about(results):
    cells = {c["sizes"]: c for c in results["panel_fixed_effects"]}
    bal, dom = cells["balanced"], cells["dominant"]
    assert bal["effective_clusters"] == pytest.approx(40.0)
    assert dom["effective_clusters"] == pytest.approx(3.9)
    rate, se = bal["cr1"]["rejection_rate"], bal["cr1"]["mc_se"]
    assert abs(rate - 0.05) < 3 * se
    assert dom["cr1"]["rejection_rate"] > 0.20
