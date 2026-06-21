"""Coverage round-2 — uncovered branches of ``statspai.synth.scm``.

Targets the classic synthetic-control estimator's still-uncovered paths:

- the ``multi_outcome`` dispatch default (``outcomes`` defaulting to the
  single outcome);
- the R-backend NotImplementedError guards (no Rscript / penalized /
  covariates / special_predictors) that fail loudly without R installed;
- ``v_method='nested'`` and ``v_method='equal'`` solver branches;
- special-predictor construction (mean / sum / slice specs) and the
  invalid-op guard;
- the ``synthplot`` trajectory / gap / both rendering paths.

DGP: one treated unit with a clean +4 post jump over imperfect donors.
Assertions check real recovery / correct loud failures — never fabricated
numbers.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import statspai as sp

TRUE_EFFECT = 4.0
T_TREAT = 11


def _scm_panel(seed=0, n_donors=8, n_t=20, effect=TRUE_EFFECT, with_cov=False):
    rng = np.random.default_rng(seed)
    units = [f"u{i}" for i in range(n_donors)] + ["treated"]
    rows = []
    for u in units:
        base = rng.normal(0, 1)
        fe = rng.normal(0, 0.5)
        w = rng.normal()
        for t in range(1, n_t + 1):
            eff = effect if (u == "treated" and t >= T_TREAT) else 0.0
            y = base + 0.2 * t + fe + eff + rng.normal(0, 0.3)
            row = {"unit": u, "time": t, "y": y}
            if with_cov:
                row["z"] = w + 0.1 * t + rng.normal(0, 0.1)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return _scm_panel()


# ---------------------------------------------------------------------------
# v_method branches
# ---------------------------------------------------------------------------
def test_v_method_equal_branch(panel):
    r = sp.synth(
        panel,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="classic",
        v_method="equal",
        placebo=False,
    )
    assert np.isfinite(r.estimate)
    assert r.estimate > 1.5


def test_v_method_nested_with_covariates():
    df = _scm_panel(seed=3, with_cov=True)
    r = sp.synth(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="classic",
        covariates=["z"],
        v_method="nested",
        placebo=False,
    )
    assert np.isfinite(r.estimate)
    # weights are a valid simplex
    wt = r.model_info.get("weights")
    w = np.asarray(wt["weight"], dtype=float)
    assert w.min() >= -1e-6
    assert abs(w.sum() - 1.0) < 1e-2


# ---------------------------------------------------------------------------
# Special predictors (mean / sum / slice / invalid op)
# ---------------------------------------------------------------------------
def test_special_predictors_mean_and_slice():
    df = _scm_panel(seed=4, with_cov=True)
    r = sp.synth(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="classic",
        placebo=False,
        special_predictors=[("z", slice(1, 5), "mean"), ("z", [6, 7], "sum")],
    )
    assert np.isfinite(r.estimate)


def test_special_predictor_invalid_op_raises():
    df = _scm_panel(seed=5, with_cov=True)
    with pytest.raises(ValueError):
        sp.synth(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            method="classic",
            placebo=False,
            special_predictors=[("z", 3, "median")],
        )


# ---------------------------------------------------------------------------
# multi_outcome default outcomes path
# ---------------------------------------------------------------------------
def test_multi_outcome_defaults_to_single_outcome(panel):
    # dispatcher defaults outcomes=[outcome]; multi_outcome_synth then
    # rejects a single outcome — covers the dispatch default branch.
    with pytest.raises(ValueError):
        sp.synth(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            method="multi_outcome",
        )


# ---------------------------------------------------------------------------
# R backend guards — no R installed → NotImplementedError / RuntimeError
# ---------------------------------------------------------------------------
def test_r_backend_rejects_penalized(panel):
    with pytest.raises(NotImplementedError):
        sp.synth(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            method="ridge",
            backend="r",
        )


def test_r_backend_rejects_covariates():
    df = _scm_panel(seed=6, with_cov=True)
    with pytest.raises(NotImplementedError):
        sp.synth(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            method="classic",
            backend="r",
            covariates=["z"],
        )


def test_r_backend_rejects_special_predictors(panel):
    with pytest.raises(NotImplementedError):
        sp.synth(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            method="classic",
            backend="synth",
            special_predictors=[("y", 3, "mean")],
        )


def test_unknown_backend_raises(panel):
    with pytest.raises(ValueError):
        sp.synth(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            method="classic",
            backend="bogus",
        )


# ---------------------------------------------------------------------------
# synthplot rendering paths
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fitted(panel):
    return sp.synth(
        panel,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="classic",
        placebo=False,
    )


def test_synthplot_trajectory(fitted):
    import matplotlib.pyplot as plt

    fig, ax = sp.synthplot(fitted, type="trajectory")
    assert fig is not None and ax is not None
    plt.close(fig)


def test_synthplot_gap(fitted):
    import matplotlib.pyplot as plt

    fig, ax = sp.synthplot(fitted, type="gap")
    assert fig is not None
    plt.close(fig)


def test_synthplot_both(fitted):
    import matplotlib.pyplot as plt

    fig, axes = sp.synthplot(fitted, type="both", title="custom")
    assert len(axes) == 2
    plt.close(fig)


# The scm module also defines its own synthplot (distinct from the
# plots.py public one); exercise it directly for coverage.
def test_scm_module_synthplot_all_types(fitted):
    import matplotlib.pyplot as plt
    from statspai.synth import scm as _scm

    fig, ax = _scm.synthplot(fitted, type="trajectory")
    assert ax is not None
    plt.close(fig)
    fig, ax = _scm.synthplot(fitted, type="gap", title="g")
    assert ax is not None
    plt.close(fig)
    fig, axes = _scm.synthplot(fitted, type="both")
    assert len(axes) == 2
    plt.close(fig)


def test_scm_module_synthplot_accepts_user_ax(fitted):
    import matplotlib.pyplot as plt
    from statspai.synth import scm as _scm

    f, a = plt.subplots()
    fig, ax = _scm.synthplot(fitted, type="trajectory", ax=a)
    assert ax is a
    plt.close("all")
    f2, a2 = plt.subplots()
    fig2, ax2 = _scm.synthplot(fitted, type="gap", ax=a2)
    assert ax2 is a2
    plt.close("all")


def test_scm_module_synthplot_requires_gap_table(fitted):
    import copy
    from statspai.synth import scm as _scm

    bad = copy.copy(fitted)
    mi = dict(fitted.model_info)
    mi["gap_table"] = None
    bad.model_info = mi
    with pytest.raises(ValueError):
        _scm.synthplot(bad)


def test_synthplot_requires_gap_table(fitted):
    import copy
    import matplotlib.pyplot as plt

    bad = copy.copy(fitted)
    mi = dict(fitted.model_info)
    mi["gap_table"] = None
    bad.model_info = mi
    with pytest.raises((ValueError, TypeError, KeyError)):
        sp.synthplot(bad)
    plt.close("all")
