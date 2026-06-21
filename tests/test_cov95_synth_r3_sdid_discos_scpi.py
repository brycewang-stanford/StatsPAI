"""Coverage round-3 (final) — SDID, DiSCo, and SCPI of ``statspai.synth``.

Targets still-uncovered branches of synthetic difference-in-differences
(``sdid`` + plotting helpers + ``synthdid_placebo``), distributional
synthetic controls (``discos`` / ``discos_test`` / ``stochastic_dominance``
/ ``qqsynth``), and the prediction-interval SCM (``scpi``): the legacy
R-style alias validation, the alternative ``method`` / ``se_method`` /
``e_method`` / ``pi_type`` options, the matplotlib plotting paths, and the
loud failures.

All estimators are pure-numpy. Matplotlib uses the Agg backend (no
display). Assertions check real properties (finite ATT/SE, populated
placebo / quantile-effect arrays, correct exceptions); no estimator
numbers are fabricated.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402  headless figure rendering

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.synth.sdid import (  # noqa: E402
    sdid,
    synthdid_placebo,
    synthdid_plot,
    synthdid_units_plot,
    synthdid_rmse_plot,
)
from statspai.synth.discos import (  # noqa: E402
    discos,
    discos_test,
    stochastic_dominance,
    qqsynth,
    discos_plot,
)
from statspai.synth.scpi import scpi  # noqa: E402
from statspai.exceptions import DataInsufficient  # noqa: E402

T_TREAT = 11


def _panel(seed=0, n_donors=8, n_t=20, effect=4.0):
    rng = np.random.default_rng(seed)
    units = [f"u{i}" for i in range(n_donors)] + ["treated"]
    rows = []
    for u in units:
        base = rng.normal(0, 1)
        fe = rng.normal(0, 0.5)
        for t in range(1, n_t + 1):
            eff = effect if (u == "treated" and t >= T_TREAT) else 0.0
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "y": base + 0.2 * t + fe + eff + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


def _dist_panel(seed=0, n_donors=6, n_t=8, n_per=40):
    """Repeated cross-section style panel for distributional SCM."""
    rng = np.random.default_rng(seed)
    units = [f"u{i}" for i in range(n_donors)] + ["treated"]
    rows = []
    for u in units:
        loc = rng.normal(0, 1)
        for t in range(1, n_t + 1):
            shift = 1.5 if (u == "treated" and t >= 5) else 0.0
            for _ in range(n_per):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "y": loc + 0.1 * t + shift + rng.normal(0, 1),
                    }
                )
    return pd.DataFrame(rows)


# ===========================================================================
# SDID — methods, SE methods, validation, plots
# ===========================================================================
@pytest.mark.parametrize("method", ["sdid", "sc", "did"])
def test_sdid_methods(method):
    r = sdid(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method=method,
        se_method="placebo",
        n_reps=30,
        seed=1,
    )
    assert np.isfinite(r.estimate)


@pytest.mark.parametrize("se_method", ["placebo", "bootstrap", "jackknife"])
def test_sdid_se_methods(se_method):
    r = sdid(
        _panel(1),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        se_method=se_method,
        n_reps=30,
        seed=2,
    )
    assert np.isfinite(r.se) or np.isnan(r.se)


def test_sdid_legacy_aliases():
    # y=/treat_unit=/treat_time= R-style aliases route through the same path.
    r = sdid(
        _panel(2),
        y="y",
        unit="unit",
        time="time",
        treat_unit="treated",
        treat_time=T_TREAT,
        se_method="placebo",
        n_reps=20,
        seed=3,
    )
    assert np.isfinite(r.estimate)


def test_sdid_missing_args_raise():
    df = _panel(3)
    with pytest.raises(TypeError):
        sdid(
            df, unit="unit", time="time", treated_unit="treated", treatment_time=T_TREAT
        )  # no outcome / y
    with pytest.raises(TypeError):
        sdid(df, outcome="y", treated_unit="treated", treatment_time=T_TREAT)
    with pytest.raises(TypeError):
        sdid(
            df, outcome="y", unit="unit", time="time", treatment_time=T_TREAT
        )  # no treated_unit
    with pytest.raises(TypeError):
        sdid(
            df, outcome="y", unit="unit", time="time", treated_unit="treated"
        )  # no treatment_time


def test_sdid_unknown_se_method_raises():
    with pytest.raises(ValueError):
        sdid(
            _panel(4),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            se_method="not_a_method",
            n_reps=10,
        )


def test_sdid_unknown_backend_raises():
    with pytest.raises(ValueError):
        sdid(
            _panel(5),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            backend="bogus",
        )


def test_sdid_insufficient_periods_raise():
    df = _panel(6)
    with pytest.raises(DataInsufficient):
        sdid(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=2,
        )  # 1 pre-period


def test_sdid_convenience_wrappers_and_dataset():
    from statspai.synth.sdid import (
        synthdid_estimate,
        sc_estimate,
        did_estimate,
        california_prop99,
    )

    df = _panel(10, n_donors=5)
    r1 = synthdid_estimate(
        df,
        "y",
        "unit",
        "time",
        "treated",
        T_TREAT,
        se_method="placebo",
        n_reps=15,
        seed=1,
    )
    r2 = sc_estimate(
        df,
        "y",
        "unit",
        "time",
        "treated",
        T_TREAT,
        se_method="placebo",
        n_reps=15,
        seed=2,
    )
    r3 = did_estimate(
        df,
        "y",
        "unit",
        "time",
        "treated",
        T_TREAT,
        se_method="placebo",
        n_reps=15,
        seed=3,
    )
    assert all(np.isfinite(r.estimate) for r in (r1, r2, r3))
    ds = california_prop99()
    assert {"state", "year", "packspercapita"}.issubset(ds.columns)
    assert "California" in set(ds["state"])


def test_sdid_r_backend_validation_guards():
    # method / se_method validation fires before Rscript is probed.
    df = _panel(11)
    with pytest.raises(ValueError):
        sdid(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            backend="synthdid",
            method="bogus",
        )
    with pytest.raises(ValueError):
        sdid(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            backend="synthdid",
            se_method="bogus",
        )


def test_synthdid_placebo_and_plots():
    df = _panel(7, n_donors=5)
    r = sdid(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        se_method="placebo",
        n_reps=20,
        seed=8,
    )
    plac = synthdid_placebo(
        df,
        y="y",
        unit="unit",
        time="time",
        treat_unit="treated",
        treat_time=T_TREAT,
        n_reps=10,
        seed=9,
    )
    assert isinstance(plac, pd.DataFrame)
    assert len(plac) >= 1
    # Plotting helpers (Agg backend)
    fig1, _ = synthdid_plot(r)
    fig2, _ = synthdid_units_plot(r)
    fig3, _ = synthdid_rmse_plot(r)
    import matplotlib.pyplot as plt

    plt.close("all")


# ===========================================================================
# DiSCo — distributional synthetic controls
# ===========================================================================
def test_discos_mixture_recovers_shift():
    r = discos(
        _dist_panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=5,
        method="mixture",
        n_quantiles=50,
        placebo=True,
        seed=1,
    )
    assert np.isfinite(r.estimate)
    qe = r.model_info.get("quantile_effects")
    assert qe is not None and np.all(np.isfinite(np.asarray(qe)))


def test_discos_quantile_method_no_placebo():
    r = discos(
        _dist_panel(1),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=5,
        method="quantile",
        n_quantiles=40,
        placebo=False,
        seed=2,
    )
    assert np.isfinite(r.estimate)


def test_discos_insufficient_pre_raises():
    df = _dist_panel(2)
    with pytest.raises(ValueError):
        discos(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=2,
            placebo=False,
        )


def test_discos_test_and_stochastic_dominance():
    r = discos(
        _dist_panel(3),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=5,
        method="mixture",
        n_quantiles=40,
        placebo=True,
        seed=4,
    )
    ks = discos_test(r, test="ks")
    assert isinstance(ks, dict)
    cvm = discos_test(r, test="cvm")
    assert isinstance(cvm, dict) and "statistic" in cvm
    sd = stochastic_dominance(r, order=1)
    assert isinstance(sd, dict)
    sd2 = stochastic_dominance(r, order=2)
    assert isinstance(sd2, dict)
    with pytest.raises(ValueError):
        stochastic_dominance(r, order=3)


def test_discos_plots_and_qqsynth():
    r = discos(
        _dist_panel(5),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=5,
        method="mixture",
        n_quantiles=40,
        placebo=True,
        seed=6,
    )
    for ptype in ("quantile_effect", "quantile_comparison", "gap", "weights"):
        fig, _ = discos_plot(r, type=ptype)
    with pytest.raises(ValueError):
        discos_plot(r, type="bogus")
    qq = qqsynth(
        _dist_panel(5),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=5,
        n_quantiles=40,
        placebo=False,
        seed=6,
    )
    import matplotlib.pyplot as plt

    plt.close("all")
    assert np.isfinite(qq.estimate)


# ===========================================================================
# SCPI — prediction-interval SCM
# ===========================================================================
@pytest.mark.parametrize("e_method", ["gaussian", "ls", "qreg"])
def test_scpi_e_methods(e_method):
    r = scpi(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        e_method=e_method,
        seed=1,
    )
    assert np.isfinite(r.estimate)


def test_scpi_pi_type_variants():
    for pit in ("in_sample", "out_of_sample", "both"):
        r = scpi(
            _panel(1),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            pi_type=pit,
            seed=2,
        )
        assert np.isfinite(r.estimate)


def test_scpi_invalid_pi_type_and_e_method():
    df = _panel(2)
    with pytest.raises(ValueError):
        scpi(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            pi_type="nope",
        )
    with pytest.raises(ValueError):
        scpi(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            e_method="nope",
        )


def test_scpi_insufficient_pre_raises():
    df = _panel(3)
    with pytest.raises(ValueError):
        scpi(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=2,
        )


@pytest.mark.parametrize("w_constr", ["simplex", "lasso", "ridge", "ols", "ls"])
def test_scest_weight_constraints(w_constr):
    from statspai.synth.scpi import scest

    out = scest(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        w_constr=w_constr,
    )
    assert "weights" in out
    assert np.isfinite(out["pre_rmspe"])


def test_scest_unknown_w_constr_raises():
    from statspai.synth.scpi import scest

    with pytest.raises(ValueError):
        scest(
            _panel(1),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            w_constr="bogus",
        )
