"""Few-treated-group inference: Conley--Taber and Ferman--Pinto.

No reference implementation ships with either paper, so the evidence here is
T1: the size of the test under a known null, measured against the analytic
cluster-robust alternative on the same panels, plus exact reconstruction of
the statistic on a fixture small enough to compute by hand.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _panel(rng, *, n_groups=30, T=8, n_treated=1, rho=0.6, effect=0.0, sizes=None):
    """Group-by-period panel with AR(1) within-group errors.

    ``sizes[j]`` scales group ``j``'s error variance as ``A + B / M_j``, the
    structure Ferman--Pinto attribute to groups being aggregates of
    ``M_j`` individuals.
    """
    rows = []
    for j in range(n_groups):
        e = np.empty(T)
        e[0] = rng.normal()
        for t in range(1, T):
            e[t] = rho * e[t - 1] + rng.normal(0, np.sqrt(1 - rho**2))
        M = 1.0 if sizes is None else float(sizes[j])
        sd = 1.0 if sizes is None else np.sqrt(0.2 + 1.0 / M)
        fe = rng.normal(0, 1)
        for t in range(T):
            d = 1.0 if (j < n_treated and t >= T // 2) else 0.0
            rows.append(
                {
                    "g": j,
                    "t": t,
                    "d": d,
                    "y": fe + 0.1 * t + effect * d + sd * e[t],
                    "M": M,
                }
            )
    return pd.DataFrame(rows)


def _cluster_robust_p(df):
    fit = sp.feols("y ~ d | g + t", df, vcov={"CRV1": "g"})
    p = fit.pvalues
    return float(p["d"]) if isinstance(p, dict) else float(np.asarray(p)[0])


def test_statistic_matches_its_definition():
    """W_j is the treated path applied to a control group's residuals."""
    rng = np.random.default_rng(5)
    df = _panel(rng, n_groups=14, T=6)
    res = sp.did_few_treated(df, y="y", unit="g", time="t", treat="d")

    # Rebuild the pieces: two-way demeaning, then the null residuals.
    piv_y = df.pivot(index="g", columns="t", values="y").to_numpy()
    piv_d = df.pivot(index="g", columns="t", values="d").to_numpy()

    def dm(a):
        return a - a.mean(1, keepdims=True) - a.mean(0, keepdims=True) + a.mean()

    yt, dt = dm(piv_y), dm(piv_d)
    denom = float((dt**2).sum())
    assert res.estimate == pytest.approx(float((dt * yt).sum() / denom), rel=1e-12)
    w = (dt[0] @ yt[1:].T) / denom  # treated group is row 0
    assert np.allclose(np.sort(res.detail["w"].to_numpy()), np.sort(w), rtol=1e-10)


def test_interval_covers_the_planted_effect_and_null_is_not_rejected():
    rng = np.random.default_rng(0)
    df = _panel(rng, effect=2.0)
    res = sp.did_few_treated(df, y="y", unit="g", time="t", treat="d")
    assert res.ci[0] < 2.0 < res.ci[1]
    at_truth = sp.did_few_treated(
        df, y="y", unit="g", time="t", treat="d", null_value=2.0
    )
    assert at_truth.pvalue > 0.10
    assert at_truth.estimate == pytest.approx(res.estimate, rel=1e-12)


def test_conley_taber_controls_size_where_cluster_robust_does_not():
    """One treated group, AR(1) errors: CRVE rejects a true null constantly."""
    rng = np.random.default_rng(11)
    reps = 150
    ct = crve = 0
    for _ in range(reps):
        df = _panel(rng)
        ct += sp.did_few_treated(df, y="y", unit="g", time="t", treat="d").pvalue < 0.05
        crve += _cluster_robust_p(df) < 0.05
    # With 29 control groups the placebo p-value lives on a grid of 1/30, so
    # the 5 percent test is conservative by construction; the cluster-robust
    # alternative rejects a true null most of the time (observed 0.74).
    assert ct / reps <= 0.07
    assert crve / reps > 0.4


def test_ferman_pinto_corrects_the_size_distortion_from_unequal_group_sizes():
    """Treated group small (noisy) => CT over-rejects; FP pulls it back."""
    rng = np.random.default_rng(3)
    controls = [int(x) for x in rng.choice([2, 4, 8, 20, 60, 200], size=39)]
    reps = 150
    ct = fp = 0
    for _ in range(reps):
        df = _panel(rng, n_groups=40, sizes=[2] + controls)
        ct += sp.did_few_treated(df, y="y", unit="g", time="t", treat="d").pvalue < 0.10
        fp += (
            sp.did_few_treated(
                df,
                y="y",
                unit="g",
                time="t",
                treat="d",
                method="ferman_pinto",
                group_size="M",
            ).pvalue
            < 0.10
        )
    assert ct / reps > 0.13  # observed 0.18
    assert fp / reps < ct / reps
    assert abs(fp / reps - 0.10) < 0.06  # observed 0.135


def test_ferman_pinto_rescales_and_records_the_variance_function():
    rng = np.random.default_rng(7)
    controls = [int(x) for x in rng.choice([5, 20, 100], size=29)]
    df = _panel(rng, n_groups=30, sizes=[5] + controls)
    ct = sp.did_few_treated(df, y="y", unit="g", time="t", treat="d")
    fp = sp.did_few_treated(
        df, y="y", unit="g", time="t", treat="d", method="ferman_pinto", group_size="M"
    )
    assert fp.estimate == pytest.approx(ct.estimate, rel=1e-12)
    fit = fp.model_info["variance_function"]
    assert fit is not None and len(fit) == 1
    (entry,) = fit.values()
    assert entry["var_treated"] > 0
    assert entry["fit"] in ("ols", "nnls")
    # The treated group is among the noisiest, so its draws are scaled up.
    assert fp.detail["w"].std() > ct.detail["w"].std()


def test_multiple_treated_groups_use_combinations():
    rng = np.random.default_rng(1)
    df = _panel(rng, n_groups=24, n_treated=2, effect=1.0)
    res = sp.did_few_treated(df, y="y", unit="g", time="t", treat="d", max_draws=500)
    assert res.model_info["n_treated_groups"] == 2
    assert res.model_info["n_draws"] == 231  # C(22, 2), all of them
    assert res.detail["groups"].str.contains(",").all()


def test_covariates_are_partialled_out():
    """The reported coefficient is the dense-OLS one, with and without x.

    With one treated group the coefficient is not consistent for the effect
    (that is the premise of both methods), so the check is the FWL identity
    against a dummy-variable regression, not closeness to the planted value.
    """
    rng = np.random.default_rng(2)
    df = _panel(rng, effect=1.0)
    df["x"] = rng.normal(size=len(df))
    df["y"] = df["y"] + 3.0 * df["x"]

    def dense(cols):
        X = np.column_stack(
            [np.ones(len(df)), df["d"].to_numpy()]
            + [df[c].to_numpy() for c in cols]
            + [
                pd.get_dummies(df[k].astype(str), drop_first=True)
                .astype(float)
                .to_numpy()
                for k in ("g", "t")
            ]
        )
        return float(np.linalg.lstsq(X, df["y"].to_numpy(), rcond=None)[0][1])

    plain = sp.did_few_treated(df, y="y", unit="g", time="t", treat="d")
    with_x = sp.did_few_treated(
        df, y="y", unit="g", time="t", treat="d", controls=["x"]
    )
    assert plain.estimate == pytest.approx(dense([]), rel=1e-10)
    assert with_x.estimate == pytest.approx(dense(["x"]), rel=1e-10)
    assert with_x.estimate != pytest.approx(plain.estimate, rel=1e-6)


@pytest.mark.parametrize(
    "kwargs, exc, match",
    [
        ({"method": "bootstrap"}, sp.MethodIncompatibility, "method must be"),
        ({"alpha": 0.0}, sp.MethodIncompatibility, "alpha"),
        ({"method": "ferman_pinto"}, sp.MethodIncompatibility, "group_size"),
    ],
)
def test_argument_errors(kwargs, exc, match):
    rng = np.random.default_rng(4)
    df = _panel(rng)
    with pytest.raises(exc, match=match):
        sp.did_few_treated(df, y="y", unit="g", time="t", treat="d", **kwargs)


def test_too_few_control_groups_raises():
    rng = np.random.default_rng(6)
    df = _panel(rng, n_groups=8)
    with pytest.raises(sp.DataInsufficient, match="control groups"):
        sp.did_few_treated(df, y="y", unit="g", time="t", treat="d")


def test_no_treated_group_raises():
    rng = np.random.default_rng(8)
    df = _panel(rng, n_treated=0)
    with pytest.raises(sp.DataInsufficient, match="no treated group"):
        sp.did_few_treated(df, y="y", unit="g", time="t", treat="d")


def test_duplicate_group_period_rows_raise():
    rng = np.random.default_rng(9)
    df = pd.concat([_panel(rng)] * 2, ignore_index=True)
    with pytest.raises(sp.MethodIncompatibility, match="more than one row"):
        sp.did_few_treated(df, y="y", unit="g", time="t", treat="d")


def test_non_binary_treatment_raises():
    rng = np.random.default_rng(10)
    df = _panel(rng)
    df.loc[df.index[:5], "d"] = 0.5
    with pytest.raises(sp.MethodIncompatibility, match="0/1 indicator"):
        sp.did_few_treated(df, y="y", unit="g", time="t", treat="d")


def test_many_treated_groups_warns():
    rng = np.random.default_rng(12)
    df = _panel(rng, n_groups=40, n_treated=15)
    with pytest.warns(UserWarning, match="one or two treated groups"):
        sp.did_few_treated(df, y="y", unit="g", time="t", treat="d")


def test_more_treated_than_control_groups_raises():
    rng = np.random.default_rng(13)
    df = _panel(rng, n_groups=40, n_treated=28)
    with pytest.raises(sp.DataInsufficient, match="one distinct control group"):
        sp.did_few_treated(df, y="y", unit="g", time="t", treat="d")


def test_audit_flags_a_few_treated_cluster_design():
    """sp.audit routes a design with few treated clusters here."""
    rng = np.random.default_rng(0)
    rows = []
    for j in range(40):
        g = 2005 if j < 3 else 0
        fe = rng.normal()
        for t in range(2003, 2008):
            d = 1.0 if (g > 0 and t >= g) else 0.0
            rows.append(
                {
                    "id": j,
                    "year": t,
                    "g": g,
                    "y": fe + 0.1 * t + 0.5 * d + rng.normal(0, 0.5),
                }
            )
    df = pd.DataFrame(rows)
    fit = sp.callaway_santanna(df, y="y", g="g", t="year", i="id")
    assert fit.model_info["n_treated_units"] == 3
    check = [c for c in sp.audit(fit)["checks"] if c["name"] == "few_treated_clusters"][
        0
    ]
    assert check["status"] == "failed"
    assert check["suggest_function"] == "sp.did_few_treated"

    wide = sp.callaway_santanna(
        df.assign(g=np.where(df["id"] < 20, 2005, 0)), y="y", g="g", t="year", i="id"
    )
    passed = [
        c for c in sp.audit(wide)["checks"] if c["name"] == "few_treated_clusters"
    ][0]
    assert passed["status"] == "passed"
