"""Relative-magnitude HonestDiD sets wider than the +/-20 sd grid.

The moments are the intensity event study of Zheng, Huang & Zhu (2026,
*World Economy* 8), Figure 3: ``reghdfe salary t2pre6..t2post4, absorb(id
id_class2 city_id#quarter_n industry4n#quarter_n) vce(cluster industry4n)``,
fitted with ``sp.hdfe_ols`` (identical to reghdfe to 1e-9). The target is the
average of the five post-period effects. At Mbar >= 1.5 its C-LF set is wider
than HonestDiD's default grid (+/-20 sd = +/-0.325); StatsPAI <= 1.32 reported
the grid end (-0.325, and a symmetric [-0.325, 0.325] at Mbar = 2).

Reference: Stata ``honestdid`` (C-LF, Delta^RM) on the same moments,
[-0.417, 0.276] at Mbar = 1.5 and [-0.531, 0.390] at Mbar = 2, printed to 3
decimals. Tolerance 2.5e-3 = the printed rounding (5e-4) + one grid step
(6.5e-4) + the simulated C-LF first-stage critical value (NumPy vs Mata
draws).
"""

import warnings

import numpy as np
import pytest

import statspai as sp

BETA = np.array(
    [
        -0.0010118918528532535,
        -0.029109978862735725,
        -0.004390456727009472,
        -0.02023699413443448,
        -0.024592128153056343,
        -0.053679008125390906,
        -0.09582057011848757,
        -0.0687853599232901,
        -0.0664913345276042,
        -0.07786267803279026,
    ]
)
SIGMA = np.array(
    [
        [
            0.00046783956463433105,
            0.0003163485816017724,
            0.00015974560674289395,
            7.988844774630456e-05,
            0.00019890474374312605,
            0.0002264780977443828,
            0.00011041208879728735,
            0.0002858558953422408,
            0.00010675343521078061,
            0.000103025501574401,
        ],
        [
            0.00031634858160177237,
            0.0008513315819000148,
            0.0002715624402625272,
            0.00012762359025083163,
            0.0001674620560402553,
            0.0002270399580585081,
            0.00015046722055610175,
            0.00033363731320659853,
            0.00016704307278869103,
            0.00017700412745051423,
        ],
        [
            0.00015974560674289403,
            0.00027156244026252727,
            0.00066669797048433,
            7.35321774753044e-05,
            0.00020170780183900153,
            0.000210929764913181,
            0.0003145847790409915,
            0.0002908356971385276,
            0.00021663964568155257,
            0.00022218790312675542,
        ],
        [
            7.988844774630431e-05,
            0.0001276235902508315,
            7.353217747530426e-05,
            0.0005323879034102567,
            9.767829600972931e-05,
            0.00020731801046711387,
            9.167776403200744e-05,
            4.1190513483530486e-05,
            4.7299532150572125e-05,
            3.681124388695085e-05,
        ],
        [
            0.00019890474374312608,
            0.0001674620560402553,
            0.00020170780183900147,
            9.767829600972947e-05,
            0.0004477692608575757,
            0.00022912917418265236,
            0.00023751958794824066,
            0.00020895764718852137,
            0.00015110790792684456,
            0.00015689805606626482,
        ],
        [
            0.0002264780977443825,
            0.00022703995805850786,
            0.00021092976491318086,
            0.00020731801046711384,
            0.0002291291741826521,
            0.000593029602076015,
            0.0002857142264253794,
            0.0002385047943675309,
            0.0001347522494817295,
            0.00013050085301020826,
        ],
        [
            0.00011041208879728741,
            0.0001504672205561017,
            0.0003145847790409915,
            9.167776403200755e-05,
            0.00023751958794824066,
            0.00028571422642537965,
            0.0006247762128798291,
            0.00032664917565996583,
            0.00023874969004303184,
            0.00018648577969069047,
        ],
        [
            0.0002858558953422408,
            0.0003336373132065986,
            0.0002908356971385276,
            4.1190513483530574e-05,
            0.00020895764718852135,
            0.00023850479436753093,
            0.00032664917565996583,
            0.0006108732484895869,
            0.00018844546667474835,
            0.000203690240148347,
        ],
        [
            0.00010675343521078032,
            0.00016704307278869073,
            0.00021663964568155228,
            4.7299532150571976e-05,
            0.00015110790792684416,
            0.00013475224948172938,
            0.00023874969004303157,
            0.0001884454666747482,
            0.00031632996157949356,
            0.00018167662269932925,
        ],
        [
            0.00010302550157440085,
            0.00017700412745051396,
            0.00022218790312675518,
            3.681124388695077e-05,
            0.00015689805606626457,
            0.00013050085301020829,
            0.0001864857796906902,
            0.00020369024014834676,
            0.00018167662269932933,
            0.00023190448564799885,
        ],
    ]
)
TIMES = [-6, -5, -4, -3, -2, 0, 1, 2, 3, 4]
AVG = np.full(5, 0.2)


def _run(m_grid, **kw):
    return sp.honest_did_from_moments(
        BETA,
        SIGMA,
        event_times=TIMES,
        m_grid=m_grid,
        method="relative_magnitude",
        l_vec=AVG,
        **kw,
    )


def test_set_wider_than_default_grid_matches_stata_honestdid():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no edge warning once the grid grows
        out = _run([1.5, 2.0])
    assert out.attrs["grid_extended_at"] == [1.5, 2.0]
    assert out.attrs["open_at"] == []
    stata = {1.5: (-0.417, 0.276), 2.0: (-0.531, 0.390)}
    for _, row in out.iterrows():
        lo, hi = stata[row["M"]]
        assert row["ci_lower"] == pytest.approx(lo, abs=2.5e-3)
        assert row["ci_upper"] == pytest.approx(hi, abs=2.5e-3)
    # the default grid stops at -0.325: the old answer is now well inside
    assert out["ci_lower"].max() < -0.40


def test_grid_expand_false_reproduces_the_truncated_honestdid_grid():
    sd = float(np.sqrt(AVG @ SIGMA[5:, 5:] @ AVG))
    with pytest.warns(UserWarning, match="edge of the test-inversion grid"):
        out = _run([2.0], grid_expand=False)
    assert out.loc[0, "ci_lower"] == pytest.approx(-20 * sd, rel=1e-12)
    assert out.attrs["open_at"] == [2.0]


def test_explicit_grid_bounds_are_honoured_not_extended():
    with pytest.warns(UserWarning, match="edge of the test-inversion grid"):
        out = _run([2.0], grid_lb=-0.45, grid_ub=0.6, grid_points=500)
    assert out.loc[0, "ci_lower"] == pytest.approx(-0.45, rel=1e-12)
    assert out.loc[0, "ci_upper"] == pytest.approx(0.39, abs=3e-3)
    assert out.attrs["grid_extended_at"] == []


def test_sets_inside_the_grid_are_unchanged_by_expansion():
    a = _run([0.0, 0.5], grid_expand=True)
    b = _run([0.0, 0.5], grid_expand=False)
    np.testing.assert_array_equal(
        a[["ci_lower", "ci_upper"]], b[["ci_lower", "ci_upper"]]
    )
    assert a.attrs["grid_extended_at"] == []


def test_grid_points_below_two_is_rejected():
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        _run([1.0], grid_points=1)


def test_fitted_regression_goes_straight_in():
    """An intensity event study fitted with sp.hdfe_ols, no manual slicing."""
    import pandas as pd

    rng = np.random.default_rng(2)
    n_u, T = 200, 8
    unit = np.repeat(np.arange(n_u), T)
    t = np.tile(np.arange(T), n_u)
    expo = rng.uniform(0, 1, n_u)[unit]
    df = pd.DataFrame({"unit": unit, "t": t})
    lead_lag = {-3: "l3", -2: "l2", 0: "f0", 1: "f1", 2: "f2", 3: "f3"}
    for k, nm in lead_lag.items():
        df[nm] = expo * (t - 4 == k)
    df["y"] = -0.2 * expo * (t >= 4) + rng.normal(size=len(df))
    fit = sp.hdfe_ols(
        "y ~ " + " + ".join(lead_lag.values()) + " | unit + t", df, cluster="unit"
    )
    mapping = {nm: k for k, nm in lead_lag.items()}
    a = sp.honest_did_from_moments(
        fit, event_times=mapping, m_grid=[0.0, 0.5], method="relative_magnitude"
    )
    names = list(mapping)
    V = pd.DataFrame(fit.vcov, index=fit.coef.index, columns=fit.coef.index).loc[
        names, names
    ]
    b = sp.honest_did_from_moments(
        fit.coef[names].to_numpy(),
        V.to_numpy(),
        event_times=list(mapping.values()),
        m_grid=[0.0, 0.5],
        method="relative_magnitude",
    )
    np.testing.assert_array_equal(
        a[["ci_lower", "ci_upper"]], b[["ci_lower", "ci_upper"]]
    )
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="dict"):
        sp.honest_did_from_moments(fit, event_times=[-3, -2, 0])
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not in the fit"):
        sp.honest_did_from_moments(fit, event_times={"nope": -2, "f0": 0})
