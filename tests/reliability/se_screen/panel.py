"""SE calibration screen: panel estimators (static, dynamic, xt* commands).

Balanced panels, 200 units by 8 periods unless the row says otherwise. The
target is the slope on ``x1`` (true value 0.5), or the autoregressive
coefficient (0.5) in the dynamic rows. Each design satisfies what the row's
variance estimator assumes: independent errors for the conventional
variance, serially correlated errors for the cluster rows, a random-walk
error for conventional first differences. Unbalanced panels are a separate
study (``tests/reliability/unbalanced_panel.py``).

    python tests/reliability/se_screen/panel.py [--only name] [--confirm]
"""

# flake8: noqa: E501
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import Row, coef, main, tidy_row, wald  # noqa: E402,F401

FAMILY = "panel"
NU, NT = 200, 8
BETA = 0.5


def _sp():
    import statspai as sp

    return sp


def _frame(nu=NU, nt=NT):
    return pd.DataFrame(
        {"id": np.repeat(np.arange(nu), nt), "t": np.tile(np.arange(nt), nu)}
    )


def _ar1(rng, nu, nt, rho, sd=1.0):
    e = np.zeros((nu, nt))
    e[:, 0] = rng.normal(size=nu) * sd / np.sqrt(1 - rho**2)
    for j in range(1, nt):
        e[:, j] = rho * e[:, j - 1] + rng.normal(size=nu) * sd
    return e.ravel()


# --------------------------------------------------------------------- DGPs
def d_fe(rng, nu=NU, nt=NT, rho=0.0, rw=False, corr=0.5):
    """Unit effect correlated with x1 (``corr``); error iid, AR(1) or a random walk."""
    df = _frame(nu, nt)
    a = rng.normal(size=nu)[df.id]
    df["x1"] = corr * a + rng.normal(size=nu * nt)
    df["x2"] = rng.normal(size=nu * nt)
    if rw:
        e = rng.normal(size=(nu, nt)).cumsum(axis=1).ravel()
    elif rho:
        e = _ar1(rng, nu, nt, rho)
    else:
        e = rng.normal(size=nu * nt)
    df["y"] = 1 + BETA * df.x1 - 0.5 * df.x2 + a + e
    return df


def d_re(rng):
    return d_fe(rng, corr=0.0)


def d_fe_ar(rng):
    return d_fe(rng, rho=0.5)


def d_fd(rng):
    return d_fe(rng, rw=True)


def d_twoway(rng):
    df = d_fe(rng)
    lam = rng.normal(size=NT)[df.t]
    df["x1"] = df.x1 + 0.5 * lam
    df["y"] = df.y + BETA * 0.5 * lam + lam
    return df


def d_dynamic(rng, nu=500, nt=8, burn=20):
    """y_it = 0.5 y_it-1 + 0.5 x_it + a_i + e_it, started from the stationary law."""
    a = rng.normal(size=nu)
    x = rng.normal(size=(nu, nt + burn))
    y = np.zeros((nu, nt + burn))
    y[:, 0] = a / (1 - BETA) + rng.normal(size=nu)
    for j in range(1, nt + burn):
        y[:, j] = BETA * y[:, j - 1] + BETA * x[:, j] + a + rng.normal(size=nu)
    df = _frame(nu, nt)
    df["y"] = y[:, burn:].ravel()
    df["x1"] = x[:, burn:].ravel()
    return df


def d_xtiv(rng, corr=0.5):
    df = _frame()
    n = len(df)
    a = rng.normal(size=NU)[df.id]
    df["z"] = rng.normal(size=n)
    v = rng.normal(size=n)
    df["x1"] = rng.normal(size=n)
    df["w"] = 0.7 * df.z + corr * a + v
    df["y"] = 1 + BETA * df.w + 0.3 * df.x1 + a + 0.5 * v + rng.normal(size=n)
    return df


def d_xtiv_rw(rng):
    """Every series is a random walk, so the first-differenced model has iid errors."""
    df = _frame()
    n = len(df)

    def walk(inc):
        return inc.reshape(NU, NT).cumsum(axis=1).ravel()

    dz, dv, de, dx = (rng.normal(size=n) for _ in range(4))
    a = rng.normal(size=NU)[df.id]
    df["z"] = walk(dz)
    df["x1"] = walk(dx)
    df["w"] = walk(0.7 * dz + dv) + 0.5 * a
    df["y"] = 1 + BETA * df.w + 0.3 * df.x1 + a + walk(0.5 * dv + de)
    return df


def d_xtiv_re(rng):
    return d_xtiv(rng, corr=0.0)


def d_pcse(rng, nu=10, nt=60):
    """Panel heteroskedasticity and a common shock: errors correlated across units."""
    df = _frame(nu, nt)
    sd = np.linspace(0.5, 1.5, nu)[df.id]
    common = rng.normal(size=nt)[df.t]
    df["x1"] = rng.normal(size=nu * nt)
    df["y"] = 1 + BETA * df.x1 + sd * (0.6 * common + 0.8 * rng.normal(size=nu * nt))
    return df


def d_fgls_het(rng, nu=20, nt=40):
    df = _frame(nu, nt)
    sd = np.linspace(0.5, 2.0, nu)[df.id]
    df["x1"] = rng.normal(size=nu * nt)
    df["y"] = 1 + BETA * df.x1 + sd * rng.normal(size=nu * nt)
    return df


def d_xtregar(rng, nu=200, nt=10):
    df = _frame(nu, nt)
    a = rng.normal(size=nu)[df.id]
    df["x1"] = 0.5 * a + _ar1(rng, nu, nt, 0.5)
    df["y"] = 1 + BETA * df.x1 + a + _ar1(rng, nu, nt, 0.5)
    return df


def d_xtregar_re(rng, nu=200, nt=10):
    df = _frame(nu, nt)
    a = rng.normal(size=nu)[df.id]
    df["x1"] = _ar1(rng, nu, nt, 0.5)
    df["y"] = 1 + BETA * df.x1 + a + _ar1(rng, nu, nt, 0.5)
    return df


def d_xttobit(rng, nt=5):
    df = _frame(NU, nt)
    a = rng.normal(size=NU)[df.id] * 0.7
    df["x1"] = rng.normal(size=NU * nt)
    ystar = 0.5 + BETA * df.x1 + a + rng.normal(size=NU * nt)
    df["y"] = np.maximum(0.0, ystar)
    df["lo"] = np.floor(ystar)
    df["hi"] = df.lo + 1.0
    return df


def d_ht(rng, nu=500, nt=5):
    """Hausman-Taylor: z2 (time-invariant) and x2 (time-varying) move with the unit effect."""
    df = _frame(nu, nt)
    a = rng.normal(size=nu)
    x1bar = rng.normal(size=nu)
    z2 = 0.8 * x1bar + 0.5 * a + rng.normal(size=nu) * 0.5
    df["x1"] = x1bar[df.id] + rng.normal(size=nu * nt)
    df["x2"] = 0.5 * a[df.id] + rng.normal(size=nu * nt)
    df["z1"] = rng.normal(size=nu)[df.id]
    df["z2"] = z2[df.id]
    df["y"] = (
        1
        + 0.5 * df.x1
        + 0.5 * df.x2
        + 0.5 * df.z1
        + BETA * df.z2
        + a[df.id]
        + rng.normal(size=nu * nt)
    )
    return df


def d_fmb(rng, nu=50, nt=100):
    """Period slopes are 0.5 plus independent period noise."""
    df = _frame(nu, nt)
    b = BETA + 0.3 * rng.normal(size=nt)
    df["x1"] = rng.normal(size=nu * nt)
    df["y"] = 0.2 + b[df.t] * df.x1 + rng.normal(size=nu * nt)
    return df


def d_ife(rng, nu=100, nt=30):
    df = _frame(nu, nt)
    lam = rng.normal(size=nu)[df.id]
    f = rng.normal(size=nt)[df.t]
    df["x1"] = 0.5 * lam * f + 0.3 * lam + 0.3 * f + rng.normal(size=nu * nt)
    df["y"] = BETA * df.x1 + lam * f + rng.normal(size=nu * nt)
    return df


def d_plogit_fe(rng):
    df = _frame()
    a = rng.normal(size=NU)[df.id]
    df["x1"] = 0.5 * a + rng.normal(size=len(df))
    df["x2"] = rng.normal(size=len(df))
    df["y"] = (rng.logistic(size=len(df)) < a + BETA * df.x1 - 0.5 * df.x2).astype(int)
    return df


def d_plogit_re(rng):
    df = _frame()
    a = rng.normal(size=NU)[df.id] * 0.7
    df["x1"] = rng.normal(size=len(df))
    df["x2"] = rng.normal(size=len(df))
    df["y"] = (rng.logistic(size=len(df)) < a + BETA * df.x1 - 0.5 * df.x2).astype(int)
    return df


def d_pprobit_re(rng):
    df = _frame()
    a = rng.normal(size=NU)[df.id] * 0.7
    df["x1"] = rng.normal(size=len(df))
    df["x2"] = rng.normal(size=len(df))
    df["y"] = (rng.normal(size=len(df)) < a + BETA * df.x1 - 0.5 * df.x2).astype(int)
    return df


def d_pois_fe(rng):
    df = _frame()
    a = rng.normal(size=NU)[df.id] * 0.5
    df["x1"] = 0.5 * a + rng.normal(size=len(df))
    df["y"] = rng.poisson(np.exp(0.5 + a + BETA * df.x1))
    return df


def d_xtnbreg_fe(rng):
    """Hausman-Hall-Griliches: lambda_it ~ Gamma(exp(xb), delta_i), y ~ Poisson(lambda)."""
    df = _frame()
    delta = np.exp(rng.normal(size=NU) * 0.5)[df.id]
    df["x1"] = rng.normal(size=len(df))
    lam = rng.gamma(shape=np.exp(1.0 + BETA * df.x1), scale=delta)
    df["y"] = rng.poisson(lam)
    return df


def d_mmq(rng):
    """Location-scale with unit effects in both; at the median the slope is 0.5."""
    df = _frame()
    a = rng.normal(size=NU)[df.id]
    df["x1"] = 0.5 * a + rng.uniform(0, 2, size=len(df))
    df["y"] = a + BETA * df.x1 + (1 + 0.3 * df.x1) * rng.normal(size=len(df))
    return df


# ------------------------------------------------- DGPs for the option rows
def d_fe_het(rng):
    """Unit effects correlated with x1; error sd exp(0.4 x1), independent over time."""
    df = _frame()
    a = rng.normal(size=NU)[df.id]
    df["x1"] = 0.5 * a + rng.normal(size=len(df))
    df["x2"] = rng.normal(size=len(df))
    df["y"] = (
        1
        + BETA * df.x1
        - 0.5 * df.x2
        + a
        + rng.normal(size=len(df)) * np.exp(0.4 * df.x1)
    )
    return df


def d_re_het(rng):
    df = d_fe_het(rng)
    a = rng.normal(size=NU)[df.id]
    df["x1"] = rng.normal(size=len(df))
    df["y"] = (
        1
        + BETA * df.x1
        - 0.5 * df.x2
        + a
        + rng.normal(size=len(df)) * np.exp(0.4 * df.x1)
    )
    return df


def d_common_shock(rng, nu=50, nt=60):
    """A period shock in both x1 and the error, serially correlated (rho 0.5)."""
    df = _frame(nu, nt)
    fx = _ar1(rng, 1, nt, 0.5)[df.t]
    fe = _ar1(rng, 1, nt, 0.5)[df.t]
    a = rng.normal(size=nu)[df.id]
    df["x1"] = 0.5 * a + fx + rng.normal(size=nu * nt)
    df["x2"] = rng.normal(size=nu * nt)
    df["y"] = 1 + BETA * df.x1 - 0.5 * df.x2 + a + fe + rng.normal(size=nu * nt)
    return df


def d_twoway_cluster(rng, nu=60, nt=60):
    """Unit and period components in both x1 and the error, none absorbed."""
    df = _frame(nu, nt)
    df["x1"] = (
        rng.normal(size=nu)[df.id]
        + rng.normal(size=nt)[df.t]
        + rng.normal(size=nu * nt)
    )
    e = (
        rng.normal(size=nu)[df.id]
        + rng.normal(size=nt)[df.t]
        + rng.normal(size=nu * nt)
    )
    df["y"] = 1 + BETA * df.x1 + e
    return df


def d_pcse_ar1(rng, nu=10, nt=60):
    df = _frame(nu, nt)
    sd = np.linspace(0.5, 1.5, nu)[df.id]
    df["x1"] = _ar1(rng, nu, nt, 0.5)
    df["y"] = 1 + BETA * df.x1 + sd * _ar1(rng, nu, nt, 0.5)
    return df


def d_fmb_ar(rng, nu=50, nt=200):
    """Period slopes follow an AR(1) around 0.5 (rho 0.5)."""
    df = _frame(nu, nt)
    b = BETA + 0.3 * _ar1(rng, 1, nt, 0.5)
    df["x1"] = rng.normal(size=nu * nt)
    df["y"] = 0.2 + b[df.t] * df.x1 + rng.normal(size=nu * nt)
    return df


def d_pois_fe_od(rng):
    """Poisson with unit effects and gamma overdispersion."""
    df = _frame()
    a = rng.normal(size=NU)[df.id] * 0.5
    df["x1"] = 0.5 * a + rng.normal(size=len(df))
    mu = np.exp(0.5 + a + BETA * df.x1)
    df["y"] = rng.poisson(mu * rng.gamma(2.0, 0.5, size=len(df)))
    return df


# ------------------------------------------------------------------- fits
def p(method, term="x1", formula="y ~ x1 + x2", **kw):
    def fit(df):
        r = _sp().panel(df, formula, entity="id", time="t", method=method, **kw)
        return coef(r, term)

    return fit


def _lag_term(r):
    return [t for t in r.params.index if "rho" in str(t).lower()][0]


def fit_xtabond(**kw):
    def fit(df):
        r = _sp().xtabond(df, y="y", x=["x1"], id="id", time="t", **kw)
        return coef(r, _lag_term(r))

    return fit


def fit_xtdpdsys(**kw):
    def fit(df):
        r = _sp().xtdpdsys(df, y="y", x=["x1"], id="id", time="t", **kw)
        return coef(r, _lag_term(r))

    return fit


def fit_xtlsdvc(**kw):
    def fit(df):
        return coef(_sp().xtlsdvc(df, y="y", x=["x1"], id="id", time="t", **kw))

    return fit


def fit_xtivreg(method, **kw):
    def fit(df):
        r = _sp().xtivreg(
            df,
            y="y",
            x=["x1"],
            endog="w",
            instruments="z",
            id="id",
            method=method,
            **kw,
        )
        return coef(r, "w")

    return fit


def fit_xtpcse(**kw):
    def fit(df):
        return coef(_sp().xtpcse(df, y="y", x=["x1"], id="id", time="t", **kw), "x1")

    return fit


def fit_fgls(**kw):
    def fit(df):
        return coef(
            _sp().panel_fgls(df, y="y", x=["x1"], id="id", time="t", **kw), "x1"
        )

    return fit


def fit_xtregar(method):
    def fit(df):
        return coef(
            _sp().xtregar(df, y="y", x=["x1"], id="id", time="t", method=method), "x1"
        )

    return fit


def fit_xttobit(df):
    return coef(_sp().xttobit(df, y="y", x=["x1"], id="id", ll=0), "x1")


def fit_xtintreg(df):
    return coef(_sp().xtintreg(df, y_lower="lo", y_upper="hi", x=["x1"], id="id"), "x1")


def fit_ht(**kw):
    def fit(df):
        r = _sp().xthtaylor(
            "y ~ x1 + x2 + z1 + z2", df, id="id", endog=["x2", "z2"], **kw
        )
        return coef(r, "z2")

    return fit


def fit_fmb(df):
    return coef(_sp().fama_macbeth("y ~ x1", df, time="t"), "x1")


def fit_hdfe(**kw):
    def fit(df):
        return tidy_row(_sp().hdfe_ols("y ~ x1 + x2 | id + t", df, **kw), term="x1")

    return fit


def fit_feols(fml="y ~ x1 + x2 | id", **kw):
    def fit(df):
        return coef(_sp().feols(fml, df, **kw), "x1")

    return fit


def fit_fepois(**kw):
    def fit(df):
        return coef(_sp().fepois("y ~ x1 | id", df, **kw), "x1")

    return fit


def fit_ife(df):
    return coef(
        _sp().interactive_fe(df, y="y", x=["x1"], id="id", time="t", n_factors=1), "x1"
    )


def fit_plogit(method):
    def fit(df):
        r = _sp().panel_logit(
            df, y="y", x=["x1", "x2"], id="id", time="t", method=method
        )
        return coef(r, "x1")

    return fit


def fit_pprobit(df):
    return coef(_sp().panel_probit(df, y="y", x=["x1", "x2"], id="id", time="t"), "x1")


def fit_xtnbreg(df):
    return coef(
        _sp().xtnbreg("y ~ x1", data=df, entity="id", time="t", model="fe"), "x1"
    )


def fit_mmqreg(df):
    return coef(_sp().mmqreg(df, y="y", x=["x1"], absorb="id", quantile=0.5), "x1")


def fit_fmb_kw(**kw):
    def fit(df):
        return coef(_sp().fama_macbeth("y ~ x1", df, time="t", **kw), "x1")

    return fit


def fit_feols_plain(**kw):
    def fit(df):
        return coef(_sp().feols("y ~ x1", df, **kw), "x1")

    return fit


def fit_hdfe_plain(**kw):
    def fit(df):
        return tidy_row(_sp().hdfe_ols("y ~ x1", df, **kw), term="x1")

    return fit


def fit_mmqreg_kw(**kw):
    def fit(df):
        return coef(_sp().mmqreg(df, y="y", x=["x1"], absorb="id", **kw), "x1")

    return fit


def fit_plogit_kw(method, **kw):
    def fit(df):
        r = _sp().panel_logit(
            df, y="y", x=["x1", "x2"], id="id", time="t", method=method, **kw
        )
        return coef(r, "x1")

    return fit


def R(name, estimator, call, dgp, fit, truth=BETA, n=NU * NT, B=None, design=""):
    return Row(name, estimator, call, n, truth, dgp, fit, B, design)


ROWS = [
    R("panel/fe", "panel", 'panel(method="fe")', d_fe, p("fe")),
    R(
        "panel/fe-cluster",
        "panel",
        'panel(method="fe", cluster="id")',
        d_fe_ar,
        p("fe", cluster="id"),
        design="AR(1) errors, rho 0.5",
    ),
    R(
        "panel/fe-robust",
        "panel",
        'panel(method="fe", robust=...)',
        d_fe_ar,
        p("fe", robust="robust"),
        design="AR(1) errors, rho 0.5",
    ),
    R("panel/re", "panel", 'panel(method="re")', d_re, p("re")),
    R(
        "panel/re-cluster",
        "panel",
        'panel(method="re", cluster="id")',
        d_re,
        p("re", cluster="id"),
    ),
    R(
        "panel/fd",
        "panel",
        'panel(method="fd")',
        d_fd,
        p("fd"),
        design="random-walk error, so first differences are independent",
    ),
    R(
        "panel/fd-cluster",
        "panel",
        'panel(method="fd", cluster="id")',
        d_fe,
        p("fd", cluster="id"),
        design="iid errors, so first differences are MA(1)",
    ),
    R("panel/between", "panel", 'panel(method="between")', d_re, p("be")),
    R(
        "panel/pooled-cluster",
        "panel",
        'panel(method="pooled", cluster="id")',
        d_re,
        p("pooled", cluster="id"),
    ),
    R("panel/twoway", "panel", 'panel(method="twoway")', d_twoway, p("twoway")),
    R("panel/mundlak", "panel", 'panel(method="mundlak")', d_fe, p("mundlak")),
    R(
        "xtabond/onestep",
        "xtabond",
        "xtabond(y, x) (one-step, robust)",
        d_dynamic,
        fit_xtabond(),
        n=4000,
        design="500 units, 8 periods, AR coefficient 0.5",
    ),
    R(
        "xtabond/twostep",
        "xtabond",
        "xtabond(twostep=True)",
        d_dynamic,
        fit_xtabond(twostep=True),
        n=4000,
        design="500 units, 8 periods; Windmeijer SE",
    ),
    R(
        "xtabond/onestep-nonrobust",
        "xtabond",
        "xtabond(robust=False)",
        d_dynamic,
        fit_xtabond(robust=False),
        n=4000,
        design="500 units, 8 periods, iid errors",
    ),
    R(
        "xtabond/collapse",
        "xtabond",
        "xtabond(collapse=True)",
        d_dynamic,
        fit_xtabond(collapse=True),
        n=4000,
        design="500 units, 8 periods",
    ),
    R(
        "xtdpdsys/onestep",
        "xtdpdsys",
        "xtdpdsys(y, x)",
        d_dynamic,
        fit_xtdpdsys(),
        n=4000,
        design="500 units, 8 periods, stationary start",
    ),
    R(
        "xtdpdsys/twostep",
        "xtdpdsys",
        "xtdpdsys(twostep=True)",
        d_dynamic,
        fit_xtdpdsys(twostep=True),
        n=4000,
        design="500 units, 8 periods, stationary start",
    ),
    R(
        "xtlsdvc/default",
        "xtlsdvc",
        "xtlsdvc(y, x)",
        d_dynamic,
        fit_xtlsdvc(),
        n=4000,
        design="500 units, 8 periods; analytic SE is the LSDV one",
    ),
    R(
        "xtlsdvc/consistent",
        "xtlsdvc",
        'xtlsdvc(plug_in="consistent")',
        d_dynamic,
        fit_xtlsdvc(plug_in="consistent"),
        n=4000,
        design="500 units, 8 periods; analytic SE is the LSDV one",
    ),
    R(
        "xtlsdvc/bootstrap",
        "xtlsdvc",
        "xtlsdvc(bootstrap=50)",
        d_dynamic,
        fit_xtlsdvc(bootstrap=50, seed=1),
        n=4000,
        B=200,
        design="500 units, 8 periods; parametric bootstrap SE; B = 200",
    ),
    R(
        "xtlsdvc/consistent-bootstrap",
        "xtlsdvc",
        'xtlsdvc(plug_in="consistent", bootstrap=50)',
        d_dynamic,
        fit_xtlsdvc(plug_in="consistent", bootstrap=50, seed=1),
        n=4000,
        B=200,
        design="500 units, 8 periods; parametric bootstrap SE; B = 200",
    ),
    R("xtivreg/fe", "xtivreg", 'xtivreg(method="fe")', d_xtiv, fit_xtivreg("fe")),
    R("xtivreg/re", "xtivreg", 'xtivreg(method="re")', d_xtiv_re, fit_xtivreg("re")),
    R(
        "xtivreg/fd",
        "xtivreg",
        'xtivreg(method="fd")',
        d_xtiv_rw,
        fit_xtivreg("fd", time="t"),
        design="random-walk errors, so the differenced errors are independent",
    ),
    R(
        "xtivreg/fd-cluster",
        "xtivreg",
        'xtivreg(method="fd", vce="cluster")',
        d_xtiv,
        fit_xtivreg("fd", time="t", vce="cluster", cluster="id"),
        design="iid level errors: differenced errors are MA(1) within unit",
    ),
    R("xtivreg/be", "xtivreg", 'xtivreg(method="be")', d_xtiv_re, fit_xtivreg("be")),
    R(
        "xtpcse/default",
        "xtpcse",
        "xtpcse(y, x, id, time)",
        d_pcse,
        fit_xtpcse(),
        n=600,
        design="10 units, 60 periods, common shock and unit-specific variances",
    ),
    R(
        "xtpcse/hetonly",
        "xtpcse",
        "xtpcse(hetonly=True)",
        d_fgls_het,
        fit_xtpcse(hetonly=True),
        n=800,
        design="20 units, 40 periods, unit-specific variances",
    ),
    R(
        "panel_fgls/heteroskedastic",
        "panel_fgls",
        "panel_fgls(y, x, id, time)",
        d_fgls_het,
        fit_fgls(),
        n=800,
        design="20 units, 40 periods, unit-specific variances",
    ),
    R(
        "xtregar/fe",
        "xtregar",
        'xtregar(method="fe")',
        d_xtregar,
        fit_xtregar("fe"),
        n=2000,
        design="200 units, 10 periods, AR(1) error rho 0.5",
    ),
    R(
        "xtregar/re",
        "xtregar",
        'xtregar(method="re")',
        d_xtregar_re,
        fit_xtregar("re"),
        n=2000,
        design="200 units, 10 periods, AR(1) error rho 0.5",
    ),
    R(
        "xttobit/default",
        "xttobit",
        "xttobit(y, x, id, ll=0)",
        d_xttobit,
        fit_xttobit,
        n=1000,
    ),
    R(
        "xtintreg/default",
        "xtintreg",
        "xtintreg(y_lower, y_upper, x, id)",
        d_xttobit,
        fit_xtintreg,
        n=1000,
        B=200,
        design="quadrature likelihood, 2.5 s a fit; B = 200",
    ),
    R(
        "xthtaylor/ht",
        "xthtaylor",
        "xthtaylor(endog=[x2, z2])",
        d_ht,
        fit_ht(),
        n=2500,
        design="500 units, 5 periods; coefficient on the endogenous time-invariant z2",
    ),
    R(
        "xthtaylor/robust",
        "xthtaylor",
        'xthtaylor(vce="robust")',
        d_ht,
        fit_ht(vce="robust"),
        n=2500,
        design="500 units, 5 periods; coefficient on z2",
    ),
    R(
        "fama_macbeth/default",
        "fama_macbeth",
        'fama_macbeth("y ~ x1", time="t")',
        d_fmb,
        fit_fmb,
        n=5000,
        design="50 units, 100 periods, period slopes vary independently",
    ),
    R(
        "hdfe_ols/default",
        "hdfe_ols",
        'hdfe_ols("y ~ x1 + x2 | id + t")',
        d_twoway,
        fit_hdfe(),
    ),
    R(
        "hdfe_ols/cluster",
        "hdfe_ols",
        'hdfe_ols(..., cluster="id")',
        d_fe_ar,
        fit_hdfe(cluster="id"),
        design="AR(1) errors, rho 0.5",
    ),
    R("feols/default", "feols", 'feols("y ~ x1 + x2 | id")', d_fe, fit_feols()),
    R(
        "feols/cluster",
        "feols",
        'feols(..., vcov={"CRV1": "id"})',
        d_fe_ar,
        fit_feols(vcov={"CRV1": "id"}),
        design="AR(1) errors, rho 0.5",
    ),
    R(
        "feols/hetero",
        "feols",
        'feols(..., vcov="hetero")',
        d_fe,
        fit_feols(vcov="hetero"),
    ),
    R("fepois/default", "fepois", 'fepois("y ~ x1 | id")', d_pois_fe, fit_fepois()),
    R(
        "interactive_fe/default",
        "interactive_fe",
        "interactive_fe(n_factors=1)",
        d_ife,
        fit_ife,
        n=3000,
        design="100 units, 30 periods, one factor correlated with x1",
    ),
    R(
        "panel_logit/fe",
        "panel_logit",
        'panel_logit(method="fe")',
        d_plogit_fe,
        fit_plogit("fe"),
    ),
    R(
        "panel_logit/re",
        "panel_logit",
        'panel_logit(method="re")',
        d_plogit_re,
        fit_plogit("re"),
    ),
    R(
        "panel_probit/re",
        "panel_probit",
        "panel_probit(y, x)",
        d_pprobit_re,
        fit_pprobit,
    ),
    R("xtnbreg/fe", "xtnbreg", 'xtnbreg(model="fe")', d_xtnbreg_fe, fit_xtnbreg),
    R(
        "mmqreg/median",
        "mmqreg",
        'mmqreg(absorb="id", quantile=0.5)',
        d_mmq,
        fit_mmqreg,
    ),
]

AR = "AR(1) errors, rho 0.5"
HET = "error sd exp(0.4 x1), independent over time"
Q25 = float(BETA + 0.3 * (-0.6744897501960817))

OPTION_ROWS = [
    R(
        "panel/fe-cr2",
        "panel",
        'panel(method="fe", cluster="id", vce="CR2")',
        d_fe_ar,
        p("fe", cluster="id", vce="CR2"),
        design=AR,
    ),
    R(
        "panel/fe-cr3",
        "panel",
        'panel(method="fe", cluster="id", vce="CR3")',
        d_fe_ar,
        p("fe", cluster="id", vce="CR3"),
        design=AR,
    ),
    R(
        "panel/fe-jackknife",
        "panel",
        'panel(method="fe", cluster="id", vce="jackknife")',
        d_fe_ar,
        p("fe", cluster="id", vce="jackknife"),
        design=AR,
    ),
    R(
        "panel/fe-driscoll-kraay",
        "panel",
        'panel(method="fe", robust="driscoll-kraay")',
        d_common_shock,
        p("fe", robust="driscoll-kraay"),
        n=3000,
        design="50 units, 60 periods, serially correlated period shocks in x1 and the error",
    ),
    R(
        "panel/re-robust",
        "panel",
        'panel(method="re", robust="robust")',
        d_re_het,
        p("re", robust="robust"),
        design=HET,
    ),
    R(
        "panel/twoway-cluster",
        "panel",
        'panel(method="twoway", cluster="id")',
        d_fe_ar,
        p("twoway", cluster="id"),
        design=AR,
    ),
    R(
        "panel/chamberlain",
        "panel",
        'panel(method="chamberlain")',
        d_fe,
        p("chamberlain"),
    ),
    R(
        "panel/between-robust",
        "panel",
        'panel(method="be", robust="robust")',
        d_re,
        p("be", robust="robust"),
    ),
    R(
        "xtabond/twostep-nonrobust",
        "xtabond",
        "xtabond(twostep=True, robust=False)",
        d_dynamic,
        fit_xtabond(twostep=True, robust=False),
        n=4000,
        design="500 units, 8 periods; two-step SE without the Windmeijer correction",
    ),
    R(
        "xtabond/ah",
        "xtabond",
        'xtabond(method="ah")',
        d_dynamic,
        fit_xtabond(method="ah"),
        n=4000,
        design="500 units, 8 periods; Anderson-Hsiao",
    ),
    R(
        "xtabond/orthogonal",
        "xtabond",
        "xtabond(orthogonal=True)",
        d_dynamic,
        fit_xtabond(orthogonal=True),
        n=4000,
        design="500 units, 8 periods",
    ),
    R(
        "xtdpdsys/collapse",
        "xtdpdsys",
        "xtdpdsys(collapse=True)",
        d_dynamic,
        fit_xtdpdsys(collapse=True),
        n=4000,
        design="500 units, 8 periods, stationary start",
    ),
    R(
        "xtivreg/ec2sls",
        "xtivreg",
        'xtivreg(method="re", ec2sls=True)',
        d_xtiv_re,
        fit_xtivreg("re", ec2sls=True),
    ),
    R(
        "xtivreg/fe-robust",
        "xtivreg",
        'xtivreg(method="fe", vce="robust")',
        d_xtiv,
        fit_xtivreg("fe", vce="robust"),
    ),
    R(
        "xtivreg/re-cluster",
        "xtivreg",
        'xtivreg(method="re", vce="cluster")',
        d_xtiv_re,
        fit_xtivreg("re", vce="cluster", cluster="id"),
    ),
    R(
        "xtpcse/ar1",
        "xtpcse",
        'xtpcse(correlation="ar1")',
        d_pcse_ar1,
        fit_xtpcse(correlation="ar1"),
        n=600,
        design="10 units, 60 periods, AR(1) errors rho 0.5, unit-specific variances",
    ),
    R(
        "xtpcse/psar1",
        "xtpcse",
        'xtpcse(correlation="psar1")',
        d_pcse_ar1,
        fit_xtpcse(correlation="psar1"),
        n=600,
        design="10 units, 60 periods, AR(1) errors rho 0.5, unit-specific variances",
    ),
    R(
        "panel_fgls/ar1",
        "panel_fgls",
        'panel_fgls(corr="ar1")',
        d_pcse_ar1,
        fit_fgls(corr="ar1"),
        n=600,
        design="10 units, 60 periods, AR(1) errors rho 0.5, unit-specific variances",
    ),
    R(
        "panel_fgls/correlated",
        "panel_fgls",
        'panel_fgls(panels="correlated")',
        d_pcse,
        fit_fgls(panels="correlated"),
        n=600,
        design="10 units, 60 periods, common shock and unit-specific variances",
    ),
    R(
        "xthtaylor/amacurdy",
        "xthtaylor",
        'xthtaylor(method="amacurdy")',
        d_ht,
        fit_ht(method="amacurdy", time="t"),
        n=2500,
        design="500 units, 5 periods; coefficient on z2",
    ),
    R(
        "fama_macbeth/newey-west",
        "fama_macbeth",
        "fama_macbeth(lags=4)",
        d_fmb_ar,
        fit_fmb_kw(lags=4),
        n=10000,
        design="50 units, 200 periods, period slopes AR(1) with rho 0.5",
    ),
    R(
        "feols/twoway-cluster",
        "feols",
        'feols("y ~ x1", vcov={"CRV1": "id + t"})',
        d_twoway_cluster,
        fit_feols_plain(vcov={"CRV1": "id + t"}),
        n=3600,
        design="60 units, 60 periods, unit and period components in x1 and the error",
    ),
    R(
        "hdfe_ols/twoway-cluster",
        "hdfe_ols",
        'hdfe_ols("y ~ x1", cluster=["id", "t"])',
        d_twoway_cluster,
        fit_hdfe_plain(cluster=["id", "t"]),
        n=3600,
        design="60 units, 60 periods, unit and period components in x1 and the error",
    ),
    R(
        "fepois/cluster",
        "fepois",
        'fepois(..., vcov={"CRV1": "id"})',
        d_pois_fe_od,
        fit_fepois(vcov={"CRV1": "id"}),
        design="overdispersed counts, unit effects",
    ),
    R(
        "fepois/hetero",
        "fepois",
        'fepois(..., vcov="hetero")',
        d_pois_fe_od,
        fit_fepois(vcov="hetero"),
        design="overdispersed counts, unit effects",
    ),
    R(
        "hdfe_ols/robust",
        "hdfe_ols",
        'hdfe_ols(..., vce="robust")',
        d_fe_het,
        fit_hdfe(vce="robust"),
        design=HET,
    ),
    R(
        "interactive_fe/nonrobust",
        "interactive_fe",
        "interactive_fe(robust=False)",
        d_ife,
        lambda df: coef(
            _sp().interactive_fe(
                df, y="y", x=["x1"], id="id", time="t", n_factors=1, robust=False
            ),
            "x1",
        ),
        n=3000,
        design="100 units, 30 periods, one factor correlated with x1",
    ),
    R(
        "panel_logit/re-cluster",
        "panel_logit",
        'panel_logit(method="re", cluster="id")',
        d_plogit_re,
        fit_plogit_kw("re", cluster="id"),
        B=200,
        design="B = 200",
    ),
    R(
        "mmqreg/q25",
        "mmqreg",
        "mmqreg(quantile=0.25)",
        d_mmq,
        fit_mmqreg_kw(quantile=0.25),
        truth=Q25,
        design="location-scale; the 0.25 quantile slope is 0.5 + 0.3 z(0.25)",
    ),
    R(
        "mmqreg/robust",
        "mmqreg",
        'mmqreg(vce="robust")',
        d_mmq,
        fit_mmqreg_kw(vce="robust"),
    ),
]

ROWS = ROWS + OPTION_ROWS

MECHANISMS: dict = {
    "xtlsdvc/default": (
        "Two things. (1) Bias: the Stata plug-in for the deterministic part of the lagged "
        "outcome uses within-demeaned residuals, which keep each unit's mean error; the "
        "first-order correction then converges to -0.2195 where the exact LSDV "
        "inconsistency is -0.2353 (rho 0.5, seven usable periods), a bias that does not "
        "shrink with N. sp.xtlsdvc matches Stata xtlsdvc here; plug_in='consistent' "
        "removes it (tests/test_lsdvc_plug_in.py). (2) SE: the analytic SE is the LSDV "
        "one and ignores the estimated correction, as in Stata; the function warns and "
        "offers bootstrap=."
    ),
    "xtlsdvc/consistent": (
        "SE only. plug_in='consistent' removes the bias (bias/SD -0.07) but the analytic "
        "SE is still the LSDV one, which ignores the estimated correction (documented, "
        "warned). Row xtlsdvc/consistent-bootstrap is the calibrated combination."
    ),
    "xtlsdvc/bootstrap": (
        "Bias only. The parametric bootstrap SE is calibrated (SE/SD 0.93) but the default "
        "Stata plug-in leaves a bias of 0.58 SD at 500 units; see xtlsdvc/default."
    ),
    "panel/fe-driscoll-kraay": (
        "Known small-T property of the Driscoll-Kraay HAC. The variance is a Bartlett HAC "
        "of the 60 period sums of the scores at the default lag 3 (a hand computation on "
        "one draw gives 0.0444 against the reported 0.0448). With serially correlated "
        "period shocks in the regressor and the error (rho 0.5 each) the kernel truncates "
        "and down-weights the autocovariances and the HAC is biased down at T = 60."
    ),
    "mmqreg/q25": (
        "Incidental-parameter bias of order 1/T, not the SE (SE/SD 0.95). The scale "
        "equation is fitted on absolute within residuals with unit effects estimated from "
        "8 observations each. Mean error at the 0.25 quantile: 0.069 with 4 periods, 0.035 "
        "with 8, 0.010 with 20, none with 40 (1,600 observations throughout). The median "
        "row is unaffected. Now stated in the mmqreg notes."
    ),
}

if __name__ == "__main__":
    main(__file__, FAMILY, ROWS, MECHANISMS)
