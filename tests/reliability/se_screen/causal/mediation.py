"""SE screen: mediation analysis.

Design constants (fixed before the first run, 2026-11-13):

* ``N = 1000``, ``B = 400``. Linear structural equations with a
  randomised binary treatment, so sequential ignorability holds and every
  effect has a closed form:

  ``M = 0.5 + 0.8 D + 0.3 x + e_M``, ``Y = 1 + 0.5 D + 1.5 M + 0.4 x + e_Y``

  gives an indirect effect of 1.2, a direct effect of 0.5 and a total
  effect of 1.7. The four-way decomposition adds a treatment-mediator
  interaction (see ``_four_way``).
* Bootstrap rows keep the function's default number of draws where a fit
  stays under a second.

Run: ``PYTHONPATH=src python tests/reliability/se_screen/causal/mediation.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from _causal_harness import Row, main  # noqa: E402
from scipy import stats  # noqa: E402

import statspai as sp  # noqa: E402

N = 1000
Z = stats.norm.ppf(0.975)
ACME, ADE, TOTAL = 0.8 * 1.5, 0.5, 0.8 * 1.5 + 0.5


def _draw(rng, n=N):
    x = rng.normal(size=n)
    d = rng.integers(0, 2, n)
    m = 0.5 + 0.8 * d + 0.3 * x + rng.normal(size=n)
    y = 1 + 0.5 * d + 1.5 * m + 0.4 * x + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "m": m, "x": x})


def _detail_row(detail, label):
    row = detail[detail["effect"].str.startswith(label)].iloc[0]
    return {
        "est": float(row["estimate"]),
        "se": float(row["se"]),
        "lo": float(row["ci_lower"]),
        "hi": float(row["ci_upper"]),
    }


def _mediate(label, **kw):
    def fit(rng):
        df = _draw(rng)
        if kw.get("inference", "bootstrap") == "bootstrap":
            kw["seed"] = int(rng.integers(2**31))
        r = sp.mediate(df, "y", "d", "m", ["x"], **kw)
        return _detail_row(r.detail, label)

    return fit


def _interventional(label):
    def fit(rng):
        df = _draw(rng)
        r = sp.mediate_interventional(
            df, "y", "d", "m", ["x"], n_boot=200, seed=int(rng.integers(2**31))
        )
        return _detail_row(r.detail, label)

    return fit


# Four-way: Y = 1 + 0.5 D + 1.5 M + 0.4 D M + 0.4 x + e, M = 0.5 + 0.8 D + 0.3 x + e
# at a0 = 0, a1 = 1, m0 = 0 and covariates at their mean (x = 0):
#   CDE = 0.5; INT_ref = 0.4 * E[M | D=0] = 0.2; INT_med = 0.4 * 0.8 = 0.32;
#   PIE = (1.5 + 0.4 * 0) * 0.8 = 1.2
FOUR_WAY = {"cde": 0.5, "int_ref": 0.2, "int_med": 0.32, "pie": 1.2}


def _four_way(component):
    def fit(rng):
        n = N
        x = rng.normal(size=n)
        d = rng.integers(0, 2, n)
        m = 0.5 + 0.8 * d + 0.3 * x + rng.normal(size=n)
        y = 1 + 0.5 * d + 1.5 * m + 0.4 * d * m + 0.4 * x + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "d": d, "m": m, "x": x})
        r = sp.four_way_decomposition(
            df, "y", "d", "m", ["x"], covariate_values={"x": 0.0}
        )
        est = float(getattr(r, component))
        se = float(r.se[component])
        return {"est": est, "se": se, "lo": est - Z * se, "hi": est + Z * se}

    return fit


# IV mediation (Dippel, Ferrara, Heblich): Z -> T, T -> M -> Y, T -> Y.
# The treatment error e_T also moves the mediator, and the mediator's own
# error v also moves the outcome; e_T does not enter the outcome equation.
IV_INDIRECT, IV_DIRECT = 0.7 * 1.2, 0.3


def _iv_mediate(which, link=0.6, **kw):
    """``link`` is the loading of the mediator's own error on the outcome."""

    def fit(rng):
        n = N
        z = rng.normal(size=n)
        e_t = rng.normal(size=n)
        v = rng.normal(size=n)
        t = 0.8 * z + e_t
        m = 0.7 * t + 0.8 * e_t + v
        y = 1.2 * m + 0.3 * t + link * v + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "d": t, "m": m, "z": z})
        r = sp.iv_mediate(df, "y", "d", "m", "z", **kw)
        fr = r.to_frame()
        row = fr.loc[which] if which in fr.index else fr[fr.iloc[:, 0] == which].iloc[0]
        cols = {c.lower(): c for c in fr.columns}
        est = float(row[cols.get("coef", cols.get("estimate"))])
        se = float(row[cols.get("se", cols.get("std.err", cols.get("std_err")))])
        lo = float(row[cols.get("ci_lower", cols.get("ci_low"))])
        hi = float(row[cols.get("ci_upper", cols.get("ci_high"))])
        return {"est": est, "se": se, "lo": lo, "hi": hi}

    return fit


# Two sequential mediators: D -> M1 -> M2 -> Y.
#   M1 = 0.8 D + e, M2 = 0.4 M1 + 0.5 D + e, Y = 1 + 0.5 D + M1 + 0.7 M2 + e
# through M1 (including M1 -> M2): 0.8 (1 + 0.4 * 0.7) = 1.024
# through M2 alone: 0.5 * 0.7 = 0.35; direct 0.5
def _paths(effect):
    def fit(rng):
        n = N
        d = rng.integers(0, 2, n)
        m1 = 0.8 * d + rng.normal(size=n)
        m2 = 0.4 * m1 + 0.5 * d + rng.normal(size=n)
        y = 1 + 0.5 * d + m1 + 0.7 * m2 + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "d": d, "m1": m1, "m2": m2})
        r = sp.mediate_paths(
            df, "y", "d", ["m1", "m2"], n_boot=200, seed=int(rng.integers(2**31))
        )
        det = r.detail
        row = det[(det["decomposition"] == "type1") & (det["effect"] == effect)].iloc[0]
        return {
            "est": float(row["estimate"]),
            "se": float(row["se"]),
            "lo": float(row["ci_lower"]),
            "hi": float(row["ci_upper"]),
        }

    return fit


D = "linear mediation model, randomised treatment"
ROWS = [
    Row(
        "mediate.acme",
        "mediate",
        "sp.mediate(df, y, d, m, ['x']), ACME",
        N,
        ACME,
        _mediate("ACME"),
        design=D,
    ),
    Row(
        "mediate.ade",
        "mediate",
        "sp.mediate(...), ADE",
        N,
        ADE,
        _mediate("ADE"),
        design=D,
    ),
    Row(
        "mediate.total",
        "mediate",
        "sp.mediate(...), total effect",
        N,
        TOTAL,
        _mediate("Total"),
        design=D,
    ),
    Row(
        "mediate.acme.delta",
        "mediate",
        "sp.mediate(..., inference='delta'), ACME",
        N,
        ACME,
        _mediate("ACME", inference="delta"),
        design=D,
    ),
    Row(
        "mediate.acme.interaction",
        "mediate",
        "sp.mediate(..., interaction=True), NIE",
        N,
        ACME,
        _mediate("NIE", interaction=True),
        design=D + " (no true interaction)",
    ),
    Row(
        "mediate_interventional.iie",
        "mediate_interventional",
        "sp.mediate_interventional(df, y, d, m, ['x'], n_boot=200), IIE",
        N,
        ACME,
        _interventional("IIE"),
        design=D,
    ),
    Row(
        "mediate_interventional.ide",
        "mediate_interventional",
        "sp.mediate_interventional(...), IDE",
        N,
        ADE,
        _interventional("IDE"),
        design=D,
    ),
    Row(
        "four_way.cde",
        "four_way_decomposition",
        "sp.four_way_decomposition(df, y, d, m, ['x']).cde +/- 1.96 se",
        N,
        FOUR_WAY["cde"],
        _four_way("cde"),
        design="linear model with a treatment-mediator interaction of 0.4",
    ),
    Row(
        "four_way.int_ref",
        "four_way_decomposition",
        "... .int_ref",
        N,
        FOUR_WAY["int_ref"],
        _four_way("int_ref"),
        design="as four_way.cde",
    ),
    Row(
        "four_way.int_med",
        "four_way_decomposition",
        "... .int_med",
        N,
        FOUR_WAY["int_med"],
        _four_way("int_med"),
        design="as four_way.cde",
    ),
    Row(
        "four_way.pie",
        "four_way_decomposition",
        "... .pie",
        N,
        FOUR_WAY["pie"],
        _four_way("pie"),
        design="as four_way.cde",
    ),
    Row(
        "iv_mediate.indirect",
        "iv_mediate",
        "sp.iv_mediate(df, y, d, m, 'z'), indirect effect",
        N,
        IV_INDIRECT,
        _iv_mediate("indirect"),
        design="one instrument; the treatment error moves the mediator, the "
        "mediator error moves the outcome",
    ),
    Row(
        "iv_mediate.direct",
        "iv_mediate",
        "sp.iv_mediate(...), direct effect",
        N,
        IV_DIRECT,
        _iv_mediate("direct"),
        design="as iv_mediate.indirect",
    ),
    Row(
        "iv_mediate.indirect.robust",
        "iv_mediate",
        "sp.iv_mediate(..., robust=True), indirect effect",
        N,
        IV_INDIRECT,
        _iv_mediate("indirect", robust=True),
        design="as iv_mediate.indirect",
    ),
    Row(
        "iv_mediate.indirect.negative_link",
        "iv_mediate",
        "sp.iv_mediate(df, y, d, m, 'z'), indirect effect",
        N,
        IV_INDIRECT,
        _iv_mediate("indirect", link=-0.6),
        design="as iv_mediate.indirect, with the mediator error entering "
        "the outcome with the opposite sign (-0.6)",
    ),
    Row(
        "iv_mediate.indirect.joint",
        "iv_mediate",
        "sp.iv_mediate(..., equations='joint'), indirect effect",
        N,
        IV_INDIRECT,
        _iv_mediate("indirect", equations="joint"),
        design="as iv_mediate.indirect",
    ),
    Row(
        "iv_mediate.indirect.joint.negative_link",
        "iv_mediate",
        "sp.iv_mediate(..., equations='joint'), indirect effect",
        N,
        IV_INDIRECT,
        _iv_mediate("indirect", link=-0.6, equations="joint"),
        design="as iv_mediate.indirect.negative_link",
    ),
    Row(
        "iv_mediate.indirect.joint.robust",
        "iv_mediate",
        "sp.iv_mediate(..., equations='joint', robust=True), indirect effect",
        N,
        IV_INDIRECT,
        _iv_mediate("indirect", link=-0.6, equations="joint", robust=True),
        design="as iv_mediate.indirect.negative_link",
    ),
    Row(
        "mediate_paths.via_m1",
        "mediate_paths",
        "sp.mediate_paths(df, y, d, ['m1', 'm2'], n_boot=200), via_M1",
        N,
        0.8 * (1 + 0.4 * 0.7),
        _paths("via_M1"),
        design="two sequential mediators, linear, randomised treatment",
    ),
    Row(
        "mediate_paths.via_m2",
        "mediate_paths",
        "sp.mediate_paths(...), via_M2",
        N,
        0.5 * 0.7,
        _paths("via_M2"),
        design="as mediate_paths.via_m1",
    ),
]

_INDEPENDENT = (
    "Reference convention, wrong in this design. ivmediate (and the default "
    "here, which equals it to 1e-11) sets the covariance between "
    "coefficients of different equations to zero, so Var(a d) = d^2 V(a) + "
    "a^2 V(d). The mediator equation's error and the outcome equation's "
    "error share the component v, so cov(a, d) is not zero: it is negative "
    "when v enters the outcome with a positive loading (the SE is too "
    "large) and positive when the loading is negative (the SE is too "
    "small). equations='joint' keeps the covariance (rows "
    "iv_mediate.indirect.joint*; equal to Stata gmm + nlcom to 2e-12). The "
    "default is unchanged because it is the documented ivmediate number. "
    "Decided 2026-11-13: a call without equations= warns that 'joint' "
    "becomes the default in a later release."
)
MECHANISMS: dict = {
    "iv_mediate.indirect": _INDEPENDENT,
    "iv_mediate.indirect.robust": _INDEPENDENT,
    "iv_mediate.indirect.negative_link": _INDEPENDENT,
}

if __name__ == "__main__":
    main(__file__, ROWS, MECHANISMS)
