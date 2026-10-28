"""``sp.simex`` against R ``simex`` 1.8 on the same simulation draws (T2).

Reference: ``_fixtures/simex_det_R.txt`` written by
``_fixtures/_generate_simex_det_R.R`` (R 4.5.2, simex 1.8). Both sides are
fed the same standard normal pseudo-errors (``simex_det_draws.csv.gz``) and
the same re-misclassified vectors (``simex_det_mc.csv.gz``), written by
``_fixtures/_generate_simex_det_data.py``: R through a replaced
``rnorm`` / ``misclass``, ``sp.simex`` through its ``_pseudo_errors`` /
``_simex_misclass.misclassify`` hooks. With the randomness removed the two
compute the same numbers, so this is deterministic parity for

* SIMEX point estimates with the quadratic, linear and rational
  ("nonlinear") extrapolants, for least squares, logit and Poisson;
* the jackknife variance (quadratic extrapolant);
* the asymptotic variance of [@carroll1996asymptotics]
  (``variance='asymptotic'``, R ``asymptotic = TRUE``);
* MC-SIMEX (R ``mcsimex``) for a misclassified regressor and a
  misclassified outcome, with the quadratic and log-linear extrapolants;
* MC-SIMEX for a three-level misclassified factor (3 x 3 matrix) entered as
  ``C(xf)``, least squares and logit (``simex_det_factor_*``).

Observed relative differences: estimates and asymptotic SEs 7e-15 or
less; jackknife SEs 8e-15 (least squares) to 1.7e-9 (logit), because R's
``glm`` returns the covariance at the weights of the iteration before its
last update (the reference runs ``glm`` with ``epsilon = 1e-16``; at the
default 1e-8 that gap is 3e-5). Tolerances: 1e-12 for estimates and
asymptotic SEs, 1e-8 for jackknife SEs. The rational extrapolant is fitted
by ``stats::nls`` in R, which stops on a relative-offset criterion of 1e-5;
observed 2.9e-7, compared at 1e-6.
"""

from __future__ import annotations

import importlib
import pathlib
import warnings
from functools import lru_cache
from typing import Dict

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp
    from statspai.measurement_error import _simex_misclass

_FIX = pathlib.Path(__file__).parent / "_fixtures"
_SIMEX = importlib.import_module("statspai.measurement_error.simex")
B = 20
# R's glm returns the covariance at the weights of the iteration before the
# last update, so a refit's model-based covariance sits ~1e-9 (relative)
# from the one at the optimum even with epsilon = 1e-16; least squares is
# exact (observed: 8e-15 gaussian, 1.7e-9 logit, 3e-11 Poisson).
JSE_TOL = 1e-8
P_X = [[0.85, 0.10], [0.15, 0.90]]
P_Y = [[0.90, 0.20], [0.10, 0.80]]


@lru_cache(maxsize=1)
def _r() -> Dict[str, float]:
    out: Dict[str, float] = {}
    text = (_FIX / "simex_det_R.txt").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip():
            key, value = line.split()
            out[key] = float(value)
    return out


@lru_cache(maxsize=1)
def _data() -> pd.DataFrame:
    df = pd.read_csv(_FIX / "simex_det_data.csv")
    df["w3"] = df["w"] + 3.0
    return df


@lru_cache(maxsize=1)
def _draws() -> np.ndarray:
    return pd.read_csv(_FIX / "simex_det_draws.csv.gz").to_numpy(dtype=float)


@lru_cache(maxsize=1)
def _mc() -> pd.DataFrame:
    return pd.read_csv(_FIX / "simex_det_mc.csv.gz")


def _replay(monkeypatch, prefix=None):
    """Feed sp.simex the fixture draws in the order R consumed them."""
    count = {"i": 0}

    def errors(rng, n, p):
        col = _draws()[:, count["i"]]
        count["i"] += 1
        return col.reshape(n, p)

    def flips(values, P_lam, rng):
        out = _mc()[f"{prefix}{count['i']}"].to_numpy(dtype=float)
        count["i"] += 1
        return out

    monkeypatch.setattr(_SIMEX, "_pseudo_errors", errors)
    monkeypatch.setattr(_simex_misclass, "misclassify", flips)
    return count


def _check(fit, case, names, tol):
    """``tol``: relative tolerance for the estimates and asymptotic SEs."""
    ref = _r()
    jack = np.asarray(fit.model_info["jackknife_cov"])
    for i, (py, r) in enumerate(names.items()):
        for kind, value, rel in (
            ("b", fit.params[py], tol),
            ("jse", np.sqrt(jack[i, i]), JSE_TOL),
            ("ase", fit.std_errors[py], tol),
        ):
            expected = ref[f"{case}_{kind}_{r}"]
            assert value == pytest.approx(expected, rel=rel, abs=0), (
                case,
                kind,
                py,
                value,
                expected,
            )


SIMEX_CASES = [
    ("gaussian", "y ~ w + z", "w", "quadratic", 1e-12),
    ("gaussian", "y ~ w + z", "w", "linear", 1e-12),
    ("gaussian1", "y ~ w3", "w3", "nonlinear", 1e-6),
    ("logit", "d ~ w + z", "w", "quadratic", 1e-12),
    ("logit", "d ~ w + z", "w", "linear", 1e-12),
    ("poisson", "cnt ~ w + z", "w", "quadratic", 1e-12),
]


@pytest.mark.parametrize("label, formula, var, extrapolant, tol", SIMEX_CASES)
def test_simex_asymptotic_matches_r(monkeypatch, label, formula, var, extrapolant, tol):
    count = _replay(monkeypatch)
    family = "gaussian" if label.startswith("gaussian") else label
    fit = sp.simex(
        formula,
        _data(),
        error_sd={var: 0.5},
        family=family,
        B=B,
        extrapolant=extrapolant,
        variance="asymptotic",
    )
    assert count["i"] == 4 * B
    terms = ["Intercept"] + [t.strip() for t in formula.split("~")[1].split("+")]
    _check(fit, f"simex_{label}_{extrapolant}", {t: t for t in terms}, tol)


MC_CASES = [
    ("x", "gaussian", "yb ~ xs + z", "quadratic"),
    ("x", "gaussian", "yb ~ xs + z", "loglinear"),
    ("x", "logit", "db ~ xs + z", "quadratic"),
    ("y", "logit", "dm ~ z", "quadratic"),
    ("y", "logit", "dm ~ z", "loglinear"),
]


@pytest.mark.parametrize("target, family, formula, extrapolant", MC_CASES)
def test_mcsimex_matches_r(monkeypatch, target, family, formula, extrapolant):
    count = _replay(monkeypatch, prefix=target)
    var, P = ("xs", P_X) if target == "x" else ("dm", P_Y)
    fit = sp.simex(
        formula,
        _data(),
        misclassification={var: P},
        family=family,
        B=B,
        extrapolant=extrapolant,
        variance="asymptotic",
    )
    assert count["i"] == 4 * B
    terms = [t.strip() for t in formula.split("~")[1].split("+")]
    names = {"Intercept": "Intercept"}
    names.update({t: ("xs1" if t == "xs" else t) for t in terms})
    _check(fit, f"mc{target}_{family}_{extrapolant}", names, 1e-12)


@lru_cache(maxsize=1)
def _factor_data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "simex_det_factor_data.csv")


@lru_cache(maxsize=1)
def _factor_mc() -> pd.DataFrame:
    return pd.read_csv(_FIX / "simex_det_factor_mc.csv.gz")


P_F = [[0.80, 0.10, 0.05], [0.10, 0.80, 0.15], [0.10, 0.10, 0.80]]
FACTOR_CASES = [
    ("gaussian", "yf ~ C(xf) + z", "quadratic"),
    ("gaussian", "yf ~ C(xf) + z", "loglinear"),
    ("logit", "df ~ C(xf) + z", "quadratic"),
]


@pytest.mark.parametrize("family, formula, extrapolant", FACTOR_CASES)
def test_mcsimex_factor_matches_r(monkeypatch, family, formula, extrapolant):
    count = {"i": 0}

    def flips(values, P_lam, rng):
        assert P_lam.shape == (3, 3)
        out = _factor_mc()[f"f{count['i']}"].to_numpy(dtype=float)
        count["i"] += 1
        return out

    monkeypatch.setattr(_simex_misclass, "misclassify", flips)
    fit = sp.simex(
        formula,
        _factor_data(),
        misclassification={"xf": P_F},
        family=family,
        B=B,
        extrapolant=extrapolant,
        variance="asymptotic",
    )
    assert count["i"] == 4 * B
    names = {
        "Intercept": "Intercept",
        "C(xf)[T.1]": "xf1",
        "C(xf)[T.2]": "xf2",
        "z": "z",
    }
    _check(fit, f"mcf_{family}_{extrapolant}", names, 1e-12)
