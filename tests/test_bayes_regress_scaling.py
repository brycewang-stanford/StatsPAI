"""How the cost of ``sp.bayes_regress`` grows with n (rOpenSci BS7.3, audit
gap #19).

What is asserted: at a fixed number of iterations, the CPU time of a fit
grows no faster than ``n ** 1.3`` over a 16-fold range of n. The samplers'
work per iteration is linear in n (the data-augmentation and Metropolis
samplers touch every observation once) or constant (the Gibbs normal
sampler works on ``X'X`` and ``X'y`` after one pass), so an exponent above
1.3 means something quadratic crept in, such as an n x n matrix.

Why CPU time and medians: wall time on a shared machine moves with the
load from other processes, CPU time of this process much less. Each point
is the median of three fits, and the exponent is the least-squares slope
of log time on log n through three points, which a single slow run cannot
push past the bound. The test is marked slow because it fits each model
nine times.
"""

import time
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

pytestmark = pytest.mark.slow

SIZES = (2_000, 8_000, 32_000)


def _data(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(n)
    out = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    eta = 0.3 + 0.5 * out.x1 - 0.2 * out.x2
    out["y"] = eta + rng.normal(size=n)
    out["yb"] = (eta + rng.normal(size=n) > 0).astype(int)
    out["yc"] = rng.poisson(np.exp(0.2 * eta))
    return out


def _cpu_seconds(formula: str, data: pd.DataFrame, model: str) -> float:
    times = []
    for rep in range(3):
        t0 = time.process_time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp.bayes_regress(
                formula, data, model=model, draws=300, burnin=100, seed=rep
            )
        times.append(time.process_time() - t0)
    return float(np.median(times))


@pytest.mark.parametrize(
    "model, formula",
    [
        ("normal", "y ~ x1 + x2"),
        ("probit", "yb ~ x1 + x2"),
        ("poisson", "yc ~ x1 + x2"),
    ],
)
def test_cost_grows_at_most_like_n_to_the_1_3(model, formula):
    # warm the imports and any lazily compiled code out of the first point
    _cpu_seconds(formula, _data(500), model)
    t = np.array([_cpu_seconds(formula, _data(n), model) for n in SIZES])
    slope = np.polyfit(np.log(SIZES), np.log(t), 1)[0]
    assert slope <= 1.3, f"{model}: time {t.round(3)} s, exponent {slope:.2f}"
