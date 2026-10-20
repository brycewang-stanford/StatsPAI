"""Write the data for the CBPS and mediation-sensitivity parity fixtures.

Run once; the CSVs are committed so the R generators
(``_generate_cbps_R.R``, ``_generate_medsens_R.R``) and the Python tests
(``test_cbps_R_parity.py``, ``test_medsens_R_parity.py``) read identical
bytes. All numbers are written with ``%.17g``.

cbps_sim.csv     n = 400, binary treatment, mildly misspecified logit.
    x1 ~ N(0, 1); x2 ~ Exp(1); x3 ~ Bernoulli(0.4)
    d ~ Bernoulli(expit(-0.3 + 0.6 x1 - 0.4 x2 + 0.5 x3 + 0.1 x1^2))
    y = 1 + 2 d + x1 + 0.5 x2 + d x1 + N(0, 1)

cbps_bound.csv   n = 400, strongly misspecified logit (cubic in x1), chosen
    so that the continuously-updated criterion along the ray of the logit
    MLE is minimised below 0.8: the two-step weighting point then sits on
    the lower end of its search interval.
    x1 ~ N(0, 1); x2 ~ Exp(1)
    d ~ Bernoulli(expit(1.5 x1 - 0.21 x1^3 + 0.2 x2));  y = 1 + d + x1 + N(0, 1)

medsens_sim.csv  n = 400, two linear mediation systems.
    x1, x2 ~ N(0, 1)
    binary treatment:      t ~ Bernoulli(0.5)
        m = 0.2 + 0.5 t + 0.3 x1 + N(0, 1)
        y = 1 + 0.6 t + 0.8 m + 0.4 x2 + N(0, 1)
    continuous treatment:  tc ~ N(0, 1)
        mc = -0.7 tc + 0.2 x1 + N(0, 1)
        yc = 0.3 tc + 1.1 mc - 0.4 x2 + N(0, 2^2)
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _expit(z):
    return 1 / (1 + np.exp(-z))


def main() -> None:
    rng = np.random.default_rng(20261020)
    n = 400
    x1, x2, x3 = rng.normal(size=n), rng.exponential(size=n), rng.binomial(1, 0.4, n)
    d = rng.binomial(1, _expit(-0.3 + 0.6 * x1 - 0.4 * x2 + 0.5 * x3 + 0.1 * x1**2))
    y = 1 + 2 * d + x1 + 0.5 * x2 + d * x1 + rng.normal(size=n)
    pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2, "x3": x3}).to_csv(
        HERE / "cbps_sim.csv", index=False, float_format="%.17g"
    )

    rng = np.random.default_rng(20261021)
    x1, x2 = rng.normal(size=n), rng.exponential(size=n)
    d = rng.binomial(1, _expit(1.5 * x1 - 0.21 * x1**3 + 0.2 * x2))
    y = 1 + d + x1 + rng.normal(size=n)
    pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2}).to_csv(
        HERE / "cbps_bound.csv", index=False, float_format="%.17g"
    )

    rng = np.random.default_rng(20261022)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    t = rng.binomial(1, 0.5, n)
    m = 0.2 + 0.5 * t + 0.3 * x1 + rng.normal(size=n)
    y = 1 + 0.6 * t + 0.8 * m + 0.4 * x2 + rng.normal(size=n)
    tc = rng.normal(size=n)
    mc = -0.7 * tc + 0.2 * x1 + rng.normal(size=n)
    yc = 0.3 * tc + 1.1 * mc - 0.4 * x2 + 2 * rng.normal(size=n)
    pd.DataFrame(
        {"t": t, "m": m, "y": y, "tc": tc, "mc": mc, "yc": yc, "x1": x1, "x2": x2}
    ).to_csv(HERE / "medsens_sim.csv", index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
