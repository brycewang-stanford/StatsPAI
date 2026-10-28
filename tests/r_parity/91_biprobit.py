"""StatsPAI bivariate probit parity (Python side) -- Module 91.

Pins ``sp.biprobit`` (the model behind
``sp.positive_correlation_test(method='biprobit')``) against Stata's
``biprobit`` and R ``GJRM::gjrm(margins = c("probit", "probit"),
model = "B", copula = "N")``. The bivariate probit has no
author-maintained reference; Stata's ``biprobit`` is the long-standing
one, so it is treated as canonical and GJRM as a bridge.

Seemingly unrelated design: different regressors in the two equations
plus a shared one, with a true error correlation of 0.4.

Rows: the six index coefficients and ``athrho`` with their
observed-information standard errors, ``rho``, the log likelihood, and
the likelihood-ratio statistic for rho = 0 against two separate probits
(Stata's ``e(chi2_c)``).

The R side is about 1e-7 away from the other two. GJRM evaluates the
bivariate normal CDF with ``VGAM::pbinorm``, which is accurate to about
1e-7 (measured against adaptive quadrature: errors of -9e-8 to 2e-7 at
four points), so its likelihood peaks at a slightly different point;
``VGAM::vglm(binom2.rho)`` lands on the same point as GJRM to 1e-10 for
the same reason. StatsPAI and Stata evaluate the CDF to 1e-16 and agree
to 1e-15. The gap is inside the 1e-6 budget; the Stata do-file uses tight
ML tolerances because at the default ones Stata stops about 3e-9 short of
the maximum.

Tolerance: rel < 1e-6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "91_biprobit"


def build_frame() -> pd.DataFrame:
    rng = np.random.default_rng(PARITY_SEED)
    n = 3000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.uniform(-1.0, 1.0, n)
    rho = 0.4
    e = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
    y1 = (0.3 + 0.8 * x1 - 0.5 * x3 + e[:, 0] > 0).astype(int)
    y2 = (-0.2 + 0.6 * x2 + 0.4 * x3 + e[:, 1] > 0).astype(int)
    return pd.DataFrame({"y1": y1, "y2": y2, "x1": x1, "x2": x2, "x3": x3})


def main() -> None:
    df = build_frame()
    dump_csv(df, MODULE)

    fit = sp.biprobit(df, "y1", "y2", ["x1", "x3"], ["x2", "x3"])
    n = int(len(df))

    def rec(stat: str, est: float, se: float | None = None) -> ParityRecord:
        return ParityRecord(
            module=MODULE,
            side="py",
            statistic=stat,
            estimate=float(est),
            se=None if se is None else float(se),
            n=n,
        )

    rows: list[ParityRecord] = []
    for eq, terms in (("eq1", ("x1", "x3", "_cons")), ("eq2", ("x2", "x3", "_cons"))):
        for t in terms:
            key = f"{eq}.{t}"
            name = "cons" if t == "_cons" else t
            rows.append(rec(f"{eq}_b_{name}", fit.params[key], fit.std_errors[key]))

    rho = float(fit.model_info["rho"])
    # athrho and its SE by the delta method from rho's (d rho / d athrho =
    # 1 - rho^2), which is how Stata transforms back the other way.
    athrho = float(np.arctanh(rho))
    athrho_se = float(fit.model_info["rho_se"]) / (1.0 - rho**2)
    rows.append(rec("athrho", athrho, athrho_se))
    rows.append(rec("rho", rho))
    ll = float(fit.diagnostics["log_likelihood"])
    rows.append(rec("loglik", ll))

    ll0 = 0.0
    for y, x in (("y1", ["x1", "x3"]), ("y2", ["x2", "x3"])):
        ll0 += float(sp.probit(f"{y} ~ {' + '.join(x)}", df).model_info["ll"])
    rows.append(rec("lr_rho0", 2.0 * (ll - ll0)))

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "equations": ["y1 = x1 x3", "y2 = x2 x3"],
            "n_rows": len(rows),
            "reference": (
                "Stata 18 biprobit (canonical); R GJRM::gjrm model='B', "
                "copula='N' (bridge)"
            ),
        },
    )


if __name__ == "__main__":
    main()
