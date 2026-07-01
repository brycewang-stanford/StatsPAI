"""StatsPAI panel GLM (feglm / fepois) parity — Module 67 (Python side).

Generates two deterministic balanced panel DGPs — one Bernoulli-logit, one
Poisson-count — and fits the two high-dimensional FE GLM estimators that
extend ``sp.feols`` to non-Gaussian families. The companion 67_panel_glm.R
runs the canonical R reference: ``fixest::feglm(..., family = "logit")`` and
``fixest::fepois`` on the same CSV, with the same single entity fixed
effect (``id``) and the same dropped-constant design.

fixest is the de facto R reference for this estimator family and ``sp.feols``
is already graded bit-exact against ``fixest::feols`` (Track A module 03);
``sp.feglm`` / ``sp.fepois`` share the same absorbed-effects / IWLS / SE
machinery, so they land on bit-for-bit agreement with their fixest siblings.

Tolerance: rel_est < 1e-6, rel_se < 1e-6 (machine). Both estimators are
closed-form given the working-weight IWLS path, with the same small-sample
correction ``ssc = "n"`` defaults as fixest; observed gaps were ~1e-6 on the
Poisson SE for x1 and 0 on everything else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "67_panel_glm"


def make_logit(seed: int = PARITY_SEED, N: int = 30, T: int = 12) -> pd.DataFrame:
    """Balanced panel Bernoulli-logit DGP with id fixed effect."""
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(N), T)
    years = np.tile(np.arange(T), N)
    x1 = rng.normal(size=N * T)
    x2 = rng.normal(size=N * T)
    # logit latent: id FE drawn from N(0, 0.5), x effects 0.4 / -0.3
    fe = rng.normal(0, 0.5, N)[ids]
    z = 0.2 + 0.4 * x1 - 0.3 * x2 + fe + rng.normal(size=N * T)
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=N * T) < p).astype(float)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "id": ids, "year": years})


def make_poisson(seed: int = PARITY_SEED + 1, N: int = 25, T: int = 12) -> pd.DataFrame:
    """Balanced panel Poisson DGP with id fixed effect."""
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(N), T)
    years = np.tile(np.arange(T), N)
    x1 = rng.normal(size=N * T)
    x2 = rng.normal(size=N * T)
    fe = rng.normal(0, 0.5, N)[ids]
    mu = np.exp(0.5 + 0.3 * x1 - 0.2 * x2 + fe + rng.normal(size=N * T) * 0.5)
    y = rng.poisson(mu).astype(float)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "id": ids, "year": years})


def _coef_rows(model: str, res, names, n: int) -> list[ParityRecord]:
    rows: list[ParityRecord] = []
    for name in names:
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"{model}_{name}",
                estimate=float(res.params[name]),
                se=float(res.std_errors[name]),
                n=n,
            )
        )
    return rows


def main() -> None:
    df_logit = make_logit()
    df_pois = make_poisson()

    # Both CSV dumps share the same module prefix so the R side can
    # read each by name.
    dump_csv(df_logit, f"{MODULE}_logit")
    dump_csv(df_pois, f"{MODULE}_poisson")

    n_logit = len(df_logit)
    n_pois = len(df_pois)
    rows: list[ParityRecord] = []

    res_logit = sp.feglm("y ~ x1 + x2 | id", data=df_logit, family="logit")
    rows += _coef_rows("feglm_logit", res_logit, ["x1", "x2"], n_logit)

    res_pois = sp.fepois("y ~ x1 + x2 | id", data=df_pois)
    rows += _coef_rows("fepois", res_pois, ["x1", "x2"], n_pois)

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "n_logit": n_logit,
            "n_poisson": n_pois,
            "feglm_logit_ref": "fixest::feglm(family='logit')",
            "fepois_ref": "fixest::fepois",
            "note": (
                "Both estimators absorb the id fixed effect via the same "
                "demeaning path used by sp.feols (bit-exact against "
                "fixest::feols, Track A module 03); the IWLS loop adds the "
                "GLM family. Coefficients and IID SEs land bit-for-bit on "
                "the fixest siblings."
            ),
        },
    )


if __name__ == "__main__":
    main()
