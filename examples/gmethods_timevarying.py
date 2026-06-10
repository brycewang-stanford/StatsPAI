"""G-methods for time-varying confounding (public-health example).

A self-contained, offline demonstration of why epidemiology needs Robins'
g-methods. We simulate the canonical structure that breaks ordinary
regression:

    L0 ~ N(0, 1)                          baseline confounder
    A0 ~ Bernoulli(sigmoid(0.5 * L0))     treatment at visit 0
    L1 = L0 + 0.5 * A0 + noise            time-varying confounder,
                                          AFFECTED by earlier treatment A0
    A1 ~ Bernoulli(sigmoid(0.3*L1 + 0.5*A0))   treatment at visit 1
    Y  = 2*A0 + 3*A1 + L0 + 2*L1 + noise  outcome (L1 affects Y directly)

L1 is simultaneously (i) a confounder of the A1 -> Y relationship
(L1 -> A1 and L1 -> Y) and (ii) a mediator of the earlier treatment
(A0 -> L1 -> Y). That dual role is the trap:

* DON'T adjust for L1  -> confounding of A1 remains  -> biased.
* DO   adjust for L1   -> the A0 -> L1 -> Y path is blocked, so an
  ordinary regression's A0 coefficient misses A0's indirect effect.

The true contrast of the *joint intervention* "always treat" (A0=A1=1)
vs "never treat" (A0=A1=0) is E[Y]=6 - 0 = 6. A naive OLS that controls
for L1 sums its A0+A1 coefficients to ~5 (it conditions away the
A0 -> L1 -> Y pathway). The parametric g-formula and the marginal
structural model both recover ~6.
"""

import numpy as np
import pandas as pd

import statspai as sp


def make_data(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    L0 = rng.normal(0, 1, n)
    A0 = rng.binomial(1, 1 / (1 + np.exp(-0.5 * L0)), n)
    L1 = L0 + 0.5 * A0 + rng.normal(0, 0.5, n)
    A1 = rng.binomial(1, 1 / (1 + np.exp(-(0.3 * L1 + 0.5 * A0))), n)
    Y = 2 * A0 + 3 * A1 + L0 + 2 * L1 + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {"id": np.arange(n), "L0": L0, "A0": A0, "L1": L1, "A1": A1, "Y": Y}
    )


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Reshape the wide DGP into the (id, time) long format `sp.msm` wants."""
    n = len(wide)
    long = pd.DataFrame(
        {
            "id": np.repeat(wide["id"].to_numpy(), 2),
            "time": np.tile([0, 1], n),
            "A": np.empty(2 * n),
            "L": np.empty(2 * n),
            "Y": np.repeat(wide["Y"].to_numpy(), 2),
        }
    )
    long.loc[0::2, "A"] = wide["A0"].to_numpy()
    long.loc[1::2, "A"] = wide["A1"].to_numpy()
    long.loc[0::2, "L"] = wide["L0"].to_numpy()
    long.loc[1::2, "L"] = wide["L1"].to_numpy()
    return long


def main() -> None:
    wide = make_data()
    print("True 'always treat' vs 'never treat' contrast = 6.0\n")

    # 1. Naive OLS that "adjusts for" the time-varying confounder L1.
    #    Biased: conditioning on L1 blocks the A0 -> L1 -> Y pathway, so the
    #    A0 coefficient misses A0's indirect effect on Y.
    import statsmodels.formula.api as smf

    naive = smf.ols("Y ~ A0 + A1 + L0 + L1", data=wide).fit()
    naive_contrast = naive.params["A0"] + naive.params["A1"]
    print(f"[naive OLS, adjusting for L1]   A0+A1 = {naive_contrast:.3f}  "
          f"(biased low — should be 6)")

    # 2. Parametric g-formula via iterative conditional expectation.
    always = sp.gformula.ice(
        data=wide, id_col="id", time_col=None,
        treatment_cols=["A0", "A1"], confounder_cols=[["L0"], ["L1"]],
        outcome_col="Y", treatment_strategy=[1, 1],
    )
    never = sp.gformula.ice(
        data=wide, id_col="id", time_col=None,
        treatment_cols=["A0", "A1"], confounder_cols=[["L0"], ["L1"]],
        outcome_col="Y", treatment_strategy=[0, 0],
    )
    print(f"[g-formula (ICE)]               contrast = "
          f"{always.value - never.value:.3f}  (≈ 6)")

    # 3. Marginal structural model via stabilized IPTW.
    msm = sp.msm(
        data=to_long(wide), y="Y", treat="A", id="id", time="time",
        time_varying=["L"], exposure="cumulative",
    )
    print(f"[MSM via IPTW]                  per-dose  = {msm.estimate:.3f}  "
          f"(cumulative-exposure slope)")

    print("\nThe g-formula and MSM recover the truth; the naive OLS does not.")


if __name__ == "__main__":
    main()
