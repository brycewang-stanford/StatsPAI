"""Double machine learning example: returns to schooling on Card (1995).

Demonstrates ``sp.dml`` — the high-dimensional / debiased-ML entry point —
on the bundled ``card_1995`` teaching dataset. Two orthogonal-score models:

* PLR  — partially linear regression (educ treated as continuous), ML
         nuisances debiasing the controls.
* PLIV — partially linear IV, instrumenting education with college
         proximity (``nearc4``), the classic Card identification.

The numbers reproduce the well-known pattern that the IV estimate of the
return to schooling exceeds the partialling-out estimate. Runs offline on
bundled data; see ``docs/guides/sp_dml_vs_doubleml.md`` for the DoubleML
alignment and ``docs/guides/rigorous_lasso_hdm.md`` for the rigorous-Lasso
(``hdm``) nuisance learners.
"""

import statspai as sp


def main() -> None:
    data = sp.datasets.card_1995()
    covariates = ["exper", "expersq", "black", "south", "smsa"]
    data = data.dropna(subset=["lwage", "educ", "nearc4"] + covariates)

    # Partially linear DML: returns to schooling with ML-debiased controls.
    plr = sp.dml(
        data=data,
        y="lwage",
        treat="educ",
        covariates=covariates,
        model="plr",
        ml_g="gbm",
        ml_m="gbm",
        n_folds=5,
        random_state=42,
    )
    print("DML partially linear (returns to schooling):")
    print(plr.summary())

    # Partially linear IV DML: instrument education with college proximity.
    pliv = sp.dml(
        data=data,
        y="lwage",
        treat="educ",
        covariates=covariates,
        instrument="nearc4",
        model="pliv",
        ml_g="gbm",
        ml_m="gbm",
        ml_r="gbm",
        n_folds=5,
        random_state=42,
    )
    print("\nDML partially linear IV (instrument = nearc4):")
    print(pliv.summary())


if __name__ == "__main__":
    main()
