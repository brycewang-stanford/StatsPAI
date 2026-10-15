"""Reference values for ``sp.dml`` options against Python ``DoubleML``.

Writes, next to this file:

* ``dml_options_data.csv`` -- one simulated sample carrying the outcome,
  treatment and instrument of each of the four models ``sp.dml``
  dispatches to, plus committed fold labels:
  ``f2`` / ``f3`` / ``f10`` (balanced random partitions into 2, 3 and 10
  folds) and ``rep{r}_kfold`` / ``rep{r}_skfold`` for r = 0, 1, 2 (the
  folds ``sp.dml`` draws itself under ``n_rep=3`` with its default
  ``random_state=42``: scikit-learn ``KFold(5, shuffle=True,
  random_state=42 + r)``, and ``StratifiedKFold`` on the binary treatment
  for the interactive model);
* ``dml_options_doubleml.json`` -- ``DoubleML`` (doubleml-for-py) run on
  those folds through ``set_sample_splitting``.

Deterministic learners only (OLS; unpenalised logit), so with the folds
shared the two engines must agree to floating point. The configurations
are the ones ``sp.validation_scope`` had no reference for:

* n_folds 2, 3 and 10 for every model (Track A uses 5, the external test 4);
* ``score='IV-type'`` for the partially linear model at those fold counts;
* ``score='ATTE'`` for the interactive model at 3 folds;
* propensity processing other than the two registered ones: trimming at
  0.1 (binding on this design) and self-normalised weights;
* ``n_rep=3``. DoubleML stores every repetition (``all_coef`` /
  ``all_se``), and those are what the per-repetition comparison uses.
  The aggregated standard error is NOT a parity row: Python DoubleML
  takes the median of the repetitions' upper 95% bounds, R DoubleML and
  StatsPAI do not, see ``test_dml_options_doubleml_parity.py``.

Run from the repository root with doubleml installed:

    python tests/reference_parity/_fixtures/_generate_dml_options_doubleml.py

Environment recorded in ``_meta``.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
N = 800
SEED = 20261015
COVARIATES = ["x1", "x2", "x3", "x4"]


def _logistic(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-v))


def make_data() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(N, 4))
    x1, x2, x3, x4 = X.T
    g = 0.8 * x1 - 0.5 * x2 + 0.3 * x3 * x4
    df = pd.DataFrame(X, columns=COVARIATES)
    # partially linear: continuous treatment
    df["d"] = 0.6 * x1 + 0.4 * x3 + rng.normal(size=N)
    df["y"] = 0.7 * df["d"] + g + rng.normal(size=N)
    # interactive: binary treatment; the index puts some propensities
    # outside (0.1, 0.9), so a 0.1 trimming threshold binds
    ps = _logistic(1.4 * x1 - 0.8 * x2 + 0.3 * x4)
    df["db"] = rng.binomial(1, ps)
    df["yb"] = (1.0 + 0.5 * x2) * df["db"] + g + rng.normal(size=N)
    # partially linear IV: continuous endogenous treatment
    z = rng.normal(size=N)
    u = rng.normal(size=N)
    df["z"] = z
    df["dc"] = 0.7 * z + 0.4 * x1 + u
    df["yc"] = 0.5 * df["dc"] + g + 0.6 * u + rng.normal(size=N)
    # interactive IV: binary instrument and treatment; two-sided
    # noncompliance, so both P(D = 1 | Z = z, X) are identified
    zb = rng.binomial(1, _logistic(1.4 * x1 - 0.6 * x3))
    take = rng.uniform(size=N) < _logistic(0.6 + 0.5 * x2)
    always = rng.uniform(size=N) < _logistic(-1.8 + 0.4 * x1)
    df["zb"] = zb
    df["di"] = np.where(zb == 1, take, always).astype(int)
    df["yi"] = 0.9 * df["di"] + g + rng.normal(size=N)

    labels = np.random.default_rng(SEED + 1)
    for k in (2, 3, 10):
        f = np.tile(np.arange(k), int(np.ceil(N / k)))[:N]
        labels.shuffle(f)
        df[f"f{k}"] = f.astype(int)

    from sklearn.model_selection import KFold, StratifiedKFold

    Xm = df[COVARIATES].to_numpy()
    for r in range(3):
        for name, splitter, args in (
            ("kfold", KFold(5, shuffle=True, random_state=42 + r), (Xm,)),
            (
                "skfold",
                StratifiedKFold(5, shuffle=True, random_state=42 + r),
                (Xm, df["db"].to_numpy()),
            ),
        ):
            col = np.empty(N, dtype=int)
            for fold, (_, test) in enumerate(splitter.split(*args)):
                col[test] = fold
            df[f"rep{r}_{name}"] = col
    return df


def _smpls(folds: np.ndarray):
    idx = np.arange(len(folds))
    return [(idx[folds != k], idx[folds == k]) for k in sorted(np.unique(folds))]


#: (key, model, score, fold column(s), extra DoubleML keyword arguments)
CASES = []
for k in (2, 3, 10):
    CASES += [
        (f"plr_po_k{k}", "plr", "partialling out", f"f{k}", {}),
        (f"plr_ivtype_k{k}", "plr", "IV-type", f"f{k}", {}),
        (f"irm_ate_k{k}", "irm", "ATE", f"f{k}", {}),
        (f"pliv_k{k}", "pliv", "partialling out", f"f{k}", {}),
        (f"iivm_k{k}", "iivm", "LATE", f"f{k}", {}),
    ]
CASES += [
    ("irm_atte_k3", "irm", "ATTE", "f3", {}),
    ("irm_ate_trim0.1_k3", "irm", "ATE", "f3", {"trimming_threshold": 0.1}),
    ("irm_ate_normalized_k3", "irm", "ATE", "f3", {"normalize_ipw": True}),
    ("irm_atte_normalized_k3", "irm", "ATTE", "f3", {"normalize_ipw": True}),
    ("iivm_trim0.1_k3", "iivm", "LATE", "f3", {"trimming_threshold": 0.1}),
    ("iivm_normalized_k3", "iivm", "LATE", "f3", {"normalize_ipw": True}),
    ("plr_po_rep3", "plr", "partialling out", "kfold", {}),
    ("plr_ivtype_rep3", "plr", "IV-type", "kfold", {}),
    ("irm_ate_rep3", "irm", "ATE", "skfold", {}),
]


def run_doubleml(df: pd.DataFrame) -> dict:
    import doubleml
    from sklearn.linear_model import LinearRegression, LogisticRegression

    def reg():
        return LinearRegression()

    def clf():
        return LogisticRegression(penalty=None, max_iter=1000)

    out = {}
    for key, model, score, fold_col, extra in CASES:
        reps = fold_col in ("kfold", "skfold")
        if model == "plr":
            data = doubleml.DoubleMLData(df, "y", "d", COVARIATES)
            learners = dict(ml_l=reg(), ml_m=reg())
            if score == "IV-type":
                learners["ml_g"] = reg()
            cls = doubleml.DoubleMLPLR
        elif model == "irm":
            data = doubleml.DoubleMLData(df, "yb", "db", COVARIATES)
            learners = dict(ml_g=reg(), ml_m=clf())
            cls = doubleml.DoubleMLIRM
        elif model == "pliv":
            data = doubleml.DoubleMLData(df, "yc", "dc", COVARIATES, z_cols="z")
            learners = dict(ml_l=reg(), ml_m=reg(), ml_r=reg())
            cls = doubleml.DoubleMLPLIV
        else:
            data = doubleml.DoubleMLData(df, "yi", "di", COVARIATES, z_cols="zb")
            learners = dict(ml_g=reg(), ml_m=clf(), ml_r=clf())
            cls = doubleml.DoubleMLIIVM
        if model in ("irm", "iivm"):
            extra = {"trimming_threshold": 0.01, "normalize_ipw": False, **extra}
        if reps:
            folds = [df[f"rep{r}_{fold_col}"].to_numpy() for r in range(3)]
        else:
            folds = [df[fold_col].to_numpy()]
        obj = cls(
            data,
            **learners,
            n_folds=len(np.unique(folds[0])),
            n_rep=len(folds),
            score=score,
            draw_sample_splitting=False,
            **extra,
        )
        obj.set_sample_splitting([_smpls(f) for f in folds])
        obj.fit()
        entry = {
            "model": model,
            "score": score,
            "folds": fold_col,
            "n_rep": len(folds),
            "theta": float(np.ravel(obj.coef)[0]),
            "se": float(np.ravel(obj.se)[0]),
            **{k: v for k, v in extra.items()},
        }
        if reps:
            entry["all_theta"] = [float(v) for v in np.ravel(obj.all_coef)]
            entry["all_se"] = [float(v) for v in np.ravel(obj.all_se)]
        out[key] = entry
    return out


def main() -> None:
    import doubleml
    import sklearn

    df = make_data()
    df.to_csv(HERE / "dml_options_data.csv", index=False, float_format="%.17g")
    # Estimate on the bytes the test will read.
    df = pd.read_csv(HERE / "dml_options_data.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = run_doubleml(df)
    out["_meta"] = {
        "generated_by": "_generate_dml_options_doubleml.py",
        "doubleml_version": doubleml.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "n": N,
        "learners": "LinearRegression; LogisticRegression(penalty=None)",
        "n_cases": len(CASES),
    }
    (HERE / "dml_options_doubleml.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(CASES)} cases")


if __name__ == "__main__":
    main()
