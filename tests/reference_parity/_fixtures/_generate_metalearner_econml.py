"""Reference CATE functions for ``sp.predict_cate`` / ``sp.compare_metalearners``.

``econml.metalearners`` ``SLearner``, ``TLearner`` and ``XLearner`` with
ridge outcome models (``alpha = 50``), least-squares effect models and the
Newton logistic propensity of ``_metalearner_learners.py``, fitted on
``metalearner_train.csv``; the S-learner also with least squares.
Recorded: the effect at the 200 rows of ``metalearner_new.csv``, and the
standard deviation and interquartile range of the effect over the training
rows. econml is MIT licensed.

Run with an interpreter that has ``econml`` installed (not the repo venv)::

    python tests/reference_parity/_fixtures/_generate_metalearner_econml.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import econml
import numpy as np
import pandas as pd
import sklearn
from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.linear_model import LinearRegression, Ridge

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from _metalearner_learners import NewtonLogit  # noqa: E402

train = pd.read_csv(HERE / "metalearner_train.csv")
new = pd.read_csv(HERE / "metalearner_new.csv")
COV = ["x1", "x2", "x3"]
X, T, Y = train[COV].to_numpy(), train["d"].to_numpy(), train["y"].to_numpy()


# Ridge outcome models with least-squares effect models: with least squares
# in both stages the T- and X-learner are the same function, and the
# comparison could not tell them apart.
def ridge():
    return Ridge(alpha=50.0, solver="cholesky")


models = {
    "s_ols": SLearner(overall_model=LinearRegression()),
    "s": SLearner(overall_model=ridge()),
    "t": TLearner(models=ridge()),
    "x": XLearner(
        models=ridge(),
        propensity_model=NewtonLogit(),
        cate_models=LinearRegression(),
    ),
}
cases = {}
for name, est in models.items():
    est.fit(Y, T, X=X)
    insample = np.asarray(est.effect(X)).ravel()
    cases[name] = {
        "effect_new": [
            float(v) for v in np.asarray(est.effect(new[COV].to_numpy())).ravel()
        ],
        "cate_mean": float(insample.mean()),
        "cate_std": float(np.std(insample)),
        "cate_iqr": float(np.percentile(insample, 75) - np.percentile(insample, 25)),
    }
out = {
    "versions": {
        "python": sys.version.split()[0],
        "econml": econml.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
    },
    "cases": cases,
}
(HERE / "metalearner_econml.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
print(out["versions"], {k: (v["cate_mean"], v["cate_std"]) for k, v in cases.items()})
