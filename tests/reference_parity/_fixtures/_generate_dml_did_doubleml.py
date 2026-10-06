"""Reference numbers for ``test_dml_did_doubleml_parity.py``.

Runs ``DoubleMLDID`` (panel) and ``DoubleMLDIDCS`` (repeated cross-sections)
from the ``DoubleML`` Python package on data simulated here, with the sample
split fixed by a ``fold`` column, and writes the data and the estimates.

Learners are a linear regression and an unpenalised logit, so that the
reference does not depend on a random forest's random stream or on the
scikit-learn version that grew it.

    python tests/reference_parity/_fixtures/_generate_dml_did_doubleml.py
"""

from __future__ import annotations

import json
from pathlib import Path

import doubleml
import numpy as np
import pandas as pd
import sklearn
from doubleml import DoubleMLDID, DoubleMLDIDCS, DoubleMLDIDData
from sklearn.linear_model import LinearRegression, LogisticRegression

HERE = Path(__file__).parent
X = ["x1", "x2", "x3", "x4"]


def main() -> None:
    rng = np.random.default_rng(20261007)
    n = 600
    x = np.round(rng.normal(size=(n, 4)), 3)
    d = rng.binomial(1, 1 / (1 + np.exp(-(0.6 * x[:, 0] - 0.4 * x[:, 1])))).astype(int)
    t = rng.binomial(1, 0.5, n).astype(int)
    trend = 1 + x[:, 0] ** 2 + 0.5 * x[:, 1]
    dy = np.round(trend + 2.0 * d + rng.normal(size=n), 3)
    y = np.round(x[:, 0] + 0.5 * d + t * (trend + 2.0 * d) + rng.normal(size=n), 3)
    fold = rng.permutation(np.arange(n) % 5)
    df = pd.DataFrame(x, columns=X).assign(d=d, t=t, dy=dy, y=y, fold=fold)
    df.to_csv(HERE / "dml_did_doubleml.csv", index=False)
    smpls = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(5)]

    def learners(score):
        g = LinearRegression()
        m = LogisticRegression(penalty=None, tol=1e-12, max_iter=5000)
        return g, (m if score == "observational" else None)

    out = {
        "meta": {
            "doubleml": doubleml.__version__,
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "panel": [],
        "rcs": [],
    }
    for score in ("observational", "experimental"):
        for norm in (True, False):
            g, m = learners(score)
            obj = DoubleMLDID(
                DoubleMLDIDData(df, "dy", "d", x_cols=X),
                g,
                m,
                n_folds=5,
                score=score,
                in_sample_normalization=norm,
                draw_sample_splitting=False,
            )
            obj.set_sample_splitting(smpls)
            obj.fit()
            out["panel"].append(
                dict(
                    score=score,
                    in_sample_normalization=norm,
                    estimate=float(obj.coef[0]),
                    se=float(obj.se[0]),
                )
            )
            g, m = learners(score)
            obj = DoubleMLDIDCS(
                DoubleMLDIDData(df, "y", "d", x_cols=X, t_col="t"),
                g,
                m,
                n_folds=5,
                score=score,
                in_sample_normalization=norm,
                draw_sample_splitting=False,
            )
            obj.set_sample_splitting(smpls)
            obj.fit()
            out["rcs"].append(
                dict(
                    score=score,
                    in_sample_normalization=norm,
                    estimate=float(obj.coef[0]),
                    se=float(obj.se[0]),
                )
            )
    (HERE / "dml_did_doubleml.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
