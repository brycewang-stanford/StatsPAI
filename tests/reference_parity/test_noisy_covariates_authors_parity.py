"""sp.noisy_covariates against the authors' code on their ADH application.

Agarwal and Singh's replication package (Zenodo record 19866751,
CC-BY-4.0; ``src/wt/stage2.R::rrr_iv``) estimates the Autor-Dorn-Hanson
(2013) effect of import competition on manufacturing employment as a
weighted partially linear IV coefficient with PCA cleaning (k = 5, five
folds, the 31 covariates of their augmented specification). Here that is
``estimand="partially_linear_iv"`` with ``weighted_fit=True`` and the
authors' fold assignment passed through ``folds=``.

Three kinds of evidence:

1. Published numbers: the package's ``results/census_*_pcr_table3.csv``
   (the clean-data point estimate and SE behind Figure 5 and Figure 6).
2. Same input, both implementations (T2): covariate matrices corrupted
   by Gaussian noise with missing cells and by the calibrated Laplace
   (differential privacy) mechanism, drawn by
   ``_fixtures/_generate_noisy_covariates_adh_data.py`` and run through
   the authors' R code by ``_fixtures/_generate_noisy_covariates_adh_R.R``.
3. The published noisy and privatised rows (S, stochastic screen): those
   rows come from one R noise draw each, which is not published as data.
   We draw the same mechanism 20 times and check that each published
   value lies within three Monte Carlo SDs of our mean (criterion fixed
   before running; at 40 draws all ten published rows have |z| < 2).

Tolerances: on standardised covariates the two implementations agree to
1e-14. The unstandardised covariates mix dollar amounts in the
thousands with dummies; the same R code on R 4.4.2 / Linux (the
published CSV) and R 4.5.2 / macOS differs by 1.8e-11 there, and our
gaps are 5e-12 to 3e-10, so those rows use rtol 1e-8 (still far inside
the 1e-6 T2 budget).

The data are the replication zip (9.6 MB), fetched on demand and pinned
by SHA-256; the test skips unless it is cached or
``STATSPAI_FETCH_DATASETS`` is set.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import statspai as sp
from statspai.datasets import _fetch

HERE = Path(__file__).parent / "_fixtures"
GOLDEN = json.loads((HERE / "noisy_covariates_adh_R.json").read_text(encoding="utf-8"))

AGARWAL_SINGH_PACKAGE = _fetch.RemoteFile(
    filename="agarwal_singh_2026_replication_package.zip",
    url="https://zenodo.org/records/19866751/files/replication_package.zip"
    "?download=1",
    sha256="92d2df42204f38e9e70d077bfa01b514ab102f9c4614b12a1761fe4a0e633348",
    size=9615512,
    licence="CC-BY-4.0 (Zenodo record 19866751); the ADH workfile inside is "
    "(c) 2013 American Economic Association, CC-BY-4.0; fetched, not "
    "redistributed",
    origin="Agarwal and Singh (2026) Econometrica replication package, "
    "doi:10.5281/zenodo.19866751",
)

# results/census_unscaled_pcr_table3.csv and census_scaled_pcr_table3.csv
PUBLISHED_CLEAN = {
    False: (-0.617993844006585, 0.0633122734665954),
    True: (-0.489855109744297, 0.0719262698272391),
}
# results/census_unscaled_semi_noise.csv and ..._private_calibration.csv
PUBLISHED_DRAWS = {
    ("gauss", 0.2): -0.630781152266829,
    ("gauss", 1.0): -0.622568745051227,
    ("dp", 100.0): -0.615854456036276,
    ("dp", 1.0): -0.676546921281723,
    ("dp", 0.01): -0.630946828712459,
}


def _design_module():
    spec = importlib.util.spec_from_file_location(
        "_noisy_adh_design", HERE / "_generate_noisy_covariates_adh_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def adh():
    spec = AGARWAL_SINGH_PACKAGE
    if not (_fetch.is_cached(spec) or _fetch.fetch_allowed_in_tests()):
        pytest.skip(
            "Agarwal-Singh replication zip not cached; set "
            "STATSPAI_FETCH_DATASETS=1 to download it (9.6 MB)."
        )
    path = _fetch.fetch(spec)
    g = _design_module()
    return g, g.load_adh(path)


def _fit(g, df, X):
    data = df.copy()
    data[g.COVARIATES] = X
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # dummies make the rank warning noisy
        return sp.noisy_covariates(
            data,
            g.Y,
            g.D,
            g.COVARIATES,
            estimand="partially_linear_iv",
            instrument=g.U,
            rank=5,
            weights=g.W,
            weighted_fit=True,
            folds=np.asarray(GOLDEN["fold"]),
        )


@pytest.mark.parametrize("scaled", [False, True])
def test_published_clean_estimate(adh, scaled):
    g, df = adh
    res = _fit(g, df, g.covariate_matrix(df, scaled))
    est, se = PUBLISHED_CLEAN[scaled]
    rtol = 1e-12 if scaled else 1e-8
    np.testing.assert_allclose(res.estimate, est, rtol=rtol)
    np.testing.assert_allclose(res.se, se, rtol=rtol)


@pytest.mark.parametrize("name", list(GOLDEN["results"]))
def test_same_input_as_authors_code(adh, name):
    g, df = adh
    res = _fit(g, df, g.corrupted(df, name))
    ref = GOLDEN["results"][name]
    rtol = 1e-12 if g.CONFIGS[name][0] else 1e-8
    np.testing.assert_allclose(res.estimate, float(ref["estimate"]), rtol=rtol)
    np.testing.assert_allclose(res.se, float(ref["se"]), rtol=rtol)


@pytest.mark.parametrize("kind,level", list(PUBLISHED_DRAWS))
def test_published_noisy_rows_screen(adh, kind, level):
    g, df = adh
    X0 = g.covariate_matrix(df, False)
    est = []
    for b in range(20):
        rng = np.random.default_rng(1000 + b)
        if kind == "gauss":
            noise = rng.normal(size=X0.shape) * np.sqrt(level * X0.var(axis=0, ddof=1))
        else:
            noise = rng.laplace(size=X0.shape) * g.dp_scale(df, X0, level)
        est.append(_fit(g, df, X0 + noise).estimate)
    est = np.asarray(est)
    published = PUBLISHED_DRAWS[(kind, level)]
    assert abs(published - est.mean()) <= 3 * est.std(ddof=1)
