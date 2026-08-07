"""`sp.genmatch` must not sell its standard error as something it isn't.

Two docstrings — the module header and `GenMatchResult` — described
`att_se` as a "bootstrap SE". The code computes
`sd(Y_t - Y_c) / sqrt(n_pairs)`, the matched-pair standard error, and there
is no bootstrap anywhere in the file. A reader would reasonably assume the
reported interval accounted for the matching and for the fitted covariate
weights; it accounts for neither.

It matters more here than the wording alone suggests. Genetic matching
matches **with replacement** — on the fixture below one control serves a
dozen treated units — so the pairs the formula treats as independent are
not. The identical formula, measured on `sp.match` over 36 designs x 1000
replications (`benchmarks/matching_se_coverage.py`), runs 0.56-0.91x the
true sampling SD and never reaches nominal coverage.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _dgp(seed: int = 0, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * x1))))
    y = 2 * d + 0.7 * x1 + rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "d": d, "y": y})


def _fit(df):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sp.genmatch(df, treat="d", y="y", covariates=["x1", "x2"])
    return res, [str(w.message) for w in caught]


class TestSEDisclosure:
    def test_warns_that_the_se_is_not_a_bootstrap(self):
        _, msgs = _fit(_dgp())
        hits = [m for m in msgs if "matched-pair standard error" in m]
        assert hits, "the SE's nature must be disclosed at call time"
        assert "not a bootstrap" in hits[0]

    def test_warning_quotes_the_measured_coverage(self):
        """Naming the number is what makes the warning actionable."""
        _, msgs = _fit(_dgp())
        hit = next(m for m in msgs if "matched-pair standard error" in m)
        assert "0.71-0.92" in hit
        assert "0.56-0.91" in hit

    def test_warning_reports_the_actual_reuse_count(self):
        """A generic caveat would not tell the user how bad their case is."""
        res, msgs = _fit(_dgp())
        hit = next(m for m in msgs if "matched-pair standard error" in m)
        matches = np.asarray(res.matches).ravel()
        n_reused = len(matches) - len(np.unique(matches))
        assert f"{n_reused} of {len(matches)} matches reuse a control" in hit

    def test_the_original_bootstrap_claims_are_gone(self):
        """Pin the exact wording that regressed, not the word 'bootstrap'.

        The docstrings legitimately mention bootstrapping now — to say the
        SE is not one, and to suggest bootstrapping the pipeline instead —
        so a blanket keyword ban would fail on its own fix.
        """
        import importlib

        # `statspai.matching.genmatch` resolves to the *function*: the
        # package __init__ re-exports it under the module's own name.
        gm = importlib.import_module("statspai.matching.genmatch")
        blob = "\n".join(
            [
                gm.__doc__ or "",
                sp.genmatch.__doc__ or "",
                gm.GenMatchResult.__doc__ or "",
            ]
        )
        for claim in (
            "the ATT estimate + bootstrap SE",
            "Holds the ATT estimate and bootstrap SE",
        ):
            assert claim not in blob, f"stale claim returned: {claim!r}"

    def test_the_docstring_says_what_it_actually_is(self):
        import importlib

        gm = importlib.import_module("statspai.matching.genmatch")
        blob = (gm.GenMatchResult.__doc__ or "").lower()
        assert "matched-pair" in blob
        assert "not* a bootstrap" in blob or "not a bootstrap" in blob


class TestTheSEIsWhatItSaysItIs:
    def test_att_se_equals_the_matched_pair_formula(self):
        """Pin the actual estimator, so the docs can be checked against it."""
        df = _dgp()
        res, _ = _fit(df)
        y = df["y"].to_numpy(dtype=float)
        t = np.where(df["d"].to_numpy() == 1)[0]
        c = np.where(df["d"].to_numpy() == 0)[0]
        diffs = y[t] - y[c[np.asarray(res.matches)]].mean(axis=1)
        expected = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
        assert res.att_se == pytest.approx(expected, rel=1e-12)

    def test_genetic_matching_reuses_controls(self):
        """The premise of the warning; if this ever changed, revisit it."""
        res, _ = _fit(_dgp())
        matches = np.asarray(res.matches).ravel()
        assert len(np.unique(matches)) < len(matches)
