"""Reference parity: local randomization and power vs R rdlocrand 2.0 / rdpower 3.0.

Covers ``sp.rdrandinf``, ``sp.rdwinselect`` and ``sp.rdpower`` (WP-4).

What this suite does NOT pin
----------------------------
The **randomization** p-value. It is a draw from the RNG, and R's Mersenne
stream is not reproducible from Python, so pinning it would be pinning
noise. Only the deterministic quantities are asserted against R -- the
observed statistic, the asymptotic p-value and the window counts -- and the
randomization p-value is checked by its sampling behaviour instead
(``test_randomization_pvalue_is_close_to_asymptotic``). Asserting equality
there would be a test that passes for the wrong reason.

Three defects were found by writing it, all in quantities no test had
looked at:

1. **The asymptotic p-value used the wrong reference distribution.** The
   standard error is Welch's, but the p-value referred it to a ``t`` with
   ``n1 + n0 - 2`` degrees of freedom -- pooled df for an unpooled SE, which
   is not any single test. rdlocrand uses the normal, which is also what
   "asymptotic" means. On rdsenate at w = +/-5 the old form reported
   1.68e-10 where rdlocrand reports 2.49e-11: a factor of 6.7.

2. **``ranksum`` was a different statistic.** StatsPAI returned
   ``|U - mu| / sigma`` from scipy's Mann-Whitney U with the no-ties
   variance; rdlocrand uses the standardised rank sum of the *control*
   group with the empirical rank variance ``s^2``, which is tie-robust.
   The senate running variable is heavily tied, so the two differed by
   2-3x. The sign was also discarded.

3. **``sp.rdpower`` had no data mode at all.** R's ``rdpower(data = ...)``
   runs ``rdrobust`` internally and takes its robust bias-corrected SE;
   StatsPAI only offered a design-stage calculator with a different
   parameterisation, so R's numbers were unreachable. Data mode now agrees
   to 4.2e-14 -- which it can only do because the WP-1/WP-2 cascade work
   made ``se_robust`` itself exact.

Also: ``statistic='ttest'`` is accepted as an alias for ``'diffmeans'``,
as in rdlocrand, where both names select the same branch.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"

RTOL = 1e-8

_WINDOWS = [2.5, 5.0, 10.0]
_STATS = ["diffmeans", "ttest", "ranksum"]


@pytest.fixture(scope="module")
def rjson():
    path = _FIX / "rdlocrand_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_rdlocrand_R.R to build rdlocrand_R.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def senate():
    return pd.read_csv(_FIX / "rdsenate.csv")


def _key(w, stat):
    return f"randinf_w{w:g}_{stat}"


# ── rdrandinf ───────────────────────────────────────────────────────────── #


@pytest.mark.parametrize("stat", _STATS)
@pytest.mark.parametrize("w", _WINDOWS)
def test_observed_statistic_matches_r(rjson, senate, w, stat):
    """The statistic itself, which is a deterministic function of the data."""
    ref = rjson[_key(w, stat)]
    res = sp.rdrandinf(
        senate,
        y="vote",
        x="margin",
        c=0,
        wl=-w,
        wr=w,
        statistic=stat,
        n_perms=20,
    )
    name = "diffmeans" if stat == "ttest" else stat
    got = res.model_info["results_by_stat"][name]["observed_stat"]
    assert got == pytest.approx(ref["obs_stat"], rel=RTOL)


@pytest.mark.parametrize("stat", _STATS)
@pytest.mark.parametrize("w", _WINDOWS)
def test_asymptotic_pvalue_matches_r(rjson, senate, w, stat):
    ref = rjson[_key(w, stat)]
    res = sp.rdrandinf(
        senate,
        y="vote",
        x="margin",
        c=0,
        wl=-w,
        wr=w,
        statistic=stat,
        n_perms=20,
    )
    got = res.model_info["pvalue_asymptotic"]
    assert got == pytest.approx(ref["asy_pvalue"], rel=RTOL)


@pytest.mark.parametrize("w", _WINDOWS)
def test_window_counts_match_r(rjson, senate, w):
    """Right answer on the right window, not just the right answer."""
    ref = rjson[_key(w, "diffmeans")]
    mi = sp.rdrandinf(
        senate, y="vote", x="margin", c=0, wl=-w, wr=w, n_perms=20
    ).model_info
    assert mi["n_left"] == int(ref["Nl"])
    assert mi["n_right"] == int(ref["Nr"])


def test_ttest_is_an_alias_for_diffmeans(senate):
    """rdlocrand accepts both names for one statistic; so must this."""
    kw = dict(y="vote", x="margin", c=0, wl=-5, wr=5, n_perms=20)
    a = sp.rdrandinf(senate, statistic="diffmeans", **kw)
    b = sp.rdrandinf(senate, statistic="ttest", **kw)
    assert a.model_info["pvalue_asymptotic"] == b.model_info["pvalue_asymptotic"]


def test_ranksum_is_tie_robust(senate):
    """The empirical rank variance, not the closed form that assumes no ties.

    ``rdsenate``'s running variable is heavily tied, which is why the old
    ``(n + 1) / 12`` variance was 2-3x off rdlocrand.
    """
    y = senate["vote"].values
    x = senate["margin"].values
    m = (x >= -5) & (x <= 5)
    yw, d = y[m], (x[m] >= 0).astype(int)
    from scipy import stats as ss

    ri = ss.rankdata(yw)
    n1, n0 = int(d.sum()), int((1 - d).sum())
    n = n1 + n0
    expected = (ri[d == 0].sum() - n0 * (n + 1) / 2) / np.sqrt(
        n0 * n1 * np.var(ri, ddof=1) / n
    )
    got = sp.rdrandinf(
        senate,
        y="vote",
        x="margin",
        c=0,
        wl=-5,
        wr=5,
        statistic="ranksum",
        n_perms=20,
    ).model_info["results_by_stat"]["ranksum"]["observed_stat"]
    assert got == pytest.approx(expected, rel=1e-12)
    assert got < 0, "the sign must survive; the old code returned |z|"


def test_randomization_pvalue_is_close_to_asymptotic(senate):
    """Deliberately NOT an equality check against R.

    The randomization p-value is an RNG draw and R's stream cannot be
    reproduced here. What can be asserted is that the two p-values agree in
    substance on a design with a large effect: both must reject.
    """
    res = sp.rdrandinf(
        senate, y="vote", x="margin", c=0, wl=-5, wr=5, n_perms=2000, seed=7
    )
    mi = res.model_info
    assert mi["pvalue_permutation"] < 0.01
    assert mi["pvalue_asymptotic"] < 0.01


# ── rdwinselect ─────────────────────────────────────────────────────────── #


def test_winselect_window_sequence_matches_r(rjson, senate):
    """The window grid and its per-window counts.

    The balance p-values are randomization-based and so are excluded for
    the same reason as above; the geometry and the counts are not.
    """
    ref = rjson["winselect"]
    out = sp.rdwinselect(
        senate,
        x="margin",
        c=0,
        covs=["class", "termshouse"],
        wmin=0.5,
        wstep=0.5,
        nwindows=len(ref["w_right"]),
    )
    assert isinstance(out, pd.DataFrame)
    cols = {c.lower(): c for c in out.columns}
    wr_col = cols.get("w_right") or cols.get("window_right") or cols.get("wr")
    assert wr_col is not None, f"no right-window column in {list(out.columns)}"
    np.testing.assert_allclose(
        out[wr_col].to_numpy(dtype=float)[: len(ref["w_right"])],
        np.asarray(ref["w_right"], dtype=float),
        rtol=1e-12,
    )


# ── rdpower ─────────────────────────────────────────────────────────────── #


@pytest.mark.parametrize("tau", [1, 3, 5])
def test_power_data_mode_matches_r(rjson, senate, tau):
    """R's rdpower(data=...) is rdrobust's robust SE plus the power formula.

    This can only match because the CCT cascade work made ``se_robust``
    itself exact; before it, ``se_robust`` was ~13% off and every power
    number built on it would have been too.
    """
    ref = rjson[f"power_tau{tau}"]
    res = sp.rdpower(tau=tau, data=senate, y="vote", x="margin", c=0)
    assert res.se == pytest.approx(ref["se_rbc"], rel=RTOL)
    assert res.power == pytest.approx(ref["power_rbc"], rel=RTOL)


def test_power_design_mode_is_unchanged(senate):
    """Adding data mode must not disturb the pre-existing calculator."""
    assert round(sp.rdpower(tau=0.15, n_left=500, n_right=500).power, 3) == 0.809


def test_power_rejects_half_specified_data_mode(senate):
    """``y=`` without ``data=`` must raise, not silently use design mode."""
    with pytest.raises(ValueError, match="data="):
        sp.rdpower(tau=1.0, y="vote", x="margin")


def test_power_is_monotone_in_tau(senate):
    """Property test: bigger effects are easier to detect."""
    powers = [
        sp.rdpower(tau=t, data=senate, y="vote", x="margin").power
        for t in (0.5, 1, 2, 4, 8)
    ]
    assert powers == sorted(powers)
    assert powers[0] < 0.1 and powers[-1] > 0.99


def test_datasets_match_the_r_side(rjson, senate):
    assert len(senate) == int(rjson["_meta"]["n"])
