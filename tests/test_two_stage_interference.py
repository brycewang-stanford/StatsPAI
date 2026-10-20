"""sp.two_stage_interference: Hudgens-Halloran two-stage randomized designs.

Known truth by exact enumeration. A finite population of six groups has
potential outcomes that depend on the unit's own treatment, the share of
its group treated and the treatment of one specific neighbour (so even
stratified interference fails). Every two-stage assignment of the design
is enumerated with its probability, which gives

* the estimands ``Ybar(z; a)`` exactly (Hudgens and Halloran's averages
  over the within-group assignment distribution);
* unbiasedness of every effect estimate (to 1e-12);
* the exact excess of the expected variance estimate over the
  randomization variance, ``S^2 / N`` with ``S^2`` the between-group
  variance of the group-level effect over all groups (to 1e-12), and
  equality when that effect is constant across groups.

The enumeration uses a vectorised transcription of the estimators; that
transcription is checked against the public function on random draws of
the same design (1e-12).
"""

from itertools import combinations, permutations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    DataInsufficient,
    MethodIncompatibility,
)

SIZES = np.array([2, 4, 2, 4, 2, 4])
LEVELS = (0.0, 0.5, 1.0)  # two groups each


def _outcome(rng, constant=False):
    """y[i](Z) for group i and assignment vector Z (0/1 of length n_i).

    ``constant=True``: unit-specific baselines, but the outcome moves with
    the own treatment and the group's treated share by the same amounts
    in every group, so every group-level effect is the same.
    """
    par = []
    for n in SIZES:
        base = rng.normal(size=n)
        tau = rng.normal(1.5, 1.0, size=n)
        gam = rng.normal(0.8, 0.5, size=n)
        eta = rng.normal(0.0, 0.7, size=n)
        par.append((base, tau, gam, eta))

    def y(i, z):
        base, tau, gam, eta = par[i]
        n = len(z)
        if constant:
            return base + 1.5 * z + 0.8 * z.mean()
        others = (z.sum() - z) / max(n - 1, 1)
        nxt = z[(np.arange(n) + 1) % n]  # identity-dependent spillover
        return base + tau * z + gam * others + eta * nxt + 0.3 * z * others

    return y


def _within(n, a):
    m = int(round(a * n))
    for idx in combinations(range(n), m):
        z = np.zeros(n)
        z[list(idx)] = 1
        yield z


def _first_stage():
    seen = set()
    for perm in permutations(range(6)):
        lab = [None] * 6
        for k, g in enumerate(perm):
            lab[g] = LEVELS[k // 2]
        key = tuple(lab)
        if key not in seen:
            seen.add(key)
            yield key


def _group_estimand(y, i, a):
    """Ybar_i(1; a), Ybar_i(0; a), Ybar_i(a)."""
    ys = [(z, y(i, z)) for z in _within(SIZES[i], a)]
    out = []
    for zz in (1, 0):
        num = np.zeros(SIZES[i])
        cnt = np.zeros(SIZES[i])
        for z, v in ys:
            num += v * (z == zz)
            cnt += z == zz
        with np.errstate(invalid="ignore"):
            out.append(np.mean(num / cnt) if cnt.all() else np.nan)
    out.append(np.mean([v.mean() for _, v in ys]))
    return out


def _estimates(lab, zs, y, w):
    """Vectorised transcription: {effect key: (estimate, variance)}."""
    g = {a: [] for a in LEVELS}
    for i, a in enumerate(lab):
        z = zs[i]
        v = y(i, z) * w[i]
        g[a].append(
            (
                v[z == 1].mean() if (z == 1).any() else np.nan,
                v[z == 0].mean() if (z == 0).any() else np.nan,
                v.mean(),
            )
        )
    arr = {a: np.array(g[a]) for a in LEVELS}

    def mv(x):
        return x.mean(), x.var(ddof=1) / len(x)

    out = {}
    d = arr[0.5][:, 0] - arr[0.5][:, 1]
    out[("direct", 0.5)] = mv(d)
    for a in (0.5, 1.0):
        pairs = {"total": (0, 1), "overall": (2, 2)}
        if a < 1:
            pairs["indirect"] = (1, 1)
        for eff, (ca, cr) in pairs.items():
            ma, va = mv(arr[a][:, ca])
            mr, vr = mv(arr[0.0][:, cr])
            out[(eff, a)] = (ma - mr, va + vr)
    return out


def _enumerate(y, w):
    """E[estimate], Var[estimate], E[variance estimate] for every effect."""
    acc = {}
    labs = list(_first_stage())
    for lab in labs:
        choices = [list(_within(SIZES[i], a)) for i, a in enumerate(lab)]
        grid = np.array(np.meshgrid(*[range(len(c)) for c in choices])).reshape(6, -1).T
        p = 1.0 / (len(labs) * len(grid))
        for pick in grid:
            zs = [choices[i][k] for i, k in enumerate(pick)]
            for key, (est, var) in _estimates(lab, zs, y, w).items():
                s = acc.setdefault(key, np.zeros(3))
                s += p * np.array([est, est**2, var])
    return {k: (v[0], v[1] - v[0] ** 2, v[2]) for k, v in acc.items()}


def _truth(y, w):
    """Population effects and the between-group variance S^2 of each."""
    G = {a: np.array([_group_estimand(y, i, a) for i in range(6)]) for a in LEVELS}
    G = {a: v * w[:, None] for a, v in G.items()}
    group_c = {("direct", 0.5): G[0.5][:, 0] - G[0.5][:, 1]}
    for a in (0.5, 1.0):
        group_c[("total", a)] = G[a][:, 0] - G[0.0][:, 1]
        group_c[("overall", a)] = G[a][:, 2] - G[0.0][:, 2]
        if a < 1:
            group_c[("indirect", a)] = G[a][:, 1] - G[0.0][:, 1]
    return {k: (c.mean(), c.var(ddof=1)) for k, c in group_c.items()}


@pytest.mark.parametrize("weighting", ["group", "individual"])
def test_unbiased_and_exact_variance_excess(weighting):
    y = _outcome(np.random.default_rng(7))
    w = SIZES / SIZES.mean() if weighting == "individual" else np.ones(6)
    mom = _enumerate(y, w)
    truth = _truth(y, w)
    assert set(mom) == set(truth)
    for key, (tau, s2) in truth.items():
        mean, var, evhat = mom[key]
        assert mean == pytest.approx(tau, abs=1e-12), key
        # E[Vhat] - Var = S^2 / N exactly
        assert evhat - var == pytest.approx(s2 / 6, abs=1e-12), key
        assert s2 > 1e-3  # heterogeneous effects: strictly conservative


def test_variance_estimate_is_unbiased_for_constant_group_effects():
    y = _outcome(np.random.default_rng(3), constant=True)
    w = np.ones(6)
    mom = _enumerate(y, w)
    for key, (tau, s2) in _truth(y, w).items():
        assert s2 == pytest.approx(0, abs=1e-24), key
        mean, var, evhat = mom[key]
        assert evhat == pytest.approx(var, rel=1e-10), key


def _draw(rng, y, lab=None):
    if lab is None:
        lab = list(rng.permutation(np.repeat(LEVELS, 2)))
    rows, zs = [], []
    for i, a in enumerate(lab):
        n = SIZES[i]
        z = np.zeros(n)
        z[rng.choice(n, int(round(a * n)), replace=False)] = 1
        zs.append(z)
        rows.append(pd.DataFrame({"g": i, "a": a, "d": z.astype(int), "y": y(i, z)}))
    return pd.concat(rows, ignore_index=True), lab, zs


@pytest.mark.parametrize("weighting", ["group", "individual"])
def test_function_equals_the_transcription(weighting):
    rng = np.random.default_rng(11)
    y = _outcome(np.random.default_rng(7))
    w = SIZES / SIZES.mean() if weighting == "individual" else np.ones(6)
    for _ in range(15):
        df, lab, zs = _draw(rng, y)
        r = sp.two_stage_interference(df, "y", "d", "g", "a", weighting=weighting)
        want = _estimates(lab, zs, y, w)
        got = {
            (e, a): (est, se)
            for e, a, est, se in r.detail[
                ["effect", "saturation", "estimate", "se"]
            ].itertuples(index=False)
        }
        assert set(got) == set(want)
        for k, (est, var) in want.items():
            assert got[k][0] == pytest.approx(est, abs=1e-12)
            assert got[k][1] == pytest.approx(np.sqrt(var), abs=1e-12)
        # headline: total effect of the highest saturation against 0
        assert r.estimate == pytest.approx(want[("total", 1.0)][0], abs=1e-12)


def test_decomposition_identity_and_reference():
    rng = np.random.default_rng(5)
    rows = []
    for g, a in enumerate(np.repeat([0.0, 0.25, 0.75], 6)):
        d = np.zeros(8, int)
        d[rng.choice(8, int(a * 8), replace=False)] = 1
        rows.append(
            pd.DataFrame(
                {"g": g, "a": a, "d": d, "y": 1 + 2 * d + a + rng.normal(size=8)}
            )
        )
    df = pd.concat(rows, ignore_index=True)
    r = sp.two_stage_interference(df, "y", "d", "g", "a", reference=0.25)
    t = r.detail.set_index(["effect", "saturation"])["estimate"]
    # direct(a) + indirect(a) = total(a) for the estimates
    assert t[("direct", 0.75)] + t[("indirect", 0.75)] == pytest.approx(
        t[("total", 0.75)], abs=1e-12
    )
    assert set(r.detail["reference"].dropna()) == {0.25}
    # relabelling groups and shuffling rows changes nothing
    df2 = df.sample(frac=1, random_state=1).assign(g=lambda f: f["g"] * 7 + 100)
    r2 = sp.two_stage_interference(df2, "y", "d", "g", "a", reference=0.25)
    pd.testing.assert_frame_equal(r.detail, r2.detail)


def test_refusals():
    rng = np.random.default_rng(0)
    df, _, _ = _draw(rng, _outcome(rng))
    f = sp.two_stage_interference
    with pytest.raises(MethodIncompatibility, match="varies within"):
        f(df.assign(a=np.where(df.index == 0, 0.3, df["a"])), "y", "d", "g", "a")
    with pytest.raises(MethodIncompatibility, match="saturation 0 but treated"):
        f(df.assign(d=np.where(df["a"] == 0, 1, df["d"])), "y", "d", "g", "a")
    with pytest.raises(MethodIncompatibility, match="0/1"):
        f(df.assign(d=df["d"] * 2), "y", "d", "g", "a")
    with pytest.raises(MethodIncompatibility, match=r"\[0, 1\]"):
        f(df.assign(a=df["a"] * 4), "y", "d", "g", "a")
    with pytest.raises(MethodIncompatibility, match="missing"):
        f(df.assign(y=np.where(df.index == 3, np.nan, df["y"])), "y", "d", "g", "a")
    with pytest.raises(MethodIncompatibility, match="reference"):
        f(df, "y", "d", "g", "a", reference=0.3)
    with pytest.raises(DataInsufficient, match="fewer than two"):
        f(df[df["g"] != df.loc[df["a"] == 1, "g"].iloc[0]], "y", "d", "g", "a")
    with pytest.raises(DataInsufficient, match="one saturation"):
        f(df[df["a"] == 0.5], "y", "d", "g", "a")


def test_count_mismatch_is_reported():
    rng = np.random.default_rng(2)
    df, _, _ = _draw(rng, _outcome(rng))
    # a size-4 group at saturation 0.5 with three treated
    g = df[(df["a"] == 0.5)].groupby("g").size().idxmax()
    rows = df.index[(df["g"] == g) & (df["d"] == 0)][:1]
    bad = df.copy()
    bad.loc[rows, "d"] = 1
    with pytest.warns(AssumptionWarning, match="treated count"):
        r = sp.two_stage_interference(bad, "y", "d", "g", "a")
    assert r.model_info["count_mismatch"] == [
        {"group": g, "n": 4, "treated": 3, "saturation": 0.5}
    ]
