"""Validation: ``sp.spillover_did`` (Butts spillover-ring DiD).

No reference implementation exists — nothing on CRAN or GitHub implements
this estimator — so this file establishes correctness the other way
available: recover a design whose direct effect and per-ring spillovers are
known by construction, and show that the estimator this replaces is biased
on the same data.

The DGP plants a treated cluster in one corner of a square, a direct effect
of 2.0, a spillover of 1.0 within distance 2 of any treated unit, 0.4
between 2 and 4, and nothing beyond. Every number the estimator reports is
checked against those.

The comparison that matters is the last test. ``sp.spatial_did`` adds a
spatial lag of treatment to a two-way fixed-effects regression, which is
the standard fix; on this design its direct effect is pulled toward zero
because the units nearest the treated — the ones the spillover reaches —
are serving as its controls. The ring estimator measures against clean
controls only and is not.

References
----------
Butts, K. (2021). "Difference-in-Differences Estimation with Spatial
Spillovers." *arXiv preprint* arXiv:2105.03737. DOI
10.48550/arXiv.2105.03737. [@butts2021difference]
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

DIRECT = 2.0
RING1 = 1.0
RING2 = 0.4
EDGES = (0.0, 2.0, 4.0)


def make_panel(seed: int, n: int = 600, noise: float = 0.3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 20, n)
    y2 = rng.uniform(0, 20, n)
    treated = (x < 5) & (y2 < 5)
    tx, ty = x[treated], y2[treated]
    dist = np.sqrt((x[:, None] - tx[None, :]) ** 2 + (y2[:, None] - ty[None, :]) ** 2)
    nearest = dist.min(axis=1)

    rows = []
    for i in range(n):
        fe = rng.normal(0, 1)
        if treated[i]:
            effect = DIRECT
        elif nearest[i] <= 2:
            effect = RING1
        elif nearest[i] <= 4:
            effect = RING2
        else:
            effect = 0.0
        for t in (1, 2):
            val = fe + 0.3 * t + (effect if t == 2 else 0.0) + rng.normal(0, noise)
            rows.append((i, t, 2 if treated[i] else 0, x[i], y2[i], val))
    return pd.DataFrame(rows, columns=["i", "t", "g", "x", "y2", "y"])


def _fit(df, ring_edges=EDGES, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.spillover_did(
            df,
            y="y",
            unit="i",
            time="t",
            cohort="g",
            coords=["x", "y2"],
            ring_edges=ring_edges,
            **kwargs,
        )


@pytest.fixture(scope="module")
def fits():
    """Ten seeds. One draw cannot separate a small bias from noise."""
    return [_fit(make_panel(s)) for s in range(10)]


def test_direct_effect_is_recovered(fits):
    est = np.array([f.direct for f in fits])
    assert abs(est.mean() - DIRECT) < 0.1, est.mean()
    # And it is not merely unbiased on average: no single draw is far off.
    assert np.max(np.abs(est - DIRECT)) < 0.3, est


def test_ring_effects_are_recovered(fits):
    for ring, truth in ((1, RING1), (2, RING2)):
        est = np.array(
            [
                float(f.rings.loc[f.rings["ring"] == ring, "estimate"].iloc[0])
                for f in fits
            ]
        )
        assert abs(est.mean() - truth) < 0.12, (ring, est.mean(), truth)


def test_spillover_decays_with_distance(fits):
    """The substantive output: the near ring is affected more than the far."""
    near = np.array(
        [float(f.rings.loc[f.rings["ring"] == 1, "estimate"].iloc[0]) for f in fits]
    )
    far = np.array(
        [float(f.rings.loc[f.rings["ring"] == 2, "estimate"].iloc[0]) for f in fits]
    )
    assert (near > far).mean() >= 0.9, near - far


def test_standard_errors_have_the_right_coverage(fits):
    """A variance estimator has to be checked, not assumed."""
    covered = [f.ci[0] <= DIRECT <= f.ci[1] for f in fits]
    assert sum(covered) >= 8, covered


def test_clean_controls_are_beyond_every_ring(fits):
    fit = fits[0]
    assert fit.n_clean_controls > 0
    assert fit.n_clean_controls < fit.n_units - fit.diagnostics["n_treated"]
    total_in_rings = int(fit.rings["n_units"].sum())
    assert (
        fit.n_clean_controls + total_in_rings
        == fit.n_units - fit.diagnostics["n_treated"]
    )


def test_twfe_spatial_lag_is_biased_where_this_is_not():
    """The reason this estimator exists.

    sp.spatial_did regresses on treatment plus its spatial lag with unit and
    period fixed effects. On this design the units it uses as controls are
    partly the ones the spillover reaches, so the direct effect is pulled
    toward zero. The ring estimator, measuring against clean controls, is
    not.
    """
    df = make_panel(0)
    ring = _fit(df)

    coords = df.drop_duplicates(subset=["i"]).sort_values("i")[["x", "y2"]].to_numpy()
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
    W = (d > 0) & (d <= 2.0)
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1)

    work = df.copy()
    work["treated_now"] = ((work["g"] == 2) & (work["t"] >= 2)).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        twfe = sp.spatial_did(
            work,
            y="y",
            treat="treated_now",
            unit="i",
            time="t",
            W=W,
        )
    twfe_direct = float(twfe.params["treated_now"])

    assert abs(ring.direct - DIRECT) < abs(twfe_direct - DIRECT), (
        ring.direct,
        twfe_direct,
    )


def test_no_clean_controls_fails_loudly():
    from statspai.exceptions import DataInsufficient

    df = make_panel(0)
    with pytest.raises(DataInsufficient, match="clean controls"):
        _fit(df, ring_edges=(0.0, 1000.0))


def test_requires_exactly_one_geometry_argument():
    from statspai.exceptions import MethodIncompatibility

    df = make_panel(0)
    with pytest.raises(MethodIncompatibility, match="coords"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp.spillover_did(df, y="y", unit="i", time="t", cohort="g")


def test_distance_matrix_matches_coordinates():
    """Passing the distances directly must give the same answer as letting
    the function build them, or one of the two paths is wrong."""
    df = make_panel(3)
    first = df.drop_duplicates(subset=["i"]).sort_values("i")
    coords = first[["x", "y2"]].to_numpy()
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_matrix = sp.spillover_did(
            df,
            y="y",
            unit="i",
            time="t",
            cohort="g",
            distances=d,
            ring_edges=EDGES,
        )
    assert by_matrix.direct == pytest.approx(_fit(df).direct, abs=1e-12)


def test_summary_flags_a_thin_control_group():
    """With almost everything inside a ring the standard errors are
    optimistic, and the summary has to say so rather than look confident."""
    df = make_panel(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sp.spillover_did(
            df,
            y="y",
            unit="i",
            time="t",
            cohort="g",
            coords=["x", "y2"],
            ring_edges=(0.0, 2.0, 18.0),
        )
    assert any("clean control" in str(w.message) for w in caught)
    assert "WARNING" in fit.summary()
