"""``sp.pcmci`` against ``tigramite`` (PCMCI with ``ParCorr``).

Reference: tigramite 5.2.10.1 (GPL-3, run as a black box),
``PCMCI(...).run_pc_stable`` and ``run_pcmci(tau_min=1)`` with
``ParCorr(significance='analytic')``, on 24 simulated vector
autoregressions (``_fixtures/_generate_pcmci_data.py``; output in
``_fixtures/pcmci_tigramite.json`` from ``_generate_pcmci_tigramite.py``).

``ParCorr`` refers the partial correlation to a t distribution with
``n - dim(Z) - 2`` degrees of freedom; ``sp.pcmci`` defaults to the Fisher
z test (:func:`sp.partial_corr_pvalue`). The t test is passed as
``ci_test=`` here so both sides run the same test. Compared:

* the MCI p-values and partial correlations of every lagged link:
  1e-10 absolute (observed 1e-15). Each is conditional on the parents the
  PC1 stage selected for both ends of the link, so agreement here means
  the parent sets agree too;
* the link decisions at the PC level: exact.

Through 1.43.1 ``sp.pcmci`` conditioned each PC-stage test on all other
candidates at once, dropped shifted parents beyond ``tau_max`` in the MCI
stage and used ``T - tau_max`` time points. On these 24 series its
p-values differed from the reference by up to 0.93 and 86 of the 1,188
link decisions were different.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "pcmci_tigramite.json").read_text(encoding="utf-8"))
_spec = importlib.util.spec_from_file_location(
    "_generate_pcmci_data", _FIX / "_generate_pcmci_data.py"
)
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)
SEEDS = sorted(REF["cases"], key=int)


def _parcorr_t(x, y, Z):
    """Partial correlation with the t reference distribution of ParCorr."""
    n = len(x)
    if Z is None:
        k, rx, ry = 0, x - x.mean(), y - y.mean()
    else:
        A = np.column_stack([np.ones(n), Z])
        k = A.shape[1] - 1
        rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
        ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    r = float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))
    df = n - k - 2
    return float(2 * stats.t.sf(abs(r) * np.sqrt(df / (1 - r * r)), df))


def _fit(seed, **kw):
    X, tau_max, pc_alpha = _gen.make(int(seed))
    ref = REF["cases"][seed]
    assert list(X.shape) == ref["shape"]
    assert float(np.abs(X).sum()) == pytest.approx(ref["checksum"], rel=1e-12)
    df = pd.DataFrame(X, columns=[f"v{i}" for i in range(X.shape[1])])
    return sp.pcmci(df, tau_max=tau_max, pc_alpha=pc_alpha, **kw), ref, X


def test_fixture_is_tigramite_output():
    assert REF["versions"]["tigramite"] == "5.2.10.1"
    assert len(SEEDS) == _gen.N_CASES == 24
    # both significant and null links are present
    p = np.concatenate(
        [np.asarray(c["p_matrix"])[:, :, 1:].ravel() for c in REF["cases"].values()]
    )
    assert (p < 0.01).sum() > 100 and (p > 0.2).sum() > 300


@pytest.mark.parametrize("seed", SEEDS)
def test_mci_p_values_and_partial_correlations_are_tigramites(seed):
    res, ref, X = _fit(seed, ci_test=_parcorr_t)
    # tigramite stores [source, target, lag]; sp.pcmci [lag, source, target]
    want_p = np.asarray(ref["p_matrix"]).transpose(2, 0, 1)
    want_v = np.asarray(ref["val_matrix"]).transpose(2, 0, 1)
    np.testing.assert_allclose(res.p_matrix[1:], want_p[1:], rtol=0, atol=1e-10)
    np.testing.assert_allclose(res.val_matrix[1:], want_v[1:], rtol=0, atol=1e-10)
    assert res.n_effective == X.shape[0] - 2 * ref["tau_max"]


@pytest.mark.parametrize("seed", SEEDS)
def test_links_are_tigramites_at_the_pc_level(seed):
    res, ref, _ = _fit(seed, ci_test=_parcorr_t)
    want = np.asarray(ref["p_matrix"]).transpose(2, 0, 1) < ref["pc_alpha"]
    assert np.array_equal(res.adjacency[1:], want[1:])
    assert not res.adjacency[0].any()
    links = res.discovered_links()
    assert len(links) == int(want[1:].sum())


@pytest.mark.parametrize("seed", ["0", "5", "13"])
def test_default_test_is_the_fisher_z_test(seed):
    res, _, _ = _fit(seed)
    explicit, _, _ = _fit(seed, ci_test=sp.partial_corr_pvalue)
    assert np.array_equal(res.p_matrix, explicit.p_matrix)
    assert np.array_equal(res.val_matrix, explicit.val_matrix)


def test_series_shorter_than_twice_the_lag_is_refused():
    X, _, _ = _gen.make(0)
    df = pd.DataFrame(X[:8], columns=[f"v{i}" for i in range(X.shape[1])])
    with pytest.raises(DataInsufficient, match="tau_max"):
        sp.pcmci(df, tau_max=3)
