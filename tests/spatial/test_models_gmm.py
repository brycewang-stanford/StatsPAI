"""GMM estimators — cross-validated against PySAL spreg on Columbus."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.spatial.models.gmm import sar_gmm, sarar_gmm, sem_gmm
from statspai.spatial.weights.core import W

FIXTURE = Path(__file__).parent / "fixtures" / "columbus_reference.json"


@pytest.fixture(scope="module")
def columbus():
    ref = json.loads(FIXTURE.read_text(encoding="utf-8"))
    neighbors = {int(k): v for k, v in ref["neighbors"].items()}
    w = W(neighbors)
    w.transform = "R"
    df = pd.DataFrame({"CRIME": ref["y"], "INC": ref["INC"], "HOVAL": ref["HOVAL"]})
    return w, df, ref


def test_sem_gmm_matches_spreg_GM_Error(columbus):
    w, df, ref = columbus
    res = sem_gmm(w, df, "CRIME ~ INC + HOVAL")
    expected = ref["gm_error"]["betas"]  # [const, INC, HOVAL, lam]
    np.testing.assert_allclose(res.params.values, expected, rtol=1e-4)


def test_sar_gmm_matches_spreg_GM_Lag(columbus):
    w, df, ref = columbus
    res = sar_gmm(w, df, "CRIME ~ INC + HOVAL", w_lags=1)
    expected = ref["gm_lag"]["betas_with_rho"]  # [const, INC, HOVAL, rho]
    np.testing.assert_allclose(res.params.values, expected, rtol=1e-4)


def test_sarar_gmm_matches_spreg_GM_Combo(columbus):
    w, df, ref = columbus
    # GM_Combo's default instrument set is [X, WX] (w_lags=1); sarar_gmm's
    # default is gstsls's [X, WX, W^2 X]. With the instruments aligned the
    # two agree up to spreg's L-BFGS-B stopping rule on the GM moments
    # (lambda ~2e-6 relative; sarar_gmm solves the moment problem exactly).
    # Before the GS2SLS rewrite this needed rtol=5e-3 because the final
    # step filtered the instruments and the SEs came from stage 1.
    res = sarar_gmm(w, df, "CRIME ~ INC + HOVAL", w_lags=1)
    expected = ref["gm_combo"]["betas_with_rho_lambda"]
    # Output order: [const, INC, HOVAL, rho, lambda] -- spreg ditto.
    np.testing.assert_allclose(res.params.values, expected, rtol=1e-5)


def test_sem_gmm_het_robust_returns_beta_se(columbus):
    w, df, _ = columbus
    res = sem_gmm(w, df, "CRIME ~ INC + HOVAL", robust="het")
    # Robust SEs should be finite and positive for beta columns
    se_beta = res.std_errors.drop("lambda").values
    assert np.all(np.isfinite(se_beta))
    assert np.all(se_beta > 0)


def test_sar_gmm_high_w_lags_still_identifies(columbus):
    w, df, _ = columbus
    res = sar_gmm(w, df, "CRIME ~ INC + HOVAL", w_lags=2)
    # With richer instruments rho should still land in a reasonable range
    rho = float(res.model_info["spatial_param_value"])
    assert -0.5 < rho < 0.9
