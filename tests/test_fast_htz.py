"""Tests for clubSandwich-equivalent HTZ Wald DOF (Pustejovsky-Tipton 2018).

See docs/superpowers/specs/2026-04-27-htz-clubsandwich-parity-design.md.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Shared panel fixture (mirrors tests/test_fast_inference.py::_ols_panel)
# ---------------------------------------------------------------------------

def _ols_panel(n_clusters=20, m=30, seed=0, beta=(0.30, -0.20), unbalanced=False):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_clusters):
        n_g = int(rng.integers(3, 50)) if unbalanced else m
        x1 = rng.normal(size=n_g)
        x2 = rng.normal(size=n_g)
        u_g = rng.normal(scale=0.5)
        eps = rng.normal(size=n_g) + u_g
        y = beta[0] * x1 + beta[1] * x2 + eps
        for i in range(n_g):
            rows.append({"g": g, "x1": x1[i], "x2": x2[i], "y": y[i]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task 1: WaldTestResult dataclass
# ---------------------------------------------------------------------------

def test_wald_test_result_is_frozen_dataclass():
    """WaldTestResult should exist, be importable from sp.fast, and be frozen."""
    res = sp.fast.WaldTestResult(
        test="HTZ", q=2, eta=18.5, F_stat=3.4, p_value=0.04, Q=7.1,
        R=np.eye(2), r=np.zeros(2), V_R=np.eye(2),
    )
    assert res.test == "HTZ"
    assert res.q == 2
    # Frozen dataclass: setting an attribute must raise.
    with pytest.raises((AttributeError, Exception)):
        res.eta = 99.0  # type: ignore[misc]
