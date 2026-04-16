"""RD power and sample-size tests."""
import numpy as np, pytest
from statspai.rd.rdpower import rdpower, rdsampsi


def test_rdpower_positive_effect_has_high_power():
    r = rdpower(tau=0.5, var_left=1, var_right=1, n_left=500, n_right=500)
    assert r.power > 0.3


def test_rdpower_zero_effect_has_alpha_power():
    r = rdpower(tau=0.0, var_left=1, var_right=1, n_left=500, n_right=500)
    assert abs(r.power - 0.05) < 0.02


def test_rdpower_larger_n_means_higher_power():
    r1 = rdpower(tau=0.3, n_left=200, n_right=200)
    r2 = rdpower(tau=0.3, n_left=2000, n_right=2000)
    assert r2.power > r1.power


def test_rdsampsi_returns_positive():
    r = rdsampsi(tau=0.3, target_power=0.80)
    assert r.n_total > 0
    assert r.n_left > 0
    assert r.n_right > 0


def test_rdsampsi_larger_effect_needs_fewer_obs():
    r1 = rdsampsi(tau=0.5)
    r2 = rdsampsi(tau=0.1)
    assert r2.n_total > r1.n_total


def test_exported():
    import statspai as sp
    assert callable(sp.rdpower)
    assert callable(sp.rdsampsi)
