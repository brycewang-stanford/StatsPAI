"""MC-SIMEX for a misclassified factor with more than two levels (T1).

The deterministic comparison with R ``mcsimex`` is in
``tests/reference_parity/test_simex_r_parity.py``; here: bias reduction
under a known truth, identities (identity matrix, labelled levels, a binary
factor equals the plain 0/1 column) and the input checks.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.measurement_error import _simex_misclass as mc

P3 = np.array([[0.80, 0.10, 0.05], [0.10, 0.80, 0.15], [0.10, 0.10, 0.80]])
BETA = {"C(xs)[T.1]": 0.5, "C(xs)[T.2]": 1.2}


def _data(seed, n=2000):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 3, n)
    z = rng.normal(size=n)
    xs = mc.misclassify(x.astype(float), P3, rng).astype(int)
    y = 1.0 + 0.5 * (x == 1) + 1.2 * (x == 2) + 0.5 * z + rng.normal(size=n)
    return pd.DataFrame({"y": y, "xs": xs, "z": z})


def _fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.simex("y ~ C(xs) + z", df, misclassification={"xs": P3}, B=30, **kw)


def test_factor_mcsimex_reduces_the_misclassification_bias():
    # 20 samples of n = 2000 (truth 0.5, 1.2). Measured: naive means 0.434
    # and 0.900, MC-SIMEX means 0.505 and 1.181 (MC SD 0.077 and 0.061;
    # mean SE 0.079 and 0.078).
    naive, corrected = [], []
    for s in range(20):
        fit = _fit(_data(s), seed=s)
        naive.append(fit.model_info["naive_params"][list(BETA)].to_numpy())
        corrected.append(fit.params[list(BETA)].to_numpy())
    truth = np.array(list(BETA.values()))
    bias_naive = np.abs(np.mean(naive, axis=0) - truth)
    bias_simex = np.abs(np.mean(corrected, axis=0) - truth)
    mc_se = np.std(corrected, axis=0, ddof=1) / np.sqrt(len(corrected))
    assert np.all(bias_naive > 0.05)
    assert np.all(bias_simex < 0.5 * bias_naive)
    assert np.all(bias_simex < 4 * mc_se)


def test_identity_matrix_returns_the_naive_fit():
    df = _data(0, 500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.simex("y ~ C(xs) + z", df, misclassification={"xs": np.eye(3)}, B=5)
        ols = sp.regress("y ~ C(xs) + z", data=df)
    np.testing.assert_allclose(fit.params.to_numpy(), ols.params.to_numpy(), atol=1e-12)
    assert fit.diagnostics["no_measurement_error"] is True


def test_labelled_levels_equal_integer_codes():
    df = _data(1, 500)
    lab = df.assign(xs=df["xs"].map({0: "a", 1: "b", 2: "c"}))
    P_lab = pd.DataFrame(P3, index=list("abc"), columns=list("abc"))
    a = _fit(df, seed=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = sp.simex(
            "y ~ C(xs) + z", lab, misclassification={"xs": P_lab}, B=30, seed=7
        )
    np.testing.assert_array_equal(a.params.to_numpy(), b.params.to_numpy())
    np.testing.assert_array_equal(a.std_errors.to_numpy(), b.std_errors.to_numpy())
    # rows of a labelled matrix are put in its column order
    shuffled = P_lab.loc[["c", "a", "b"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = sp.simex(
            "y ~ C(xs) + z", lab, misclassification={"xs": shuffled}, B=30, seed=7
        )
    np.testing.assert_array_equal(a.params.to_numpy(), c.params.to_numpy())


def test_binary_factor_equals_the_plain_column():
    rng = np.random.default_rng(2)
    n = 400
    xb = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {"xb": xb, "z": rng.normal(size=n), "y": 1 + xb + rng.normal(size=n)}
    )
    P = [[0.85, 0.10], [0.15, 0.90]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = sp.simex("y ~ xb + z", df, misclassification={"xb": P}, B=20, seed=3)
        fac = sp.simex("y ~ C(xb) + z", df, misclassification={"xb": P}, B=20, seed=3)
    np.testing.assert_array_equal(plain.params.to_numpy(), fac.params.to_numpy())
    np.testing.assert_array_equal(
        plain.std_errors.to_numpy(), fac.std_errors.to_numpy()
    )
    assert list(fac.params.index) == ["Intercept", "C(xb)[T.1]", "z"]


def test_misclassify_draws_follow_the_matrix_columns():
    rng = np.random.default_rng(0)
    codes = np.repeat([0.0, 1.0, 2.0], 100_000)
    new = mc.misclassify(codes, P3, rng)
    for i in range(3):
        freq = np.bincount(new[codes == i].astype(int), minlength=3) / 100_000
        np.testing.assert_allclose(freq, P3[:, i], atol=0.005)


@pytest.mark.parametrize(
    "P, match",
    [
        ([[0.9, 0, 0], [0.1, 0.9, 0], [0, 0.1, 1.0]], "matrix logarithm"),
        ([[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]], "eigenvalues"),
        ([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.8]], "sum to one"),
    ],
)
def test_invalid_matrices_are_refused(P, match):
    with pytest.raises(MethodIncompatibility, match=match):
        sp.simex("y ~ C(xs) + z", _data(0, 200), misclassification={"xs": P}, B=5)


def test_factor_input_checks():
    df = _data(0, 300)
    with pytest.raises(MethodIncompatibility, match="must enter as a factor"):
        sp.simex("y ~ xs + z", df, misclassification={"xs": P3}, B=5)
    with pytest.raises(MethodIncompatibility, match="0/1 with a 2 x 2"):
        sp.simex("xs ~ z", df, misclassification={"xs": P3}, B=5)
    with pytest.raises(MethodIncompatibility, match="also appears in"):
        sp.simex("y ~ C(xs) + C(xs):z", df, misclassification={"xs": P3}, B=5)
    bad = df.assign(xs=df["xs"].where(df["xs"] != 2, 5))
    with pytest.raises(MethodIncompatibility, match="level"):
        sp.simex("y ~ C(xs) + z", bad, misclassification={"xs": P3}, B=5)
    two = df[df["xs"] != 2]
    with pytest.raises(MethodIncompatibility, match="every level"):
        sp.simex("y ~ C(xs) + z", two, misclassification={"xs": P3}, B=5)
    with pytest.raises(MethodIncompatibility, match="same labels"):
        sp.simex(
            "y ~ C(xs) + z",
            df,
            misclassification={"xs": pd.DataFrame(P3, index=[0, 1, 3])},
            B=5,
        )


def test_a_repeated_eigenvalue_split_into_a_conjugate_pair_is_accepted():
    # P3's eigenvalue 0.7 is defective (a Jordan block), so rounding splits
    # it by ~1e-9: into two reals on macOS Accelerate, into 0.7 +- ~1e-9 i on
    # Linux OpenBLAS, which an exact "imag > 0" test refused. A perturbation
    # at rounding size with zero column sums gives the complex case on every
    # platform.
    K = np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    P_pair = P3 + 1e-14 * K
    ev = np.linalg.eig(P_pair)[0]
    assert 1e-10 < np.max(np.abs(np.imag(ev))) < 1e-8  # the case being tested
    P_checked, levels = mc.mc_matrix_levels("x", P_pair)
    assert levels == [0, 1, 2]
    for lam in (0.5, 1.0, 2.0):
        np.testing.assert_allclose(
            mc.mc_power(P_checked, lam), mc.mc_power(P3, lam), atol=1e-7
        )
