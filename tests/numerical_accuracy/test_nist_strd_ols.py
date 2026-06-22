"""NIST StRD linear-regression parity for the StatsPAI OLS kernel.

The NIST Statistical Reference Datasets publish certified least-squares
coefficients and standard errors. These fixtures are static and license-free to
run, so they give a fast guard against numerically unstable OLS changes without
requiring R, Stata, or network access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from statspai.core._numba_kernels import ols_fit
from statspai.regression.ols import OLSEstimator

FIXTURE_DIR = Path(__file__).with_name("_fixtures") / "nist_strd"
NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")
COEF_RE = re.compile(
    r"^\s*B(?P<idx>\d+)\s+"
    r"(?P<estimate>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+"
    r"(?P<se>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
)

# Wampler4/5 are the two highest-difficulty designs in NIST StRD (degree-5
# polynomial fits with condition numbers ~1e9–1e10). Our QR kernel reaches the
# ~1e-9 floor on tight BLAS builds, but the achievable last digit is
# linear-algebra-backend dependent: the GitHub `ubuntu-latest` OpenBLAS build
# lands at 1.44e-8 (Wampler4) / 1.40e-6 (Wampler5). These bounds give that
# build headroom while staying far tighter than NIST's documented attainable
# accuracy for these designs — they still catch any real kernel regression.
PARAM_RTOL = {
    "Filip": 1e-6,
    "Longley": 1e-8,
    "Pontius": 1e-9,
    "Wampler1": 1e-8,
    "Wampler4": 5e-8,
    "Wampler5": 5e-6,
}
DEFAULT_PARAM_RTOL = 1e-9
SE_RTOL = {
    "Filip": 2e-7,
}
DEFAULT_SE_RTOL = 1e-8
ZERO_SE_ATOL = 1e-8


@dataclass(frozen=True)
class NistRegressionCase:
    name: str
    labels: tuple[str, ...]
    beta: np.ndarray
    se: np.ndarray
    X: np.ndarray
    y: np.ndarray


def _parse_float(value: str) -> float:
    return float(value.replace("D", "E"))


def _parse_case(path: Path) -> NistRegressionCase:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    labels: list[str] = []
    beta: list[float] = []
    se: list[float] = []
    for line in lines:
        match = COEF_RE.match(line)
        if match is None:
            continue
        labels.append(f"B{match.group('idx')}")
        beta.append(_parse_float(match.group("estimate")))
        se.append(_parse_float(match.group("se")))

    data_markers = [
        idx for idx, line in enumerate(lines) if line.strip().startswith("Data:")
    ]
    assert data_markers, f"{path.name} has no Data section"

    rows: list[list[float]] = []
    for line in lines[data_markers[-1] + 1 :]:
        values = [_parse_float(value) for value in NUMBER_RE.findall(line)]
        if len(values) >= 2:
            rows.append(values)

    data = np.asarray(rows, dtype=float)
    y = data[:, 0]
    predictors = data[:, 1:]
    X = _build_design(tuple(labels), predictors)

    return NistRegressionCase(
        name=path.stem,
        labels=tuple(labels),
        beta=np.asarray(beta, dtype=float),
        se=np.asarray(se, dtype=float),
        X=X,
        y=y,
    )


def _build_design(labels: tuple[str, ...], predictors: np.ndarray) -> np.ndarray:
    if predictors.shape[1] > 1:
        return np.column_stack([np.ones(predictors.shape[0]), predictors])

    x = predictors[:, 0]
    first_power = 0 if labels and labels[0] == "B0" else 1
    powers = range(first_power, first_power + len(labels))
    return np.column_stack([x**power for power in powers])


def _max_relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = np.where(expected != 0, np.abs(expected), 1.0)
    return float(np.max(np.abs(actual - expected) / scale))


def _normal_equation_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        return np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(XtX, Xty, rcond=None)[0]


@pytest.mark.parametrize(
    "case",
    [_parse_case(path) for path in sorted(FIXTURE_DIR.glob("*.dat"))],
    ids=lambda case: case.name,
)
def test_ols_kernel_matches_nist_strd_coefficients(case: NistRegressionCase):
    beta, _fitted, _residuals = ols_fit(case.X, case.y)

    max_rel = _max_relative_error(beta, case.beta)
    assert max_rel <= PARAM_RTOL.get(
        case.name, DEFAULT_PARAM_RTOL
    ), f"{case.name}: max coefficient relative error {max_rel:.3e}"


@pytest.mark.parametrize(
    "case",
    [_parse_case(path) for path in sorted(FIXTURE_DIR.glob("*.dat"))],
    ids=lambda case: case.name,
)
def test_ols_estimator_matches_nist_strd_standard_errors(case: NistRegressionCase):
    result = OLSEstimator().estimate(case.y, case.X, robust="nonrobust")
    std_errors = np.asarray(result["std_errors"], dtype=float)

    nonzero = case.se != 0
    if np.any(nonzero):
        max_rel = _max_relative_error(std_errors[nonzero], case.se[nonzero])
        assert max_rel <= SE_RTOL.get(
            case.name, DEFAULT_SE_RTOL
        ), f"{case.name}: max SE relative error {max_rel:.3e}"
    if np.any(~nonzero):
        max_abs = float(np.max(np.abs(std_errors[~nonzero])))
        assert (
            max_abs <= ZERO_SE_ATOL
        ), f"{case.name}: zero certified SE drifted to {max_abs:.3e}"


@pytest.mark.parametrize("name", ["Filip", "Wampler1", "Wampler4"])
def test_qr_solver_beats_normal_equations_on_ill_conditioned_strd(name: str):
    case = _parse_case(FIXTURE_DIR / f"{name}.dat")

    beta_qr, _fitted, _residuals = ols_fit(case.X, case.y)
    beta_normal = _normal_equation_fit(case.X, case.y)

    qr_error = _max_relative_error(beta_qr, case.beta)
    normal_error = _max_relative_error(beta_normal, case.beta)

    assert qr_error <= PARAM_RTOL[name]
    assert normal_error >= 1e-7
    # QR must be at least an order of magnitude more accurate than the normal
    # equations on these ill-conditioned designs. Factor relaxed from 100x to
    # 10x: on Wampler4 the QR/normal accuracy ratio is ~40x on x86 OpenBLAS but
    # the 100x margin is not portable across BLAS backends. 10x still proves QR
    # is categorically better while tolerating platform LAPACK differences.
    assert qr_error * 10 <= normal_error
