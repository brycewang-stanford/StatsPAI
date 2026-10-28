"""Inputs for the Agarwal-Singh ADH parity test of sp.noisy_covariates.

The design follows the authors' replication package (Zenodo record
19866751, CC-BY-4.0; ``scripts/03_run_census.R``,
``src/census/specifications.R``, ``src/census/semi.R``): the Autor-Dorn-
Hanson (2013) commuting-zone panel, outcome ``d_sh_empl_mfg``, treatment
``d_tradeusch_pw``, instrument ``d_tradeotch_pw_lag``, weights
``timepwt48`` and the 31 covariates of their augmented specification.

The corrupted covariate matrices are drawn here with NumPy and passed to
the authors' R code as raw doubles (``.bin``), so both sides read the
same bits. The test imports this module to redraw them.

Regenerate the golden file (needs R with dplyr + foreign and the
unpacked replication package)::

    python tests/reference_parity/_fixtures/_generate_noisy_covariates_adh_data.py \\
        <zip or unpacked package> <work dir>
    Rscript tests/reference_parity/_fixtures/_generate_noisy_covariates_adh_R.R \\
        <unpacked replication_package dir> <work dir> \\
        tests/reference_parity/_fixtures/noisy_covariates_adh_R.json
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

DTA_MEMBER = "replication_package/data/workfile_china.dta"
DTA_SHA256 = "58f2ced39352287266d5c9c9d76b89939cdc4973955cebf69d528966653a2971"

X1 = [
    "l_shind_manuf_cbp",
    "l_sh_popedu_c",
    "l_sh_popfborn",
    "l_sh_empl_f",
    "l_sh_routine33",
    "l_task_outsource",
]
X2 = [
    "reg_midatl",
    "reg_encen",
    "reg_wncen",
    "reg_satl",
    "reg_escen",
    "reg_wscen",
    "reg_mount",
    "reg_pacif",
]
X3 = [
    "l_sh_empl_mfg",
    "l_sh_empl_nmfg",
    "l_sh_unempl",
    "l_sh_nilf",
    "l_sh_ssadiswkrs",
    "l_avg_lnwkwage_mfg",
    "l_avg_lnwkwage_nmfg",
    "l_trans_totindiv_pc",
    "l_trans_ssaret_pc",
    "l_trans_ssadis_pc",
    "l_trans_totmed_pc",
    "l_trans_fedinc_pc",
    "l_trans_unemp_pc",
    "l_trans_taaimp_pc",
    "l_avg_hhincsum_pc_pw",
    "l_avg_hhincwage_pc_pw",
]
COVARIATES = X1 + X2 + X3 + ["t2"]
Y, D, U, W = "d_sh_empl_mfg", "d_tradeusch_pw", "d_tradeotch_pw_lag", "timepwt48"

# Census privacy budget used by the authors (Appendix M): rho = 2.56 *
# 447 / 4099 zero-concentrated DP, epsilon = sqrt(2 rho).
EPSILON_OFFICIAL = float(np.sqrt(447 / 4099 * 2.56 * 2))

# name -> (scaled?, kind, level, missing share, numpy seed)
CONFIGS: Dict[str, Tuple[bool, str, float, float, int]] = {
    "clean_unscaled": (False, "clean", 0.0, 0.0, 0),
    "clean_scaled": (True, "clean", 0.0, 0.0, 0),
    "gauss40_miss10_unscaled": (False, "gauss", 0.4, 0.1, 20261028),
    "gauss100_scaled": (True, "gauss", 1.0, 0.0, 20261029),
    "dp_1eps_unscaled": (False, "dp", 1.0, 0.0, 20261030),
    "dp_0.01eps_unscaled": (False, "dp", 0.01, 0.0, 20261031),
}


def load_adh(source: Path) -> pd.DataFrame:
    """The ADH workfile from the replication zip or its unpacked folder."""
    source = Path(source)
    if source.is_dir():
        raw = (source / "data" / "workfile_china.dta").read_bytes()
    else:
        with zipfile.ZipFile(source) as zf:
            raw = zf.read(DTA_MEMBER)
    import hashlib

    got = hashlib.sha256(raw).hexdigest()
    if got != DTA_SHA256:
        raise ValueError(f"workfile_china.dta SHA-256 {got} != {DTA_SHA256}")
    df = pd.read_stata(io.BytesIO(raw))
    # float32 columns in the .dta: widen before any arithmetic.
    cols = COVARIATES + [Y, D, U, W, "l_popcount"]
    return df[cols].astype("float64")


def covariate_matrix(df: pd.DataFrame, scaled: bool) -> np.ndarray:
    X = df[COVARIATES].to_numpy(dtype=float)
    if scaled:  # R scale(): centre, divide by the n - 1 standard deviation
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    return X


def dp_scale(df: pd.DataFrame, X: np.ndarray, level: float) -> np.ndarray:
    """Laplace scale per column, the authors' ``run_calibration_unscaled``.

    Sensitivities: 100 for percentages, 1 for dummies, twice the range
    for the offshorability index and twice the maximum for wage, transfer
    and income levels; divided by the smallest CZ population and scaled
    by p / epsilon (Proposition 6.1).
    """
    x1 = df[X1].to_numpy(dtype=float)
    x3 = df[X3].to_numpy(dtype=float)
    s1 = np.full(len(X1), 100.0)
    s1[5] = 2 * x1[:, 5].max() - 2 * x1[:, 5].min()
    s2 = np.ones(len(X2))
    s3 = np.full(len(X3), 100.0)
    s3[5:] = 2 * x3[:, 5:].max(axis=0)
    s = np.concatenate([s1, s2, s3, [1.0]])
    S = s / df["l_popcount"].min()
    eps = level * EPSILON_OFFICIAL
    return S * X.shape[1] / eps


def corrupted(df: pd.DataFrame, name: str) -> np.ndarray:
    """Covariate matrix of one configuration (NaN marks a missing cell)."""
    scaled, kind, level, miss, seed = CONFIGS[name]
    X = covariate_matrix(df, scaled)
    rng = np.random.default_rng(seed)
    if kind == "gauss":  # authors' add_gaussian_vec: variance level * var(col)
        sd = np.sqrt(level * X.var(axis=0, ddof=1))
        X = X + rng.normal(size=X.shape) * sd
    elif kind == "dp":
        X = X + rng.laplace(size=X.shape) * dp_scale(df, X, level)
    if miss > 0:
        X = np.where(rng.uniform(size=X.shape) < miss, np.nan, X)
    return X


def main(source: str, out: str) -> None:
    df = load_adh(Path(source))
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in CONFIGS:
        # R reads native little-endian doubles column-major (readBin + matrix).
        corrupted(df, name).ravel(order="F").astype("<f8").tofile(
            out_dir / f"{name}.bin"
        )
    meta = {"configs": list(CONFIGS), "n": int(df.shape[0]), "p": len(COVARIATES)}
    (out_dir / "configs.json").write_text(json.dumps(meta), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
