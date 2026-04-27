"""HDFE Poisson baseline datasets.

Three reproducible synthetic panels with two-way crossed fixed effects.
Design choices:

* DGP is **Poisson with two-way crossed FE** so the same dataset works for
  ``feols`` (linear) and ``fepois`` (Poisson) backends — though the
  baseline focuses on Poisson, since that is the user-reported gap.
* FE codes are drawn i.i.d. (not panel) so cardinalities are decoupled
  from N and easy to scale.
* Coefficient ground truth is fixed (``TRUE_BETA``) so every backend
  can be evaluated against the same target — but the **baseline metric
  is cross-backend agreement**, not absolute bias.
* Single seed per config -> bit-identical CSV across machines.

Sizes (calibrated against the user's 100k-individual scenario):

==========  ==========  ===========  ===========
config      N           fe1 (large)  fe2 (small)
==========  ==========  ===========  ===========
small       100,000     1,000        50
medium      1,000,000   100,000      1,000
large       10,000,000  1,000,000    10,000
==========  ==========  ===========  ===========

CSV (gzip) is the lowest-common-denominator format — R reads it via
``data.table::fread`` and Python via ``pandas.read_csv``.  Feather/
Arrow would be faster, but the local R install has no ``arrow`` pkg
and we want the script to run out of the box.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# True coefficients on x1, x2 (intercept absorbed into FE means)
TRUE_BETA: Dict[str, float] = {"x1": 0.30, "x2": -0.20}

CONFIGS: Dict[str, Dict[str, int]] = {
    "small":  dict(n=100_000,    fe1=1_000,     fe2=50,     seed=42),
    "medium": dict(n=1_000_000,  fe1=100_000,   fe2=1_000,  seed=43),
    "large":  dict(n=10_000_000, fe1=1_000_000, fe2=10_000, seed=44),
}


def make_dataset(name: str) -> Tuple[pd.DataFrame, Dict]:
    """Generate a Poisson HDFE dataset.

    Returns
    -------
    df : pd.DataFrame
        Columns ``y`` (int64), ``x1`` (float64), ``x2`` (float64),
        ``fe1`` (int32), ``fe2`` (int32).
    meta : dict
        Provenance for the JSON sidecar.
    """
    if name not in CONFIGS:
        raise ValueError(f"unknown config {name!r}; choose from {list(CONFIGS)}")
    cfg = CONFIGS[name]
    rng = np.random.default_rng(cfg["seed"])
    n, fe1_card, fe2_card = cfg["n"], cfg["fe1"], cfg["fe2"]

    fe1 = rng.integers(0, fe1_card, size=n, dtype=np.int32)
    fe2 = rng.integers(0, fe2_card, size=n, dtype=np.int32)

    # FE effects with modest variance so exp(eta) stays in a sane range
    alpha = rng.normal(0.0, 0.5, size=fe1_card)
    gamma = rng.normal(0.0, 0.3, size=fe2_card)

    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)

    eta = (
        0.5
        + TRUE_BETA["x1"] * x1
        + TRUE_BETA["x2"] * x2
        + alpha[fe1]
        + gamma[fe2]
    )
    # Cap eta so exp doesn't overflow on the tails (very rare with
    # the variances above, but cheap insurance).
    np.clip(eta, -10.0, 10.0, out=eta)
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(np.int64)

    df = pd.DataFrame(
        {
            "y": y,
            "x1": x1.astype(np.float64),
            "x2": x2.astype(np.float64),
            "fe1": fe1,
            "fe2": fe2,
        }
    )
    meta = {
        "name": name,
        "n": int(n),
        "fe1_cardinality": int(fe1_card),
        "fe2_cardinality": int(fe2_card),
        "seed": int(cfg["seed"]),
        "true_beta": TRUE_BETA,
        "y_mean": float(df["y"].mean()),
        "y_var": float(df["y"].var()),
        "y_zero_share": float((df["y"] == 0).mean()),
        "fe1_unique_observed": int(df["fe1"].nunique()),
        "fe2_unique_observed": int(df["fe2"].nunique()),
    }
    return df, meta


def write_csv(name: str, *, force: bool = False) -> Path:
    """Materialise one dataset to ``data/<name>.csv.gz`` + meta sidecar.

    Returns the CSV path. Skips re-generation if both files already
    exist and ``force`` is False (datasets are deterministic in seed).
    """
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / f"{name}.csv.gz"
    meta_path = DATA_DIR / f"{name}.meta.json"
    if csv_path.exists() and meta_path.exists() and not force:
        print(f"[datasets] {name}: cached at {csv_path}", file=sys.stderr)
        return csv_path

    t0 = time.perf_counter()
    df, meta = make_dataset(name)
    t1 = time.perf_counter()
    df.to_csv(csv_path, index=False, compression="gzip")
    t2 = time.perf_counter()
    meta["bytes_compressed"] = csv_path.stat().st_size
    meta["gen_seconds"] = round(t1 - t0, 3)
    meta["write_seconds"] = round(t2 - t1, 3)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"[datasets] {name}: n={meta['n']:,}  fe1={meta['fe1_cardinality']:,}  "
        f"fe2={meta['fe2_cardinality']:,}  size={meta['bytes_compressed']/1e6:.1f}MB  "
        f"gen={meta['gen_seconds']}s  write={meta['write_seconds']}s",
        file=sys.stderr,
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HDFE baseline datasets")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["small", "medium"],
        help="Which configs to materialise (default: small medium). "
        "Add 'large' explicitly — it is ~500MB+ on disk and ~10min to write.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if cached")
    args = parser.parse_args()
    for name in args.configs:
        write_csv(name, force=args.force)


if __name__ == "__main__":
    main()
