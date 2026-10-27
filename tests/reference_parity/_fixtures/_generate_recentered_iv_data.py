"""Fixtures for tests/reference_parity/test_recentered_iv_stata.py.

Two data sets, written to this directory, rounded to 6 decimals so that
Stata ``import delimited ..., asdouble`` and pandas
``read_csv(float_precision="round_trip")`` read identical doubles:

* ``recentered_iv_hdfe.csv`` -- a "who marries whom" style share instrument.
  Children carry known, child-specific probabilities of being a girl; an
  adult's instrument is the share of opposite-sex children among the
  other-group children of their neighbourhood. The expected instrument
  ``mu`` (the closed form: the mean of the opposite-sex probabilities) varies
  across neighbourhoods and is correlated with the error. Columns: id, y, d,
  z, mu, x1, f (absorbed FE), c (cluster = neighbourhood). Used for the
  ivreghdfe comparison with ``mu`` given.
* ``recentered_iv_ri.csv`` + ``recentered_iv_ri_perms.csv.gz`` -- an instrument
  whose values are permuted within strata ``g``; the second file holds 200
  fixed within-stratum permutations (columns ``zs1`` .. ``zs200``, matched on
  ``id``) that both StatsPAI and Stata ``ritest`` (``samplingsourcefile``)
  use, so the randomisation counts can be compared exactly. It is stored
  gzip-compressed (written without a timestamp, so reruns are byte-identical)
  because the plain CSV is above the repository's 500 KB file limit.

Run from the repository root::

    python tests/reference_parity/_fixtures/_generate_recentered_iv_data.py

then ``_generate_recentered_iv_Stata.do`` from this directory.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent


def share_instrument(
    female: np.ndarray,
    group: np.ndarray,
    hood: np.ndarray,
    child_sex: np.ndarray,
    child_group: np.ndarray,
    child_hood: np.ndarray,
) -> np.ndarray:
    """Share of opposite-sex children among other-group children in the hood.

    ``child_sex`` is 1 for a girl. Works for 0/1 sexes and for girl
    probabilities alike (it is linear in ``child_sex``), so passing the
    probabilities returns the closed-form expected instrument.
    """
    n_hood = int(max(hood.max(), child_hood.max())) + 1
    girls = np.zeros((n_hood, 2))
    count = np.zeros((n_hood, 2))
    np.add.at(girls, (child_hood, child_group), child_sex)
    np.add.at(count, (child_hood, child_group), 1.0)
    other = 1 - group
    m = count[hood, other]
    share_girls = girls[hood, other] / m
    return np.where(female == 1, 1.0 - share_girls, share_girls)


def make_hdfe(seed: int = 20261028) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    n_hood, n_adult, n_child = 60, 900, 3000
    child_hood = rng.integers(0, n_hood, n_child)
    child_group = rng.integers(0, 2, n_child)
    # Known, heterogeneous girl probabilities: neighbourhood-level sex ratios.
    hood_p = rng.uniform(0.35, 0.65, n_hood)
    child_p = np.clip(hood_p[child_hood] + rng.normal(0, 0.03, n_child), 0.05, 0.95)
    child_sex = (rng.random(n_child) < child_p).astype(float)
    hood = rng.integers(0, n_hood, n_adult)
    group = rng.integers(0, 2, n_adult)
    female = rng.integers(0, 2, n_adult)
    z = share_instrument(female, group, hood, child_sex, child_group, child_hood)
    mu = share_instrument(female, group, hood, child_p, child_group, child_hood)
    eps = 2.0 * (mu - mu.mean()) + rng.normal(size=n_adult)
    x1 = rng.normal(size=n_adult)
    f = rng.integers(0, 12, n_adult)
    d = 3.0 * z + 0.5 * x1 + 0.6 * eps + rng.normal(size=n_adult)
    y = 1.5 * d + 0.3 * x1 + 0.2 * f + eps
    df = pd.DataFrame(
        {
            "id": np.arange(1, n_adult + 1),
            "y": y,
            "d": d,
            "z": z,
            "mu": mu,
            "x1": x1,
            "f": f,
            "c": hood,
        }
    )
    keep = np.isfinite(df["z"]) & np.isfinite(df["mu"])
    df = df.loc[keep].reset_index(drop=True)
    for col in ["y", "d", "z", "mu", "x1"]:
        df[col] = df[col].round(6)
    design = {
        "child_hood": child_hood,
        "child_group": child_group,
        "child_p": child_p,
    }
    return df, design


def make_ri(seed: int = 7, reps: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = 300
    g = np.repeat(np.arange(6), n // 6)
    # Within-stratum intensities: different levels and spreads per stratum.
    zs = np.round(rng.gamma(2.0, 1.0, n) * (1 + 0.3 * g), 6)
    x = rng.normal(size=n)
    e = rng.normal(size=n)
    d = 0.7 * zs + 0.4 * x + 0.5 * e + rng.normal(size=n)
    y = 1.2 * d + 0.3 * x + 0.5 * g + e
    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "y": np.round(y, 6),
            "d": np.round(d, 6),
            "zs": zs,
            "x": np.round(x, 6),
            "g": g,
        }
    )
    perms = {"id": df["id"].to_numpy()}
    for r in range(1, reps + 1):
        out = zs.copy()
        for s in np.unique(g):
            idx = np.flatnonzero(g == s)
            out[idx] = zs[idx][rng.permutation(idx.size)]
        perms[f"zs{r}"] = out
    return df, pd.DataFrame(perms)


def main() -> None:
    hdfe, _ = make_hdfe()
    hdfe.to_csv(HERE / "recentered_iv_hdfe.csv", index=False, float_format="%.6f")
    ri, perms = make_ri()
    ri.to_csv(HERE / "recentered_iv_ri.csv", index=False, float_format="%.6f")
    perms.to_csv(
        HERE / "recentered_iv_ri_perms.csv.gz",
        index=False,
        float_format="%.6f",
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )


if __name__ == "__main__":
    main()
