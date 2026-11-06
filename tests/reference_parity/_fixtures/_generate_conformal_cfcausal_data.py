"""Data and splits for ``test_conformal_cfcausal_r_parity.py``.

Writes ``conformal_cfcausal_data.csv`` (one long table, ``block`` column
names the case) and ``conformal_cfcausal_spec.json`` (the train / calibration
rows of every split, 1-based for R, plus the constants of the weight and
propensity functions). ``_generate_conformal_cfcausal_R.R`` reads both and
runs ``cfcausal`` on exactly these rows.

The splits are the ones StatsPAI draws for the seeds the test uses; they
are reproduced here from the documented rules so that the reference can be
given the same rows:

* ``sp.weighted_conformal_prediction`` takes the train / calibration arrays
  directly (no split to reproduce).
* ``sp.conformal_counterfactual`` / ``sp.conformal_ite_interval``:
  ``rng = np.random.default_rng(seed)``; arm 1 then arm 0, each
  ``rng.permutation(n_arm)``, the first ``max(int(calib_frac * n_arm), 2)``
  positions are the calibration rows (positions within the arm, rows in data
  order).
* ``sp.conformal_regression(method="split", newdata=...)``:
  ``np.random.default_rng(seed).permutation(n)``, the first
  ``round(train_frac * n)`` rows fit the model.
* ``sp.conformal_cate``: ``np.random.RandomState(seed).permutation(n)``, the
  first ``max(int(n * calib_fraction), 10)`` rows calibrate.
* ``sp.conformal_continuous``: ``np.random.default_rng(seed).shuffle`` of
  ``0..n-1``, the first ``int(calibration_frac * n)`` rows calibrate.

If any of these rules changes, the test fails, as it should: the fixture
would no longer describe the same computation.

Run from the repository root::

    python tests/reference_parity/_fixtures/_generate_conformal_cfcausal_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

#: density ratio of the covariate-shift case, w(x) = exp(SHIFT * x1)
SHIFT = 0.8
#: known propensity e(x) = plogis(PS_A + PS_B * x1)
PS_A, PS_B = -0.3, 0.8


def known_ps(x1: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(PS_A + PS_B * x1)))


def main() -> None:
    rng = np.random.default_rng(20261106)
    frames = []

    # ---- block "wcp": train / calibration / test arrays -------------------
    n_tr, n_cal, n_te = 60, 41, 16
    n = n_tr + n_cal + n_te
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x1[-1] = 4.0  # a test point whose own weight forces an infinite band
    y = 1.0 + x1 - 0.5 * x2 + rng.normal(scale=0.7, size=n)
    role = np.array(["train"] * n_tr + ["calib"] * n_cal + ["test"] * n_te)
    frames.append(
        pd.DataFrame({"block": "wcp", "role": role, "x1": x1, "x2": x2, "y": y})
    )

    # ---- block "cf": observational data with a known propensity -----------
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    t = (rng.uniform(size=n) < known_ps(x1)).astype(int)
    y = np.where(t == 1, 2.0 + x1 + 0.5 * x2, x1) + rng.normal(scale=0.8, size=n)
    frames.append(
        pd.DataFrame(
            {"block": "cf", "role": "data", "x1": x1, "x2": x2, "t": t, "y": y}
        )
    )
    # test points, the last with e(x) = 0.024: an infinite band for Y(1)
    xt1 = np.append(np.linspace(-2.0, 2.0, 11), -4.2)
    xt2 = np.append(np.linspace(1.0, -1.0, 11), 0.3)
    frames.append(pd.DataFrame({"block": "cf", "role": "test", "x1": xt1, "x2": xt2}))

    # ---- block "rct": randomized, e = 1/2, for sp.conformal_cate ----------
    n = 240
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    t = rng.integers(0, 2, size=n)
    y = 1.0 + 0.5 * x1 + t * (1.0 + 0.5 * x2) + rng.normal(scale=0.5, size=n)
    frames.append(
        pd.DataFrame(
            {"block": "rct", "role": "data", "x1": x1, "x2": x2, "t": t, "y": y}
        )
    )

    # ---- block "dose": continuous treatment, for sp.conformal_continuous --
    n = 150
    x1 = rng.normal(size=n)
    dose = rng.uniform(0.0, 5.0, size=n)
    y = 2.0 + 0.7 * dose + 0.5 * x1 + rng.normal(scale=0.5, size=n)
    frames.append(
        pd.DataFrame({"block": "dose", "role": "data", "x1": x1, "t": dose, "y": y})
    )
    frames.append(
        pd.DataFrame(
            {
                "block": "dose",
                "role": "test",
                "x1": np.linspace(-1.5, 1.5, 7),
                "t": np.linspace(0.5, 4.5, 7),
            }
        )
    )

    data = pd.concat(frames, ignore_index=True)
    data.to_csv(HERE / "conformal_cfcausal_data.csv", index=False, float_format="%.17g")

    # Read back what was written, so the splits below are drawn on the same
    # rows the test will see.
    data = pd.read_csv(HERE / "conformal_cfcausal_data.csv")
    spec: dict = {"shift": SHIFT, "ps_a": PS_A, "ps_b": PS_B}

    # counterfactual / ITE splits
    cf = data[(data.block == "cf") & (data.role == "data")]
    t = cf["t"].to_numpy(dtype=int)
    spec["cf"] = {}
    for seed, calib_frac in ((11, 0.3), (12, 0.5)):
        r = np.random.default_rng(seed)
        arms = {}
        for arm in (1, 0):
            n_arm = int(np.sum(t == arm))
            perm = r.permutation(n_arm)
            n_c = max(int(calib_frac * n_arm), 2)
            arms[str(arm)] = {"train": (perm[n_c:] + 1).tolist()}
        spec["cf"][f"{seed}_{calib_frac}"] = arms

    # conformal_regression split on the wcp training + calibration rows
    w = data[(data.block == "wcp") & (data.role != "test")]
    spec["reg"] = {}
    for seed, train_frac in ((3, 0.6), (4, 0.5)):
        perm = np.random.default_rng(seed).permutation(len(w))
        n_fit = int(round(train_frac * len(w)))
        spec["reg"][f"{seed}_{train_frac}"] = (perm[:n_fit] + 1).tolist()

    # conformal_cate split
    rct = data[data.block == "rct"]
    d = rct["t"].to_numpy(dtype=int)
    spec["cate"] = {}
    for seed, frac, alpha in (
        (42, 0.25, 0.05),
        (7, 0.4, 0.05),
        (7, 0.4, 0.2),
        (42, 0.5, 0.1),
    ):
        n = len(rct)
        perm = np.random.RandomState(seed).permutation(n)
        n_c = max(int(n * frac), 10)
        cal, trn = perm[:n_c], perm[n_c:]
        arms = {}
        for arm in (1, 0):
            arms[str(arm)] = {
                "train": (trn[d[trn] == arm] + 1).tolist(),
                "calib": (cal[d[cal] == arm] + 1).tolist(),
            }
        arms["alpha"] = alpha
        spec["cate"][f"{seed}_{frac}_{alpha}"] = arms

    # conformal_continuous split: shuffle 0..n-1, first int(frac * n) calibrate
    dose = data[(data.block == "dose") & (data.role == "data")]
    spec["dose"] = {}
    for seed, frac, alpha in ((0, 0.5, 0.1), (5, 0.4, 0.2)):
        idx = np.arange(len(dose))
        np.random.default_rng(seed).shuffle(idx)
        n_c = int(frac * len(dose))
        spec["dose"][f"{seed}_{frac}_{alpha}"] = {
            "train": (idx[n_c:] + 1).tolist(),
            "calib": (idx[:n_c] + 1).tolist(),
            "alpha": alpha,
        }

    (HERE / "conformal_cfcausal_spec.json").write_text(
        json.dumps(spec, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
