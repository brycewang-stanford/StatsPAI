"""Balanced staggered panel for the ``sp.etwfe`` option-by-option reference
(``etwfe_options_R.json`` / ``etwfe_options_Stata.json``).

240 units x 8 years (2001-2008), cohorts 2004 / 2006 / 2007 and 80
never-treated units (``g = 0``).  Columns:

* ``y``   continuous outcome with cohort x period effect heterogeneity, a
  time-varying control ``x1`` and an effect moderated by ``xv``;
* ``yc``  a Poisson count and ``yb`` a 0/1 outcome on the same design, for
  the nonlinear families;
* ``x1``  time-varying control, correlated with the cohort;
* ``xv``  time-varying continuous moderator;
* ``w``   positive observation weights that vary across and within units,
  so the two ``agg_weights`` rules give different aggregates;
* ``xcat`` a three-level categorical moderator (``a`` / ``b`` / ``c``) cut
  from ``xv``.

    python tests/reference_parity/_fixtures/_generate_etwfe_options_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "etwfe_options_panel.csv"


def main() -> None:
    rng = np.random.default_rng(20261015)
    n_units, years = 240, np.arange(2001, 2009)
    T = len(years)
    g_unit = np.repeat([0, 2004, 2006, 2007], [80, 60, 60, 40])
    rng.shuffle(g_unit)
    d = pd.DataFrame(
        {
            "id": np.repeat(np.arange(1, n_units + 1), T),
            "year": np.tile(years, n_units),
            "g": np.repeat(g_unit, T),
        }
    )
    a_i = np.repeat(rng.normal(size=n_units), T)
    w_i = np.repeat(rng.uniform(0.5, 2.0, n_units), T)
    d["x1"] = 0.3 * (d["g"] > 0) + 0.1 * (d["year"] - 2001) + rng.normal(size=len(d))
    d["xv"] = 0.5 * a_i + rng.normal(size=len(d))
    d["w"] = w_i * rng.uniform(0.8, 1.25, len(d))
    treated = (d["g"] > 0) & (d["year"] >= d["g"])
    rel = np.where(treated, d["year"] - d["g"], 0)
    tau = np.where(
        treated,
        1.0 + 0.25 * rel + 0.4 * (d["g"] == 2006) + 0.5 * d["xv"],
        0.0,
    )
    lam = 0.15 * (d["year"] - 2001) + 0.2 * (d["g"] == 2007)
    d["y"] = a_i + lam + 0.5 * d["x1"] + tau + rng.normal(size=len(d))
    eta_c = 0.3 + 0.4 * a_i + 0.05 * (d["year"] - 2001) + 0.1 * d["x1"]
    d["yc"] = rng.poisson(np.exp(eta_c + 0.3 * tau / (1 + np.abs(tau))))
    eta_b = -0.4 + 0.6 * a_i + 0.08 * (d["year"] - 2001) + 0.2 * d["x1"]
    p_b = 1.0 / (1.0 + np.exp(-(eta_b + 0.6 * treated)))
    d["yb"] = (rng.uniform(size=len(d)) < p_b).astype(int)
    # Three-level categorical moderator, cut from xv (no further draws).
    d["xcat"] = np.where(d["xv"] > 0.5, "c", np.where(d["xv"] < -0.5, "a", "b"))
    d.to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
