"""A pre-built CDLZ stack with population weights, for ``sp.stacked_did``.

Built from ``stacked_did_panel.csv`` (cohorts 6, 9, 12 plus never-treated):
one sub-experiment per cohort over event window [-3, 3], controls the
never-treated units plus units first treated after the window (a clean-control
rule), and a time-invariant population weight per unit.

    python tests/reference_parity/_fixtures/_generate_stacked_did_prebuilt_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent


def main() -> None:
    df = pd.read_csv(HERE / "stacked_did_panel.csv")
    ft = df["first_treat"].replace(0, np.inf)
    rng = np.random.default_rng(20260929)
    pop = pd.Series(
        rng.integers(50, 5000, df["id"].nunique()), index=sorted(df["id"].unique())
    )
    frames = []
    for g in sorted(ft[np.isfinite(ft)].unique()):
        treated_ids = set(df.loc[ft == g, "id"])
        ctrl_ids = set(
            df.loc[ft > g + 3, "id"]
        )  # never, or first treated after the window
        sub = df[
            df["id"].isin(treated_ids | ctrl_ids) & df["year"].between(g - 3, g + 3)
        ].copy()
        sub["event"] = int(g)
        sub["treated"] = sub["id"].isin(treated_ids).astype(int)
        sub["rel"] = sub["year"] - int(g)
        frames.append(sub)
    st = pd.concat(frames, ignore_index=True)
    st["pop"] = st["id"].map(pop).astype(float)
    st[["id", "year", "event", "treated", "rel", "y", "pop"]].to_csv(
        HERE / "stacked_did_prebuilt.csv", index=False, float_format="%.17g"
    )
    print(f"wrote stacked_did_prebuilt.csv ({len(st)} rows)")


if __name__ == "__main__":
    main()
