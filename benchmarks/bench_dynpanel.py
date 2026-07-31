"""Benchmark: dynamic-panel GMM (``sp.xtabond`` / ``sp.xtdpdsys``) scaling.

The cost of these estimators is driven by two things that grow at different
rates: the number of units ``N`` (rows, and the number of variance groups)
and the number of periods ``T`` (which drives the *instrument count*
quadratically unless collapsed). Both are varied here, and the collapsed
variant is timed alongside the full one so the O(T^2) instrument growth is
visible rather than inferred.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import statspai as sp

from _utils import bench, fmt_ms


def _make_dynamic(n_units: int, n_periods: int, seed: int = 0) -> pd.DataFrame:
    """AR(1) panel with a unit fixed effect, generated period-by-period.

    Vectorised over units so the fixture build does not dominate the timing
    of the estimator it is meant to measure.
    """
    rng = np.random.default_rng(seed)
    rho = 0.5
    alpha = rng.normal(size=n_units)
    y = alpha / (1 - rho) + rng.normal(size=n_units)
    burn = 15
    frames = []
    for t in range(n_periods + burn):
        x = rng.normal(size=n_units)
        y = rho * y + x + alpha + rng.normal(size=n_units)
        if t >= burn:
            frames.append(
                pd.DataFrame(
                    {"id": np.arange(n_units), "time": t - burn, "y": y, "x": x}
                )
            )
    return pd.concat(frames, ignore_index=True)


def _fit(df: pd.DataFrame, **kwargs):
    with warnings.catch_warnings():
        # Instrument-count advisories are the point of some of these
        # configurations; they are not what is being timed.
        warnings.simplefilter("ignore")
        return sp.xtabond(df, y="y", x=["x"], id="id", time="time", **kwargs)


def run(
    sizes: List[Tuple[int, int]] = ((2_000, 8), (10_000, 10), (20_000, 15))
) -> Dict:
    out = []
    for n_units, n_periods in sizes:
        df = _make_dynamic(n_units, n_periods)
        probe = _fit(df, lags=1, twostep=True)
        cases = {
            "diff_1step": dict(lags=1),
            "diff_2step": dict(lags=1, twostep=True),
            "diff_2step_collapse": dict(lags=1, twostep=True, collapse=True),
            "system_2step": dict(lags=1, twostep=True, method="system"),
        }
        row = {
            "n_units": n_units,
            "n_periods": n_periods,
            "n_rows": int(probe.model_info["n_obs"]),
            "n_instruments": int(probe.model_info["n_instruments"]),
        }
        for name, kwargs in cases.items():
            row[name] = bench(lambda kw=kwargs: _fit(df, **kw), n_runs=2)
        out.append(row)
    return {"name": "dynamic_panel_gmm", "rows": out}


def report(result: Dict) -> str:
    lines = [
        "| units | periods | rows | instruments | diff 1-step | diff 2-step "
        "| diff 2-step collapsed | system 2-step |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"]:
        lines.append(
            "| {n_units} | {n_periods} | {n_rows} | {n_instruments} | "
            "{a} | {b} | {c} | {d} |".format(
                a=fmt_ms(row["diff_1step"]["mean_s"]),
                b=fmt_ms(row["diff_2step"]["mean_s"]),
                c=fmt_ms(row["diff_2step_collapse"]["mean_s"]),
                d=fmt_ms(row["system_2step"]["mean_s"]),
                **row,
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    print(report(run()))
