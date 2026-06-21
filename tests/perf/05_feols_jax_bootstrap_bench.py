"""Track-C accelerator benchmark — sp.fast.feols_jax_bootstrap.

Times the four bootstrap variants (pairs / cluster / wild /
wild_cluster) at a fixed (n, B, n_clusters, p) DGP, comparing:

  * CPU sequential — a Python loop that calls ``sp.fast.feols``
    once per bootstrap iteration. This is the baseline you would
    get without any of the v1.14 acceleration work.
  * CPU JAX vmap — ``sp.fast.feols_jax_bootstrap`` on whatever
    device JAX picks (CPU here). Documents the JIT + vmap
    overhead absorbing the parallelism win on CPU.
  * GPU vmap — same call on a CUDA / MPS device. This is the
    cell that documents the headline GPU speedup; expect 10-100x
    over the CPU sequential baseline at B >= 1000 on n >= 1e5.

Results are written to ``results/05_feols_jax_bootstrap_<side>.json``
where ``<side>`` is ``cpu_seq``, ``cpu_jax``, or one of the GPU
device platforms reported by ``jax.devices()[0].platform``. The JSS
table at Table~\\ref{tab:accel-bench} is populated from these files
by ``tests/perf/compare_perf.py`` (a small extension to that script
maps the ``05_*`` files into the multirow accelerator table).

Usage
-----
On the CPU baseline machine (Mac / Linux laptop)::

    python tests/perf/05_feols_jax_bootstrap_bench.py --modes cpu_seq cpu_jax

On a Colab / Lambda Labs / RunPod GPU runtime, after a one-time
``pip install jax[cuda12] jaxlib statspai``::

    python tests/perf/05_feols_jax_bootstrap_bench.py --modes cpu_jax gpu

The CPU-sequential mode is platform-bound (it is the same loop
regardless of where it runs), so to get a clean speedup ratio we
recommend running ``cpu_seq`` once on the GPU machine alongside the
``gpu`` mode, then comparing within the same hardware tier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import statspai as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    TimingResult,
    gpu_hbm_peak_mb,
    hardware_record,
    time_repeat,
)

# DGP — fixed across modes so cross-machine numbers are comparable.
N = 1_000_000
N_FIRMS = 5_000
N_YEARS = 20
B = 2_000
N_REPS = 3
SEED = 42
VMAP_CHUNK_SIZE = 200
VARIANTS: tuple[str, ...] = ("pairs", "cluster", "wild", "wild_cluster")


def make_panel(
    n: int = N, n_firms: int = N_FIRMS, n_years: int = N_YEARS, seed: int = SEED
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, n_firms, size=n)
    year = rng.integers(0, n_years, size=n)
    firm_fe = rng.normal(0, 1.0, size=n_firms)
    year_fe = rng.normal(0, 0.5, size=n_years)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (
        2.0 * x1
        - 1.5 * x2
        + firm_fe[firm]
        + year_fe[year]
        + rng.normal(scale=0.5, size=n)
    )
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm, "year": year})


def _cpu_sequential_bootstrap(
    df: pd.DataFrame,
    formula: str,
    variant: str,
    n_boot: int,
    cluster: str | None,
    fe_cols: Sequence[str],
    y_col: str = "y",
    x_cols: Sequence[str] = ("x1", "x2"),
) -> None:
    """Reference CPU loop — calls sp.fast.feols once per bootstrap draw.

    All four variants do real work:

    * ``pairs``        — resample (y, X) row indices with replacement;
                          one full FE-OLS fit per draw.
    * ``cluster``      — block-resample by cluster id; one fit per draw.
    * ``wild``         — Rademacher (±1) signs per observation; reconstruct
                          ``y_b = y_hat + signs * resid`` from a one-shot
                          base fit, then refit FE-OLS per draw.
    * ``wild_cluster`` — Rademacher signs assigned per cluster, same
                          reconstruction + refit pattern as ``wild``.

    The wild variants therefore measure the **true** per-iteration cost of
    a refit, which is what a naive user without v1.14 acceleration would
    pay. Base fit + within-transform happens **once** outside the loop
    and is amortised over ``n_boot`` iterations (which is what any
    competent baseline implementation would also do).
    """
    rng = np.random.default_rng(0)
    n = len(df)
    if variant == "pairs":
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            sp.fast.feols(formula, df.iloc[idx], vcov="iid")
        return
    if variant == "cluster":
        if cluster is None:
            raise ValueError("cluster variant requires a cluster column")
        cluster_ids = df[cluster].to_numpy()
        uniq = np.unique(cluster_ids)
        groups = {c: np.where(cluster_ids == c)[0] for c in uniq}
        for _ in range(n_boot):
            picked = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([groups[c] for c in picked])
            sp.fast.feols(formula, df.iloc[idx], vcov="iid")
        return
    # wild / wild_cluster — one base fit to recover residuals + fitted,
    # then refit per draw on the reconstructed y.
    stacked = np.column_stack(
        [df[y_col].to_numpy(dtype=np.float64)]
        + [df[c].to_numpy(dtype=np.float64) for c in x_cols]
    )
    demeaned, _info = sp.fast.demean(stacked, df[list(fe_cols)])
    y_w = demeaned[:, 0]
    X_w = demeaned[:, 1:]
    beta, *_ = np.linalg.lstsq(X_w, y_w, rcond=None)
    resid = y_w - X_w @ beta
    y_hat = df[y_col].to_numpy(dtype=np.float64) - resid
    df_local = df.copy(deep=False)
    if variant == "wild":
        for _ in range(n_boot):
            signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=n)
            df_local[y_col] = y_hat + signs * resid
            sp.fast.feols(formula, df_local, vcov="iid")
        return
    if variant == "wild_cluster":
        if cluster is None:
            raise ValueError("wild_cluster variant requires a cluster column")
        cluster_ids = df[cluster].to_numpy()
        _uniq, inv = np.unique(cluster_ids, return_inverse=True)
        n_clusters = int(_uniq.size)
        for _ in range(n_boot):
            cluster_signs = rng.choice(
                np.array([-1.0, 1.0], dtype=np.float64),
                size=n_clusters,
            )
            signs = cluster_signs[inv]
            df_local[y_col] = y_hat + signs * resid
            sp.fast.feols(formula, df_local, vcov="iid")
        return
    raise ValueError(f"unknown variant: {variant!r}")


def run_mode(
    mode: str,
    df: pd.DataFrame,
    formula: str,
    cluster: str,
    fe_cols: Sequence[str],
    vmap_chunk_size: int = VMAP_CHUNK_SIZE,
) -> list[TimingResult]:
    rows: list[TimingResult] = []
    for variant in VARIANTS:
        cluster_kw = cluster if variant in ("cluster", "wild_cluster") else None
        if mode == "cpu_seq":
            label = "cpu_sequential"

            def runner(_v: str = variant, _c: str | None = cluster_kw) -> None:
                _cpu_sequential_bootstrap(
                    df,
                    formula,
                    _v,
                    n_boot=B,
                    cluster=_c,
                    fe_cols=fe_cols,
                )

        else:  # cpu_jax / gpu — both use feols_jax_bootstrap, device decided by JAX
            label = mode

            def runner(
                _v: str = variant,
                _c: str | None = cluster_kw,
                _chunk: int = vmap_chunk_size,
            ) -> None:
                kw: dict = dict(n_boot=B, seed=0, bootstrap=_v, vmap_chunk_size=_chunk)
                if _c is not None:
                    kw["cluster"] = _c
                sp.fast.feols_jax_bootstrap(formula, df, **kw)

        hbm_before = gpu_hbm_peak_mb() if mode != "cpu_seq" else None
        median, iqr, t_min, t_max, peak_mem = time_repeat(
            runner,
            n_reps=N_REPS,
            warmup=1,
        )
        hbm_after = gpu_hbm_peak_mb() if mode != "cpu_seq" else None
        if hbm_before is not None and hbm_after is not None:
            peak_hbm = max(hbm_before, hbm_after)
        else:
            peak_hbm = hbm_after if hbm_before is None else hbm_before
        rows.append(
            TimingResult(
                estimator="feols_jax_bootstrap",
                side=label,
                n=N,
                n_reps=N_REPS,
                median_time_s=median,
                iqr_time_s=iqr,
                min_time_s=t_min,
                max_time_s=t_max,
                peak_mem_mb=peak_mem,
                peak_gpu_hbm_mb=peak_hbm,
                extra={
                    "variant": variant,
                    "B": B,
                    "n_clusters": N_FIRMS,
                    "p": 2,
                    "fe": list(fe_cols),
                    "vmap_chunk_size": vmap_chunk_size,
                    "n": N,
                    "n_firms": N_FIRMS,
                    "n_years": N_YEARS,
                },
            )
        )
    return rows


def _jax_device_label() -> str:
    try:
        import jax

        return f"jax_{jax.devices()[0].platform}"
    except ImportError:
        return "jax_unavailable"


def main(argv: Sequence[str] | None = None) -> None:
    global B  # noqa: PLW0603 — intentional rebind so ``run_mode`` sees override
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=["cpu_seq", "cpu_jax", "gpu"],
        default=["cpu_seq", "cpu_jax"],
        help="Which timing modes to run. 'gpu' is identical to 'cpu_jax' "
        "but written under a separate JSON file label that records "
        "the JAX device platform.",
    )
    ap.add_argument(
        "--n-boot", type=int, default=B, help=f"Override bootstrap draws (default {B})"
    )
    ap.add_argument(
        "--vmap-chunk-size",
        type=int,
        default=VMAP_CHUNK_SIZE,
        help=f"JAX vmap chunk size (default {VMAP_CHUNK_SIZE}); "
        "lower to fit on smaller GPUs.",
    )
    args = ap.parse_args(argv)
    B = args.n_boot

    df = make_panel()
    formula = "y ~ x1 + x2 | firm + year"
    cluster = "firm"
    fe_cols = ("firm", "year")

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in args.modes:
        print(f"--- mode={mode!r} ---", flush=True)
        rows = run_mode(
            mode, df, formula, cluster, fe_cols, vmap_chunk_size=args.vmap_chunk_size
        )
        side = mode if mode != "gpu" else _jax_device_label()
        out_path = out_dir / f"05_feols_jax_bootstrap_{side}.json"
        out_path.write_text(
            json.dumps(
                {
                    "estimator": "feols_jax_bootstrap",
                    "side": side,
                    "mode": mode,
                    "rows": [r.to_dict() for r in rows],
                    "hardware": hardware_record(),
                    "extra": {
                        "n": N,
                        "n_firms": N_FIRMS,
                        "n_years": N_YEARS,
                        "B": B,
                        "formula": formula,
                        "cluster": cluster,
                        "vmap_chunk_size": args.vmap_chunk_size,
                        "n_reps": N_REPS,
                        "wild_baseline": "true_wild",
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        for r in rows:
            hbm = f" gpu_hbm={r.peak_gpu_hbm_mb:.0f}MiB" if r.peak_gpu_hbm_mb else ""
            print(f"  {r.extra['variant']:14s} median={r.median_time_s:8.3f}s{hbm}")
        print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
