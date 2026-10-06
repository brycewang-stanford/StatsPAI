"""Monte Carlo study of the bootstrap LR test for the number of regimes.

Writes ``mswitch_lrtest_mc.json`` next to this file. Run from the repository
root (about 2 CPU-hours; ``--workers`` processes)::

    PYTHONPATH=src python \
        tests/reference_parity/_fixtures/_generate_mswitch_lrtest_mc.py

``--only NAME`` reruns one design and keeps the others from the existing
file. Seeds depend on the design and the sample only, so the statistics do
not depend on what is run together; ``seconds_per_fit_pair`` does (it was
recorded on a loaded machine and is not a benchmark).

Designs (one regime against two, switching constant, common variance)

* ``size_wn``: white noise, T = 100, ``model='dr'``.
* ``size_ar1``: AR(1) with coefficient 0.5, T = 150, ``model='ar', ar=1``.
* ``power_clear`` / ``power_moderate``: means -1.5 / 1.5 and -0.75 / 0.75,
  staying probability 0.95, unit variance, T = 100.
* ``double_wn``: the white-noise design with the full bootstrap in every
  sample, to check the device used for the others.

All but the last use the "warp-speed" device: one bootstrap series per Monte
Carlo sample, and the observed statistics are referred to the quantiles of
the pooled bootstrap statistics. Every fit uses the search of
``mswitch_lrtest`` (its private ``_pair``), so the rates are those of the
function as shipped.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

DESIGNS: Dict[str, Dict[str, Any]] = {
    "size_wn": dict(n=100, ar=0, phi=0.0, mu=(0.0, 0.0), mc=400, reps=1, starts=10),
    "size_ar1": dict(n=150, ar=1, phi=0.5, mu=(0.0, 0.0), mc=200, reps=1, starts=10),
    "power_clear": dict(
        n=100, ar=0, phi=0.0, mu=(-1.5, 1.5), mc=100, reps=1, starts=10
    ),
    "power_moderate": dict(
        n=100, ar=0, phi=0.0, mu=(-0.75, 0.75), mc=100, reps=1, starts=10
    ),
    "double_wn": dict(n=100, ar=0, phi=0.0, mu=(0.0, 0.0), mc=200, reps=19, starts=6),
}
ORDER = list(DESIGNS)
BASE_SEED = 20261006
LEVELS = (0.10, 0.05)


def draw(design: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    n, burn = int(design["n"]), 50
    state = np.zeros(n + burn, dtype=int)
    for t in range(1, n + burn):
        stay = rng.uniform() < 0.95
        state[t] = state[t - 1] if stay else 1 - state[t - 1]
    dev = np.zeros(n + burn)
    shock = rng.standard_normal(n + burn)
    for t in range(1, n + burn):
        dev[t] = design["phi"] * dev[t - 1] + shock[t]
    return (np.asarray(design["mu"])[state] + dev)[burn:]


def one(task: Tuple[str, int]) -> Dict[str, Any]:
    from statspai.timeseries import _mswitch_core as core
    from statspai.timeseries.mswitch_lrtest import _pair, _simulate

    name, i = task
    design = DESIGNS[name]
    seq = np.random.SeedSequence([BASE_SEED, ORDER.index(name), i])
    data_seq, fit_seq, *boot = seq.spawn(2 + int(design["reps"]))
    y = draw(design, np.random.default_rng(data_seq))
    p = int(design["ar"])
    empty = np.empty((len(y), 0))
    model = "ar" if p else "dr"
    spec0 = core.Spec(1, model, p, 0, 0, "switch", False, False)
    spec1 = core.Spec(2, model, p, 0, 0, "switch", False, False)
    dat = core.Data(y, empty, empty, p)
    starts = int(design["starts"])
    t0 = time.perf_counter()
    with np.errstate(all="ignore"):
        obs = _pair(spec0, spec1, dat, fit_seq, starts, 500, 1e-9)
        stars, flags, raw = [], [], [obs["lr_raw"]]
        for b in boot:
            sim_seq, refit_seq = b.spawn(2)
            ystar = _simulate(
                spec0, dat, obs["theta_null"], np.random.default_rng(sim_seq)
            )
            sim = core.Data(ystar, empty, empty, p)
            res = _pair(spec0, spec1, sim, refit_seq, starts, 500, 1e-9)
            stars.append(max(res["lr_raw"], 0.0))
            raw.append(res["lr_raw"])
            flags.append([res["lr_raw"] < 0.0, res["n_maxima"] > 1])
    return {
        "design": name,
        "i": i,
        "lr": max(obs["lr_raw"], 0.0),
        "lr_raw_min": float(min(raw)),
        "stars": stars,
        "floored": int(obs["lr_raw"] < 0.0) + int(sum(f[0] for f in flags)),
        "multiple": int(obs["n_maxima"] > 1) + int(sum(f[1] for f in flags)),
        "seconds": time.perf_counter() - t0,
    }


def summarise(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    design = DESIGNS[name]
    lr = np.array([r["lr"] for r in rows])
    stars = np.array([r["stars"] for r in rows])
    mc, reps = stars.shape
    out: Dict[str, Any] = {k: v for k, v in design.items()}
    out["mu"] = list(design["mu"])
    out["n_fits"] = int(mc * (reps + 1))
    out["floored"] = int(sum(r["floored"] for r in rows))
    out["multiple_maxima"] = int(sum(r["multiple"] for r in rows))
    out["seconds_per_fit_pair"] = float(
        sum(r["seconds"] for r in rows) / (mc * (reps + 1))
    )
    out["lr_raw_min"] = float(min(r["lr_raw_min"] for r in rows))
    out["lr_mean"] = float(lr.mean())
    out["lr_quantiles_90_95_99"] = [
        float(q) for q in np.quantile(lr, [0.9, 0.95, 0.99])
    ]
    for level in LEVELS:
        tag = f"{int(round(100 * level))}"
        if reps == 1:
            crit = float(np.quantile(stars[:, 0], 1.0 - level))
            rej = lr > crit
            out[f"warp_critical_{tag}"] = crit
        else:
            pval = (1.0 + (stars >= lr[:, None]).sum(axis=1)) / (reps + 1.0)
            rej = pval <= level + 1e-12
            # the same samples by the warp-speed device (first replicate)
            crit = float(np.quantile(stars[:, 0], 1.0 - level))
            out[f"warp_reject_{tag}"] = float((lr > crit).mean())
        out[f"reject_{tag}"] = float(rej.mean())
        out[f"reject_{tag}_mc_se"] = float(np.sqrt(level * (1 - level) / mc))
        for df in (1, 3):
            out[f"chi2_df{df}_reject_{tag}"] = float(
                (lr > stats.chi2.ppf(1.0 - level, df)).mean()
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--scale", type=float, default=1.0, help="shrink mc (trial)")
    parser.add_argument("--only", default="", help="rerun one design, keep the rest")
    parser.add_argument(
        "--out", default=str(Path(__file__).with_name("mswitch_lrtest_mc.json"))
    )
    args = parser.parse_args()
    names = [args.only] if args.only else ORDER
    tasks = [
        (name, i)
        for name in names
        for i in range(max(int(DESIGNS[name]["mc"] * args.scale), 4))
    ]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(one, tasks, chunksize=2))
    import statspai

    result = {
        "generator": Path(__file__).name,
        "statspai_version": statspai.__version__,
        "base_seed": BASE_SEED,
        "scale": args.scale,
        "designs": {
            name: summarise(name, [r for r in rows if r["design"] == name])
            for name in names
        },
    }
    if args.only:
        old = json.loads(Path(args.out).read_text(encoding="utf-8"))
        old["designs"].update(result["designs"])
        result = old
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
