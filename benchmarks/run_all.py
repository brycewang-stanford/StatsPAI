"""Run all benchmarks and write RESULTS.md + results.json.

Usage:
    python benchmarks/run_all.py --quick   (small N, ~ 30s)
    python benchmarks/run_all.py --full    (up to n=500k, minutes)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

# Local imports
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_regression
import bench_hdfe
import bench_did


def main(mode: str = 'quick') -> None:
    print(f"\n== StatsPAI benchmark ({mode}) ==\n")

    import statspai as sp

    meta = {
        'statspai_version': sp.__version__,
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'mode': mode,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }
    print(f"StatsPAI {meta['statspai_version']} · Python "
          f"{meta['python']} · {meta['platform']}\n")

    if mode == 'quick':
        regression_sizes = (1_000, 10_000)
        hdfe_sizes = [
            {'n_units': 500, 'n_periods': 10},
            {'n_units': 2_000, 'n_periods': 10},
        ]
        did_sizes = (200, 1_000)
    else:
        regression_sizes = (1_000, 10_000, 100_000, 500_000)
        hdfe_sizes = [
            {'n_units': 500, 'n_periods': 10},
            {'n_units': 2_000, 'n_periods': 10},
            {'n_units': 10_000, 'n_periods': 10},
            {'n_units': 50_000, 'n_periods': 10},
        ]
        did_sizes = (200, 1_000, 5_000, 20_000)

    print("-- Regression --")
    reg_res = bench_regression.run(sizes=regression_sizes)
    print()

    print("-- HDFE --")
    hdfe_res = bench_hdfe.run(sizes=hdfe_sizes)
    print()

    print("-- DID --")
    did_res = bench_did.run(sizes=did_sizes)
    print()

    all_results = {
        'meta': meta,
        'regression': reg_res,
        'hdfe': hdfe_res,
        'did': did_res,
    }

    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    _render_markdown(all_results, os.path.join(HERE, 'RESULTS.md'))
    print(f"\nResults written to {HERE}/RESULTS.md and results.json")


def _render_markdown(results: dict, path: str) -> None:
    meta = results['meta']
    lines = []
    lines.append("# StatsPAI Benchmark Results")
    lines.append("")
    lines.append(f"- **StatsPAI**: {meta['statspai_version']}")
    lines.append(f"- **Python**: {meta['python']}")
    lines.append(f"- **Platform**: {meta['platform']}")
    lines.append(f"- **Mode**: {meta['mode']}")
    lines.append(f"- **Timestamp (UTC)**: {meta['timestamp_utc']}")
    lines.append("")

    # Regression panel
    reg = results['regression']
    lines.append(f"## {reg['name']}")
    lines.append("")
    if reg['has_statsmodels']:
        lines.append("| n | sp.regress | statsmodels | vs statsmodels |")
        lines.append("|---:|---:|---:|:---:|")
        for row in reg['rows']:
            lines.append(
                f"| {row['n']:,} | {row['sp_regress_s']*1000:.1f} ms "
                f"| {row['statsmodels_s']*1000:.1f} ms "
                f"| {row.get('speedup_vs_sm','—')} |"
            )
    else:
        lines.append("| n | sp.regress |")
        lines.append("|---:|---:|")
        for row in reg['rows']:
            lines.append(
                f"| {row['n']:,} | {row['sp_regress_s']*1000:.1f} ms |"
            )
    lines.append("")

    # HDFE panel
    hdfe = results['hdfe']
    lines.append(f"## {hdfe['name']}")
    lines.append("")
    if hdfe['has_linearmodels']:
        lines.append("| units × T | n | sp.absorb_ols | linearmodels | vs lm |")
        lines.append("|---|---:|---:|---:|:---:|")
        for row in hdfe['rows']:
            lm_s = row.get('linearmodels_panelols_s')
            lm_fmt = f"{lm_s*1000:.0f} ms" if lm_s else "—"
            speedup = row.get('speedup_vs_lm', '—')
            lines.append(
                f"| {row['n_units']:,} × {row['n_periods']} "
                f"| {row['n_obs']:,} "
                f"| {row['sp_absorb_ols_s']*1000:.1f} ms "
                f"| {lm_fmt} "
                f"| {speedup} |"
            )
    else:
        lines.append("| units × T | n | sp.absorb_ols |")
        lines.append("|---|---:|---:|")
        for row in hdfe['rows']:
            lines.append(
                f"| {row['n_units']:,} × {row['n_periods']} "
                f"| {row['n_obs']:,} "
                f"| {row['sp_absorb_ols_s']*1000:.1f} ms |"
            )
    lines.append("")

    # DID panel
    did = results['did']
    lines.append(f"## {did['name']}")
    lines.append("")
    lines.append("| units | obs | CS 2021 | Wooldridge |")
    lines.append("|---:|---:|---:|---:|")
    for row in did['rows']:
        lines.append(
            f"| {row['n_units']:,} | {row['n_obs']:,} "
            f"| {row['cs2021_s']*1000:.0f} ms "
            f"| {row['wooldridge_s']*1000:.0f} ms |"
        )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Generated by `python benchmarks/run_all.py`.")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true',
                   help='Small N, ~30s total')
    p.add_argument('--full', action='store_true',
                   help='Full sweep up to n=500k')
    args = p.parse_args()
    mode = 'full' if args.full else 'quick'
    main(mode)
