"""Write the CausalAgentBench data files and their SHA-256 manifest.

Every trial copies its prompt's file(s) from ``tests/agent_bench/data/`` into
a fresh sandbox. The files are part of the pre-registered design: their
hashes go into ``data/MANIFEST.json`` and from there into the
pre-registration lock, so a changed byte is detected before any trial runs.

Sources
-------
* ``card.csv``, ``mpdta.csv``, ``lee.csv``, ``basque.csv``, ``nsw_dw.csv``,
  ``california.csv``: the datasets shipped in ``statspai.datasets`` (some
  are simulated replicas; the benchmark scores the analysis of the file
  given, not a claim about the original study).
* The others are drawn here from documented, seeded data-generating
  processes, so several tasks also have a known true value (``TRUTHS``).

``trade.csv`` (200,000 rows) is not committed; it is regenerated
byte-for-byte from its seed and checked against the manifest.

Usage::

    python tests/agent_bench/runners/build_data.py          # write
    python tests/agent_bench/runners/build_data.py --check  # verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
MANIFEST = DATA / "MANIFEST.json"
#: Files regenerated rather than committed (size).
NOT_COMMITTED = ("trade.csv",)

#: True parameters of the simulated designs (used by build_golds.py).
TRUTHS = {
    "synth_survival.csv": {"beta_x1": 0.5, "beta_x2": -0.6, "beta_treat": -0.4},
    "panel.csv": {"beta_intercept": 2.0, "beta_x1": 1.5, "icc": 2.25 / 3.25},
    "logit_panel.csv": {"beta_intercept": -0.5, "beta_x1": 0.6},
    "sfa.csv": {"beta_lnk": 0.6, "beta_lnl": 0.4},
    "mediation.csv": {"acme": 0.2, "ade": 0.3},
    "trade.csv": {"beta_x1": 2.0, "beta_x2": -1.5},
}


def _sp():
    sys.path.insert(0, str(HERE.parents[2] / "src"))
    import statspai as sp

    return sp


def _shipped(
    name: str, rename: Dict[str, str] | None = None
) -> Callable[[], pd.DataFrame]:
    def make() -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = getattr(_sp().datasets, name)()
        df = pd.DataFrame(df)
        df.attrs = {}
        return df.rename(columns=rename or {})

    return make


def _synth_survival() -> pd.DataFrame:
    rng = np.random.default_rng(101)
    n = 800
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    treat = rng.integers(0, 2, n)
    rate = 0.1 * np.exp(0.5 * x1 - 0.6 * x2 - 0.4 * treat)
    t_event = rng.exponential(1 / rate)
    t_cens = rng.uniform(0, 25, n)
    return pd.DataFrame(
        {
            "time": np.minimum(t_event, t_cens).round(6),
            "event": (t_event <= t_cens).astype(int),
            "x1": x1.round(6),
            "x2": x2.round(6),
            "treat": treat,
        }
    )


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(102)
    groups, per = 40, 25
    gid = np.repeat(np.arange(1, groups + 1), per)
    u = rng.normal(0, 1.5, groups)[gid - 1]
    x1 = rng.normal(size=groups * per)
    y = 2.0 + 1.5 * x1 + u + rng.normal(size=groups * per)
    return pd.DataFrame({"gid": gid, "x1": x1.round(6), "y": y.round(6)})


def _logit_panel() -> pd.DataFrame:
    rng = np.random.default_rng(103)
    groups, per = 60, 20
    gid = np.repeat(np.arange(1, groups + 1), per)
    u = rng.normal(0, 1.0, groups)[gid - 1]
    x1 = rng.normal(size=groups * per)
    p = 1 / (1 + np.exp(-(-0.5 + 0.6 * x1 + u)))
    y = (rng.uniform(size=groups * per) < p).astype(int)
    return pd.DataFrame({"gid": gid, "x1": x1.round(6), "y": y})


def _sfa() -> pd.DataFrame:
    rng = np.random.default_rng(104)
    n = 600
    lnk, lnl = rng.normal(2, 0.5, n), rng.normal(1.5, 0.5, n)
    v = rng.normal(0, 0.2, n)
    u = np.abs(rng.normal(0, 0.35, n))
    lny = 1.0 + 0.6 * lnk + 0.4 * lnl + v - u
    return pd.DataFrame({"lny": lny.round(6), "lnk": lnk.round(6), "lnl": lnl.round(6)})


def _wage_gap() -> pd.DataFrame:
    rng = np.random.default_rng(105)
    n = 2000
    female = rng.integers(0, 2, n)
    educ = np.clip(rng.normal(13 - 0.4 * female, 2.5), 6, 20).round()
    exper = np.clip(rng.normal(18 - 2.0 * female, 8), 0, 45).round()
    log_wage = (
        1.4
        + 0.08 * educ
        + 0.02 * exper
        - female * (0.12 + 0.004 * exper)
        + rng.normal(0, 0.4, n)
    )
    return pd.DataFrame(
        {"log_wage": log_wage.round(6), "educ": educ, "exper": exper, "female": female}
    )


def _mediation() -> pd.DataFrame:
    rng = np.random.default_rng(106)
    n = 1200
    x = rng.normal(size=n)
    treat = rng.integers(0, 2, n)
    m = 0.5 + 0.4 * treat + 0.3 * x + rng.normal(size=n)
    y = 1.0 + 0.3 * treat + 0.5 * m + 0.2 * x + rng.normal(size=n)
    return pd.DataFrame(
        {"y": y.round(6), "m": m.round(6), "treat": treat, "x": x.round(6)}
    )


def _trade() -> pd.DataFrame:
    rng = np.random.default_rng(107)
    firms, years = 20000, 10
    firm = np.repeat(np.arange(firms), years)
    year = np.tile(np.arange(2010, 2010 + years), firms)
    a_f = rng.normal(size=firms)[firm]
    a_t = rng.normal(0, 0.5, years)[year - 2010]
    x1 = rng.normal(size=firms * years) + 0.5 * a_f
    x2 = rng.normal(size=firms * years) - 0.3 * a_t
    e = rng.normal(0, 0.5, firms)[firm] + rng.normal(size=firms * years)
    y = 2.0 * x1 - 1.5 * x2 + a_f + a_t + e
    return pd.DataFrame(
        {
            "firm": firm,
            "year": year,
            "x1": x1.round(6),
            "x2": x2.round(6),
            "y": y.round(6),
        }
    )


def _macro() -> pd.DataFrame:
    rng = np.random.default_rng(108)
    T, burn = 300, 100
    A1 = np.array([[0.5, 0.1], [0.3, 0.4]])
    A2 = np.array([[-0.1, 0.0], [0.1, -0.1]])
    c = np.array([0.2, 0.1])
    chol = np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 0.8]]))
    y = np.zeros((T + burn, 2))
    for t in range(2, T + burn):
        y[t] = c + A1 @ y[t - 1] + A2 @ y[t - 2] + chol @ rng.normal(size=2)
    y = y[burn:]
    return pd.DataFrame(
        {"t": np.arange(1, T + 1), "y1": y[:, 0].round(6), "y2": y[:, 1].round(6)}
    )


FILES: Dict[str, Callable[[], pd.DataFrame]] = {
    "card.csv": _shipped("card_1995"),
    "mpdta.csv": _shipped("mpdta"),
    "lee.csv": _shipped("lee_2008_senate", {"x": "margin", "y": "voteshare_next"}),
    "basque.csv": _shipped("basque_terrorism"),
    "nsw_dw.csv": _shipped("nsw_dw"),
    "california.csv": _shipped("california_prop99"),
    "synth_survival.csv": _synth_survival,
    "panel.csv": _panel,
    "logit_panel.csv": _logit_panel,
    "sfa.csv": _sfa,
    "wage_gap.csv": _wage_gap,
    "mediation.csv": _mediation,
    "trade.csv": _trade,
    "macro.csv": _macro,
}


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, lineterminator="\n", float_format="%.10g").encode(
        "utf-8"
    )


def build(check: bool) -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    old = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    manifest = {}
    bad = []
    for name, make in FILES.items():
        df = make()
        raw = _csv_bytes(df)
        digest = hashlib.sha256(raw).hexdigest()
        manifest[name] = {
            "sha256": digest,
            "rows": int(len(df)),
            "columns": list(map(str, df.columns)),
            "committed": name not in NOT_COMMITTED,
        }
        if check:
            want = (old.get(name) or {}).get("sha256")
            if want != digest:
                bad.append(name)
        else:
            (DATA / name).write_bytes(raw)
    if check:
        for name in bad:
            print(f"DRIFT {name}: regenerated bytes differ from MANIFEST.json")
        print("OK" if not bad else f"{len(bad)} file(s) drifted")
        return 1 if bad else 0
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} files and {MANIFEST.name}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    sys.exit(build(ap.parse_args().check))
