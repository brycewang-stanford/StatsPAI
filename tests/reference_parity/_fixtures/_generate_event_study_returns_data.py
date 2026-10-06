"""Synthetic daily returns for test_event_study_returns_parity.py.

Twelve securities over 320 business days, a market factor and two more
factors, and one event date per security (spread over 40 days, so event
windows overlap in calendar time for some pairs and not for others). An
abnormal return of 2% is planted on the event day of the first six
securities, and the event-day variance of all of them is doubled.

Writes ``event_study_returns.csv`` (long: id, date, ret, mkt, smb, hml),
``event_study_events.csv`` (id, event_date) and, for Stata's ``estudy``,
which wants one column per security, ``event_study_returns_wide.csv``.

Run from this folder: ``python _generate_event_study_returns_data.py``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
rng = np.random.default_rng(20261007)
T, N = 320, 12
dates = pd.bdate_range("2021-01-04", periods=T)
mkt = rng.normal(0.0004, 0.010, T)
smb = rng.normal(0.0, 0.006, T)
hml = rng.normal(0.0, 0.006, T)
ids = [f"s{j:02d}" for j in range(1, N + 1)]
event_pos = 250 + rng.permutation(np.arange(0, 40, 3))[:N]
rows, wide = [], {"date": dates.strftime("%Y-%m-%d"), "mkt": mkt, "smb": smb, "hml": hml}
for j, name in enumerate(ids):
    alpha = rng.normal(0, 0.0003)
    beta, b_s, b_h = rng.uniform(0.6, 1.5), rng.normal(0, 0.4), rng.normal(0, 0.4)
    eps = rng.normal(0, rng.uniform(0.008, 0.02), T)
    eps[event_pos[j]] *= np.sqrt(2.0)
    ret = alpha + beta * mkt + b_s * smb + b_h * hml + eps
    if j < 6:
        ret[event_pos[j]] += 0.02
    wide[name] = ret
    rows.append(pd.DataFrame({"id": name, "date": wide["date"], "ret": ret}))
long = pd.concat(rows, ignore_index=True)
factors = pd.DataFrame({"date": wide["date"], "mkt": mkt, "smb": smb, "hml": hml})
long = long.merge(factors, on="date")
# two securities with a few missing returns in the estimation window
gone = long.index[(long["id"].isin(["s03", "s09"]))][[20, 21, 400 - 320 + 55]]
long.loc[gone, "ret"] = np.nan
long.round(10).to_csv(HERE / "event_study_returns.csv", index=False)
events = pd.DataFrame({"id": ids, "event_date": dates[event_pos].strftime("%Y-%m-%d")})
events.to_csv(HERE / "event_study_events.csv", index=False)
w = long.pivot(index="date", columns="id", values="ret").reset_index().merge(factors, on="date")
w.round(10).to_csv(HERE / "event_study_returns_wide.csv", index=False)
print(long.shape, events.shape, w.shape, int(long["ret"].isna().sum()))
