"""Chapter 15 of Stock & Watson, *Introduction to Econometrics* (4th ed.):
forecasting US GDP growth with autoregressions and the term spread.

The chapter's replication program is written in RATS and ships with its
output (``chapter15/ch15_replication_files.out``). The numbers below are
copied from that file; the comment on each gives where it is printed. They
cover an AR(1), an AR(2), ADL(2,1) and ADL(2,2) with heteroskedasticity-
robust standard errors (RATS ``linreg(robust)``, which is HC0), their
forecasts of 2017:Q4, the Granger test, the BIC / AIC table, the augmented
Dickey-Fuller regression, the QLR break statistic and the pseudo
out-of-sample forecast errors.

The data are not redistributed here. Set ``STATSPAI_SW4E_DIR`` to the
folder holding ``SW_4E_Replication_Data`` to run this; skipped otherwise.
"""

import os
import re
import warnings
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

ROOT = os.environ.get("STATSPAI_SW4E_DIR")
pytestmark = pytest.mark.skipif(
    not ROOT or not Path(ROOT).is_dir(),
    reason="set STATSPAI_SW4E_DIR to the Stock & Watson 4E replication files",
)

SAMPLE = ("1962Q1", "2017Q3")
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _read_xlsx(path):
    """First sheet of a workbook. (openpyxl rejects these two files.)"""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            tag = f"{{{_NS['m']}}}t"
            for si in root.findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(tag)))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.iter(f"{{{_NS['m']}}}row"):
        cells = {}
        for c in row.findall("m:c", _NS):
            idx = 0
            for ch in re.match(r"[A-Z]+", c.get("r")).group(0):
                idx = idx * 26 + ord(ch) - 64
            v = c.find("m:v", _NS)
            if v is not None:
                text = c.get("t") == "s"
                cells[idx - 1] = shared[int(v.text)] if text else float(v.text)
        rows.append(cells)
    width = max(max(r) for r in rows if r) + 1
    table = [[r.get(j) for j in range(width)] for r in rows]
    return pd.DataFrame(table[1:], columns=table[0])


@pytest.fixture(scope="module")
def macro():
    folder = next(Path(ROOT).rglob("us_macro_quarterly.xlsx")).parent
    q = _read_xlsx(folder / "us_macro_quarterly.xlsx")
    m = _read_xlsx(folder / "us_macro_monthly.xlsx")
    for frame in (q, m):
        date = pd.to_datetime("1899-12-30") + pd.to_timedelta(frame.freq, unit="D")
        frame["quarter"] = date.dt.to_period("Q")
    rates = m.groupby("quarter")[["GS10", "TB3MS"]].mean()  # compact=average
    out = q.set_index("quarter").join(rates)
    out["y"] = np.log(out.GDPC1)
    out["ygrowth"] = 400 * out.y.diff()
    out["rspread"] = out.GS10 - out.TB3MS
    return out[["y", "ygrowth", "rspread"]]


def _fit(macro, x=None, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.ardl(macro, "ygrowth", x, sample=SAMPLE, vce="hc0", **kw)


# (coefficients, standard errors) as printed, and the forecast of 2017:Q4
PUBLISHED = {
    # "AR(1) Forecast of GDP Growth"
    (1, None): (
        [1.9500623574, 0.3408366195],
        [0.3223718808, 0.0730091500],
        3.00908,
    ),
    # "AR(2) Forecast of GDP Growth"
    (2, None): (
        [1.6027511368, 0.2792349114, 0.1767336051],
        [0.3691031958, 0.0763803458, 0.0767317752],
        3.00303,
    ),
    # "ADL(2,2) Forecast of GDP Grow"
    (2, 2): (
        [0.943747185, 0.245801239, 0.175741767, -0.128684428, 0.617507725],
        [0.456779207, 0.075134697, 0.075070496, 0.415056016, 0.422871986],
        2.92992,
    ),
}


@pytest.mark.parametrize("order", sorted(PUBLISHED, key=str))
def test_regressions_and_forecasts(macro, order):
    p, q = order
    coef, se, forecast = PUBLISHED[order]
    res = _fit(macro, "rspread" if q else None, lags=p, x_lags=q or 1)
    assert res.nobs == 223  # "Usable Observations"
    # RATS prints 9 to 10 decimals
    np.testing.assert_allclose(res.params.to_numpy(), coef, atol=6e-10)
    np.testing.assert_allclose(res.std_errors.to_numpy(), se, atol=6e-10)
    assert round(res.forecast()["forecast"].item(), 5) == forecast


def test_adl21_forecast(macro):
    # "ADL(2,1) Forecast of GDP Grow      2.85412"
    res = _fit(macro, "rspread", lags=2, x_lags=1)
    assert round(res.forecast()["forecast"].item(), 5) == 2.85412


def test_granger_statistic(macro):
    # "Chi-Squared(2)=      8.121102 or F(2,*)=      4.06055"
    out = _fit(macro, "rspread", lags=2, x_lags=2).granger()
    assert round(out["chi2"], 6) == 8.121102
    assert round(out["statistic"], 5) == 4.06055


def test_information_criteria_tables(macro):
    # the two "p SSR(p)/T ln(SSR(p)/T) (p+1)ln(T)/T BIC(p) AIC(p) R**2" reports
    ar = {  # p: (BIC, AIC, R2)
        0: (2.373, 2.358, 0.000), 1: (2.273, 2.242, 0.117), 2: (2.265, 2.219, 0.145),
        3: (2.289, 2.228, 0.145), 4: (2.310, 2.233, 0.149), 5: (2.319, 2.227, 0.161),
        6: (2.342, 2.235, 0.162),
    }  # fmt: skip
    adl = {
        0: (2.373, 2.358, 0.000), 1: (2.274, 2.228, 0.138), 2: (2.272, 2.196, 0.180),
        3: (2.310, 2.203, 0.188), 4: (2.346, 2.208, 0.199), 5: (2.380, 2.212, 0.210),
        6: (2.424, 2.225, 0.214),
    }  # fmt: skip
    for x, published, chosen in ((None, ar, 2), ("rspread", adl, 2)):
        res = _fit(macro, x, lags="bic", x_lags="same", max_lags=6)
        assert res.lags == chosen
        for p, (bic, aic, r2) in published.items():
            row = res.ic_table.loc[p]
            assert abs(row["bic"] - bic) < 5.1e-4
            assert abs(row["aic"] - aic) < 5.1e-4
            assert abs(row["r2"] - r2) < 5.1e-4


def test_pseudo_out_of_sample_errors(macro):
    # "Statistics on Series FCST_ERR_AR1" / "..._ERR2_AR1" (44 observations,
    # 2007:01 to 2017:04) and the AR(2) pair
    for p, mean, msfe in ((1, -1.093844, 6.738487), (2, -0.879367, 6.359684)):
        out = _fit(macro, lags=p).poos("2007Q1")
        assert out.attrs["n_forecasts"] == 44
        assert round(out.attrs["bias"], 6) == mean
        assert round(out.attrs["rmsfe"] ** 2, 6) == msfe


def test_dickey_fuller_regression(macro):
    # "linreg dy ... # constant trend y{1} dy{1 to 2}":
    # Y{1}  -0.019306733  0.009878802  -1.95436
    series = macro.loc["1961Q2":"2017Q3", "y"]
    res = sp.unitroot(series, test="adf", trend="ct", lags=2)
    assert res.n_obs == 223
    assert round(res.rho, 9) == -0.019306733
    assert round(res.se, 9) == 0.009878802
    assert round(res.statistic, 5) == -1.95436
    assert not res.reject


def test_qlr_statistic(macro):
    # "Maximum Value is 6.47456237664 at 1980:04"; the program scales the
    # HC0 Wald by ndf/nobs, which is the HC1 statistic
    data = macro.assign(
        yg1=macro.ygrowth.shift(1), yg2=macro.ygrowth.shift(2),
        rs1=macro.rspread.shift(1), rs2=macro.rspread.shift(2),
    ).loc[SAMPLE[0] : SAMPLE[1]]  # fmt: skip
    res = sp.structural_break(
        data, y="ygrowth", x=["yg1", "yg2", "rs1", "rs2"], method="sup-f",
        break_vars=["const", "rs1", "rs2"], vce="hc1",
    )  # fmt: skip
    np.testing.assert_allclose(res.f_stats, 6.47456237664, rtol=1e-10)
    assert data.index[res.sup_break - 1] == pd.Period("1980Q4")
    assert res.p_values < 0.01  # above the 1% critical value of 6.02
