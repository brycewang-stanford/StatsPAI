"""Convert the data of Neusser (2016), *Time Series Econometrics*, to CSV.

The companion files (Excel, EViews and MATLAB) are not redistributed.
Download them from the author's page into one folder and run::

    python tests/external_parity/neusser_convert.py <folder>

which writes ``<folder>/_statspai/*.csv``, the inputs of
``test_neusser_time_series.py`` and ``test_neusser_quarterly_gdp.py``.

Three of the ``.xlsx`` files were written by EViews with backslashes in
the archive member names (``xl\\worksheets\\sheet1.xml``), which pandas and
openpyxl refuse; they are repacked in memory first. The ``.xls`` files
need ``xlrd``.
"""

from __future__ import annotations

import glob
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def repack(path: Path) -> io.BytesIO:
    """An .xlsx whose member names use backslashes, with forward slashes."""
    src = zipfile.ZipFile(path)
    fixed = {n.replace("\\", "/"): n for n in src.namelist() if "\\" in n}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as out:
        for name in src.namelist():
            if "\\" in name or name in fixed:
                continue
            if fixed and name.startswith("xl/worksheets/sheet"):
                continue  # empty placeholder sheets
            out.writestr(name, src.read(name))
        for good, bad in fixed.items():
            out.writestr(good, src.read(bad))
    buf.seek(0)
    return buf


def main(folder: str) -> None:
    root = Path(folder)
    out = root / "_statspai"
    out.mkdir(exist_ok=True)

    def save(frame: pd.DataFrame, name: str) -> None:
        frame.to_csv(out / name, index=False)
        print(f"{name:24s} {frame.shape}")

    bip = pd.read_excel(
        root / "BIPBeispiel" / "BIPBeispiel.xls", header=None, names=["q", "bip"]
    )
    save(bip, "bipbeispiel.csv")

    smi = pd.read_excel(root / "SwissMarketIndexTSE" / "smi.XLS").dropna()
    ret = 100 * np.diff(np.log(smi["SMI"].to_numpy(float)))
    save(pd.DataFrame({"t": np.arange(ret.size), "r": ret}), "smi_ret.csv")

    lead = pd.read_excel(
        root / "LeadingIndicatorConsumerSentiment" / "leadingindicators.xlsx"
    )
    save(lead, "leading.csv")

    us = pd.read_excel(root / "VARUS" / "VARUS.xlsx")
    us.columns = [str(c).strip() for c in us.columns]
    save(
        pd.DataFrame(
            {
                "y": 100 * np.log(us["GDPPC"]),
                "p": 100 * np.log(us["CPI"]),
                "m": 100 * np.log(us["M1"]),
                "r": us["TB3M"],
            }
        ),
        "varus_t.csv",
    )

    adv = pd.read_excel(root / "ADVERSALES" / "ADVERSALES.XLSX")
    save(np.log(adv[["adver", "sales"]]), "adv_log.csv")

    bq = pd.read_excel(repack(root / "BlanchardQuah" / "BlanchardQuah.xlsx"))
    save(bq.dropna()[["dgdp", "ur"]].reset_index(drop=True), "bq_clean.csv")

    bl = pd.read_excel(repack(root / "Blanchard" / "Blanchard.xlsx"))
    save(
        bl.dropna()[["dy", "ur", "dp", "dw", "dm"]].reset_index(drop=True),
        "bl_clean.csv",
    )

    co = pd.read_excel(repack(root / "USAcointegration" / "USAcointegration.XLSX"))
    save(
        pd.DataFrame(
            {
                "c": np.log(co["cons"]),
                "i": np.log(co["inv"]),
                "y": np.log(co["gdp"]),
                "rr": co["rr"],
            }
        ),
        "coint_t.csv",
    )

    hits = glob.glob(str(root / "BeispielQuartalschaetzung" / "*" / "data.xls"))
    if hits:
        q = pd.read_excel(hits[0], sheet_name="Quartalsweise")
        q.columns = ["date", "bip", "ip", "sent", "bip_seco"]
        save(q, "quarterly_gdp.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
