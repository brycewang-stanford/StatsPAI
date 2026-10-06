"""``sp.read_data`` reads a workbook written with backslashes in its
archive member names (as EViews does), which pandas alone refuses."""

import io
import zipfile

import pandas as pd
import pytest

import statspai as sp


def _eviews_style(tmp_path):
    good = tmp_path / "good.xlsx"
    frame = pd.DataFrame(
        {"date": ["1990Q1", "1990Q2", "1990Q3"], "gdp": [1.5, 2.5, 0.25]}
    )
    frame.to_excel(good, index=False)
    moved = {"xl/worksheets/sheet1.xml", "xl/sharedStrings.xml", "xl/workbook.xml"}
    empty = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/'
        '2006/main"><sheetData/></worksheet>'
    )
    bad = tmp_path / "eviews.xlsx"
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(bad, "w") as out:
        present = set(src.namelist())
        assert moved <= present
        for name in src.namelist():
            if name in moved:
                out.writestr(name.replace("/", "\\"), src.read(name))
            else:
                out.writestr(name, src.read(name))
        # the placeholder sheets EViews leaves under the regular names
        out.writestr("xl/worksheets/sheet2.xml", empty)
        out.writestr("xl/worksheets/sheet3.xml", empty)
    return frame, good, bad


def test_backslash_member_names_are_repaired(tmp_path):
    frame, good, bad = _eviews_style(tmp_path)
    with pytest.raises(Exception):
        pd.read_excel(bad)  # the failure being worked around
    got = sp.read_data(str(bad))
    pd.testing.assert_frame_equal(got, frame)
    pd.testing.assert_frame_equal(sp.read_data(str(good)), frame)


def test_file_on_disk_is_not_modified(tmp_path):
    _, _, bad = _eviews_style(tmp_path)
    before = bad.read_bytes()
    sp.read_data(str(bad))
    assert bad.read_bytes() == before
    with zipfile.ZipFile(io.BytesIO(before)) as z:
        assert any("\\" in n for n in z.namelist())
