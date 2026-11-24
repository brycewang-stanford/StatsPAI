"""``sp.winsor`` percentiles against Stata ``_pctile`` and R ``farr``.

Two references, both from real runs.

Stata 18 (``_fixtures/_generate_winsor_pctile_Stata.do``): ``_pctile``,
the routine ``winsor2`` calls, on ``x = 1..n`` for ``n = 2..120`` at every
whole percentile and at 0.5, 2.5, 97.5 and 99.5, and ``winsor2`` itself
on ``1..100`` at ``cuts(7 93)`` and ``cuts(29 57)``.  With ``P = n p / 100``
the percentile is the mean of two order statistics when ``P`` is a whole
number.  ``np.percentile(method="averaged_inverted_cdf")`` forms
``n * (p / 100)`` instead, 7.000000000000001 for ``n = 100, p = 7``, and
returned the 8th value where Stata returns 7.5.  The values are whole
numbers or halves, so the comparison is exact equality.

R 4.5.2 with farr 1.0.9000 (``_fixtures/_generate_winsor_infinite_R.R``):
``farr::winsorize`` and ``farr::truncate`` on vectors holding ``-Inf`` and
``Inf``, which R counts as the smallest and largest values
(``include_infinite=True``).  Cut-offs are order statistics or the mean of
two, so the comparison is exact equality as well.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
_R = json.loads((_FIX / "winsor_infinite_R.json").read_text(encoding="utf-8"))
_CASES = sorted(k for k in _R if k != "versions")


def _num(values):
    """R's "Inf" / "-Inf" / "NA" / "NaN" strings back to floats."""
    return np.array([np.nan if v == "NA" else float(v) for v in values], dtype=float)


def _cuts(case):
    # round(): 100 * 0.025 is 2.5 but 100 * 0.07 is not 7
    return (round(100 * case["lo"], 10), round(100 * case["hi"], 10))


def test_percentile_is_stata_pctile_on_the_whole_grid():
    grid = pd.read_csv(_FIX / "winsor_pctile_Stata.csv")
    assert len(grid) == 119 * 103
    wrong = []
    for n, rows in grid.groupby("n"):
        df = pd.DataFrame({"x": np.arange(1.0, n + 1)})
        for p, want in zip(rows["p"], rows["pctile"]):
            # cuts=(p, 100): the minimum of the result is the p-th percentile
            got = sp.winsor(df, ["x"], cuts=(p, 100))["x_w"].min()
            if got != want:
                wrong.append((n, p, got, want))
    assert not wrong, wrong[:5]


def test_winsor2_where_n_p_over_100_is_whole():
    ref = pd.read_csv(_FIX / "winsor_pctile_winsor2_Stata.csv")
    df = ref[["x"]]
    a = sp.winsor(df, ["x"], cuts=(7, 93))["x_w"]
    b = sp.winsor(df, ["x"], cuts=(29, 57))["x_w"]
    c = sp.winsor(df, ["x"], cuts=(7, 93), trim=True)["x_tr"]
    pd.testing.assert_series_equal(a, ref["x_a"], check_names=False)
    pd.testing.assert_series_equal(b, ref["x_b"], check_names=False)
    pd.testing.assert_series_equal(c, ref["x_c"], check_names=False)
    assert (a.min(), a.max()) == (7.5, 93.5)  # 7 and 8, 93 and 94 averaged


@pytest.mark.parametrize("name", _CASES)
@pytest.mark.parametrize("replace", [False, True])
def test_include_infinite_equals_farr(name, replace):
    case = _R[name]
    df = pd.DataFrame({"x": _num(case["x"])})
    for key, trim in (("w", False), ("t", True)):
        out = sp.winsor(
            df,
            ["x"],
            cuts=_cuts(case),
            trim=trim,
            replace=replace,
            include_infinite=True,
        )
        col = "x" if replace else "x_tr" if trim else "x_w"
        np.testing.assert_array_equal(out[col].to_numpy(), _num(case[key]))


@pytest.mark.parametrize("trim", [False, True])
def test_include_infinite_by_group_equals_farr_group_by_group(trim):
    names = [k for k in _CASES if (_R[k]["lo"], _R[k]["hi"]) == (0.01, 0.99)]
    assert len(names) >= 4
    df = pd.concat(
        [pd.DataFrame({"g": k, "x": _num(_R[k]["x"])}) for k in names],
        ignore_index=True,
    )
    want = np.concatenate([_num(_R[k]["t" if trim else "w"]) for k in names])
    out = sp.winsor(df, ["x"], by="g", trim=trim, replace=True, include_infinite=True)
    np.testing.assert_array_equal(out["x"].to_numpy(), want)


def test_include_infinite_changes_nothing_without_infinite_values():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    x[::37] = np.nan
    df = pd.DataFrame({"x": x, "g": np.arange(500) % 3})
    for kwargs in ({}, {"by": "g"}, {"trim": True}, {"cuts": (2.5, 97.5)}):
        pd.testing.assert_frame_equal(
            sp.winsor(df, ["x"], include_infinite=True, **kwargs),
            sp.winsor(df, ["x"], **kwargs),
        )


def test_include_infinite_needs_the_stata_percentile():
    df = pd.DataFrame({"x": [1.0, 2.0, np.inf]})
    with pytest.raises(sp.MethodIncompatibility, match="method='stata'"):
        sp.winsor(df, ["x"], method="linear", include_infinite=True)


def test_include_infinite_does_not_warn():
    import warnings

    df = pd.DataFrame({"x": [-np.inf, *range(100), np.inf]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sp.winsor(df, ["x"], replace=True, include_infinite=True)
    assert (out["x"].min(), out["x"].max()) == (0.0, 99.0)


def test_transform_data_winsor_op_passes_include_infinite():
    from statspai.agent.workflow_tools import _apply_transform

    case = _R["both_tails"]
    df = pd.DataFrame({"x": _num(case["x"])})
    step = {"op": "winsor", "columns": ["x"], "include_infinite": True}
    out = _apply_transform(df, step)
    np.testing.assert_array_equal(out["x"].to_numpy(), _num(case["w"]))
