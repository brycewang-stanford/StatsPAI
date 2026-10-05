#!/usr/bin/env python3
"""Build the frozen Stata-translation holdout: data, corpus and do-file.

The corpus is written against Stata's documented grammar, one command per
feature (abbreviations, ``if`` / ``in``, weights, factor variables and
interactions, ``vce()`` spellings, prefixes, display options), on two
synthetic datasets generated here. No command comes from a paper's
replication package, and none was used to develop the translator: the
set is fixed by this file and the numbers by running ``holdout.do`` in a
licensed Stata.

    python tests/stata_translation_holdout/build_holdout.py   # data + do-file
    stata -b do holdout.do                                    # -> holdout_Stata.json

``tests/test_stata_translation_holdout.py`` then scores ``sp.from_stata``
/ ``sp.stata`` on five layers per command.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent

#: (id, dataset, command, expectation)
#: expectation: "run" (must execute and match Stata) or "refuse" (must not
#: execute: a prefix, macro or option the translator cannot honour).
CORPUS = [
    ("ols", "cross", "regress y x1 x2", "run"),
    ("ols_robust", "cross", "regress y x1 x2, vce(robust)", "run"),
    ("ols_abbrev_r", "cross", "reg y x1 x2, r", "run"),
    ("ols_cluster", "cross", "regress y x1 x2, vce(cluster g)", "run"),
    ("ols_abbrev_cl", "cross", "reg y x1 x2, cl(g)", "run"),
    ("ols_hc3", "cross", "regress y x1 x2, vce(hc3)", "run"),
    ("ols_if", "cross", "regress y x1 x2 if d == 1", "run"),
    ("ols_in", "cross", "regress y x1 x2 in 1/200", "run"),
    ("ols_if_and", "cross", "regress y x1 x2 if d == 1 & x1 > 0", "run"),
    ("ols_aweight", "cross", "regress y x1 x2 [aweight=w]", "run"),
    ("ols_pweight", "cross", "regress y x1 x2 [pweight=w]", "run"),
    ("ols_fweight", "cross", "regress y x1 x2 [fweight=fw]", "run"),
    ("ols_interaction", "cross", "regress y c.x1##c.x2", "run"),
    ("ols_factor", "cross", "regress y i.k x1", "run"),
    ("ols_factor_slope", "cross", "regress y x1 i.d#c.x2", "run"),
    ("ols_noconstant", "cross", "regress y x1 x2, noconstant", "run"),
    ("ols_level", "cross", "regress y x1 x2, level(90)", "run"),
    ("ols_beta_display", "cross", "regress y x1 x2, beta", "run"),
    ("ols_quietly", "cross", "quietly regress y x1 x2", "run"),
    ("ols_missing", "cross", "regress y x1 xm", "run"),
    ("iv_2sls", "cross", "ivregress 2sls y x1 (x2 = z)", "run"),
    ("iv_2sls_robust", "cross", "ivregress 2sls y x1 (x2 = z), vce(robust)", "run"),
    ("iv_liml", "cross", "ivregress liml y x1 (x2 = z z2)", "run"),
    ("logit", "cross", "logit yb x1 x2", "run"),
    ("probit_robust", "cross", "probit yb x1 x2, vce(robust)", "run"),
    ("poisson", "cross", "poisson cnt x1 x2", "run"),
    ("poisson_robust", "cross", "poisson cnt x1 x2, vce(robust)", "run"),
    ("qreg", "cross", "qreg y x1 x2", "run"),
    ("areg", "cross", "areg y x1 x2, absorb(g)", "run"),
    ("reghdfe_cluster", "cross", "reghdfe y x1 x2, absorb(g) vce(cluster g)", "run"),
    ("xtreg_fe", "panel", "xtreg y x, fe", "run"),
    ("xtreg_fe_cluster", "panel", "xtreg y x, fe vce(cluster id)", "run"),
    ("xtreg_re", "panel", "xtreg y x, re", "run"),
    ("refuse_unknown_option", "cross", "regress y x1 x2, weirdopt(3)", "refuse"),
    # `bootstrap:` was the refused prefix here until sp.stata learned to run
    # it by resampling the command (2026-10); its standard errors depend on
    # the draws, so it cannot be a "run" entry either
    ("refuse_rolling", "cross", "rolling, window(20): regress y x1 x2", "refuse"),
    ("refuse_by", "cross", "by d: regress y x1 x2", "refuse"),
    ("refuse_svy", "cross", "svy: regress y x1 x2", "refuse"),
    ("refuse_macro", "cross", "regress y `controls'", "refuse"),
    ("refuse_unknown_command", "cross", "frobnicate y x1 x2", "refuse"),
]


def make_data() -> None:
    rng = np.random.default_rng(20261003)
    n = 400
    x1, x2, z, z2 = (rng.normal(size=n) for _ in range(4))
    u = rng.normal(size=n)
    x2 = 0.7 * z + 0.4 * z2 + 0.5 * u + x2 * 0.6
    d = (rng.uniform(size=n) < 0.5).astype(int)
    y = 1 + 0.8 * x1 - 0.5 * x2 + 0.6 * d + u + rng.normal(size=n)
    xm = rng.normal(size=n)
    xm[::9] = np.nan
    cross = pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "z": z,
            "z2": z2,
            "d": d,
            "k": rng.integers(1, 4, size=n),
            "g": rng.integers(1, 41, size=n),
            "w": rng.uniform(0.5, 2.0, size=n),
            "fw": rng.integers(1, 4, size=n),
            "xm": xm,
            "yb": (y > np.median(y)).astype(int),
            "cnt": rng.poisson(np.exp(0.3 + 0.25 * x1 - 0.2 * x2)),
        }
    )
    cross.to_csv(HERE / "holdout_cross.csv", index=False, float_format="%.17g")
    units, periods = 60, 6
    rows = []
    for i in range(units):
        a = rng.normal()
        for t in range(1, periods + 1):
            x = rng.normal() + 0.3 * a
            rows.append((i + 1, t, 1 + 0.9 * x + a + 0.1 * t + rng.normal(), x))
    pd.DataFrame(rows, columns=["id", "t", "y", "x"]).to_csv(
        HERE / "holdout_panel.csv", index=False, float_format="%.17g"
    )


_DUMP = """
program define _dump
    args fh id comma
    if _rc {
        file write `fh' `"  ""' "`id'" `"": {"rc": "' (_rc) "}`comma'" _n
        exit
    }
    matrix HB = e(b)
    matrix HV = e(V)
    local k = colsof(HB)
    file write `fh' `"  ""' "`id'" `"": {"rc": 0, "N": "' (e(N)) `", "b": ["'
    local first 1
    forvalues j = 1/`k' {
        local se = sqrt(HV[`j', `j'])
        if !(HB[1, `j'] == 0 & `se' == 0) {
            if !`first' file write `fh' ", "
            file write `fh' %24.16e (HB[1, `j'])
            local first 0
        }
    }
    file write `fh' `"], "se": ["'
    local first 1
    forvalues j = 1/`k' {
        local se = sqrt(HV[`j', `j'])
        if !(HB[1, `j'] == 0 & `se' == 0) {
            if !`first' file write `fh' ", "
            file write `fh' %24.16e (`se')
            local first 0
        }
    }
    file write `fh' `"], "names": ["'
    local names : colfullnames HB
    local first 1
    forvalues j = 1/`k' {
        local se = sqrt(HV[`j', `j'])
        if !(HB[1, `j'] == 0 & `se' == 0) {
            local nm : word `j' of `names'
            if !`first' file write `fh' ", "
            file write `fh' (char(34)) "`nm'" (char(34))
            local first 0
        }
    }
    file write `fh' "]}`comma'" _n
end
"""


def make_do_file() -> None:
    runnable = [c for c in CORPUS if c[3] == "run"]
    lines = [
        "*! holdout.do -- Stata side of the translation holdout.",
        "*! Written by build_holdout.py; do not edit. Requires reghdfe, ftools.",
        "version 17",
        "clear all",
        "set more off",
        "* Double precision: see tests/stata_parity/option_parity/README.md.",
        "set type double",
        _DUMP,
        "tempname fh",
        'file open `fh\' using "holdout_Stata.json", write replace',
        'file write `fh\' "{" _n',
    ]
    current = None
    for cid, dataset, command, _ in runnable:
        if dataset != current:
            lines.append(
                f'import delimited "holdout_{dataset}.csv", clear asdouble case(preserve)'
            )
            if dataset == "panel":
                lines.append("xtset id t")
            current = dataset
        lines.append(f"capture noisily {command}")
        lines.append(f'_dump `fh\' "{cid}" ","')
    lines += [
        'file write `fh\' `"  "_meta": {"stata": "18 MP", "precision": "double"}"\' _n',
        'file write `fh\' "}" _n',
        "file close `fh'",
        "set type float",
    ]
    (HERE / "holdout.do").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    make_data()
    make_do_file()
    (HERE / "corpus.json").write_text(
        json.dumps(
            [
                {"id": cid, "dataset": ds, "command": cmd, "expect": exp}
                for cid, ds, cmd, exp in CORPUS
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote data, holdout.do and corpus.json ({len(CORPUS)} commands)")


if __name__ == "__main__":
    main()
