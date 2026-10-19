"""``booktabs=True`` on the LaTeX exports (2026-10-19 audit).

The default keeps plain ``\\hline`` rules (compiles without extra
packages); ``booktabs=True`` swaps them for ``\\toprule`` / ``\\midrule`` /
``\\bottomrule`` and changes nothing else.
"""

from __future__ import annotations

import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

RULES = ("\\hline\\hline", "\\hline", "\\toprule", "\\midrule", "\\bottomrule")


@pytest.fixture(scope="module")
def results():
    r = np.random.default_rng(0)
    df = pd.DataFrame({"x": r.normal(size=200)})
    df["d"] = (r.random(200) < 0.5).astype(int)
    df["y"] = df.x + df.d + r.normal(size=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = sp.regress("y ~ x + d", data=df)
        ipw = sp.ipw(df, y="y", treat="d", covariates=["x"], n_bootstrap=5)
    return {
        "regtable": (lambda **k: sp.regtable(ols, ols).to_latex(**k)),
        "econometric": (lambda **k: ols.to_latex(**k)),
        "causal": (lambda **k: ipw.to_latex(**k)),
    }


def _without_rules(tex: str) -> list:
    return [ln for ln in tex.split("\n") if ln.strip() not in RULES]


@pytest.mark.parametrize("kind", ["regtable", "econometric", "causal"])
def test_booktabs_swaps_only_the_rules(results, kind):
    plain = results[kind]()
    booked = results[kind](booktabs=True)
    assert "\\hline" in plain and "\\toprule" not in plain
    assert "\\hline" not in booked
    assert booked.count("\\toprule") == 1 and booked.count("\\bottomrule") == 1
    assert booked.index("\\toprule") < booked.index("\\bottomrule")
    assert _without_rules(plain) == _without_rules(booked)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
@pytest.mark.parametrize("kind", ["regtable", "causal"])
def test_booktabs_output_compiles(results, kind, tmp_path):
    doc = (
        "\\documentclass{article}\\usepackage{booktabs}\\usepackage{threeparttable}"
        "\\begin{document}" + results[kind](booktabs=True) + "\\end{document}"
    )
    (tmp_path / "t.tex").write_text(doc, encoding="utf-8")
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout[-2000:]
