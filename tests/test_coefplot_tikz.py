"""Tests for coefficient-plot export.

``sp.coefplot`` returns a Matplotlib ``(fig, ax)`` — PNG/PDF come free via
``fig.savefig``. ``sp.coefplot_tikz`` is the LaTeX-native counterpart: a
``pgfplots`` forest plot as editable source. These tests lock the pgfplots
structure and the data→coordinate mapping.
"""

import re

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def models():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["y"] = 1.0 + 2.0 * df["x"] - 0.5 * df["z"] + rng.normal(size=n)
    df["y2"] = 0.5 + 1.5 * df["x"] + rng.normal(size=n)
    m1 = sp.regress("y ~ x + z", data=df)
    m2 = sp.regress("y2 ~ x", data=df)
    return m1, m2


class TestStructure:

    def test_returns_tikzpicture(self, models):
        t = sp.coefplot_tikz(*models)
        assert "\\begin{tikzpicture}" in t
        assert "\\end{tikzpicture}" in t
        assert "\\begin{axis}[" in t
        assert "\\end{axis}" in t

    def test_error_bar_plot(self, models):
        t = sp.coefplot_tikz(*models)
        assert "error bars/.cd, x dir=both, x explicit" in t
        assert "+- (" in t  # explicit per-point error

    def test_one_legend_entry_per_model(self, models):
        t = sp.coefplot_tikz(*models, model_names=["OLS", "Alt"])
        assert t.count("\\addlegendentry{") == 2
        assert "\\addlegendentry{OLS}" in t
        assert "\\addlegendentry{Alt}" in t

    def test_zero_reference_line(self, models):
        t = sp.coefplot_tikz(*models)
        assert "\\draw[gray,dashed] (axis cs:0," in t

    def test_y_axis_reversed_like_matplotlib(self, models):
        t = sp.coefplot_tikz(*models)
        assert "y dir=reverse," in t


class TestLabels:

    def test_coef_labels_applied_and_escaped(self, models):
        t = sp.coefplot_tikz(models[0], coef_labels={"x": "Treatment_effect"})
        # underscore must be escaped for LaTeX
        assert "Treatment\\_effect" in t
        assert "Treatment_effect" not in t.replace("Treatment\\_effect", "")

    def test_title_and_xlabel(self, models):
        t = sp.coefplot_tikz(models[0], title="My plot", xlabel="Beta")
        assert "title={My plot}" in t
        assert "xlabel={Beta}" in t

    def test_variables_filter(self, models):
        t = sp.coefplot_tikz(models[0], variables=["x"])
        assert "yticklabels={x}" in t


class TestStandalone:

    def test_standalone_is_compilable_doc(self, models):
        t = sp.coefplot_tikz(models[0], standalone=True)
        assert t.startswith("\\documentclass{standalone}")
        assert "\\usepackage{pgfplots}" in t
        assert t.strip().endswith("\\end{document}")

    def test_non_standalone_is_bare_picture(self, models):
        t = sp.coefplot_tikz(models[0])
        assert "documentclass" not in t
        assert t.lstrip().startswith("\\begin{tikzpicture}")


class TestNumerics:

    def test_error_width_tracks_level(self, models):
        """A 99% interval must be wider than a 90% interval."""
        t90 = sp.coefplot_tikz(models[0], variables=["x"], level=0.90)
        t99 = sp.coefplot_tikz(models[0], variables=["x"], level=0.99)

        def first_err(t):
            return float(re.search(r"\+- \(([0-9.]+),0\)", t).group(1))

        assert first_err(t99) > first_err(t90)

    def test_point_estimate_in_coordinates(self, models):
        # known DGP: x coefficient ~ 2.0 in model 1
        t = sp.coefplot_tikz(models[0], variables=["x"])
        x_coord = float(re.search(r"\(([0-9.]+),", t).group(1))
        assert x_coord == pytest.approx(2.0, abs=0.25)

    def test_single_model_no_offset(self, models):
        t = sp.coefplot_tikz(models[0], variables=["x"])
        # single model: y position is the integer index 0 (no offset)
        assert re.search(r",0\) \+-", t)


class TestCausalResult:

    def test_accepts_causal_result(self):
        rng = np.random.default_rng(1)
        n = 200
        df = pd.DataFrame({"x": rng.normal(size=n)})
        df["y"] = 1 + 2 * df["x"] + rng.normal(size=n)
        cr = sp.qreg(df, "y ~ x")  # returns a CausalResult
        t = sp.coefplot_tikz(cr)
        assert "\\begin{tikzpicture}" in t
        assert "\\addplot+" in t


class TestMatplotlibSavefig:

    def test_savefig_png_pdf(self, models, tmp_path):
        plt = pytest.importorskip("matplotlib.pyplot")  # noqa: F841
        fig, ax = sp.coefplot(*models)
        for ext in ("png", "pdf"):
            p = tmp_path / f"cp.{ext}"
            fig.savefig(str(p))
            assert p.stat().st_size > 500
