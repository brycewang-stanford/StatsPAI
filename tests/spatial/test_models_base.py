import numpy as np
import pandas as pd
from statspai.spatial.models._base import build_design_matrix


def test_build_design_matrix_adds_intercept():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x1": [4.0, 5.0, 6.0]})
    y, X, names = build_design_matrix("y ~ x1", df)
    assert y.shape == (3,)
    assert X.shape == (3, 2)
    assert names[0] in {"Intercept", "(Intercept)", "1"}  # formulaic convention
    assert "x1" in names


def test_build_design_matrix_no_intercept():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x1": [4.0, 5.0, 6.0]})
    y, X, names = build_design_matrix("y ~ x1 - 1", df)
    assert X.shape == (3, 1)
    assert names == ["x1"]
