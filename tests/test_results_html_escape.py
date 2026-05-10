"""Regression tests for rich HTML result display escaping."""

import pandas as pd

from statspai.core.results import CausalResult, EconometricResults


def test_econometric_results_repr_html_escapes_display_strings():
    term = 'x<script>alert("term")</script>'
    result = EconometricResults(
        params=pd.Series([1.0], index=[term]),
        std_errors=pd.Series([0.1], index=[term]),
        model_info={
            "model_type": "<b>OLS</b>",
            "method": '<img src=x onerror="alert(1)">',
        },
        data_info={
            "dependent_var": "<y>",
            "nobs": "unknown",
            "df_resid": 10,
        },
        diagnostics={
            "unsafe<script>": "<svg/onload=alert(1)>",
            "R-squared": 0.5,
        },
    )

    html = result._repr_html_()

    assert term not in html
    assert "<b>OLS</b>" not in html
    assert "<img src=x" not in html
    assert "<svg/onload" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;b&gt;OLS&lt;/b&gt;" in html
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html
    assert "&lt;svg/onload=alert(1)&gt;" in html
    assert "unknown" in html


def test_causal_result_repr_html_escapes_matching_detail_and_metadata():
    detail = pd.DataFrame(
        {
            "variable": ['cov<script>alert("detail")</script>'],
            "mean_treated": [1.0],
            "mean_control": [0.5],
            "smd": [0.2],
        }
    )
    result = CausalResult(
        method="<b>Matching</b>",
        estimand="<ATE>",
        estimate=1.0,
        se=0.1,
        pvalue=0.02,
        ci=(0.8, 1.2),
        alpha=0.05,
        n_obs="unknown",
        detail=detail,
        model_info={
            "distance": "<logit>",
            "method": "<nearest>",
            "n_treated": "<10>",
            "n_control": 5,
        },
    )

    html = result._repr_html_()

    assert "<b>Matching</b>" not in html
    assert '<script>alert("detail")</script>' not in html
    assert "&lt;b&gt;Matching&lt;/b&gt;" in html
    assert "&lt;ATE&gt;" in html
    assert "cov&lt;script&gt;" in html
    assert "&lt;logit&gt;" in html
    assert "&lt;nearest&gt;" in html
    assert "&lt;10&gt;" in html
    assert "unknown" in html
