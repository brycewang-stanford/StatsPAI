"""Exports and plots for neural causal estimators."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def _small_df(seed=123, n=160):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.4 * x1 - 0.2 * x2)))
    d = rng.binomial(1, p).astype(float)
    y = 1.5 * d + np.sin(x1) + 0.3 * x2 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


def test_dragonnet_neural_exports_and_plots(tmp_path):
    import matplotlib.pyplot as plt
    import statspai as sp

    df = _small_df()
    result = sp.dragonnet(
        df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        repr_layers=(16,),
        head_layers=(8,),
        epochs=8,
        n_bootstrap=10,
        validation_fraction=0.2,
        early_stopping=True,
        patience=3,
    )

    effects = sp.neural_effects_frame(result)
    summary = sp.neural_summary_frame(result)
    training = sp.neural_training_frame(result)
    np.testing.assert_allclose(len(effects), result.n_obs)
    np.testing.assert_allclose(summary.loc[0, "estimate"], result.estimate)
    np.testing.assert_allclose(summary.loc[0, "n_obs"], result.n_obs)
    np.testing.assert_allclose(len(training), result.model_info["n_epochs_trained"])
    assert {"cate", "mu0", "mu1", "propensity", "treatment"} <= set(effects.columns)
    assert summary.loc[0, "architecture"] == "DragonNet"
    assert len(training) >= 1

    md_path = tmp_path / "dragonnet.md"
    html_path = tmp_path / "dragonnet.html"
    xlsx_path = tmp_path / "dragonnet.xlsx"
    assert "DragonNet" in result.to_markdown(str(md_path))
    assert "Unit-Level Effects" in sp.neural_causal_to_html(result, str(html_path))
    assert sp.neural_causal_to_excel(result, str(xlsx_path)) == str(xlsx_path)
    assert md_path.exists() and html_path.exists() and xlsx_path.exists()

    for plot_type in ("cate", "effects", "propensity", "loss"):
        fig, ax = result.plot(type=plot_type)
        assert fig is ax.get_figure()
        plt.close(fig)


def test_neural_functions_are_registered():
    import statspai as sp

    names = set(sp.list_functions())
    assert {"tarnet", "cfrnet", "dragonnet"} <= names
    schema = sp.function_schema("dragonnet")
    assert "validation_fraction" in schema["parameters"]["properties"]


def test_cevae_result_exports_and_plot(tmp_path):
    import matplotlib.pyplot as plt
    import statspai as sp

    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 2))
    t = rng.binomial(1, 0.5, size=80)
    y = 1.0 * t + x[:, 0] + rng.normal(scale=0.2, size=80)
    res = sp.cevae(x, t, y, z_dim=1, n_epochs=5, seed=0)
    assert "estimate" in res.tidy().columns
    assert res.to_excel(str(tmp_path / "cevae.xlsx")).endswith(".xlsx")
    assert "CEVAE" in res.to_markdown(str(tmp_path / "cevae.md"))
    fig, ax = res.plot("loss")
    assert fig is ax.get_figure()
    plt.close(fig)
