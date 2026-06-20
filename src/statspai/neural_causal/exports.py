"""Export helpers for neural causal treatment-effect results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def _as_array(value: Any, n: Optional[int] = None) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if n is not None and len(arr) != n:
        return None
    return arr


def _is_neural_result(result: Any) -> bool:
    return bool(getattr(result, "model_info", {}).get("neural_causal"))


def _require_neural_result(result: Any) -> None:
    if not _is_neural_result(result):
        raise ValueError(
            "Expected a StatsPAI neural causal CausalResult "
            "(TARNet, CFRNet, or DragonNet)."
        )


def neural_effects_frame(result: Any, *, sort_by: Optional[str] = None) -> pd.DataFrame:
    """Return unit-level neural causal predictions as a tidy DataFrame.

    Parameters
    ----------
    result : CausalResult
        Result returned by ``sp.tarnet``, ``sp.cfrnet``, or ``sp.dragonnet``.
    sort_by : {"cate", "propensity", None}, optional
        Sort rows by a diagnostic column.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Operates on a fitted neural result; the `neural` extra
    >>> # (pip install statspai[neural]) is needed to produce one:
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x1, x2 = rng.normal(size=n), rng.normal(size=n)
    >>> d = rng.binomial(1, 0.5, size=n)
    >>> y = 2.0 * d + x1 + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
    >>> res = sp.tarnet(df, y='y', treat='d',
    ...                 covariates=['x1', 'x2'], epochs=50)  # doctest: +SKIP
    >>> sp.neural_effects_frame(res, sort_by='cate').head()  # doctest: +SKIP
    """
    _require_neural_result(result)
    mi = result.model_info
    cate = _as_array(mi.get("cate"))
    if cate is None:
        raise ValueError("Neural result does not contain CATE estimates.")
    n = len(cate)
    data = {
        "unit": np.arange(n),
        "cate": cate,
    }
    mu0 = _as_array(mi.get("mu0"), n)
    mu1 = _as_array(mi.get("mu1"), n)
    treatment = _as_array(mi.get("treatment"), n)
    propensity = _as_array(mi.get("propensity"), n)
    aipw = _as_array(mi.get("aipw_scores"), n)
    if mu0 is not None:
        data["mu0"] = mu0
    if mu1 is not None:
        data["mu1"] = mu1
    if treatment is not None:
        data["treatment"] = treatment.astype(int)
    if propensity is not None:
        data["propensity"] = propensity
    if aipw is not None:
        data["aipw_score"] = aipw
    df = pd.DataFrame(data)
    if sort_by is not None:
        if sort_by not in df.columns:
            raise ValueError(f"sort_by={sort_by!r} is not available.")
        df = df.sort_values(sort_by).reset_index(drop=True)
    return df


def neural_summary_frame(result: Any) -> pd.DataFrame:
    """Return a one-row summary table for a neural causal result.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Operates on a fitted neural result; the `neural` extra
    >>> # (pip install statspai[neural]) is needed to produce one:
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x1, x2 = rng.normal(size=n), rng.normal(size=n)
    >>> d = rng.binomial(1, 0.5, size=n)
    >>> y = 2.0 * d + x1 + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
    >>> res = sp.dragonnet(df, y='y', treat='d',
    ...                    covariates=['x1', 'x2'],
    ...                    epochs=50)  # doctest: +SKIP
    >>> sp.neural_summary_frame(res)  # doctest: +SKIP
    """
    _require_neural_result(result)
    mi = result.model_info
    keys = [
        "architecture",
        "device",
        "n_epochs_trained",
        "validation_fraction",
        "early_stopping",
        "n_covariates",
        "n_treated",
        "n_control",
        "se_method",
        "cate_mean",
        "cate_std",
        "cate_min",
        "cate_q05",
        "cate_q25",
        "cate_median",
        "cate_q75",
        "cate_q95",
        "cate_max",
        "propensity_mean",
        "propensity_std",
        "propensity_min",
        "propensity_max",
        "ate_plugin",
        "ate_aipw",
    ]
    row = {
        "method": result.method,
        "estimand": result.estimand,
        "estimate": result.estimate,
        "std_error": result.se,
        "p_value": result.pvalue,
        "conf_low": result.ci[0],
        "conf_high": result.ci[1],
        "n_obs": result.n_obs,
    }
    for key in keys:
        if key in mi:
            row[key] = mi[key]
    return pd.DataFrame([row])


def neural_training_frame(result: Any) -> pd.DataFrame:
    """Return per-epoch training diagnostics if recorded.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Operates on a fitted neural result; the `neural` extra
    >>> # (pip install statspai[neural]) is needed to produce one:
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x1, x2 = rng.normal(size=n), rng.normal(size=n)
    >>> d = rng.binomial(1, 0.5, size=n)
    >>> y = 2.0 * d + x1 + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
    >>> res = sp.tarnet(df, y='y', treat='d', covariates=['x1', 'x2'],
    ...                 epochs=50)  # doctest: +SKIP
    >>> sp.neural_training_frame(res).tail()  # doctest: +SKIP
    """
    _require_neural_result(result)
    history = result.model_info.get("training_history")
    if isinstance(history, pd.DataFrame):
        return history.copy()
    return pd.DataFrame(result.model_info.get("loss_history", []))


def neural_causal_to_markdown(
    result: Any,
    path: Optional[str] = None,
    *,
    effects_head: int = 20,
    digits: int = 4,
) -> str:
    """Render a neural causal result to GitHub-flavoured Markdown.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Operates on a fitted neural result; the `neural` extra
    >>> # (pip install statspai[neural]) is needed to produce one:
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x1, x2 = rng.normal(size=n), rng.normal(size=n)
    >>> d = rng.binomial(1, 0.5, size=n)
    >>> y = 2.0 * d + x1 + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
    >>> res = sp.tarnet(df, y='y', treat='d', covariates=['x1', 'x2'],
    ...                 epochs=50)  # doctest: +SKIP
    >>> md = sp.neural_causal_to_markdown(res)  # doctest: +SKIP
    """
    summary = neural_summary_frame(result).round(digits)
    effects = neural_effects_frame(result).head(effects_head).round(digits)
    training = neural_training_frame(result).round(digits)
    parts = [
        f"# {result.method}",
        "",
        "## Summary",
        summary.to_markdown(index=False),
        "",
        "## Unit-Level Effects",
        effects.to_markdown(index=False),
    ]
    if not training.empty:
        parts.extend(["", "## Training Diagnostics", training.to_markdown(index=False)])
    text = "\n".join(parts) + "\n"
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def neural_causal_to_html(
    result: Any,
    path: Optional[str] = None,
    *,
    effects_head: int = 50,
    digits: int = 4,
) -> str:
    """Render a neural causal result to a compact HTML report.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Operates on a fitted neural result; the `neural` extra
    >>> # (pip install statspai[neural]) is needed to produce one:
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x1, x2 = rng.normal(size=n), rng.normal(size=n)
    >>> d = rng.binomial(1, 0.5, size=n)
    >>> y = 2.0 * d + x1 + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
    >>> res = sp.cfrnet(df, y='y', treat='d', covariates=['x1', 'x2'],
    ...                 epochs=50)  # doctest: +SKIP
    >>> html = sp.neural_causal_to_html(res)  # doctest: +SKIP
    """
    summary = neural_summary_frame(result).round(digits)
    effects = neural_effects_frame(result).head(effects_head).round(digits)
    training = neural_training_frame(result).round(digits)
    blocks = [
        "<html><body>",
        f"<h1>{result.method}</h1>",
        "<h2>Summary</h2>",
        summary.to_html(index=False),
        "<h2>Unit-Level Effects</h2>",
        effects.to_html(index=False),
    ]
    if not training.empty:
        blocks.extend(["<h2>Training Diagnostics</h2>", training.to_html(index=False)])
    blocks.append("</body></html>")
    html = "\n".join(blocks)
    if path is not None:
        Path(path).write_text(html, encoding="utf-8")
    return html


def neural_causal_to_excel(
    result: Any,
    path: str,
    *,
    digits: int = 6,
) -> str:
    """Write a multi-sheet Excel workbook for a neural causal result.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Operates on a fitted neural result; the `neural` extra
    >>> # (pip install statspai[neural]) is needed to produce one:
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x1, x2 = rng.normal(size=n), rng.normal(size=n)
    >>> d = rng.binomial(1, 0.5, size=n)
    >>> y = 2.0 * d + x1 + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
    >>> res = sp.dragonnet(df, y='y', treat='d', covariates=['x1', 'x2'],
    ...                    epochs=50)  # doctest: +SKIP
    >>> sp.neural_causal_to_excel(res, 'neural.xlsx')  # doctest: +SKIP
    """
    summary = neural_summary_frame(result).round(digits)
    effects = neural_effects_frame(result).round(digits)
    training = neural_training_frame(result).round(digits)
    with pd.ExcelWriter(path) as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        effects.to_excel(writer, sheet_name="Effects", index=False)
        if not training.empty:
            training.to_excel(writer, sheet_name="Training", index=False)
    return path


__all__ = [
    "neural_effects_frame",
    "neural_summary_frame",
    "neural_training_frame",
    "neural_causal_to_markdown",
    "neural_causal_to_html",
    "neural_causal_to_excel",
]
