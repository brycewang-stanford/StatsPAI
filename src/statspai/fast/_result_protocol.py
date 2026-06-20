"""Small JSON-safe helpers for ``sp.fast`` result payloads."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def jsonable(value: Any) -> Any:
    """Convert NumPy/Pandas values into strict JSON-compatible objects."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value_float = float(value)
        return value_float if np.isfinite(value_float) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, pd.DataFrame):
        return jsonable(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def tidy_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return JSON-safe tidy records from a result table."""
    records = (
        df.reset_index().rename(columns={"index": "term"}).to_dict(orient="records")
    )
    return [
        {str(key): jsonable(value) for key, value in record.items()}
        for record in records
    ]


def distribution_summary(values: Any) -> dict[str, Any]:
    """Compact finite-value summary for bootstrap/simulation draws."""
    arr = np.asarray(values, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    out: dict[str, Any] = {"n": int(arr.size)}
    if finite.size == 0:
        out.update(
            {
                "mean": None,
                "sd": None,
                "q025": None,
                "q50": None,
                "q975": None,
            }
        )
        return out
    out.update(
        {
            "mean": float(np.mean(finite)),
            "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
            "q025": float(np.quantile(finite, 0.025)),
            "q50": float(np.quantile(finite, 0.5)),
            "q975": float(np.quantile(finite, 0.975)),
        }
    )
    return out
