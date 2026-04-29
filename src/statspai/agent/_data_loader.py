"""Data-file loading for the MCP layer.

Pulled out of ``mcp_server.py`` so the JSON-RPC dispatch file stays
focused on protocol concerns. Public surface mirrors what the server
needs:

* :func:`load_dataframe(path, columns=, sample_n=)` — entry point.
* :func:`max_data_bytes` / :func:`is_remote_url` — config / predicate.

Supported formats: ``.csv`` / ``.tsv`` / ``.txt`` / ``.parquet`` /
``.pq`` / ``.feather`` / ``.arrow`` / ``.xlsx`` / ``.xls`` / ``.dta``
(Stata) / ``.json`` / ``.jsonl``. Schemes: ``file://``, ``s3://``,
``gs://``, ``https://``, ``http://``.

Local loads are LRU-cached by ``(path, mtime, columns_key)`` so
repeated tools/call invocations on the same file are O(1).
"""
from __future__ import annotations

import functools
import os
from typing import List, Optional


#: Default max file size (bytes) the server will load. A misconfigured
#: client pointing at a 50GB parquet will OOM the host otherwise.
#: Override via ``STATSPAI_MCP_MAX_DATA_BYTES`` (e.g. ``5_000_000_000``);
#: set to ``0`` to disable the check.
DEFAULT_MAX_DATA_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


def max_data_bytes() -> int:
    raw = os.environ.get("STATSPAI_MCP_MAX_DATA_BYTES")
    if raw is None:
        return DEFAULT_MAX_DATA_BYTES
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_DATA_BYTES


def is_remote_url(path: str) -> bool:
    return path.startswith(("s3://", "gs://", "https://", "http://",
                             "file://"))


def load_dataframe(path: str,
                    columns: Optional[List[str]] = None,
                    sample_n: Optional[int] = None):
    """Load a DataFrame from a local path or remote URL.

    Parameters
    ----------
    path : str
        Absolute filesystem path or one of: ``file://``, ``s3://``,
        ``gs://``, ``https://``, ``http://``.
    columns : list of str, optional
        Column projection. Honoured by parquet / feather / stata
        readers; for CSV we read all columns then sub-select (read_csv's
        ``usecols`` would also work but mismatched names raise; we want
        the server to be permissive and let the estimator surface
        column-name errors with rich remediation).
    sample_n : int, optional
        Uniform random subsample size (seed=0, deterministic).
        Applied AFTER the projection.

    Notes
    -----
    Caches the materialised frame keyed by ``(path, mtime, columns)``
    so repeated tool calls on the same file are O(1) after the first
    load.
    """
    if is_remote_url(path):
        # Remote — defer all guard rails to the underlying loader; we
        # can't ``os.path.exists`` an s3 URL, and pandas/storage_options
        # error messages are rich enough.
        df = _load_remote(path, columns=columns)
    else:
        if not os.path.isabs(path):
            raise ValueError(
                f"data_path must be absolute or a URL, got {path!r}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file: {path}")
        size = os.path.getsize(path)
        cap = max_data_bytes()
        if cap and size > cap:
            raise ValueError(
                f"data file is {size:,} bytes; exceeds "
                f"STATSPAI_MCP_MAX_DATA_BYTES={cap:,}. "
                f"Pass data_sample_n=<N> for a random subsample, or "
                f"raise the limit with the env var."
            )
        mtime = os.path.getmtime(path)
        df = _load_local_cached(path, mtime, tuple(columns or ()))

    if columns:
        keep = [c for c in columns if c in df.columns]
        if keep:
            df = df[keep]
    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=int(sample_n), random_state=0).reset_index(drop=True)
    return df


@functools.lru_cache(maxsize=8)
def _load_local_cached(path: str, mtime: float,
                       columns_key: tuple):  # noqa: ARG001 — mtime invalidates
    """LRU-cached local loader. ``mtime`` busts the cache on file edits."""
    import pandas as pd
    lower = path.lower()
    cols = list(columns_key) or None
    if lower.endswith((".csv", ".tsv", ".txt")):
        sep = "\t" if lower.endswith(".tsv") else ","
        return pd.read_csv(path, sep=sep, usecols=cols)
    if lower.endswith((".parquet", ".pq")):
        return pd.read_parquet(path, columns=cols)
    if lower.endswith((".feather", ".arrow")):
        return pd.read_feather(path, columns=cols)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, usecols=cols)
    if lower.endswith(".dta"):
        # Stata native — alignment with Stata is StatsPAI's tagline,
        # so being able to read .dta is non-negotiable.
        return pd.read_stata(path, columns=cols)
    if lower.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
        return df[cols] if cols else df
    if lower.endswith(".json"):
        df = pd.read_json(path)
        return df[cols] if cols else df
    raise ValueError(
        f"Unsupported file extension: {path!r}. Supported: "
        ".csv/.tsv/.txt/.parquet/.pq/.feather/.arrow/.xlsx/.xls/.dta/"
        ".json/.jsonl"
    )


def _load_remote(url: str, columns: Optional[List[str]] = None):
    """Load a DataFrame from a remote URL via pandas storage backends.

    Pandas dispatches s3:// / gs:// / https:// to fsspec. Authentication
    is configured by the host environment (e.g. AWS credentials chain);
    we don't smuggle secrets through the MCP layer.
    """
    import pandas as pd
    lower = url.split("?", 1)[0].lower()
    cols = list(columns) if columns else None
    if lower.endswith((".csv", ".tsv", ".txt")):
        sep = "\t" if lower.endswith(".tsv") else ","
        return pd.read_csv(url, sep=sep, usecols=cols)
    if lower.endswith((".parquet", ".pq")):
        return pd.read_parquet(url, columns=cols)
    if lower.endswith((".feather", ".arrow")):
        return pd.read_feather(url, columns=cols)
    if lower.endswith(".dta"):
        return pd.read_stata(url, columns=cols)
    if lower.endswith(".jsonl"):
        df = pd.read_json(url, lines=True)
        return df[cols] if cols else df
    if lower.endswith(".json"):
        df = pd.read_json(url)
        return df[cols] if cols else df
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(url, usecols=cols)
    raise ValueError(
        f"Unsupported remote extension in {url!r}. "
        f"See load_dataframe docs for supported formats."
    )


__all__ = [
    "DEFAULT_MAX_DATA_BYTES",
    "max_data_bytes",
    "is_remote_url",
    "load_dataframe",
]
