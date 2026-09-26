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
import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import unquote, urlparse, urlunparse

from ..exceptions import MethodIncompatibility

if TYPE_CHECKING:
    import pandas as pd


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
    return path.startswith(("s3://", "gs://", "https://", "http://", "file://"))


def _sanitize_url(url: str) -> str:
    """Drop query/fragment tokens before echoing a remote URL to clients."""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


@functools.lru_cache(maxsize=8)
def _sha256_file(path: str, mtime_ns: int, size: int) -> str:
    # mtime_ns and size are cache keys; the reader only needs the path.
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def data_provenance(
    path: str,
    columns: Optional[List[str]] = None,
    sample_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a compact provenance record for an MCP ``data_path``.

    Local files get size, mtime, and a full SHA-256 hash. Remote URLs are
    deliberately not hashed by StatsPAI because the bytes may require
    provider-specific auth and repeated downloads; the sanitized URL is
    still recorded so table notes can point back to the upstream source.
    """
    # Local vs remote must NOT be decided via urlparse's scheme detection:
    # urlparse(r"C:\data\x.csv") reads the Windows drive letter as a
    # single-char scheme ("c") and strips the drive from .path, which made
    # data_provenance tag a local Windows file as remote and skip the
    # SHA-256 (a real bug surfaced by the 2026-06-23 full-matrix run on
    # windows-latest — local paths worked on POSIX only because an empty
    # urlparse scheme falls back to "file"). Decide with the same prefix
    # test load_dataframe uses; treat file:// as local since it is hashable.
    if path.startswith(("s3://", "gs://", "https://", "http://")):
        suffix_source = unquote(urlparse(path).path) or path
        out: Dict[str, Any] = {
            "source": _sanitize_url(path),
            "scheme": urlparse(path).scheme,
            "source_type": "remote",
            "format": Path(suffix_source).suffix.lower().lstrip("."),
        }
        if columns:
            out["columns_requested"] = list(columns)
        if sample_n is not None:
            out["sample_n"] = int(sample_n)
            out["sample_seed"] = 0
            out["sample_method"] = SAMPLE_METHOD
        out["hash_status"] = "not_hashed_remote"
        return out

    # Local: a bare absolute path (POSIX or Windows) or a file:// URL.
    # Echo a bare caller path verbatim so provenance round-trips exactly
    # (no drive-letter case folding); only file:// is unwrapped to a path.
    if path.startswith("file://"):
        local_path = unquote(urlparse(path).path)
    else:
        local_path = path
    out = {
        "source": local_path,
        "scheme": "file",
        "source_type": "local",
        "format": Path(local_path).suffix.lower().lstrip("."),
    }
    if columns:
        out["columns_requested"] = list(columns)
    if sample_n is not None:
        out["sample_n"] = int(sample_n)
        out["sample_seed"] = 0
        out["sample_method"] = SAMPLE_METHOD

    try:
        stat = os.stat(local_path)
        sha256 = _sha256_file(local_path, stat.st_mtime_ns, stat.st_size)
    except OSError as e:
        out["hash_status"] = f"unavailable: {type(e).__name__}"
        return out

    out["size_bytes"] = stat.st_size
    out["mtime_ns"] = stat.st_mtime_ns
    out["sha256"] = sha256
    out["hash_status"] = "sha256"
    return out


#: Row-chunk size for streamed sampling of over-cap files.
_STREAM_CHUNK_ROWS = 250_000

#: Formats that can be read in row chunks without materialising the file.
_STREAMABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".parquet", ".pq", ".jsonl", ".dta")

#: Sampling rule recorded in provenance. One rule for every code path, so
#: the rows returned for a given ``data_sample_n`` never depend on whether
#: the file happened to fit under the byte cap.
SAMPLE_METHOD = "uniform_min_key_seed0"


def _is_streamable(path: str) -> bool:
    return path.lower().endswith(_STREAMABLE_SUFFIXES)


def estimated_load_bytes(path: str, columns: Optional[List[str]] = None) -> int:
    """Best cheap estimate of the bytes a full local load materialises.

    Parquet is compressed on disk, so its on-disk size can understate the
    in-memory frame several-fold; use the footer's *uncompressed* column
    sizes (restricted to the projection when one is given).  Other formats
    fall back to the on-disk size.
    """
    size = os.stat(path).st_size
    if path.lower().endswith((".parquet", ".pq")):
        try:
            import pyarrow.parquet as pq

            meta = pq.ParquetFile(path).metadata
            want = set(columns) if columns else None
            total = 0
            for g in range(meta.num_row_groups):
                rg = meta.row_group(g)
                for c in range(rg.num_columns):
                    col = rg.column(c)
                    top = col.path_in_schema.split(".")[0]
                    if want is None or top in want:
                        total += col.total_uncompressed_size
            return max(total, 1) if want else max(total, size)
        except Exception:  # noqa: BLE001 — estimate only; fall back to disk size
            return size
    return size


def _sample_keys(rng: Any, n_rows: int) -> Any:
    return rng.random(n_rows)


def _uniform_sample(df: "pd.DataFrame", n: int) -> "pd.DataFrame":
    """Keep the ``n`` rows with the smallest seed-0 uniform keys, file order."""
    import numpy as np

    if len(df) <= n:
        return df
    keys = _sample_keys(np.random.default_rng(0), len(df))
    idx = np.sort(np.argpartition(keys, n - 1)[:n])
    return df.iloc[idx].reset_index(drop=True)


def _iter_chunks(path: str, columns: Optional[List[str]]) -> Any:
    import pandas as pd

    lower = path.lower()
    cols = list(columns) if columns else None
    if lower.endswith((".csv", ".tsv", ".txt")):
        sep = "\t" if lower.endswith(".tsv") else ","
        yield from pd.read_csv(
            path, sep=sep, usecols=cols, chunksize=_STREAM_CHUNK_ROWS
        )
    elif lower.endswith((".parquet", ".pq")):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=_STREAM_CHUNK_ROWS, columns=cols):
            yield batch.to_pandas()
    elif lower.endswith(".jsonl"):
        for chunk in pd.read_json(path, lines=True, chunksize=_STREAM_CHUNK_ROWS):
            yield chunk[cols] if cols else chunk
    elif lower.endswith(".dta"):
        with pd.read_stata(path, columns=cols, chunksize=_STREAM_CHUNK_ROWS) as reader:
            yield from reader
    else:  # pragma: no cover — guarded by _is_streamable
        raise MethodIncompatibility(f"{path!r} cannot be streamed")


def _stream_sample(path: str, columns: Optional[List[str]], n: int) -> "pd.DataFrame":
    """Uniform sample of ``n`` rows in one pass, O(n + chunk) memory.

    Row ``i`` gets the ``i``-th draw of ``default_rng(0).random``; the
    generator's stream does not depend on how it is chunked, so this
    returns exactly the rows :func:`_uniform_sample` returns on the fully
    loaded frame.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    kept: Optional[pd.DataFrame] = None
    kept_keys: np.ndarray = np.empty(0)
    kept_pos: np.ndarray = np.empty(0, dtype=np.int64)
    offset = 0
    for chunk in _iter_chunks(path, columns):
        m = len(chunk)
        keys = _sample_keys(rng, m)
        pos = np.arange(offset, offset + m, dtype=np.int64)
        offset += m
        chunk = chunk.reset_index(drop=True)
        cand = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
        cand_keys = np.concatenate([kept_keys, keys])
        cand_pos = np.concatenate([kept_pos, pos])
        if len(cand_keys) > n:
            sel = np.argpartition(cand_keys, n - 1)[:n]
            cand = cand.iloc[sel].reset_index(drop=True)
            cand_keys, cand_pos = cand_keys[sel], cand_pos[sel]
        kept, kept_keys, kept_pos = cand, cand_keys, cand_pos
    if kept is None:
        return pd.DataFrame(columns=list(columns or []))
    order = np.argsort(kept_pos)
    return kept.iloc[order].reset_index(drop=True)


def load_dataframe(
    path: str, columns: Optional[List[str]] = None, sample_n: Optional[int] = None
) -> "pd.DataFrame":
    """Load a DataFrame from a local path or remote URL.

    Parameters
    ----------
    path : str
        Absolute filesystem path or one of: ``file://``, ``s3://``,
        ``gs://``, ``https://``, ``http://``.
    columns : list of str, optional
        Column projection, passed to every reader that supports it
        (CSV ``usecols``, Parquet / Feather / Stata ``columns``).
    sample_n : int, optional
        Uniform random subsample size without replacement. Deterministic:
        row ``i`` of the file gets the ``i``-th draw of
        ``numpy.random.default_rng(0).random`` and the ``sample_n`` rows
        with the smallest draws are kept, in file order.  The same rows
        are returned whether the file is loaded whole or streamed.

    Notes
    -----
    Local files larger than ``STATSPAI_MCP_MAX_DATA_BYTES`` (estimated
    in-memory size; Parquet uses its uncompressed column sizes) are
    rejected *unless* ``sample_n`` is given and the format is streamable
    (.csv/.tsv/.txt/.parquet/.pq/.jsonl/.dta): then the sample is drawn in
    a single chunked pass without materialising the file.

    Caches the materialised frame keyed by ``(path, mtime, columns)``
    so repeated tool calls on the same file are O(1) after the first
    load.
    """
    sample_size = None
    if sample_n is not None:
        try:
            sample_size = int(sample_n)
        except (TypeError, ValueError):
            raise MethodIncompatibility(
                f"data_sample_n must be a positive integer, got {sample_n!r}"
            )
        if sample_size < 1:
            raise MethodIncompatibility(
                f"data_sample_n must be a positive integer, got {sample_n!r}"
            )

    if is_remote_url(path):
        # Remote — defer all guard rails to the underlying loader; we
        # can't ``os.path.exists`` an s3 URL, and pandas/storage_options
        # error messages are rich enough.
        df = _load_remote(path, columns=columns)
    else:
        if not os.path.isabs(path):
            raise MethodIncompatibility(
                f"data_path must be absolute or a URL, got {path!r}"
            )
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"No such file: {path}")
        except OSError as e:
            raise MethodIncompatibility(
                f"Could not read data file metadata: {path!r}: {e}"
            ) from e
        cap = max_data_bytes()
        est = estimated_load_bytes(path, columns) if cap else stat.st_size
        if cap and est > cap:
            if sample_size is not None and _is_streamable(path):
                df = _stream_sample(path, columns, sample_size)
                return df
            raise MethodIncompatibility(_over_cap_message(path, est, cap, columns))
        mtime = stat.st_mtime
        df = _load_local_cached(path, mtime, tuple(columns or ()))

    if columns:
        keep = [c for c in columns if c in df.columns]
        if keep:
            df = df[keep]
    if sample_size is not None:
        df = _uniform_sample(df, sample_size)
    return df


def _over_cap_message(
    path: str, est: int, cap: int, columns: Optional[List[str]]
) -> str:
    what = (
        "estimated in-memory size (uncompressed Parquet columns)"
        if path.lower().endswith((".parquet", ".pq"))
        else "file size"
    )
    head = (
        f"data {what} is {est:,} bytes; exceeds "
        f"STATSPAI_MCP_MAX_DATA_BYTES={cap:,}. "
    )
    if _is_streamable(path):
        tips = (
            "Pass data_sample_n=<N> to draw a uniform sample in one streamed "
            "pass (the file is never fully loaded)"
        )
        if not columns:
            tips += ", and/or data_columns=[...] to read only the columns used"
        return head + tips + "; or raise the limit with the env var."
    return (
        head + f"'{Path(path).suffix}' files must be parsed whole, so data_sample_n "
        "cannot reduce peak memory here. Convert to Parquet/CSV (which "
        "support streamed sampling), pre-sample offline, or raise the limit "
        "with the env var."
    )


@functools.lru_cache(maxsize=8)
def _load_local_cached(
    path: str,
    mtime: float,
    columns_key: tuple,  # noqa: ARG001 — mtime invalidates
) -> "pd.DataFrame":
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
    raise MethodIncompatibility(
        f"Unsupported file extension: {path!r}. Supported: "
        ".csv/.tsv/.txt/.parquet/.pq/.feather/.arrow/.xlsx/.xls/.dta/"
        ".json/.jsonl"
    )


def _load_remote(url: str, columns: Optional[List[str]] = None) -> "pd.DataFrame":
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
    raise MethodIncompatibility(
        f"Unsupported remote extension in {url!r}. "
        f"See load_dataframe docs for supported formats."
    )


__all__ = [
    "DEFAULT_MAX_DATA_BYTES",
    "max_data_bytes",
    "is_remote_url",
    "data_provenance",
    "load_dataframe",
    "estimated_load_bytes",
    "SAMPLE_METHOD",
]
