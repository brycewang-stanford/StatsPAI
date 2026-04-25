"""Reproducibility metadata for publication tables.

Stata and R's standard table tools (``esttab``, ``modelsummary``,
``fixest::etable``) do not record the package version, the random seed,
or a hash of the data alongside the rendered table. The result is that a
co-author six months later cannot tell whether your ``Table 3`` came
from the same data + code as today.

This module gives :func:`statspai.regtable` (and the wider output stack)
a small, portable footer line that captures:

- StatsPAI version (``sp.__version__``).
- Python version (``sys.version_info``).
- An optional fingerprint of the input ``DataFrame``: row count, column
  count, and a SHA-256 of its contents (truncated for display).
- An optional seed (any integer the caller chooses to expose).
- The wall-clock timestamp the table was rendered.

The output format is a single-line string suitable for embedding directly
into a table footer note. There is intentionally no schema versioning
or structured-output mode — this is a *human-readable* trail, not a
formal artifact contract.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import sys
from typing import Any, Optional


def _statspai_version() -> str:
    try:
        from .. import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def _python_version() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


def _hash_dataframe(df, length: int = 10) -> Optional[str]:
    """Return a short SHA-256 fingerprint of *df* (or ``None`` on failure).

    We hash the byte representation of ``pd.util.hash_pandas_object`` which
    is order-sensitive and column-name-sensitive — exactly the granularity
    we want for "is this the same dataset?".
    """
    if df is None:
        return None
    try:
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            return None
        h = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        digest = hashlib.sha256(h).hexdigest()
        return digest[:length]
    except Exception:
        return None


def build_repro_note(
    *,
    data: Optional[Any] = None,
    seed: Optional[int] = None,
    timestamp: bool = True,
    package_version: bool = True,
    python_version: bool = False,
    extra: Optional[str] = None,
) -> str:
    """Build a single-line reproducibility note.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        If given, hash the frame and include ``rows × cols, SHA256:abcdef`` in
        the line. Anything other than a ``DataFrame`` is silently ignored.
    seed : int, optional
        Random seed to record. Use whatever seed your script passed to
        :class:`numpy.random.default_rng` / ``random.seed``.
    timestamp : bool, default True
        Whether to append a ``"YYYY-MM-DD HH:MM"`` wall-clock timestamp.
    package_version : bool, default True
        Whether to include ``StatsPAI vX.Y.Z``. Almost always True.
    python_version : bool, default False
        Append ``Python 3.11.5``. Off by default to keep the note compact.
    extra : str, optional
        Free-form text appended verbatim at the end (e.g. git commit hash).

    Returns
    -------
    str
        A single line, semicolon-separated. Returns ``""`` when every
        component is disabled.
    """
    parts: list[str] = []

    if package_version:
        parts.append(f"StatsPAI v{_statspai_version()}")

    if python_version:
        parts.append(f"Python {_python_version()}")

    if data is not None:
        try:
            import pandas as pd

            if isinstance(data, pd.DataFrame):
                n_rows = len(data)
                n_cols = len(data.columns)
                digest = _hash_dataframe(data)
                if digest:
                    parts.append(f"data {n_rows}×{n_cols} SHA256:{digest}")
                else:
                    parts.append(f"data {n_rows}×{n_cols}")
        except Exception:
            pass

    if seed is not None:
        parts.append(f"seed={int(seed)}")

    if timestamp:
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        parts.append(ts)

    if extra:
        parts.append(str(extra))

    return "Reproducibility: " + "; ".join(parts) if parts else ""
