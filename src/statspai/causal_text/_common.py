"""Shared helpers for ``causal_text``: text embedding + projection.

All embedders return an ``(n, k)`` numeric array suitable as confounder
features.  No heavy dependencies are imported at top level — torch /
sentence-transformers are pulled in lazily when the user picks an
embedder that requires them.
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable, Optional, Sequence, Union

import numpy as np


__all__ = [
    'hash_embed_texts',
    'embed_texts',
]


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list:
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())


def hash_embed_texts(
    texts: Sequence[str],
    *,
    n_components: int = 32,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic hashing-vectoriser embedding.

    Each token is hashed (md5) and bucketed into one of ``n_components``
    dimensions; document vectors are L2-normalised counts in those
    buckets.  Reproducible, dependency-free, and good enough for a
    confounder-projection MVP.

    Parameters
    ----------
    texts : sequence of str
    n_components : int, default 32
        Embedding dimensionality.
    seed : int, default 0
        Salt for the bucket hash; changing the seed changes the
        embedding.  Pinned ``seed=0`` yields the documented default.

    Returns
    -------
    np.ndarray of shape (n_texts, n_components)
    """
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    n = len(texts)
    out = np.zeros((n, n_components), dtype=np.float64)
    salt = str(int(seed)).encode("utf-8")
    for i, text in enumerate(texts):
        for tok in _tokenize(text):
            h = hashlib.md5(
                salt + tok.encode("utf-8"), usedforsecurity=False,
            ).digest()
            bucket = int.from_bytes(h[:4], "little") % n_components
            out[i, bucket] += 1.0
        norm = np.linalg.norm(out[i])
        if norm > 0:
            out[i] /= norm
    return out


def embed_texts(
    texts: Sequence[str],
    *,
    embedder: Union[str, Callable[[Sequence[str]], np.ndarray]] = 'hash',
    n_components: int = 32,
    seed: int = 0,
) -> np.ndarray:
    """Dispatch to the named embedder.

    Parameters
    ----------
    texts : sequence of str
    embedder : {'hash', 'sbert'} or callable, default 'hash'
        - ``'hash'``: deterministic dependency-free hashing vectoriser
          (see :func:`hash_embed_texts`).
        - ``'sbert'``: lazy-import ``sentence-transformers`` and use
          ``all-MiniLM-L6-v2`` by default.  Optional dep — install
          with ``pip install statspai[text]``.
        - Callable: ``f(texts) -> np.ndarray`` of shape ``(n, k)``.
    n_components : int, default 32
        Used by 'hash' only; sbert dimensionality is fixed by the model.
    seed : int, default 0

    Returns
    -------
    np.ndarray of shape (n_texts, k)
    """
    if callable(embedder):
        out = embedder(list(texts))
        out = np.asarray(out, dtype=np.float64)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        if out.shape[0] != len(texts):
            raise ValueError(
                "Custom embedder returned shape "
                f"{out.shape} (expected first dim = {len(texts)})"
            )
        return out
    if embedder == 'hash':
        return hash_embed_texts(
            texts, n_components=n_components, seed=seed,
        )
    if embedder == 'sbert':
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover (lazy dep)
            raise ImportError(
                "embedder='sbert' requires sentence-transformers. "
                "Install via `pip install sentence-transformers` or "
                "`pip install statspai[text]`."
            ) from exc
        model = SentenceTransformer("all-MiniLM-L6-v2")
        out = np.asarray(model.encode(list(texts)), dtype=np.float64)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        return out
    raise ValueError(
        f"Unknown embedder={embedder!r}. Use 'hash', 'sbert', or a callable."
    )
