"""Principal components of a set of variables.

The components are the eigenvectors of the correlation matrix (or, on
request, the covariance matrix) ordered by eigenvalue. The ``j``-th
eigenvalue is the variance of the ``j``-th component, so the eigenvalues
divided by their sum are the shares of total variance.
"""

from __future__ import annotations

from typing import Any, ClassVar, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["pca", "PCAResult"]


def _columns(
    data: pd.DataFrame, variables: Optional[Sequence[str]], who: str
) -> Tuple[List[str], pd.DataFrame]:
    """The analysis columns and their complete rows."""
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            f"sp.{who}: data must be a DataFrame.",
            recovery_hint="Pass the variables as columns of a DataFrame.",
        )
    names = (
        [str(v) for v in variables]
        if variables is not None
        else [str(c) for c in data.select_dtypes("number").columns]
    )
    missing = [v for v in names if v not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"sp.{who}: {missing} are not columns of the data.",
            recovery_hint="Check the variable names.",
        )
    if len(names) < 2:
        raise MethodIncompatibility(
            f"sp.{who}: at least two variables are needed.",
            recovery_hint="List the variables to analyse.",
        )
    frame = data[names].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) <= len(names):
        raise DataInsufficient(
            f"sp.{who}: {len(frame)} complete rows for {len(names)} variables.",
            recovery_hint="Use fewer variables or more observations.",
        )
    constant = [v for v in names if frame[v].std(ddof=1) == 0]
    if constant:
        raise MethodIncompatibility(
            f"sp.{who}: {constant} do not vary.",
            recovery_hint="Drop the constant variables.",
        )
    return names, frame


def _ordered_eigen(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues in decreasing order; each eigenvector signed so that its
    elements sum to a positive number (Stata's convention)."""
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    signs = np.where(vectors.sum(axis=0) < 0, -1.0, 1.0)
    return values, vectors * signs


class PCAResult(ResultProtocolMixin):
    """Outcome of :func:`pca`.

    Attributes
    ----------
    eigenvalues : pandas.DataFrame
        One row per component: ``eigenvalue``, ``difference`` to the next,
        ``proportion`` of total variance and ``cumulative`` proportion.
    loadings : pandas.DataFrame
        The eigenvectors, variables in rows and retained components in
        columns. Each has unit length.
    unexplained : pandas.Series
        Share of each variable's variance not reproduced by the retained
        components (zero when all are kept).
    n_obs : int
        Complete rows used.
    matrix : {'correlation', 'covariance'}
        What was decomposed.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> f = rng.normal(size=300)
    >>> df = pd.DataFrame({f"x{j}": f + rng.normal(size=300) for j in range(3)})
    >>> res = sp.pca(df, n_components=1)
    >>> res.loadings.shape
    (3, 1)
    >>> bool((res.unexplained > 0).all())
    True
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        eigenvalues: pd.DataFrame,
        loadings: pd.DataFrame,
        unexplained: pd.Series,
        n_obs: int,
        matrix: str,
        means: pd.Series,
        scales: pd.Series,
    ) -> None:
        self.method = f"Principal components / {matrix}"
        self.eigenvalues = eigenvalues
        self.loadings = loadings
        self.unexplained = unexplained
        self.n_obs = n_obs
        self.matrix = matrix
        self.n_components = loadings.shape[1]
        self._means = means
        self._scales = scales

    def scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """Component scores: the centred (and, for the correlation matrix,
        standardised) variables times the eigenvectors. The mean and
        standard deviation are those of the estimation sample. A row with
        a missing variable gets missing scores."""
        names = list(self.loadings.index)
        z = (data[names].astype(float) - self._means) / self._scales
        out = z.to_numpy() @ self.loadings.to_numpy()
        return pd.DataFrame(out, index=data.index, columns=self.loadings.columns)

    def summary(self) -> str:
        lines = [
            f"{self.method:<48}Number of obs   = {self.n_obs:>8,}",
            f"{'':<48}Number of comp. = {self.n_components:>8}",
            "",
            self.eigenvalues.to_string(float_format=lambda v: f"{v:.4f}"),
            "",
            "Principal components (eigenvectors)",
            self.loadings.assign(Unexplained=self.unexplained).to_string(
                float_format=lambda v: f"{v:.4f}"
            ),
        ]
        return "\n".join(lines)

    def plot(self, ax: Any = None) -> Any:
        """Scree plot: eigenvalues against component number."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        values = self.eigenvalues["eigenvalue"].to_numpy()
        ax.plot(np.arange(1, values.size + 1), values, marker="o")
        ax.set_xlabel("Component")
        ax.set_ylabel("Eigenvalue")
        ax.set_title("Scree plot")
        return ax

    def __repr__(self) -> str:
        return self.summary()


def pca(
    data: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    *,
    n_components: Optional[int] = None,
    min_eigenvalue: Optional[float] = None,
    covariance: bool = False,
) -> PCAResult:
    """Principal component analysis.

    Parameters
    ----------
    data : pandas.DataFrame
        The data. Rows with a missing value in any analysed variable are
        dropped.
    variables : list of str, optional
        Columns to analyse. Default: every numeric column.
    n_components : int, optional
        Keep this many components. Default: all.
    min_eigenvalue : float, optional
        Keep the components whose eigenvalue exceeds this (1 is the Kaiser
        rule for a correlation matrix). Ignored when ``n_components`` is
        given.
    covariance : bool, default False
        Decompose the covariance matrix instead of the correlation matrix.
        The components then depend on the units of the variables.

    Returns
    -------
    PCAResult
        ``eigenvalues`` (with shares of variance), ``loadings`` (unit-length
        eigenvectors), ``unexplained``, and ``scores(data)``.

    Notes
    -----
    An eigenvector is defined up to sign; here each is signed so that its
    elements sum to a positive number, which is what Stata's ``pca``
    prints. R's ``prcomp`` may show the opposite sign on some columns.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> f = rng.normal(size=500)
    >>> df = pd.DataFrame({f"x{j}": f + rng.normal(size=500) for j in range(4)})
    >>> res = sp.pca(df)
    >>> round(float(res.eigenvalues["proportion"].sum()), 12)
    1.0
    >>> res.scores(df).shape
    (500, 4)
    """
    names, frame = _columns(data, variables, "pca")
    p = len(names)
    x = frame.to_numpy(dtype=float)
    means = frame.mean()
    sd = frame.std(ddof=1)
    matrix = np.cov(x, rowvar=False) if covariance else np.corrcoef(x, rowvar=False)
    values, vectors = _ordered_eigen(matrix)
    values = np.clip(values, 0.0, None)

    if n_components is not None:
        keep = int(n_components)
        if not 1 <= keep <= p:
            raise MethodIncompatibility(
                f"sp.pca: n_components={n_components} is not between 1 and {p}.",
                recovery_hint="Ask for at most as many components as variables.",
            )
    elif min_eigenvalue is not None:
        keep = max(int((values > float(min_eigenvalue)).sum()), 1)
    else:
        keep = p

    labels = [f"Comp{j + 1}" for j in range(p)]
    total = values.sum()
    table = pd.DataFrame(
        {
            "eigenvalue": values,
            "difference": np.append(values[:-1] - values[1:], np.nan),
            "proportion": values / total,
            "cumulative": np.cumsum(values) / total,
        },
        index=labels,
    )
    loadings = pd.DataFrame(vectors[:, :keep], index=names, columns=labels[:keep])
    diagonal = np.diag(matrix)
    explained = (vectors[:, :keep] ** 2) @ values[:keep]
    unexplained = pd.Series(np.clip(1.0 - explained / diagonal, 0.0, None), index=names)
    scales = pd.Series(1.0, index=names) if covariance else sd
    return PCAResult(
        eigenvalues=table,
        loadings=loadings,
        unexplained=unexplained,
        n_obs=len(frame),
        matrix="covariance" if covariance else "correlation",
        means=means,
        scales=scales,
    )
