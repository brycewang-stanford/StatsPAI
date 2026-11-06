"""Fitted values and residuals lined up with the estimation sample.

Estimators drop incomplete rows before fitting, so the stored fitted values
and residuals are shorter than the data and carry no row labels. The fits
record the labels of the rows they kept (``data_info['sample_index']`` on
:class:`~statspai.core.results.EconometricResults`); this module turns the
stored arrays into ``pd.Series`` on those labels.

The arrays stay the default return type of ``residuals()`` and
``fitted_values()``: they are read as plain arrays across the package and
in user code (positional indexing, ``@`` products), and a Series on the
data's labels would change what ``r[0]`` means. ``as_series=True`` asks for
the labelled form.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility


def sample_series(values: Any, index: Any, name: str, who: str) -> pd.Series:
    """``values`` as a Series on the estimation-sample labels ``index``.

    Raises :class:`MethodIncompatibility` when the fit stored no values,
    recorded no sample, or the two disagree in length (which would pair a
    residual with another row's label).
    """
    if values is None:
        raise MethodIncompatibility(
            f"{who}: this fit stores no {name.replace('_', ' ')}.",
            recovery_hint="Use predict(data) on the estimation data instead.",
        )
    arr = np.asarray(values).ravel()
    if index is None:
        raise MethodIncompatibility(
            f"{who}: this fit did not record which data rows it used, so its "
            f"{name.replace('_', ' ')} cannot be labelled.",
            recovery_hint=(
                "Call it with as_series=False for the array, or fit from a "
                "formula and a DataFrame."
            ),
        )
    labels = pd.Index(index)
    if len(labels) != len(arr):
        raise MethodIncompatibility(
            f"{who}: {len(arr)} stored values for {len(labels)} sample rows.",
            recovery_hint="Call it with as_series=False for the array.",
            diagnostics={"n_values": len(arr), "n_rows": len(labels)},
        )
    return pd.Series(arr, index=labels, name=name)


class StoredFitMixin:
    """``fitted_values()`` / ``residuals()`` / ``sample_index`` for results
    that are not :class:`EconometricResults` (``sp.qreg``, ``sp.tobit``).

    The estimator calls :meth:`_store_fit` with the arrays in the order of
    the estimation sample and that sample's row labels.
    """

    _fit_fitted: Optional[np.ndarray] = None
    _fit_resid: Optional[np.ndarray] = None
    _fit_index: Optional[pd.Index] = None
    _fit_who: str = "result"

    def _store_fit(
        self, fitted: Any, resid: Any, index: Optional[pd.Index], who: str
    ) -> None:
        self._fit_fitted = np.asarray(fitted, dtype=float).ravel()
        self._fit_resid = np.asarray(resid, dtype=float).ravel()
        self._fit_index = None if index is None else pd.Index(index)
        self._fit_who = who

    @property
    def sample_index(self) -> Optional[pd.Index]:
        """Labels of the data rows the fit used, in the order of
        :meth:`fitted_values` / :meth:`residuals`; ``None`` if unrecorded."""
        return self._fit_index

    def fitted_values(self, as_series: bool = False) -> Any:
        """Fitted values on the estimation sample.

        Parameters
        ----------
        as_series : bool, default False
            ``False`` gives the stored array (``None`` if the fit kept
            none). ``True`` gives a ``pd.Series`` indexed by
            :attr:`sample_index`, the labels of the rows the fit used.
        """
        if not as_series:
            return self._fit_fitted
        return sample_series(
            self._fit_fitted, self._fit_index, "fitted_values", self._fit_who
        )

    def residuals(self, as_series: bool = False) -> Any:
        """Response residuals ``y - fitted_values()`` on the estimation sample.

        Parameters
        ----------
        as_series : bool, default False
            ``False`` gives the stored array (``None`` if the fit kept
            none). ``True`` gives a ``pd.Series`` indexed by
            :attr:`sample_index`.
        """
        if not as_series:
            return self._fit_resid
        return sample_series(
            self._fit_resid, self._fit_index, "residuals", self._fit_who
        )
