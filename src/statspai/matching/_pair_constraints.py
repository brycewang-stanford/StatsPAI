"""Which treated-control pairs nearest-neighbour matching may form.

``sp.match(method='nearest', exact=..., covariate_calipers=..., mahvars=...)``
restricts the candidate pairs in the way R ``MatchIt::matchit(method =
"nearest")`` does:

* ``exact`` -- a pair is allowed only when the two units agree on every
  listed variable (``matchit(exact = ~ ...)``);
* ``covariate_calipers`` -- a pair is allowed only when the two units are
  within the given width on each listed variable (the *named* entries of
  ``matchit(caliper = c(x = ...))``); with ``caliper_scale='sd'`` the width
  is in standard deviations of that variable over the matching sample
  (``std.caliper = TRUE``, ``sd()`` with divisor ``n - 1``);
* ``mahvars`` -- the propensity score is used only as a caliper and the
  matching itself runs on the Mahalanobis distance of these variables
  (``matchit(mahvars = ~ ...)``), with the pooled within-arm covariance.

A forbidden pair gets an infinite distance, which every matching routine in
:mod:`statspai.matching.match` treats as "not a candidate". The fast
searches (sorted score, k-d tree) do not see the distance matrix, so the
caller switches them off whenever a constraint is active.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility


def column_list(value: Any, *, name: str) -> List[str]:
    """``None`` -> ``[]``; a string -> one name; otherwise a list of names."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        names = [str(v) for v in value]
    except TypeError as exc:
        raise MethodIncompatibility(
            f"match: {name} must be a column name or a list of column names; "
            f"got {type(value).__name__}.",
            recovery_hint=f"Pass {name}=['col_a', 'col_b'].",
        ) from exc
    if not names:
        return []
    return names


def caliper_map(value: Any) -> Dict[str, float]:
    """Validate ``covariate_calipers``: a mapping column -> positive width."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MethodIncompatibility(
            "match: covariate_calipers must map a column name to a width, "
            f"e.g. {{'age': 0.25}}; got {type(value).__name__}.",
            recovery_hint="Pass a dict such as covariate_calipers={'age': 0.25}.",
        )
    out: Dict[str, float] = {}
    for key, width in value.items():
        try:
            w = float(width)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                f"match: covariate_calipers[{key!r}] must be a number; got "
                f"{width!r}.",
                recovery_hint="Give every caliper a positive finite width.",
            ) from exc
        if not np.isfinite(w) or w <= 0:
            raise MethodIncompatibility(
                f"match: covariate_calipers[{key!r}] must be positive and "
                f"finite; got {width!r}.",
                recovery_hint="Give every caliper a positive finite width.",
            )
        out[str(key)] = w
    return out


@dataclass
class PairConstraints:
    """The allowed-pair rule over the rows of the matching sample."""

    exact_codes: Optional[np.ndarray] = None
    cal_X: Optional[np.ndarray] = None
    cal_width: Optional[np.ndarray] = None
    cal_names: List[str] = field(default_factory=list)
    #: With ``mahvars``: the caliper on the propensity score and the score.
    ps_caliper: Optional[float] = None
    pscore: Optional[np.ndarray] = None

    @property
    def active(self) -> bool:
        return (
            self.exact_codes is not None
            or self.cal_X is not None
            or self.ps_caliper is not None
        )

    def allowed(self, idx_from: np.ndarray, idx_to: np.ndarray) -> np.ndarray:
        """Boolean ``(len(idx_from), len(idx_to))`` matrix of allowed pairs."""
        ok = np.ones((len(idx_from), len(idx_to)), dtype=bool)
        if self.exact_codes is not None:
            ok &= (
                self.exact_codes[idx_from][:, None] == self.exact_codes[idx_to][None, :]
            )
        if self.cal_X is not None and self.cal_width is not None:
            for j, width in enumerate(self.cal_width):
                col = self.cal_X[:, j]
                ok &= np.abs(col[idx_from][:, None] - col[idx_to][None, :]) <= width
        if self.ps_caliper is not None and self.pscore is not None:
            ps = self.pscore
            ok &= np.abs(ps[idx_from][:, None] - ps[idx_to][None, :]) <= self.ps_caliper
        return ok

    def apply(
        self, block: np.ndarray, idx_from: np.ndarray, idx_to: np.ndarray
    ) -> np.ndarray:
        """``block`` with every forbidden pair set to ``inf``."""
        if not self.active:
            return block
        out = np.array(block, dtype=float, copy=True)
        out[~self.allowed(idx_from, idx_to)] = np.inf
        return out

    def summary(self, treated: np.ndarray, idx_t: np.ndarray) -> Dict[str, Any]:
        """Counts for ``model_info``: treated units with no exact partner."""
        info: Dict[str, Any] = {}
        if self.exact_codes is not None:
            codes = self.exact_codes
            ctrl_cells = set(codes[np.asarray(treated) == 0].tolist())
            no_cell = int(sum(1 for i in idx_t if codes[i] not in ctrl_cells))
            info["n_exact_cells"] = int(len(np.unique(codes)))
            info["n_treated_without_exact_control"] = no_cell
        if self.cal_names and self.cal_width is not None:
            info["covariate_caliper_widths"] = dict(
                zip(self.cal_names, [float(w) for w in self.cal_width])
            )
        return info


def build_constraints(
    clean: pd.DataFrame,
    *,
    exact: Sequence[str],
    calipers: Mapping[str, float],
    caliper_scale: str,
) -> PairConstraints:
    """Exact-cell codes and caliper columns over the matching sample."""
    pc = PairConstraints()
    if exact:
        pc.exact_codes = (
            clean.groupby(list(exact), sort=True, dropna=False)
            .ngroup()
            .to_numpy(dtype=np.int64)
        )
    if calipers:
        names = list(calipers)
        bad = [
            c
            for c in names
            if not pd.api.types.is_numeric_dtype(clean[c])
            or pd.api.types.is_bool_dtype(clean[c])
        ]
        if bad:
            raise MethodIncompatibility(
                f"match: covariate_calipers need numeric columns; {bad} are not.",
                recovery_hint=(
                    "Use exact= for a categorical variable; a caliper is a "
                    "distance on a numeric one (MatchIt refuses it too)."
                ),
                diagnostics={"non_numeric": bad},
            )
        X = clean[names].to_numpy(dtype=float)
        width = np.array([calipers[c] for c in names], dtype=float)
        if caliper_scale == "sd":
            sd = np.std(X, axis=0, ddof=1)
            width = width * sd
        pc.cal_X = X
        pc.cal_width = width
        pc.cal_names = names
    return pc
