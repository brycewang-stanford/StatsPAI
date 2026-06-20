"""Estimand specification: the engine-neutral description of *what* to fit.

A :class:`EstimandSpec` captures one estimand (OLS / IV / fixed-effects / DiD)
in a form every backend adapter can consume, regardless of whether the user
expressed it as a fixest-style formula or as structured ``y`` / ``treatment`` /
``covariates`` arguments. The spec carries both representations and keeps them
in sync so a pyfixest adapter can hand the formula straight through while a
linearmodels adapter reads the structured fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Canonical estimand aliases → internal key. Keeps the public surface forgiving
# (``"2sls"``, ``"tsls"``, ``"reg"`` all land on the same adapter logic).
_ESTIMAND_ALIASES = {
    "reg": "ols",
    "regress": "ols",
    "lm": "ols",
    "feols": "feols",
    "hdfe": "feols",
    "fe": "feols",
    "panel_fe": "feols",
    "twfe": "feols",
    "iv": "iv",
    "ivreg": "iv",
    "ivreg2": "iv",
    "2sls": "iv",
    "tsls": "iv",
    "ivreghdfe": "iv",
    "did": "did",
    "did_2x2": "did",
    "poisson": "poisson",
    "fepois": "poisson",
    "ppml": "poisson",
    "ppmlhdfe": "poisson",
    "dml": "dml",
    "double_ml": "dml",
}


def canonical_estimand(name: str) -> str:
    key = (name or "").strip().lower()
    return _ESTIMAND_ALIASES.get(key, key)


@dataclass
class EstimandSpec:
    """Engine-neutral description of one estimand.

    Either ``formula`` (fixest 1-3 part syntax) or the structured fields
    (``y`` + ``treatment`` + ``covariates`` …) must pin down the model. The
    constructor helpers :meth:`from_kwargs` and :meth:`from_result` fill both
    representations so downstream adapters can pick whichever they prefer.

    Parameters
    ----------
    estimand : str
        Canonical estimand key (``"ols"``, ``"feols"``, ``"iv"``, ``"did"``,
        ``"poisson"``, ``"dml"`` …).
    data : pandas.DataFrame
    formula : str, optional
        fixest-style: ``"y ~ x1 + x2"`` (OLS), ``"y ~ x | fe1 + fe2"`` (FE),
        ``"y ~ x | fe | endog ~ z1 + z2"`` (IV with FE).
    y, treatment : str, optional
        Outcome and focal regressor (the coefficient cross-validation
        reconciles by default).
    covariates : list of str, optional
    fixed_effects : list of str, optional
    endog : list of str, optional
        Endogenous regressors (IV).
    instruments : list of str, optional
    cluster : list of str, optional
    weights : str, optional
    vcov : str, optional
        Requested variance estimator (``"iid"``, ``"HC1"``, ``"cluster"`` …).
    term : str, optional
        Focal coefficient to reconcile. Defaults to ``treatment`` (or the
        first endogenous regressor for IV).
    extra : dict
        Estimand-specific extras forwarded verbatim (e.g. DiD ``time`` / ``unit``
        / ``gname`` columns).
    """

    estimand: str
    data: pd.DataFrame
    formula: Optional[str] = None
    y: Optional[str] = None
    treatment: Optional[str] = None
    covariates: List[str] = field(default_factory=list)
    fixed_effects: List[str] = field(default_factory=list)
    endog: List[str] = field(default_factory=list)
    instruments: List[str] = field(default_factory=list)
    cluster: List[str] = field(default_factory=list)
    weights: Optional[str] = None
    vcov: Optional[str] = None
    term: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- construction ----------------------------------------------------- #
    @classmethod
    def from_kwargs(
        cls,
        data: pd.DataFrame,
        estimand: str,
        *,
        formula: Optional[str] = None,
        y: Optional[str] = None,
        treatment: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        fixed_effects: Optional[List[str]] = None,
        endog: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None,
        cluster: Optional[Union[str, List[str]]] = None,
        weights: Optional[str] = None,
        vcov: Optional[str] = None,
        term: Optional[str] = None,
        **extra: Any,
    ) -> "EstimandSpec":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a pandas DataFrame.")
        est = canonical_estimand(estimand)
        spec = cls(
            estimand=est,
            data=data,
            formula=formula,
            y=y,
            treatment=treatment,
            covariates=list(covariates or []),
            fixed_effects=list(fixed_effects or []),
            endog=list(endog or []),
            instruments=list(instruments or []),
            cluster=_aslist(cluster),
            weights=weights,
            vcov=vcov,
            term=term,
            extra=dict(extra),
        )
        spec._reconcile_formula_and_fields()
        spec._validate()
        return spec

    @classmethod
    def from_result(cls, result: Any) -> "EstimandSpec":
        """Best-effort recovery of a spec from a fitted StatsPAI result.

        Reads the metadata StatsPAI result objects commonly carry
        (``estimand`` / ``formula`` / ``data`` / treatment column). Raises a
        clear error when the result does not expose enough to re-run it
        elsewhere — rather than guessing and silently cross-validating the
        wrong model.
        """
        data = _first_attr(result, ["data", "_data", "df", "_df"])
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Cannot recover the dataset from this result object; pass "
                "`data=` and the estimand explicitly to sp.cross_validate()."
            )
        formula = _first_attr(result, ["formula", "_formula", "model_formula"])
        estimand = _first_attr(
            result, ["estimand", "_estimand", "method", "model_type"]
        )
        treatment = _first_attr(
            result, ["treatment", "_treatment", "treatment_var", "focal_var"]
        )
        y = _first_attr(result, ["y", "_y", "outcome", "depvar"])
        if estimand is None and formula is None:
            raise ValueError(
                "Result does not expose an estimand or formula; pass them "
                "explicitly to sp.cross_validate()."
            )
        return cls.from_kwargs(
            data,
            estimand or "ols",
            formula=formula,
            y=y,
            treatment=treatment,
        )

    # -- derived ---------------------------------------------------------- #
    def focal_term(self) -> str:
        """Name of the coefficient cross-validation reconciles."""
        if self.term:
            return self.term
        if self.estimand == "did":
            # CS-DID reconciles the overall aggregated ATT, not a regressor.
            return "ATT"
        if self.estimand == "iv" and self.endog:
            return self.endog[0]
        if self.treatment:
            return self.treatment
        # Fall back to the first RHS term parsed from the formula.
        if self.formula:
            rhs = _rhs_terms(self.formula)
            if rhs:
                return rhs[0]
        raise ValueError(
            "Could not determine the focal coefficient to reconcile; pass "
            "`term=` (the coefficient name) to sp.cross_validate()."
        )

    def build_formula(self) -> str:
        """Construct a fixest-style formula from the structured fields."""
        if self.formula:
            return self.formula
        if not self.y:
            raise ValueError("Need `y` (or a `formula`) to build a model.")
        rhs_main = [self.treatment] if self.treatment else []
        rhs_main += [c for c in self.covariates if c not in rhs_main]
        main = " + ".join(rhs_main) if rhs_main else "1"
        parts = [f"{self.y} ~ {main}"]
        if self.fixed_effects:
            parts.append(" + ".join(self.fixed_effects))
        if self.endog:
            iv = f"{' + '.join(self.endog)} ~ {' + '.join(self.instruments)}"
            if not self.fixed_effects:
                parts.append("0")  # no FE block placeholder
            parts.append(iv)
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimand": self.estimand,
            "formula": self.formula,
            "y": self.y,
            "treatment": self.treatment,
            "covariates": list(self.covariates),
            "fixed_effects": list(self.fixed_effects),
            "endog": list(self.endog),
            "instruments": list(self.instruments),
            "cluster": list(self.cluster),
            "weights": self.weights,
            "vcov": self.vcov,
            "term": self.term,
            "n_obs": int(len(self.data)),
            "extra": dict(self.extra),
        }

    # -- internals -------------------------------------------------------- #
    def _reconcile_formula_and_fields(self) -> None:
        """Fill structured fields from a formula (and pin a focal term).

        A formula and explicit structured args (e.g. ``treatment=``) can be
        passed together — the explicit arg pins the focal term while the
        formula still fills ``y`` / fixed effects / IV blocks. We therefore
        always parse a supplied formula, overriding only the fields the user
        left unset.
        """
        if self.formula:
            parsed = _parse_fixest_formula(self.formula)
            if self.y is None:
                self.y = parsed["y"]
            if not self.fixed_effects and parsed["fixed_effects"]:
                self.fixed_effects = parsed["fixed_effects"]
            if not self.endog and parsed["endog"]:
                self.endog = parsed["endog"]
                self.instruments = parsed["instruments"]
            exog = parsed["exog"]
            if self.treatment is None and exog:
                self.treatment = exog[0]
            if not self.covariates:
                self.covariates = [c for c in exog if c != self.treatment]
        elif self.estimand != "did":
            # Structured input → synthesise a formula for formula-native engines.
            # CS-DID is not formula-based (it takes g/t/i columns), so skip.
            try:
                self.formula = self.build_formula()
            except ValueError:
                self.formula = None

    def _validate(self) -> None:
        cols = set(self.data.columns)
        if self.estimand == "did":
            self._validate_did(cols)
            return
        needed: List[str] = []
        for f in (self.y, self.treatment, self.weights):
            if f:
                needed.append(f)
        needed += [c for c in self.covariates]
        needed += [c for c in self.fixed_effects]
        needed += [c for c in self.endog]
        needed += [c for c in self.instruments]
        needed += [c for c in self.cluster]
        missing = [c for c in needed if c and c not in cols]
        if missing:
            raise ValueError(
                f"Columns not found in data: {sorted(set(missing))}. "
                f"Available: {sorted(cols)[:20]}..."
            )
        if self.estimand == "iv" and not self.endog:
            raise ValueError(
                "IV estimand requires endogenous regressors; pass `endog=` and "
                "`instruments=` or a 3-part fixest formula."
            )

    def _validate_did(self, cols: set) -> None:
        """Callaway–Sant'Anna needs y plus group (g), time (t), unit (i)."""
        g = self.extra.get("g")
        t = self.extra.get("t")
        i = self.extra.get("i")
        if not all([self.y, g, t, i]):
            raise ValueError(
                "DiD (Callaway–Sant'Anna) cross-check needs `y`, `g` (cohort / "
                "first-treatment period, 0 = never treated), `t` (time) and "
                "`i` (unit id). Example: sp.cross_validate(df, 'did', y='lemp', "
                "g='first_treat', t='year', i='countyreal')."
            )
        missing = [c for c in (self.y, g, t, i) if c not in cols]
        if missing:
            avail = sorted(str(c) for c in cols)
            miss = sorted(str(c) for c in set(missing))
            raise ValueError(
                f"DiD columns not found in data: {miss}. " f"Available: {avail[:20]}..."
            )


# --------------------------------------------------------------------------- #
# Formula parsing (fixest 1-3 part syntax)
# --------------------------------------------------------------------------- #


def _parse_fixest_formula(formula: str) -> Dict[str, Any]:
    """Parse ``y ~ exog | fe | endog ~ instr`` into its components."""
    out: Dict[str, Any] = {
        "y": None,
        "exog": [],
        "fixed_effects": [],
        "endog": [],
        "instruments": [],
    }
    parts = [p.strip() for p in formula.split("|")]
    lhs_rhs = parts[0].split("~")
    if len(lhs_rhs) != 2:
        raise ValueError(f"Malformed formula (need one '~'): {formula!r}")
    out["y"] = lhs_rhs[0].strip()
    out["exog"] = _split_terms(lhs_rhs[1])
    # Blocks after the first: a block containing '~' is the IV block
    # (``endog ~ instruments``); any other non-empty block is fixed effects.
    # This disambiguates ``y ~ x | fe`` from ``y ~ x | endog ~ z``.
    for block in parts[1:]:
        if "~" in block:
            endog, instr = block.split("~")
            out["endog"] = _split_terms(endog)
            out["instruments"] = _split_terms(instr)
        elif block not in ("", "0", "1"):
            out["fixed_effects"] = _split_terms(block)
    return out


def _split_terms(s: str) -> List[str]:
    return [t.strip() for t in re.split(r"\+", s) if t.strip() not in ("", "1")]


def _rhs_terms(formula: str) -> List[str]:
    try:
        return [str(t) for t in _parse_fixest_formula(formula)["exog"]]
    except ValueError:
        return []


def _aslist(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _first_attr(obj: Any, names: List[str]) -> Any:
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None
