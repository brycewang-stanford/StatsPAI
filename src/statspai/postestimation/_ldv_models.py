"""Adapters for the limited-dependent-variable margins.

Each adapter writes a model's Stata ``predict()`` statistics as a function
of its equation indices ``x'b_e`` and ancillary parameters; the engine in
:mod:`._ldv_margins` differentiates and averages them. :func:`ldv_model`
picks the adapter for a fitted result (``None`` when it is not one of these
models). A new estimator plugs in by storing ``result._ml_state`` (its
estimation-scale ``theta``, ``vcov`` and per-equation term names) and adding
an adapter here.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import special, stats

from ..exceptions import MethodIncompatibility
from ._design import (
    _FormulaDesign,
    _simple_part,
    _structural_formula,
    _TermDesign,
    model_formula,
)

__all__ = ["ldv_model"]


# ---------------------------------------------------------------------------
# prediction options
# ---------------------------------------------------------------------------

_LIMITS_RE = re.compile(
    r"^\s*(e|ystar|pr)\s*(?:\(\s*([^,]*?)\s*,\s*([^)]*?)\s*\))?\s*$"
)


def _limit_value(text: str, default: float) -> float:
    text = text.strip()
    if text in ("", "."):
        return default
    try:
        return float(text)
    except ValueError as exc:
        raise MethodIncompatibility(
            f"margins: cannot read the limit {text!r} in predict=.",
            recovery_hint="Write limits as numbers or '.', e.g. predict='e(0,.)'.",
        ) from exc


class _Option:
    """A parsed ``predict=`` option: name plus optional (a, b) limits."""

    def __init__(self, name: str, a: float = -np.inf, b: float = np.inf) -> None:
        self.name = name
        self.a = a
        self.b = b

    def __repr__(self) -> str:
        if self.name in ("e", "ystar", "pr") and (
            np.isfinite(self.a) or np.isfinite(self.b)
        ):
            fmt = lambda v: "." if not np.isfinite(v) else f"{v:g}"  # noqa: E731
            return f"{self.name}({fmt(self.a)},{fmt(self.b)})"
        return self.name


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------


class _LDVModel:
    """What the engine needs from a fitted limited-dependent-variable model."""

    kind: str = ""
    options: Tuple[str, ...] = ()
    default: str = "xb"
    labels: Dict[str, str] = {}

    def __init__(
        self,
        result: Any,
        theta: np.ndarray,
        V: np.ndarray,
        eq_terms: List[List[str]],
        eq_pos: List[np.ndarray],
        formula: Optional[str],
        nobs: Optional[int],
    ) -> None:
        self.result = result
        self.theta = np.asarray(theta, dtype=float)
        self.V = np.asarray(V, dtype=float)
        self.eq_terms = eq_terms
        self.eq_pos = [np.asarray(p, dtype=int) for p in eq_pos]
        self.formula = formula
        self.nobs = nobs
        if self.V.shape != (self.theta.size, self.theta.size):
            raise MethodIncompatibility(
                "margins: the stored covariance does not match the parameter "
                f"vector ({self.V.shape} for {self.theta.size} parameters).",
            )

    # -- options -----------------------------------------------------------
    def parse(self, predict: Optional[str]) -> _Option:
        text = self.default if predict is None else str(predict).strip()
        name = text.split("(")[0].strip().lower()
        if name not in self.options:
            raise MethodIncompatibility(
                f"margins: predict={predict!r} is not available after "
                f"{self.kind}; options are {list(self.options)}.",
                recovery_hint=f"Stata's default after {self.kind} is "
                f"predict({self.default}).",
                diagnostics={"predict": predict, "available": list(self.options)},
            )
        if "(" in text and name not in ("e", "ystar", "pr"):
            raise MethodIncompatibility(
                f"margins: predict={predict!r} takes no arguments after {self.kind}."
            )
        return _Option(name)

    def outcomes(self, opt: _Option) -> List[Any]:
        return [None]

    def expression(self, opt: _Option) -> str:
        return self.labels.get(opt.name, repr(opt))

    def scale(self, theta: np.ndarray) -> float:
        return 1.0

    def predict(self, theta: np.ndarray, idx: List[np.ndarray], opt: _Option) -> Any:
        raise NotImplementedError

    # -- data ----------------------------------------------------------------
    def designs(self, frame: pd.DataFrame) -> List[Any]:
        out = []
        for terms in self.eq_terms:
            if all(_simple_part(p) for t in terms for p in t.split(":")):
                out.append(_TermDesign(terms, frame))
            elif self.formula is not None:
                out.append(
                    _FormulaDesign(_structural_formula(self.formula), frame, terms)
                )
            else:
                raise MethodIncompatibility(
                    f"margins: the terms {terms} of this {self.kind} fit are "
                    "transformations and the fit did not record its formula.",
                    recovery_hint="Create the transformed columns in the data "
                    "and fit on them.",
                )
        return out

    def sample_mask(self, frame: pd.DataFrame, variables: Sequence[str]) -> np.ndarray:
        cols = [v for v in variables if v in frame.columns]
        dep = self.dependent()
        if dep is not None and dep in frame.columns:
            cols.append(dep)
        mask = (
            frame[cols].notna().all(axis=1).to_numpy()
            if cols
            else np.ones(len(frame), bool)
        )
        # a copy: under pandas copy-on-write .to_numpy() is read-only, and
        # the subclasses narrow this mask in place
        return np.array(mask, dtype=bool)

    def dependent(self) -> Optional[str]:
        dep = (getattr(self.result, "data_info", None) or {}).get("dependent_var")
        return dep if isinstance(dep, str) else None


def _cdf_pdf(link: str, z: np.ndarray) -> np.ndarray:
    if link == "logit":
        return np.asarray(special.expit(z))
    return np.asarray(special.ndtr(z))


class _Multinomial(_LDVModel):
    kind = "mlogit"
    options = ("pr",)
    default = "pr"

    def __init__(self, result: Any, categories: List[Any], base: Any, **kw: Any):
        super().__init__(result, **kw)
        self.categories = list(categories)
        self.base_pos = [i for i, c in enumerate(categories) if _same(c, base)][0]

    def outcomes(self, opt: _Option) -> List[Any]:
        return list(self.categories)

    def expression(self, opt: _Option) -> str:
        return "Pr(y==outcome)"

    def predict(self, theta: np.ndarray, idx: List[np.ndarray], opt: _Option) -> Any:
        n = idx[0].shape[0]
        cols = list(idx)
        cols.insert(self.base_pos, np.zeros(n))
        Z = np.column_stack(cols)
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)


class _Ordered(_LDVModel):
    options = ("pr", "xb")
    default = "pr"
    labels = {"xb": "Linear prediction (cutpoints excluded)"}

    def __init__(
        self, result: Any, categories: List[Any], link: str, cut_pos: Any, **kw: Any
    ) -> None:
        super().__init__(result, **kw)
        self.categories = list(categories)
        self.link = link
        self.kind = "ologit" if link == "logit" else "oprobit"
        self.cut_pos = np.asarray(cut_pos, dtype=int)

    def outcomes(self, opt: _Option) -> List[Any]:
        return list(self.categories) if opt.name == "pr" else [None]

    def expression(self, opt: _Option) -> str:
        return "Pr(y==outcome)" if opt.name == "pr" else self.labels["xb"]

    def predict(self, theta: np.ndarray, idx: List[np.ndarray], opt: _Option) -> Any:
        xb = idx[0]
        if opt.name == "xb":
            return xb[:, None]
        cuts = theta[self.cut_pos]
        n = xb.shape[0]
        cum = [np.zeros(n)] + [_cdf_pdf(self.link, c - xb) for c in cuts] + [np.ones(n)]
        return np.diff(np.column_stack(cum), axis=1)


def _tobit_moments(
    m: np.ndarray, s: float, a: float, b: float, what: str
) -> np.ndarray:
    """``Pr(a < y* < b)``, ``E(y* | a < y* < b)`` or ``E(max(a, min(b, y*)))``
    for ``y* ~ N(m, s^2)`` (Stata's ``predict, pr / e / ystar``)."""
    lo = (a - m) / s if np.isfinite(a) else np.full_like(m, -np.inf)
    hi = (b - m) / s if np.isfinite(b) else np.full_like(m, np.inf)
    P = special.ndtr(hi) - special.ndtr(lo)
    phi_lo = np.where(np.isfinite(lo), stats.norm.pdf(lo), 0.0)
    phi_hi = np.where(np.isfinite(hi), stats.norm.pdf(hi), 0.0)
    if what == "pr":
        return np.asarray(P)
    if what == "e":
        return np.asarray(m + s * (phi_lo - phi_hi) / P)
    below = a * special.ndtr(lo) if np.isfinite(a) else 0.0
    above = b * special.ndtr(-hi) if np.isfinite(b) else 0.0
    return np.asarray(below + above + m * P + s * (phi_lo - phi_hi))


class _Tobit(_LDVModel):
    kind = "tobit"
    options = ("xb", "e", "ystar", "pr")
    default = "xb"

    def __init__(
        self, result: Any, ll: Optional[float], ul: Optional[float], **kw: Any
    ):
        super().__init__(result, **kw)
        self.ll = -np.inf if ll is None else float(ll)
        self.ul = np.inf if ul is None else float(ul)

    def parse(self, predict: Optional[str]) -> _Option:
        text = self.default if predict is None else str(predict).strip()
        m = _LIMITS_RE.match(text.lower())
        if m is None:
            return super().parse(predict)
        name, a, b = m.group(1), m.group(2), m.group(3)
        if a is None:
            # bare 'e' / 'ystar' / 'pr': the model's own censoring limits
            return _Option(name, self.ll, self.ul)
        return _Option(name, _limit_value(a, -np.inf), _limit_value(b, np.inf))

    def expression(self, opt: _Option) -> str:
        if opt.name == "xb":
            return "Linear prediction"
        return {
            "e": "E(y | a < y < b)",
            "ystar": "E(y*), y* = max(a, min(y, b))",
            "pr": "Pr(a < y < b)",
        }[opt.name] + f", predict({opt!r})"

    def scale(self, theta: np.ndarray) -> float:
        return float(np.exp(theta[-1]))

    def predict(self, theta: np.ndarray, idx: List[np.ndarray], opt: _Option) -> Any:
        xb = idx[0]
        if opt.name == "xb":
            return xb[:, None]
        s = float(np.exp(theta[-1]))
        return _tobit_moments(xb, s, opt.a, opt.b, opt.name)[:, None]


class _IntReg(_Tobit):
    """``intreg``: the tobit statistics with no censoring limits of its
    own; rows with both bounds missing are outside ``e(sample)``."""

    kind = "intreg"

    def dependent(self) -> Optional[str]:
        return None

    def sample_mask(self, frame: pd.DataFrame, variables: Sequence[str]) -> np.ndarray:
        mask = super().sample_mask(frame, variables)
        bounds = (getattr(self.result, "_ml_state", None) or {}).get("bounds")
        if bounds and all(b in frame.columns for b in bounds):
            mask &= frame[list(bounds)].notna().any(axis=1).to_numpy()
        return mask


class _GOLogit(_LDVModel):
    """``gologit``: Pr(Y > c_j) = F(a_j + x'b_j), one equation per cut."""

    kind = "gologit"
    options = ("pr",)
    default = "pr"

    def __init__(self, result: Any, categories: List[Any], link: str, **kw: Any):
        super().__init__(result, **kw)
        self.categories = list(categories)
        self.link = link

    def outcomes(self, opt: _Option) -> List[Any]:
        return list(self.categories)

    def expression(self, opt: _Option) -> str:
        return "Pr(y==outcome)"

    def predict(self, theta: np.ndarray, idx: List[np.ndarray], opt: _Option) -> Any:
        n = idx[0].shape[0]
        F = [_cdf_pdf(self.link, z) for z in idx]
        cum = np.column_stack([np.ones(n)] + F + [np.zeros(n)])
        return cum[:, :-1] - cum[:, 1:]


class _Heckman(_LDVModel):
    kind = "heckman"
    options = ("xb", "ycond", "yexpected", "psel", "xbsel")
    default = "xb"
    labels = {
        "xb": "Linear prediction",
        "ycond": "E(y | selected), predict(ycond)",
        "yexpected": "E(y*) with y* = 0 when not selected, predict(yexpected)",
        "psel": "Pr(selected), predict(psel)",
        "xbsel": "Linear prediction of the selection equation, predict(xbsel)",
    }

    def __init__(self, result: Any, select: Optional[str], y: str, **kw: Any):
        super().__init__(result, **kw)
        self.select = select
        self.y = y

    def dependent(self) -> Optional[str]:
        return None  # y is missing for the rows not selected

    def sample_mask(self, frame: pd.DataFrame, variables: Sequence[str]) -> np.ndarray:
        mask = super().sample_mask(frame, variables)
        if self.select is not None and self.select in frame.columns:
            mask &= frame[self.select].notna().to_numpy()
        return mask

    def predict(self, theta: np.ndarray, idx: List[np.ndarray], opt: _Option) -> Any:
        xb, zg = idx
        if opt.name == "xb":
            return xb[:, None]
        if opt.name == "xbsel":
            return zg[:, None]
        if opt.name == "psel":
            return special.ndtr(zg)[:, None]
        rho = np.tanh(theta[-2])
        sigma = np.exp(theta[-1])
        mills = np.exp(stats.norm.logpdf(zg) - special.log_ndtr(zg))
        ycond = xb + rho * sigma * mills
        if opt.name == "ycond":
            return ycond[:, None]
        return (special.ndtr(zg) * ycond)[:, None]


class _HeckmanTwostep(_LDVModel):
    kind = "heckman (twostep)"
    options = ("xb",)
    default = "xb"
    labels = {"xb": "Linear prediction"}

    def dependent(self) -> Optional[str]:
        return None

    def predict(self, theta: np.ndarray, idx: List[np.ndarray], opt: _Option) -> Any:
        return idx[0][:, None]


class _HeckProbit(_LDVModel):
    kind = "heckprobit"
    options = ("pmargin", "p11", "p10", "p01", "p00", "psel", "pcond", "xb", "xbsel")
    default = "pmargin"
    labels = {
        "pmargin": "Pr(y=1), predict(pmargin)",
        "p11": "Pr(y=1, selected), predict(p11)",
        "p10": "Pr(y=1, not selected), predict(p10)",
        "p01": "Pr(y=0, selected), predict(p01)",
        "p00": "Pr(y=0, not selected), predict(p00)",
        "psel": "Pr(selected), predict(psel)",
        "pcond": "Pr(y=1 | selected), predict(pcond)",
        "xb": "Linear prediction",
        "xbsel": "Linear prediction of the selection equation, predict(xbsel)",
    }

    def __init__(self, result: Any, select: Optional[str], y: str, **kw: Any):
        super().__init__(result, **kw)
        self.select = select
        self.y = y

    def dependent(self) -> Optional[str]:
        return None

    def sample_mask(self, frame: pd.DataFrame, variables: Sequence[str]) -> np.ndarray:
        mask = super().sample_mask(frame, variables)
        if self.select is not None and self.select in frame.columns:
            mask &= frame[self.select].notna().to_numpy()
        return mask

    def predict(self, theta: np.ndarray, idx: List[np.ndarray], opt: _Option) -> Any:
        from ..regression._bivariate_normal import binormal_cdf

        xb, zg = idx
        name = opt.name
        if name == "xb":
            return xb[:, None]
        if name == "xbsel":
            return zg[:, None]
        if name == "pmargin":
            return special.ndtr(xb)[:, None]
        if name == "psel":
            return special.ndtr(zg)[:, None]
        rho = float(np.tanh(theta[-1]))
        sign_y = 1.0 if name in ("p11", "p10", "pcond") else -1.0
        if name in ("p11", "p01", "pcond"):
            p = binormal_cdf(sign_y * xb, zg, sign_y * rho)
            if name == "pcond":
                p = p / special.ndtr(zg)
            return np.asarray(p)[:, None]
        # not selected: Pr(y = ., s = 0) splits by the latent outcome
        p = binormal_cdf(sign_y * xb, -zg, -sign_y * rho)
        return np.asarray(p)[:, None]


class _IVProbit(_LDVModel):
    """``ivprobit`` (ML): the linear prediction, or ``Pr(y=1)`` as Stata's
    average structural function (``asf``, conditional on the reduced-form
    errors at their observed values) or fixed ASF (``fixedasf``, the
    outcome equation alone)."""

    kind = "ivprobit"
    options = ("xb", "pr", "fixedasf")
    default = "xb"
    labels = {
        "xb": "Fitted values (linear prediction)",
        "pr": "Average structural function probabilities, predict(pr)",
        "fixedasf": "Fixed average structural function probabilities, "
        "predict(pr fixedasf)",
    }

    def __init__(
        self,
        result: Any,
        endog: List[str],
        rf_terms: List[List[str]],
        rf_pos: List[np.ndarray],
        anc: Dict[str, int],
        **kw: Any,
    ) -> None:
        super().__init__(result, **kw)
        self.endog = list(endog)
        self.rf_terms = rf_terms
        self.rf_pos = [np.asarray(p, dtype=int) for p in rf_pos]
        self.anc = anc
        self._base: Optional[Tuple[List[np.ndarray], np.ndarray]] = None

    def parse(self, predict: Optional[str]) -> _Option:
        text = (self.default if predict is None else str(predict)).lower().split()
        words = set(text)
        if words in ({"pr", "fixedasf"}, {"fixedasf"}):
            return _Option("fixedasf")
        if words in ({"pr", "asf"}, {"asf"}):
            return _Option("pr")
        return super().parse(predict)

    def bind_base(self, frame: pd.DataFrame) -> None:
        """Reduced-form designs and endogenous values at the observed data:
        the ASF holds the reduced-form errors there (Stata keeps copies of
        the original variables while margins sets counterfactual values)."""
        Xs = [_TermDesign(t, frame).build(frame) for t in self.rf_terms]
        Y2 = frame[self.endog].to_numpy(dtype=float)
        self._base = (Xs, Y2)

    def _errors_cov(self, theta: np.ndarray) -> np.ndarray:
        p = len(self.endog)
        sd = np.ones(p + 1)
        for j in range(p):
            sd[j + 1] = np.exp(theta[self.anc[f"/lnsigma{j + 2}"]])
        S = np.diag(sd**2)
        for j in range(p):
            for i in range(j + 1):
                r = np.tanh(theta[self.anc[f"/athrho{j + 2}_{i + 1}"]])
                S[j + 1, i] = S[i, j + 1] = r * sd[j + 1] * sd[i]
        return S

    def predict(self, theta: np.ndarray, idx: List[np.ndarray], opt: _Option) -> Any:
        xb = idx[0]
        if opt.name == "xb":
            return xb[:, None]
        if opt.name == "fixedasf":
            return special.ndtr(xb)[:, None]
        if self._base is None or self._base[1].shape[0] != xb.shape[0]:
            raise MethodIncompatibility(
                "margins: the average structural function after ivprobit "
                "averages over the observed reduced-form errors, so it is "
                "defined for averages over the sample (method='ame'), not "
                "at the means.",
                recovery_hint="Use method='ame', or predict='pr fixedasf'.",
            )
        Xs, Y2 = self._base
        Vres = np.column_stack(
            [Y2[:, j] - X @ theta[p] for j, (X, p) in enumerate(zip(Xs, self.rf_pos))]
        )
        S = self._errors_cov(theta)
        coef = np.linalg.solve(S[1:, 1:], S[1:, 0])
        resid_var = S[0, 0] - S[0, 1:] @ coef
        return special.ndtr((xb + Vres @ coef) / np.sqrt(resid_var))[:, None]


def _same(a: Any, b: Any) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def ldv_model(result: Any) -> Optional[_LDVModel]:
    """The margins adapter for ``result``, or None when it is not an LDV fit."""
    state = getattr(result, "_ml_state", None)
    if isinstance(state, dict):
        return _from_state(result, state)
    pred = getattr(result, "predict", None)
    args = getattr(pred, "args", None)
    func = getattr(pred, "func", None)
    if args and getattr(func, "__name__", "") == "_bound_category_probabilities":
        return _from_category_model(result, args)
    mi = getattr(result, "model_info", None) or {}
    if mi.get("model_type") == "IV probit":
        return _from_ivprobit(result, mi)
    return None


def _from_state(result: Any, st: Dict[str, Any]) -> _LDVModel:
    model = st["model"]
    theta = np.asarray(st["theta"], dtype=float)
    V = np.asarray(st["vcov"], dtype=float)
    formula = st.get("formula")
    nobs = st.get("nobs")
    if model in ("tobit", "intreg"):
        k = len(st["terms"])
        cls = _Tobit if model == "tobit" else _IntReg
        return cls(
            result,
            st.get("ll"),
            st.get("ul"),
            theta=theta,
            V=V,
            eq_terms=[list(st["terms"])],
            eq_pos=[np.arange(k)],
            formula=formula,
            nobs=nobs,
        )
    if model in ("heckman", "heckprobit"):
        kx, kz = len(st["out_terms"]), len(st["sel_terms"])
        cls = _Heckman if model == "heckman" else _HeckProbit
        return cls(
            result,
            st.get("select"),
            st["y"],
            theta=theta,
            V=V,
            eq_terms=[list(st["out_terms"]), list(st["sel_terms"])],
            eq_pos=[np.arange(kx), np.arange(kx, kx + kz)],
            formula=None,
            nobs=nobs,
        )
    if model == "gologit":
        terms = list(st["terms"])
        k = len(terms)
        J = len(st["categories"]) - 1
        return _GOLogit(
            result,
            list(st["categories"]),
            st["link"],
            theta=theta,
            V=V,
            eq_terms=[terms for _ in range(J)],
            eq_pos=[np.arange(j * k, (j + 1) * k) for j in range(J)],
            formula=formula,
            nobs=nobs,
        )
    if model == "heckman_twostep":
        k = len(st["out_terms"])
        return _HeckmanTwostep(
            result,
            theta=theta[:k],
            V=V[:k, :k],
            eq_terms=[list(st["out_terms"])],
            eq_pos=[np.arange(k)],
            formula=None,
            nobs=nobs,
        )
    raise MethodIncompatibility(f"margins: unknown model state {model!r}.")


def _from_category_model(result: Any, args: Tuple[Any, ...]) -> Optional[_LDVModel]:
    kind, _coef, _cuts, categories, names, formula = args[:6]
    params = result.params
    V = np.asarray((result.data_info or {}).get("var_cov"), dtype=float)
    nobs = (result.data_info or {}).get("nobs")
    mi = result.model_info or {}
    if kind == "mlogit":
        theta = params.to_numpy(dtype=float)
        if mi.get("rrr"):
            # rrr=True stores exp(b) and its delta-method covariance
            V = V / np.outer(theta, theta)
            theta = np.log(theta)
        base = mi.get("base_category")
        eq_terms: List[List[str]] = []
        eq_pos: List[np.ndarray] = []
        for cat in categories:
            if _same(cat, base):
                continue
            prefix = f"[{cat}]"
            pos = [i for i, n in enumerate(params.index) if str(n).startswith(prefix)]
            eq_terms.append([str(params.index[i])[len(prefix) :] for i in pos])
            eq_pos.append(np.asarray(pos))
        return _Multinomial(
            result,
            list(categories),
            base,
            theta=theta,
            V=V,
            eq_terms=eq_terms,
            eq_pos=eq_pos,
            formula=formula,
            nobs=nobs,
        )
    if kind in ("ologit", "oprobit"):
        names_all = [str(n) for n in params.index]
        pos = [i for i, n in enumerate(names_all) if not n.startswith("/cut")]
        cut_pos = [i for i, n in enumerate(names_all) if n.startswith("/cut")]
        return _Ordered(
            result,
            list(categories),
            "logit" if kind == "ologit" else "probit",
            cut_pos,
            theta=params.to_numpy(dtype=float),
            V=V,
            eq_terms=[[names_all[i] for i in pos]],
            eq_pos=[np.asarray(pos)],
            formula=formula,
            nobs=nobs,
        )
    return None


def _from_ivprobit(result: Any, mi: Dict[str, Any]) -> _LDVModel:
    di = result.data_info or {}
    y = di.get("dependent_var")
    names = [str(n) for n in result.params.index]
    if mi.get("method") == "twostep":
        raise MethodIncompatibility(
            "margins after ivprobit, twostep is not available: the two-step "
            "coefficients are scaled by Var(u | v) = 1, so neither Pr(y=1) "
            "nor the coefficients' covariance with the first stage is "
            "defined on the structural scale.",
            recovery_hint="Refit with method='mle' (Stata's default ivprobit).",
        )
    prefix = f"{y}:"
    pos = [i for i, n in enumerate(names) if n.startswith(prefix)]
    endog = list(mi.get("endog") or [])
    rf_terms, rf_pos = [], []
    for e in endog:
        ep = [i for i, n in enumerate(names) if n.startswith(f"{e}:")]
        rf_terms.append([names[i][len(e) + 1 :] for i in ep])
        rf_pos.append(np.asarray(ep))
    anc = {n: i for i, n in enumerate(names) if n.startswith("/")}
    return _IVProbit(
        result,
        endog,
        rf_terms,
        rf_pos,
        anc,
        theta=result.params.to_numpy(dtype=float),
        V=np.asarray(di.get("var_cov"), dtype=float),
        eq_terms=[[names[i][len(prefix) :] for i in pos]],
        eq_pos=[np.asarray(pos)],
        formula=model_formula(result),
        nobs=di.get("nobs"),
    )
