"""
Conditional moment tests for probit and tobit fits.

Both models are consistent only if the latent error is normal and
homoskedastic, and neither leaves a residual to look at: the probit
error is never observed, the tobit error only above the limit. The
tests of Newey (1985) and Tauchen (1985) replace every power of the
unobserved error by its expectation given what is observed and ask
whether the resulting sample moments are zero.

With ``M`` the ``(n, J)`` moment contributions, ``G`` the scores, ``H``
the Hessian and ``W = d m / d theta'``, the statistic is

    m' [(M - G H^{-1} W')' (M - G H^{-1} W')]^{-1} m  ~  chi2(J)

(Skeels and Vella 1999, eq. 2.13). ``opg=True`` replaces ``H`` and ``W``
by their outer-product estimates, which turns the statistic into the
explained sum of squares of a regression of ones on ``[G, M]``; that
version is easy to compute by hand and over-rejects in small samples.

References
----------
[@newey1985maximum], [@tauchen1985diagnostic], [@pagan1989diagnostic],
[@skeels1999monte]
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import special, stats

from ..exceptions import MethodIncompatibility

__all__ = ["cmtest"]

_TESTS = ("normality", "heterosc", "skewness", "kurtosis")
_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))

ObsFn = Callable[[np.ndarray], np.ndarray]


def _log_phi(z: np.ndarray) -> np.ndarray:
    return np.asarray(-_LOG_SQRT_2PI - 0.5 * z * z)


def _probit_pieces(
    y: np.ndarray, X: np.ndarray, test: str
) -> Tuple[ObsFn, ObsFn, List[str]]:
    q = 2.0 * y - 1.0

    def obs(t: np.ndarray) -> np.ndarray:
        return np.asarray(special.log_ndtr(q * (X @ t)))

    def psi(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xb = X @ t
        return xb, q * np.exp(_log_phi(xb) - special.log_ndtr(q * xb))

    # E(eps^k | y, x) = (k - 1) E(eps^(k-2) | .) + (-xb)^(k-1) psi, so the
    # k-th moment condition is xb^(k-1) psi up to terms that vanish.
    def mom(t: np.ndarray) -> np.ndarray:
        xb, g = psi(t)
        if test == "heterosc":
            return np.asarray((xb * g)[:, None] * X[:, 1:])
        cols = {"skewness": [xb**2 * g], "kurtosis": [xb**3 * g]}.get(
            test, [xb**2 * g, xb**3 * g]
        )
        return np.column_stack(cols)

    return obs, mom, []


def _tobit_pieces(
    y: np.ndarray,
    X: np.ndarray,
    ll: Optional[float],
    ul: Optional[float],
    test: str,
) -> Tuple[ObsFn, ObsFn, List[str]]:
    k = X.shape[1]
    lo = y <= ll if ll is not None else np.zeros(len(y), dtype=bool)
    hi = y >= ul if ul is not None else np.zeros(len(y), dtype=bool)
    mid = ~lo & ~hi

    def obs(t: np.ndarray) -> np.ndarray:
        xb = X @ t[:k]
        s = np.exp(t[k])
        out = np.zeros(len(y), dtype=np.result_type(t, float))
        out[mid] = _log_phi((y[mid] - xb[mid]) / s) - t[k]
        if lo.any():
            out[lo] = special.log_ndtr((ll - xb[lo]) / s)
        if hi.any():
            out[hi] = special.log_ndtr((xb[hi] - ul) / s)
        return out

    def eps_moment(t: np.ndarray, order: int) -> Tuple[np.ndarray, Any]:
        """E(eps^order | observed): the power itself when uncensored, the
        truncated-normal moment (by recursion) when censored."""
        xb = X @ t[:k]
        s = np.exp(t[k])
        dtype = np.result_type(t, float)
        out = np.zeros(len(y), dtype=dtype)
        out[mid] = (y[mid] - xb[mid]) ** order
        for mask, limit, sign in ((lo, ll, -1.0), (hi, ul, 1.0)):
            if not mask.any():
                continue
            # eps <= a (left limit) or eps >= a (right limit), a = limit - xb.
            a = limit - xb[mask]
            ratio = np.exp(_log_phi(a / s) - special.log_ndtr(-sign * a / s))
            m_prev = np.ones(int(mask.sum()), dtype=dtype)
            m_cur = sign * s * ratio
            for j in range(2, order + 1):
                m_prev, m_cur = (
                    m_cur,
                    (j - 1) * s * s * m_prev + sign * s * a ** (j - 1) * ratio,
                )
            out[mask] = m_cur if order >= 1 else m_prev
        return out, s

    def mom(t: np.ndarray) -> np.ndarray:
        if test == "heterosc":
            m2, s = eps_moment(t, 2)
            return np.asarray((m2 - s * s)[:, None] * X[:, 1:])
        cols = []
        if test in ("normality", "skewness"):
            cols.append(eps_moment(t, 3)[0])
        if test in ("normality", "kurtosis"):
            m4, s = eps_moment(t, 4)
            cols.append(m4 - 3.0 * s**4)
        return np.column_stack(cols)

    return obs, mom, []


def _design_of(result: Any) -> Dict[str, Any]:
    design = getattr(result, "_cm_design", None)
    if design is not None:
        return dict(design)
    info = getattr(result, "model_info", None) or {}
    data = getattr(result, "data_info", None) or {}
    if info.get("family") == "binomial" and info.get("link") == "probit":
        if info.get("weights") is not None:
            raise MethodIncompatibility(
                "cmtest: the probit was fitted with weights; the test is "
                "derived for an unweighted likelihood."
            )
        X, y = data.get("X"), data.get("y")
        if X is None or y is None:
            raise MethodIncompatibility(
                "cmtest: this probit result does not carry its design matrix."
            )
        names = list(data.get("var_names") or [])
        return {
            "model": "probit",
            "y": np.asarray(y, dtype=float),
            "X": np.asarray(X, dtype=float),
            "names": names,
            "theta": np.asarray(result.params.values, dtype=float),
        }
    raise MethodIncompatibility(
        "cmtest: needs a fit from sp.probit or sp.tobit. The moment "
        "conditions are those of a normal latent error; a logit has no "
        "counterpart, and other models are not covered.",
        diagnostics={"model_type": info.get("model_type") or info.get("method")},
    )


def cmtest(result: Any, test: str = "normality", opg: bool = False) -> Dict[str, Any]:
    """
    Conditional moment test of a probit or tobit fit.

    Parameters
    ----------
    result
        A fit returned by :func:`statspai.probit` or :func:`statspai.tobit`
        (unweighted).
    test : {'normality', 'heterosc', 'skewness', 'kurtosis'}
        ``'normality'`` tests the third and fourth moments of the latent
        error jointly (2 degrees of freedom); ``'skewness'`` and
        ``'kurtosis'`` test them one at a time. ``'heterosc'`` tests
        ``E[(eps^2 - sigma^2) x_j] = 0`` for every regressor.
    opg : bool, default False
        Use the outer-product-of-the-gradient form. It is the one usually
        shown in textbooks because it needs no derivatives, and it is
        known to over-reject in finite samples; the default uses the
        Hessian and the derivatives of the moments.

    Returns
    -------
    dict
        ``statistic``, ``df``, ``pvalue``, ``test``, ``model``, ``opg`` and
        ``moments`` (the sample moments, each divided by n). For the
        one-moment tests ``z`` is the signed square root of the statistic.

    Notes
    -----
    For the probit, the normality moments are ``xb^2 psi`` and
    ``xb^3 psi`` with ``psi`` the generalised residual, which are also the
    moments of a RESET test on the squared and cubed index. A rejection
    therefore cannot tell non-normality from a misspecified index.

    Examples
    --------
    >>> import statspai as sp
    >>> fit = sp.tobit(df, y="hours", x=["wage", "kids"], ll=0)
    >>> sp.cmtest(fit, test="normality")  # doctest: +SKIP
    >>> sp.cmtest(fit, test="heterosc")  # doctest: +SKIP

    References
    ----------
    [@newey1985maximum], [@tauchen1985diagnostic], [@pagan1989diagnostic],
    [@skeels1999monte]
    """
    from ..regression._optim_helpers import ml_scores_hessian

    key = str(test).lower()
    key = {"heteroskedasticity": "heterosc", "het": "heterosc"}.get(key, key)
    if key not in _TESTS:
        raise MethodIncompatibility(
            f"cmtest: test={test!r} is not available; use one of {list(_TESTS)}."
        )
    d = _design_of(result)
    y, X, theta = d["y"], d["X"], np.asarray(d["theta"], dtype=float)
    if d["model"] == "probit":
        obs, mom, _ = _probit_pieces(y, X, key)
    else:
        obs, mom, _ = _tobit_pieces(y, X, d.get("ll"), d.get("ul"), key)
    if key == "heterosc" and X.shape[1] < 2:
        raise MethodIncompatibility(
            "cmtest: the heteroskedasticity test needs at least one regressor."
        )

    G, H = ml_scores_hessian(obs, theta)
    M = np.real(mom(theta.astype(complex)))
    m = M.sum(axis=0)
    if opg:
        A = M - G @ np.linalg.solve(G.T @ G, G.T @ M)
    else:
        h = 1e-30
        W = np.column_stack(
            [
                np.imag(mom(theta + 1j * h * np.eye(theta.size)[j]).sum(axis=0)) / h
                for j in range(theta.size)
            ]
        )
        A = M - G @ np.linalg.solve(H, W.T)
    V = A.T @ A
    if np.linalg.matrix_rank(V) < V.shape[0]:
        raise MethodIncompatibility(
            "cmtest: the moment conditions are collinear with the scores "
            "(a regressor that is a function of the others, such as a full "
            "set of dummies); the statistic is not defined."
        )
    stat = float(m @ np.linalg.solve(V, m))
    df = int(M.shape[1])
    out: Dict[str, Any] = {
        "statistic": stat,
        "df": df,
        "pvalue": float(stats.chi2.sf(stat, df)),
        "test": f"Conditional moment test ({key})",
        "model": d["model"],
        "opg": bool(opg),
        "moments": (m / len(y)).tolist(),
    }
    if df == 1:
        out["z"] = float(np.sign(m[0]) * np.sqrt(stat))
    if key == "heterosc":
        names = d.get("names") or []
        out["variables"] = list(names[1:]) if len(names) == X.shape[1] else None
    return out
