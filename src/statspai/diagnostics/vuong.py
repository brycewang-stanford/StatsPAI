"""
Vuong's test for choosing between two non-nested likelihood models.

Both models are fitted to the same observations. With ``l1_i`` and
``l2_i`` their per-observation log-likelihoods and ``m_i = l1_i - l2_i``,

    z = sqrt(n) * mean(m) / sd(m)  ->  N(0, 1)

when the two models are equally close to the truth. A large positive
``z`` favours the first model, a large negative one the second.

References
----------
[@vuong1989likelihood], [@wilson2015misuse]
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats

from ..exceptions import MethodIncompatibility

__all__ = ["vuong"]


def _pieces(result: Any, label: str) -> Tuple[np.ndarray, int, Optional[np.ndarray]]:
    data = getattr(result, "data_info", None) or {}
    info = getattr(result, "model_info", None) or {}
    ll = data.get("llobs")
    if ll is None:
        raise MethodIncompatibility(
            f"vuong: {label} does not carry per-observation log-likelihoods. "
            "Fits from sp.poisson, sp.nbreg, sp.zip_model, sp.zinb, "
            "sp.hurdle, sp.logit and sp.probit do.",
            diagnostics={"model_type": info.get("model_type")},
        )
    if info.get("weights") is not None or data.get("weights") is not None:
        raise MethodIncompatibility(
            f"vuong: {label} was fitted with weights; the test is derived "
            "for an unweighted likelihood."
        )
    k = int(data.get("n_params") or len(result.params))
    y = data.get("y")
    return np.asarray(ll, dtype=float), k, None if y is None else np.asarray(y)


def vuong(model1: Any, model2: Any) -> Dict[str, Any]:
    """
    Vuong test between two non-nested models fitted by maximum likelihood.

    Parameters
    ----------
    model1, model2
        Fits on the same observations from :func:`statspai.poisson`,
        :func:`statspai.nbreg`, :func:`statspai.zip_model`,
        :func:`statspai.zinb`, :func:`statspai.hurdle`,
        :func:`statspai.logit` or :func:`statspai.probit`, without
        weights.

    Returns
    -------
    dict
        ``statistic`` (z, positive when ``model1`` fits better),
        ``pvalue`` (two-sided), ``preferred`` (``'model1'``, ``'model2'``
        or ``None`` when the two-sided test does not reject at 5%), and
        the same three under ``aic`` and ``bic``, where the mean
        log-likelihood ratio is first reduced by ``(k1 - k2) / n`` and
        ``(k1 - k2) log(n) / (2 n)``. ``k1`` and ``k2`` count every
        estimated parameter, a negative binomial dispersion included.

    Notes
    -----
    The normal reference distribution needs the two models to be
    non-nested and not to coincide at the truth.

    * **Nested models.** If one model is a restriction of the other
      (Poisson inside negative binomial), use a likelihood-ratio test
      (:func:`statspai.lrtest`).
    * **Zero inflation.** A zero-inflated model and its plain counterpart
      are a boundary case and the statistic is not standard normal under
      the null of no inflation, so the test should not be read as a test
      for zero inflation [@wilson2015misuse]. The comparison
      remains informative as a description of which model fits closer.

    R's ``pscl::vuong`` counts parameters with ``length(coef())``, which
    leaves out a negative binomial dispersion. Its corrected statistics
    therefore differ from these when exactly one of the two models is
    negative binomial; the uncorrected statistic is the same.

    Examples
    --------
    >>> import statspai as sp
    >>> nb = sp.nbreg(data=df, y="trips", x=["income", "dist"])
    >>> hd = sp.hurdle(data=df, y="trips", x=["income", "dist"])
    >>> sp.vuong(nb, hd)["statistic"]  # doctest: +SKIP

    References
    ----------
    [@vuong1989likelihood], [@wilson2015misuse]
    """
    l1, k1, y1 = _pieces(model1, "model1")
    l2, k2, y2 = _pieces(model2, "model2")
    if l1.shape != l2.shape:
        raise MethodIncompatibility(
            f"vuong: the two models use {l1.size} and {l2.size} observations; "
            "the test compares them observation by observation, so fit both "
            "on the same rows.",
            diagnostics={"n1": int(l1.size), "n2": int(l2.size)},
        )
    if y1 is not None and y2 is not None and not np.array_equal(y1, y2):
        raise MethodIncompatibility(
            "vuong: the two models have different outcome vectors; they must "
            "be fitted to the same dependent variable on the same rows."
        )
    m = l1 - l2
    n = m.size
    sd = float(np.std(m, ddof=1))
    if not np.isfinite(sd) or sd < 1e-12 * max(1.0, float(np.max(np.abs(l1)))):
        raise MethodIncompatibility(
            "vuong: the two models give the same likelihood for every "
            "observation; there is nothing to compare."
        )

    def block(shift: float) -> Dict[str, Any]:
        z = float(np.sqrt(n) * (np.mean(m) - shift) / sd)
        p = float(2.0 * stats.norm.sf(abs(z)))
        preferred = None if p >= 0.05 else ("model1" if z > 0 else "model2")
        return {"statistic": z, "pvalue": p, "preferred": preferred}

    out: Dict[str, Any] = block(0.0)
    out["aic"] = block((k1 - k2) / n)
    out["bic"] = block((k1 - k2) * np.log(n) / (2.0 * n))
    out.update(
        test="Vuong non-nested test",
        n_obs=int(n),
        k1=k1,
        k2=k2,
        loglik1=float(l1.sum()),
        loglik2=float(l2.sum()),
    )
    return out
