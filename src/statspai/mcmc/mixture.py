"""
Mixtures of normal regressions: ``sp.bayes_mixture``.

    y_i | z_i = k  ~  N(x_i' beta_k, sigma_k^2)

with either a fixed number of components (Dirichlet weights) or a
Dirichlet process, which lets the data choose how many components are
occupied. ``"y ~ 1"`` is the mixture of normals used for density
estimation and clustering.

The prior of each component is conjugate,
``sigma_k^2 ~ IG(a0 / 2, d0 / 2)`` and
``beta_k | sigma_k^2 ~ N(b0, sigma_k^2 B0)``, so coefficients, variances
and weights integrate out and the sampler only moves the allocations (a
collapsed Gibbs sampler). Component parameters are drawn afterwards from
their conditional posterior.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.utils import create_design_matrices
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from ._core import check_mcmc_args, spawn_rngs
from ._results import posterior_table
from .regress import BayesRegressResult

_KERNEL: Dict[str, Any] = {}


def _log_marginal_py(
    n: float,
    sxx: np.ndarray,
    sxy: np.ndarray,
    syy: float,
    b0inv: np.ndarray,
    b0inv_b0: np.ndarray,
    b0_quad: float,
    logdet_b0inv: float,
    a0: float,
    d0: float,
) -> float:
    """Log marginal likelihood of the observations of one component."""
    lam = b0inv + sxx
    chol = np.linalg.cholesky(lam)
    v = np.linalg.solve(chol, b0inv_b0 + sxy)
    dn = d0 + syy + b0_quad - (v * v).sum()
    an = a0 + n
    val: float = (
        -0.5 * n * math.log(math.pi)
        + 0.5 * logdet_b0inv
        - np.log(np.diag(chol)).sum()
        + math.lgamma(0.5 * an)
        - math.lgamma(0.5 * a0)
        + 0.5 * a0 * math.log(d0)
        - 0.5 * an * math.log(dn)
    )
    return val


def _sweep_py(
    z: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    cnt: np.ndarray,
    sxx: np.ndarray,
    sxy: np.ndarray,
    syy: np.ndarray,
    lml: np.ndarray,
    b0inv: np.ndarray,
    b0inv_b0: np.ndarray,
    b0_quad: float,
    logdet_b0inv: float,
    a0: float,
    d0: float,
    weight: float,
    dp: bool,
    lml_empty: float,
    u: np.ndarray,
    lm: Any,
) -> None:
    """Reallocate every observation once, in place.

    ``weight`` is the Dirichlet parameter of each component (finite
    mixture) or the concentration of the Dirichlet process.
    """
    n = y.shape[0]
    cap = cnt.shape[0]
    logw = np.empty(cap)
    cand = np.empty(cap)
    for i in range(n):
        c = z[i]
        xi = X[i]
        xx = np.outer(xi, xi)
        cnt[c] -= 1.0
        sxx[c] -= xx
        sxy[c] -= xi * y[i]
        syy[c] -= y[i] * y[i]
        if cnt[c] < 0.5:
            cnt[c] = 0.0
            sxx[c] = 0.0
            sxy[c] = 0.0
            syy[c] = 0.0
            lml[c] = lml_empty
        else:
            lml[c] = lm(
                cnt[c],
                sxx[c],
                sxy[c],
                syy[c],
                b0inv,
                b0inv_b0,
                b0_quad,
                logdet_b0inv,
                a0,
                d0,
            )
        single = lm(
            1.0,
            xx,
            xi * y[i],
            y[i] * y[i],
            b0inv,
            b0inv_b0,
            b0_quad,
            logdet_b0inv,
            a0,
            d0,
        )
        used_empty = False
        top = -np.inf
        for k in range(cap):
            if cnt[k] > 0.5:
                cand[k] = lm(
                    cnt[k] + 1.0,
                    sxx[k] + xx,
                    sxy[k] + xi * y[i],
                    syy[k] + y[i] * y[i],
                    b0inv,
                    b0inv_b0,
                    b0_quad,
                    logdet_b0inv,
                    a0,
                    d0,
                )
                prior_w = cnt[k] + (0.0 if dp else weight)
                logw[k] = math.log(prior_w) + cand[k] - lml[k]
            elif dp and used_empty:
                logw[k] = -np.inf
            else:
                used_empty = True
                cand[k] = single
                logw[k] = math.log(weight) + single - lml_empty
            if logw[k] > top:
                top = logw[k]
        total = 0.0
        for k in range(cap):
            logw[k] = math.exp(logw[k] - top)
            total += logw[k]
        target = u[i] * total
        acc = 0.0
        new = cap - 1
        for k in range(cap):
            acc += logw[k]
            if target < acc:
                new = k
                break
        z[i] = new
        cnt[new] += 1.0
        sxx[new] += xx
        sxy[new] += xi * y[i]
        syy[new] += y[i] * y[i]
        lml[new] = cand[new]


def _kernels() -> Tuple[Any, Any]:
    if "sweep" not in _KERNEL:
        from numba import njit  # type: ignore[import-untyped]

        lm = njit(cache=True)(_log_marginal_py)
        _KERNEL["lm"] = lm
        _KERNEL["sweep"] = njit(cache=True)(_sweep_py)
    return _KERNEL["sweep"], _KERNEL["lm"]


def _component_posterior(
    cnt: float,
    sxx: np.ndarray,
    sxy: np.ndarray,
    syy: float,
    b0inv: np.ndarray,
    b0inv_b0: np.ndarray,
    b0_quad: float,
    a0: float,
    d0: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Posterior mean, precision factor and inverse-gamma parameters of a
    component given its sufficient statistics."""
    lam = b0inv + sxx
    mean = np.linalg.solve(lam, b0inv_b0 + sxy)
    dn = d0 + syy + b0_quad - mean @ lam @ mean
    return mean, lam, a0 + cnt, float(dn)


def bayes_mixture(
    formula: str,
    data: pd.DataFrame,
    components: Union[int, str] = 2,
    alpha: float = 1.0,
    alpha_prior: Optional[Tuple[float, float]] = None,
    prior_mean: Any = None,
    prior_scale: Any = None,
    sigma2_prior: Optional[Tuple[float, float]] = None,
    max_components: Optional[int] = None,
    grid: int = 200,
    draws: int = 5000,
    burnin: int = 1000,
    thin: int = 1,
    chains: int = 1,
    seed: Optional[int] = None,
    level: float = 0.95,
) -> BayesRegressResult:
    """Bayesian mixture of normal regressions, finite or Dirichlet process.

    Each observation belongs to an unobserved component with its own
    coefficients and error variance. ``"y ~ 1"`` gives a mixture of
    normals: a flexible density estimate and a model-based clustering.

    Parameters
    ----------
    formula : str
        ``"y ~ 1"`` or ``"y ~ x1 + x2"``.
    data : DataFrame
    components : int or 'dp', default 2
        Number of components, or ``'dp'`` for a Dirichlet process mixture
        in which the number of occupied components is inferred.
    alpha : float, default 1
        Finite mixture: the weights are ``Dirichlet(alpha, ..., alpha)``.
        Dirichlet process: the concentration; the prior expected number
        of clusters grows like ``alpha * log(n)``.
    alpha_prior : (shape, rate), optional
        Dirichlet process only: a Gamma prior for the concentration,
        which is then sampled and reported.
    prior_mean : array, optional
        Prior mean ``b0`` of every component's coefficients. Default: the
        pooled least-squares coefficients.
    prior_scale : float or matrix, optional
        ``B0`` in ``beta_k | sigma_k^2 ~ N(b0, sigma_k^2 B0)``. Default
        ``100 * n * inv(X'X)``: a component mean can sit about ten
        component standard deviations away from the pooled fit.
    sigma2_prior : (a0, d0), optional
        ``sigma_k^2 ~ IG(a0 / 2, d0 / 2)``. Default ``a0 = 4`` and ``d0``
        such that the prior mean of a component variance is a quarter of
        the pooled residual variance.
    max_components : int, optional
        Dirichlet process only: cap on the number of occupied clusters
        (memory). Default ``min(n, 50)``.
    grid : int, default 200
        Points of the density estimate (``"y ~ 1"`` only).
    draws, burnin, thin, chains, seed, level
        As in :func:`statspai.bayes_regress`.

    Returns
    -------
    BayesRegressResult
        ``model='mixture'``. For a finite mixture the table has, for each
        component, its weight, coefficients and variance; components are
        ordered by their first coefficient within every draw. For a
        Dirichlet process it has the number of occupied clusters (and
        ``alpha``). ``model_info`` holds

        * ``'n_clusters'``: posterior distribution of the number of
          occupied components,
        * ``'similarity'``: the matrix of posterior probabilities that
          two observations share a component,
        * ``'cluster'``: a point estimate of the partition, the sampled
          partition closest to ``'similarity'`` in squared error,
        * ``'density'``: posterior mean and band of the density on a
          grid (``"y ~ 1"`` only).

    Notes
    -----
    The default prior depends on the data through the pooled fit. It is
    printed in ``result.prior``; pass the three prior arguments to fix it.

    Component labels are not identified. Ordering by the first
    coefficient is a convention that works when components differ in
    that coefficient; the similarity matrix, the partition and the
    density do not depend on labels.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = np.r_[rng.normal(-2, 0.5, 100), rng.normal(2, 0.5, 100)]
    >>> fit = sp.bayes_mixture("y ~ 1", pd.DataFrame({"y": y}),
    ...                        draws=300, burnin=200, seed=1)
    >>> list(fit.params.index)[:2]
    ['weight[1]', 'weight[2]']
    >>> int(np.unique(fit.model_info["cluster"]).size)
    2

    References
    ----------
    neal2000markov, escobar1995bayesian
    """
    check_mcmc_args(draws, burnin, thin, chains)
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    dp = isinstance(components, str)
    if dp and str(components).lower() != "dp":
        raise MethodIncompatibility(
            f"components must be an integer or 'dp'; got {components!r}."
        )
    if not dp and (int(components) != components or int(components) < 2):
        raise MethodIncompatibility(
            f"components must be an integer of at least 2 or 'dp'; got {components!r}."
        )
    if alpha <= 0:
        raise MethodIncompatibility("alpha must be positive.")
    if alpha_prior is not None and not dp:
        raise MethodIncompatibility("alpha_prior applies to components='dp' only.")
    if alpha_prior is not None and min(alpha_prior) <= 0:
        raise MethodIncompatibility("alpha_prior must be (shape, rate), both positive.")
    y_df, X_df = create_design_matrices(formula, data)
    y = np.ascontiguousarray(np.asarray(y_df, dtype=float).reshape(-1))
    X = np.ascontiguousarray(np.asarray(X_df, dtype=float))
    xnames = [str(c) for c in X_df.columns]
    n, k = X.shape
    if n < 2 * k + 4:
        raise DataInsufficient(f"{n} observations are too few for a mixture model.")
    if np.linalg.matrix_rank(X) < k:
        raise MethodIncompatibility("The regressors are collinear.")
    xtx = X.T @ X
    ols = np.linalg.solve(xtx, X.T @ y)
    s2 = float(((y - X @ ols) ** 2).sum() / max(n - k, 1))
    if s2 <= 0:
        raise MethodIncompatibility("The outcome is fitted exactly by the regressors.")
    b0 = (
        ols
        if prior_mean is None
        else np.broadcast_to(np.asarray(prior_mean, dtype=float), (k,)).copy()
    )
    if prior_scale is None:
        B0 = 100.0 * n * np.linalg.inv(xtx)
    else:
        sc = np.asarray(prior_scale, dtype=float)
        B0 = np.eye(k) * float(sc) if sc.ndim == 0 else 0.5 * (sc + sc.T)
        if B0.shape != (k, k) or np.linalg.eigvalsh(B0).min() <= 0:
            raise MethodIncompatibility(
                "prior_scale must be a positive number or a positive "
                f"definite {k} x {k} matrix."
            )
    if sigma2_prior is None:
        a0, d0 = 4.0, 0.5 * s2
    else:
        a0, d0 = (float(v) for v in sigma2_prior)
        if a0 <= 0 or d0 <= 0:
            raise MethodIncompatibility("sigma2_prior must be two positive numbers.")
    b0inv = np.ascontiguousarray(np.linalg.inv(B0))
    b0inv_b0 = b0inv @ b0
    b0_quad = float(b0 @ b0inv_b0)
    logdet_b0inv = float(np.linalg.slogdet(b0inv)[1])
    if dp:
        cap = min(n, 50) if max_components is None else int(max_components)
        if cap < 2:
            raise MethodIncompatibility("max_components must be at least 2.")
        cap += 1  # one empty slot for a new cluster
    else:
        cap = int(components)
    sweep, lm = _kernels()
    hyper = (b0inv, b0inv_b0, b0_quad, logdet_b0inv, a0, d0)
    lml_empty = 0.0
    density_only = k == 1 and np.allclose(X, 1.0)
    if density_only:
        pad = 0.15 * (y.max() - y.min())
        ygrid = np.linspace(y.min() - pad, y.max() + pad, int(grid))

    def predictive(cnt: float, sxx: Any, sxy: Any, syy: float, pts: np.ndarray) -> Any:
        mean, lam, an, dn = _component_posterior(
            cnt, sxx, sxy, syy, b0inv, b0inv_b0, b0_quad, a0, d0
        )
        scale = np.sqrt(dn / an * (1.0 + 1.0 / lam[0, 0]))
        return stats.t.pdf(pts, an, loc=mean[0], scale=scale)

    n_iter = burnin + draws * thin
    rngs = spawn_rngs(seed, chains)
    K = cap if not dp else 0
    col_names: List[str] = []
    if not dp:
        col_names += [f"weight[{j + 1}]" for j in range(K)]
        for j in range(K):
            col_names += [f"comp{j + 1}:{c}" for c in xnames] + [f"comp{j + 1}:sigma2"]
    col_names += ["n_clusters"]
    if dp and alpha_prior is not None:
        col_names += ["alpha"]
    pieces = []
    z_keep = np.empty((chains * draws, n), dtype=np.int16)
    dens = np.empty((chains * draws, int(grid))) if density_only else None
    capped = False
    row = 0
    for ch in range(chains):
        rng = rngs[ch]
        # start from a k-means-like split on the fitted values' residual order
        start_k = min(cap - 1 if dp else cap, 4)
        order = np.argsort(y - X @ ols if not density_only else y)
        z = np.empty(n, dtype=np.int64)
        z[order] = (np.arange(n) * start_k // n).astype(np.int64)
        cnt = np.zeros(cap)
        sxx = np.zeros((cap, k, k))
        sxy = np.zeros((cap, k))
        syy = np.zeros(cap)
        lml = np.zeros(cap)
        for c in range(cap):
            idx = z == c
            if idx.any():
                cnt[c] = idx.sum()
                sxx[c] = X[idx].T @ X[idx]
                sxy[c] = X[idx].T @ y[idx]
                syy[c] = y[idx] @ y[idx]
                lml[c] = lm(cnt[c], sxx[c], sxy[c], syy[c], *hyper)
        a_cur = float(alpha)
        out = np.empty((draws, len(col_names)))
        kept = 0
        for it in range(n_iter):
            sweep(
                z,
                X,
                y,
                cnt,
                sxx,
                sxy,
                syy,
                lml,
                *hyper,
                a_cur,
                dp,
                lml_empty,
                rng.random(n),
                lm,
            )
            occ = int((cnt > 0.5).sum())
            if dp and occ >= cap - 1:
                capped = True
            if dp and alpha_prior is not None:
                # Escobar and West's auxiliary-variable update
                sh, rt = alpha_prior
                eta = rng.beta(a_cur + 1.0, n)
                odds = (sh + occ - 1.0) / (n * (rt - np.log(eta)))
                shape = (
                    sh + occ if rng.random() < odds / (1.0 + odds) else sh + occ - 1.0
                )
                a_cur = rng.gamma(shape, 1.0 / (rt - np.log(eta)))
            if it >= burnin and (it - burnin) % thin == 0:
                z_keep[row] = z
                vals: List[float] = []
                if not dp:
                    w = rng.dirichlet(cnt + a_cur)
                    comps = []
                    for c in range(K):
                        mean, lam, an, dn = _component_posterior(
                            cnt[c],
                            sxx[c],
                            sxy[c],
                            syy[c],
                            b0inv,
                            b0inv_b0,
                            b0_quad,
                            a0,
                            d0,
                        )
                        sig2 = (dn / 2.0) / rng.gamma(an / 2.0)
                        chol = np.linalg.cholesky(lam)
                        beta = mean + np.sqrt(sig2) * np.linalg.solve(
                            chol.T, rng.standard_normal(k)
                        )
                        comps.append((beta, sig2, w[c]))
                    comps.sort(key=lambda t: t[0][0])
                    vals += [c[2] for c in comps]
                    for beta, sig2, _ in comps:
                        vals += list(beta) + [sig2]
                vals.append(float(occ))
                if dp and alpha_prior is not None:
                    vals.append(a_cur)
                out[kept] = vals
                if dens is not None:
                    total = n + (a_cur if dp else K * a_cur)
                    f = np.zeros(int(grid))
                    empty_w = 0.0
                    for c in range(cap):
                        if cnt[c] > 0.5:
                            wt = (cnt[c] + (0.0 if dp else a_cur)) / total
                            f += wt * predictive(cnt[c], sxx[c], sxy[c], syy[c], ygrid)
                        elif not dp:
                            empty_w += a_cur / total
                    if dp:
                        empty_w = a_cur / total
                    if empty_w > 0:
                        f += empty_w * predictive(
                            0.0, sxx[0] * 0, sxy[0] * 0, 0.0, ygrid
                        )
                    dens[row] = f
                kept += 1
                row += 1
        pieces.append(out)
    d_df = pd.DataFrame(np.vstack(pieces), columns=col_names)
    chain_idx = np.repeat(np.arange(chains), draws)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table, diag = posterior_table(d_df, chain_idx, chains, level)
    # posterior similarity and the sampled partition closest to it
    take = np.linspace(0, len(z_keep) - 1, min(len(z_keep), 4000)).astype(int)
    zs = z_keep[take]
    info: Dict[str, Any] = {
        "components": "dp" if dp else K,
        "n_clusters": d_df["n_clusters"]
        .astype(int)
        .value_counts(normalize=True)
        .sort_index(),
    }
    if n <= 3000:
        sim = np.zeros((n, n), dtype=np.float32)
        for zz in zs:
            sim += zz[:, None] == zz[None, :]
        sim /= len(zs)
        best, best_loss = 0, np.inf
        for j, zz in enumerate(zs[:: max(1, len(zs) // 200)]):
            loss = float((((zz[:, None] == zz[None, :]) - sim) ** 2).sum())
            if loss < best_loss:
                best, best_loss = j, loss
        zbest = zs[:: max(1, len(zs) // 200)][best]
        _, relabel = np.unique(zbest, return_inverse=True)
        # label clusters by their mean outcome
        means = np.array([y[relabel == c].mean() for c in range(relabel.max() + 1)])
        rank = np.empty_like(means, dtype=int)
        rank[np.argsort(means)] = np.arange(means.size)
        info["similarity"] = sim
        info["cluster"] = pd.Series(rank[relabel] + 1, index=X_df.index, name="cluster")
    if dens is not None:
        lo = (1.0 - level) / 2.0
        info["density"] = pd.DataFrame(
            {
                "y": ygrid,
                "density": dens.mean(axis=0),
                "lower": np.quantile(dens, lo, axis=0),
                "upper": np.quantile(dens, 1.0 - lo, axis=0),
            }
        )
    if not dp:
        weights_text = f"Dirichlet({alpha:g}, ..., {alpha:g})"
    elif alpha_prior is None:
        weights_text = f"Dirichlet process, concentration {alpha:g}"
    else:
        weights_text = (
            "Dirichlet process, concentration ~ "
            f"Gamma({alpha_prior[0]:g}, rate {alpha_prior[1]:g})"
        )
    prior: Dict[str, Any] = {
        "weights": weights_text,
        "prior_mean": pd.Series(b0, index=xnames),
        "prior_scale": B0,
        "sigma2": f"IG({a0 / 2:g}, {d0 / 2:g})",
        "data_dependent": prior_mean is None
        or prior_scale is None
        or sigma2_prior is None,
    }
    res = BayesRegressResult(
        model="mixture",
        formula=formula,
        params=table["mean"].copy(),
        std_errors=table["sd"].copy(),
        table=table,
        draws=d_df,
        chain=chain_idx,
        n_obs=n,
        n_draws=draws * chains,
        burnin=burnin,
        thin=thin,
        chains=chains,
        sampler="collapsed Gibbs over the allocations",
        acceptance_rate=None,
        prior=prior,
        level=level,
        model_info=info,
        diagnostics_info=diag,
        _model=None,
    )
    if capped:
        text = (
            "The number of occupied clusters reached max_components; the "
            "posterior of the number of clusters is truncated. Raise "
            "max_components or lower alpha."
        )
        diag["warnings"].append(text)
        warnings.warn(text, ConvergenceWarning, stacklevel=2)
    if not dp:
        gaps = [
            abs(
                table.loc[f"comp{j + 2}:{xnames[0]}", "mean"]
                - table.loc[f"comp{j + 1}:{xnames[0]}", "mean"]
            )
            / max(table.loc[f"comp{j + 1}:{xnames[0]}", "sd"], 1e-12)
            for j in range(K - 1)
        ]
        if min(gaps) < 2.0:
            text = (
                "Two components are not separated in their first coefficient, "
                "so ordering by it does not identify the labels and the "
                "per-component rows mix components. Use the similarity "
                "matrix, the partition and the density, which do not depend "
                "on labels."
            )
            diag["warnings"].append(text)
            warnings.warn(text, ConvergenceWarning, stacklevel=2)
    return res
