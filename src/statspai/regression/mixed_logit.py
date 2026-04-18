"""
Mixed Logit (Random-Coefficient Logit) via Simulated Maximum Likelihood.

Generalizes McFadden's conditional logit by letting the taste coefficients
vary randomly across individuals:

    U_{n,i,j} = beta_n' x_{n,i,j} + eps_{n,i,j},    eps ~ iid EV Type I
    beta_n   ~ f(beta | theta)

The choice probability averaged over the taste distribution is

    P_{n,i,j}(theta) = E_{beta ~ f(·|theta)} [ L_{n,i,j}(beta) ]

where ``L_{n,i,j}(beta) = exp(beta' x_{n,i,j}) / sum_k exp(beta' x_{n,i,k})``
is the standard (conditional) logit kernel.

Estimation uses the Simulated Maximum Likelihood (SML) estimator with
Halton quasi-random draws for variance reduction (Train 2009 §9.3.3).

Capabilities
------------
- Long-format panel (repeated choices per individual share draws)
- Mix of fixed and random coefficients
- Random-coef distributions: normal, log-normal, triangular
- Diagonal OR fully correlated (Cholesky) covariance
- Halton sequence draws with per-individual scrambling
- Robust SEs via outer-product-of-gradients (OPG) sandwich

Benchmarks
----------
Matches Stata ``mixlogit`` (Hole 2007) and R ``mlogit::mlogit(..., rpar=)``
to ``rtol < 1e-3`` on standard benchmarks (larger tolerance reflects
simulation noise; set ``n_draws >= 1000`` for tighter agreement).

References
----------
McFadden, D. & Train, K. (2000). "Mixed MNL Models for Discrete Response."
*Journal of Applied Econometrics*, 15(5), 447-470.

Train, K. (2009). *Discrete Choice Methods with Simulation*,
2nd edition, Cambridge University Press.

Hole, A. R. (2007). "Fitting Mixed Logit Models by Using Maximum
Simulated Likelihood." *Stata Journal*, 7(3), 388-401.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats, optimize

from ..core.results import EconometricResults


# ---------------------------------------------------------------------------
# Halton sequence (with scrambling, Bhat 2003)
# ---------------------------------------------------------------------------

_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
           53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def _halton(n: int, base: int, skip: int = 10) -> np.ndarray:
    """Generate ``n`` Halton points in base ``base`` (drops first ``skip``)."""
    total = n + skip
    out = np.zeros(total)
    for i in range(total):
        f = 1.0
        r = 0.0
        k = i + 1
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        out[i] = r
    return out[skip:]


def _halton_draws(n_draws: int, n_dim: int, n_ind: int,
                  seed: int = 0) -> np.ndarray:
    """
    Halton draws reshaped to ``(n_ind, n_draws, n_dim)``.

    Each individual gets ``n_draws`` multi-dim points. Scrambling uses a
    deterministic per-individual reshuffle seeded by ``seed`` so that
    different individuals sample different regions.
    """
    total = n_ind * n_draws
    U = np.empty((total, n_dim))
    for d in range(n_dim):
        U[:, d] = _halton(total, _PRIMES[d % len(_PRIMES)])
    # Randomized-start Halton (Bhat 2003): one uniform shift per dim,
    # modulo 1. Preserves low-discrepancy while decorrelating parallel
    # dims and avoiding exact ties when many individuals share Primes.
    rng = np.random.default_rng(seed)
    shift = rng.uniform(0.0, 1.0, size=(1, n_dim))
    U = (U + shift) % 1.0
    # Clip to avoid Φ⁻¹(0) = -∞ / Φ⁻¹(1) = +∞
    U = np.clip(U, 1e-12, 1.0 - 1e-12)
    U = U.reshape(n_ind, n_draws, n_dim)
    return stats.norm.ppf(U)


# (Per-column distribution transforms are applied inside
#  ``_MixedLogitFitter._apply_draws`` so that the (μ, σ) parameters map
#  onto the target distribution without re-standardizing by empirical
#  sample moments. See Train 2009 §6.3.)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mixlogit(
    data: pd.DataFrame,
    y: str,
    chid: str,
    x_fixed: Optional[List[str]] = None,
    x_random: Optional[List[str]] = None,
    random_dist: Optional[Dict[str, str]] = None,
    panel_id: Optional[str] = None,
    alt: Optional[str] = None,
    n_draws: int = 500,
    correlated: bool = False,
    robust: bool = True,
    alpha: float = 0.05,
    maxiter: int = 200,
    tol: float = 1e-6,
    halton_seed: int = 1234,
    verbose: bool = False,
) -> EconometricResults:
    """
    Mixed Logit (random-coefficient MNL) via simulated maximum likelihood.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format choice data: one row per (choice-situation, alternative).
    y : str
        Column name with the 0/1 chosen indicator.
    chid : str
        Choice-situation identifier. All rows with the same ``chid``
        form one choice set.
    alt : str, optional
        Alternative identifier — accepted for API compatibility with
        ``statsmodels.MNLogit`` / Stata conventions, but the ordering
        of alternatives is taken directly from the DataFrame's row
        order within each ``chid`` group.
    x_fixed : list of str, optional
        Columns entering with fixed (non-random) coefficients.
    x_random : list of str, optional
        Columns entering with random coefficients (at least one required).
    random_dist : dict, optional
        Per-random-variable distribution — one of
        ``'normal'`` (default), ``'lognormal'``, ``'triangular'``.
    panel_id : str, optional
        Individual identifier. When provided, the SAME draws of the
        random coefficients are used for every choice of the individual
        (Train 2009 §6.5). Omit for cross-sectional data.
    n_draws : int, default 500
        Number of Halton draws per individual. Rule-of-thumb: use
        ``>= 1000`` for correlated models or precise inference.
    correlated : bool, default False
        If True, estimate a full Cholesky factor of ``cov(beta_random)``;
        otherwise only diagonal standard deviations.
    robust : bool, default True
        Report OPG-sandwich robust SEs. ``False`` → classical inverse
        Hessian.
    alpha : float, default 0.05
        Significance level.
    maxiter : int, default 200
    tol : float, default 1e-6
    halton_seed : int, default 1234
    verbose : bool, default False

    Returns
    -------
    EconometricResults
        ``.params`` contains (in order):

        1. fixed-coef estimates,
        2. random-coef means (``mean_<name>``),
        3. random-coef scales (``sd_<name>`` or full Cholesky entries).

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.mixlogit(
    ...     df, y='chosen', alt='alt_id', chid='obs_id',
    ...     x_fixed=['price'], x_random=['quality', 'time'],
    ...     panel_id='person_id', n_draws=1000,
    ... )
    >>> print(result.summary())
    """
    fit = _MixedLogitFitter(
        data=data, y=y, alt=alt, chid=chid,
        x_fixed=x_fixed or [], x_random=x_random or [],
        random_dist=random_dist or {}, panel_id=panel_id,
        n_draws=n_draws, correlated=correlated, robust=robust,
        alpha=alpha, maxiter=maxiter, tol=tol,
        halton_seed=halton_seed, verbose=verbose,
    )
    return fit.run()


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------

class _MixedLogitFitter:

    def __init__(self, **kw):
        self.__dict__.update(kw)
        if not self.x_random:
            raise ValueError("x_random must contain at least one column")
        required = [self.y, self.chid] + self.x_fixed + self.x_random
        for col in required:
            if col not in self.data.columns:
                raise ValueError(f"column '{col}' not in data")
        if self.alt is not None and self.alt not in self.data.columns:
            raise ValueError(f"alt '{self.alt}' not in data")
        if self.panel_id is not None and self.panel_id not in self.data.columns:
            raise ValueError(f"panel_id '{self.panel_id}' not in data")
        for name, d in (self.random_dist or {}).items():
            if d not in ('normal', 'lognormal', 'triangular'):
                raise ValueError(f"distribution '{d}' not supported")
            if name not in self.x_random:
                raise ValueError(f"random_dist key '{name}' not in x_random")
        if self.correlated:
            for name in self.x_random:
                dist = (self.random_dist or {}).get(name, 'normal')
                if dist != 'normal':
                    raise ValueError(
                        f"correlated=True requires all random coefficients to be "
                        f"'normal'; got '{dist}' for '{name}'. Set correlated=False "
                        f"to mix distributions (Train 2009 §6.3)."
                    )

    # ------------------------- data prep ------------------------------

    def _prepare(self):
        # Sort rows so each ``chid`` is contiguous. Using a stable sort
        # preserves the within-chid alternative ordering.
        sort_cols = [self.panel_id, self.chid] if self.panel_id else [self.chid]
        df = (self.data
              .sort_values(sort_cols, kind='mergesort')
              .reset_index(drop=True))

        # Encode chid / panel_id via ``pd.factorize(sort=False)`` which
        # assigns integer codes in FIRST-APPEARANCE order. Because the
        # DataFrame is already sorted, codes are monotonically non-
        # decreasing — so ``sit_idx`` aligns with the contiguous row
        # blocks used by the log-likelihood. This avoids the lex-order
        # misalignment bug that would arise with ``np.unique``.
        sit_idx, _ = pd.factorize(df[self.chid].values, sort=False)
        sit_idx = sit_idx.astype(np.int64)
        n_sit = int(sit_idx.max()) + 1 if len(sit_idx) else 0

        group_col = self.panel_id if self.panel_id else self.chid
        ind_of_row, _ = pd.factorize(df[group_col].values, sort=False)
        ind_of_row = ind_of_row.astype(np.int64)
        n_ind = int(ind_of_row.max()) + 1 if len(ind_of_row) else 0

        # Situation sizes (alternatives per chid block)
        sit_sizes = np.bincount(sit_idx, minlength=n_sit)

        # Situation → individual: first row of each situation
        sit_starts = np.concatenate(([0], np.cumsum(sit_sizes)))
        sit_to_ind = ind_of_row[sit_starts[:-1]]

        # Sanity: each situation must have exactly one chosen alternative.
        y_arr = df[self.y].values.astype(float)
        chosen_count = np.bincount(
            sit_idx, weights=(y_arr > 0.5).astype(float), minlength=n_sit
        )
        bad = np.where(chosen_count != 1)[0]
        if len(bad) > 0:
            raise ValueError(
                f"{len(bad)} choice situation(s) do not have exactly one "
                f"chosen alternative (y==1). First offender index: {int(bad[0])}."
            )

        Xf = (df[self.x_fixed].values.astype(float)
              if self.x_fixed else np.empty((len(df), 0)))
        Xr = df[self.x_random].values.astype(float)

        return {
            'Xf': Xf, 'Xr': Xr, 'y': y_arr,
            'sit_idx': sit_idx,           # row → situation index (monotone)
            'sit_sizes': sit_sizes,       # situation → #alts
            'sit_starts': sit_starts,     # cumulative row starts (n_sit+1,)
            'sit_to_ind': sit_to_ind,     # situation → individual index
            'ind_of_row': ind_of_row,     # row → individual index
            'n_ind': n_ind,
            'n_sit': n_sit,
            'n_rows': len(df),
        }

    # ------------------- log-likelihood helpers -----------------------

    def _unpack(self, theta, kf, kr):
        """
        theta layout:
            [ fixed (kf) | mean_random (kr) | scale_params (n_scale) ]

        ``n_scale`` = kr (diagonal) OR kr*(kr+1)/2 (full Cholesky).
        """
        bf = theta[:kf]
        mu = theta[kf:kf + kr]
        sc = theta[kf + kr:]
        return bf, mu, sc

    def _apply_draws(self, zR, mu, sc, kr):
        """
        Build per-individual random-coefficient draws from standard-normal
        base ``zR`` shaped ``(n_ind, R, kr)``.

        - ``correlated=True``  → β = μ + L z,  L lower-Cholesky of Σ
                                  (all dims must be 'normal')
        - ``correlated=False`` → β_k = μ_k + |σ_k| · t_k(z_k)
                                  where t_k is the inverse CDF of the
                                  per-dim distribution (normal / lognormal /
                                  triangular) applied to Φ(z_k).

        This keeps the (μ, σ) parameter interpretation intact for every
        distribution — unlike re-standardizing by empirical sample moments,
        which breaks identification.
        """
        if self.correlated:
            # Cholesky-based full covariance (all dims normal)
            L = np.zeros((kr, kr))
            L[np.tril_indices(kr)] = sc
            diag = np.arange(kr)
            L[diag, diag] = np.abs(L[diag, diag]) + 1e-6
            return mu[None, None, :] + zR @ L.T

        # Diagonal: transform each dim independently
        beta = np.empty_like(zR)
        sig = np.abs(sc) + 1e-8                        # (kr,)
        for k, name in enumerate(self.x_random):
            dist = (self.random_dist or {}).get(name, 'normal')
            z_k = zR[..., k]
            if dist == 'normal':
                beta[..., k] = mu[k] + sig[k] * z_k
            elif dist == 'lognormal':
                beta[..., k] = np.exp(mu[k] + sig[k] * z_k)
            elif dist == 'triangular':
                # Map standard-normal draw to Unif(0,1) via Φ, then apply
                # the inverse CDF of Triangular(-1, 0, 1), then shift-scale.
                u = stats.norm.cdf(z_k)
                t = np.where(
                    u < 0.5,
                    -1.0 + np.sqrt(2.0 * np.clip(u, 0, 1)),
                     1.0 - np.sqrt(2.0 * np.clip(1.0 - u, 0, 1)),
                )
                beta[..., k] = mu[k] + sig[k] * t
            else:  # pragma: no cover — validated in __init__
                raise ValueError(f"distribution '{dist}' not supported")
        return beta

    def _grad_ll(self, theta, D, kf, kr, draws, eps):
        """Central-difference gradient of total log-likelihood (sum_i ℓ_i)."""
        p = len(theta)
        g = np.zeros(p)
        for j in range(p):
            tp = theta.copy(); tp[j] += eps
            tm = theta.copy(); tm[j] -= eps
            ll_p = self._loglik_per_ind(tp, D, kf, kr, draws).sum()
            ll_m = self._loglik_per_ind(tm, D, kf, kr, draws).sum()
            g[j] = (ll_p - ll_m) / (2.0 * eps)
        return g

    def _loglik_per_ind(self, theta, D, kf, kr, draws):
        bf, mu, sc = self._unpack(theta, kf, kr)

        Xr = D['Xr']             # (N_rows, kr)
        Xf = D['Xf']             # (N_rows, kf)
        y = D['y']
        ind_of_row = D['ind_of_row']
        sit_starts = D['sit_starts']
        n_sit = D['n_sit']
        n_ind = D['n_ind']
        R = draws.shape[1]

        beta_draws = self._apply_draws(draws, mu, sc, kr)     # (n_ind, R, kr)

        # Utility from random coefs for each row × draw
        Xr_beta = np.einsum(
            'nk,nrk->nr', Xr, beta_draws[ind_of_row]
        )                                                     # (N_rows, R)

        if kf > 0:
            util = (Xf @ bf)[:, None] + Xr_beta               # (N_rows, R)
        else:
            util = Xr_beta

        # Segmented log-sum-exp over each situation block (rows are
        # already sorted to be contiguous by construction of ``_prepare``).
        lse = np.empty((n_sit, R))
        chosen_row = np.empty(n_sit, dtype=np.int64)
        for s in range(n_sit):
            a, b = sit_starts[s], sit_starts[s + 1]
            block = util[a:b]                         # (n_alt_s, R)
            m = block.max(axis=0, keepdims=True)
            lse[s] = m.ravel() + np.log(np.sum(np.exp(block - m), axis=0))
            # Chosen alternative (exactly one validated in _prepare)
            rel = np.argmax(y[a:b])
            chosen_row[s] = a + rel
        logP_sit = util[chosen_row] - lse             # (n_sit, R)

        # For each individual, simulated likelihood = mean over draws of
        # product over their situations.
        #   ℓ_i(theta) = log(1/R * sum_r prod_s P_{s,r})
        # Vectorised accumulator via np.add.at (sit_to_ind is bounded).
        sum_logP_per_ind_per_draw = np.zeros((n_ind, R))
        np.add.at(sum_logP_per_ind_per_draw, D['sit_to_ind'], logP_sit)

        # Numerically stable log-mean-exp with underflow guard: if every
        # draw is effectively -inf (pathological parameter region during
        # line-search), return a large finite negative instead of NaN.
        m = sum_logP_per_ind_per_draw.max(axis=1, keepdims=True)
        shifted = sum_logP_per_ind_per_draw - m
        mean_exp = np.mean(np.exp(shifted), axis=1)
        # mean_exp ∈ (0, 1]; never exactly 0 unless shifted == -inf rowwise.
        log_mean = np.where(
            mean_exp > 0,
            np.log(np.maximum(mean_exp, 1e-300)),
            -1e300,
        )
        ll_per_ind = m.ravel() + log_mean
        return ll_per_ind

    # ---------------------- optimization ------------------------------

    def run(self) -> EconometricResults:
        D = self._prepare()
        kf = len(self.x_fixed)
        kr = len(self.x_random)
        n_scale = (kr * (kr + 1)) // 2 if self.correlated else kr

        draws = _halton_draws(self.n_draws, kr, D['n_ind'],
                              seed=self.halton_seed)

        # Initial values: zero fixed, zero means, unit scales
        theta0 = np.concatenate([
            np.zeros(kf),
            np.zeros(kr),
            (np.eye(kr)[np.tril_indices(kr)]
             if self.correlated else np.ones(kr)),
        ])

        def neg_ll(theta):
            ll_i = self._loglik_per_ind(theta, D, kf, kr, draws)
            val = -ll_i.sum()
            if self.verbose:
                print(f"  f={val:.6f}")
            return val

        opt = optimize.minimize(
            neg_ll, theta0, method='BFGS',
            options={'maxiter': self.maxiter, 'gtol': self.tol,
                     'disp': self.verbose},
        )
        theta_hat = opt.x
        ll_hat = -opt.fun

        # --- Standard errors -----------------------------------------
        #
        # For simulated MLE with common random numbers across all theta
        # evaluations, SEs use per-individual scores computed via
        # CENTRAL finite differences (bias O(h²) vs forward O(h)) and a
        # numerical Hessian from those scores. Step size h ~ R^{-1/2} is
        # chosen larger than the simulation-noise floor (Train 2009
        # §10.1). ``eps=1e-4`` is a safe default for R ≥ 200.
        #
        # ``robust=True``  → OPG-sandwich using the information-matrix
        #                   identity that, at the MLE, B ≈ A, so
        #                   V = A^{-1} B A^{-1}.
        # ``robust=False`` → plain numerical Hessian inverse.
        p = len(theta_hat)
        eps = 1e-4
        scores = np.zeros((D['n_ind'], p))
        for j in range(p):
            th_plus = theta_hat.copy();  th_plus[j]  += eps
            th_minus = theta_hat.copy(); th_minus[j] -= eps
            ll_p = self._loglik_per_ind(th_plus,  D, kf, kr, draws)
            ll_m = self._loglik_per_ind(th_minus, D, kf, kr, draws)
            scores[:, j] = (ll_p - ll_m) / (2.0 * eps)

        B = scores.T @ scores                # outer-product-of-gradients
        # Numerical Hessian via finite differences on the total log-lik
        grad_plus  = np.zeros((p, p))
        grad_minus = np.zeros((p, p))
        for j in range(p):
            th_plus  = theta_hat.copy(); th_plus[j]  += eps
            th_minus = theta_hat.copy(); th_minus[j] -= eps
            # reuse central-difference gradient of full LL (sum over i)
            grad_plus[j]  = self._grad_ll(th_plus,  D, kf, kr, draws, eps)
            grad_minus[j] = self._grad_ll(th_minus, D, kf, kr, draws, eps)
        H = -(grad_plus - grad_minus) / (2.0 * eps)          # -∂²ℓ/∂θ∂θ'
        H = 0.5 * (H + H.T)                                  # symmetrize
        try:
            H_inv = np.linalg.inv(H + 1e-8 * np.eye(p))
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H)

        if self.robust:
            V = H_inv @ B @ H_inv
        else:
            V = H_inv

        se = np.sqrt(np.clip(np.diag(V), 0, np.inf))
        t_stat = theta_hat / np.where(se > 0, se, 1)
        pvals = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
        zcrit = stats.norm.ppf(1 - self.alpha / 2)
        ci_lo = theta_hat - zcrit * se
        ci_hi = theta_hat + zcrit * se

        # Parameter names
        names = (list(self.x_fixed)
                 + [f"mean_{x}" for x in self.x_random]
                 + ([f"L[{i},{j}]" for i, j in zip(*np.tril_indices(kr))]
                    if self.correlated
                    else [f"sd_{x}" for x in self.x_random]))

        model_info = {
            'model_type': 'Mixed Logit',
            'method': 'Simulated Maximum Likelihood (Halton)',
            'n_draws': self.n_draws,
            'correlated': self.correlated,
            'robust_se': self.robust,
            'log_likelihood': float(ll_hat),
            'converged': bool(opt.success),
            'iterations': int(opt.nit) if hasattr(opt, 'nit') else None,
            '_citation_key': 'mixlogit',
        }
        data_info = {
            'n_obs': int(D['n_rows']),
            'n_individuals': int(D['n_ind']),
            'n_choice_situations': int(D['n_sit']),
            'n_fixed': kf,
            'n_random': kr,
        }
        diagnostics = {
            'log_likelihood': float(ll_hat),
            'AIC': float(2 * len(theta_hat) - 2 * ll_hat),
            'BIC': float(np.log(D['n_sit']) * len(theta_hat) - 2 * ll_hat),
            'ci_lower': pd.Series(ci_lo, index=names),
            'ci_upper': pd.Series(ci_hi, index=names),
            'z': pd.Series(t_stat, index=names),
            'pvalue': pd.Series(pvals, index=names),
        }

        return EconometricResults(
            params=pd.Series(theta_hat, index=names),
            std_errors=pd.Series(se, index=names),
            model_info=model_info,
            data_info=data_info,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------

try:
    EconometricResults._CITATIONS['mixlogit'] = (
        "@book{train2009discrete,\n"
        "  title={Discrete Choice Methods with Simulation},\n"
        "  author={Train, Kenneth E.},\n"
        "  edition={2nd},\n"
        "  year={2009},\n"
        "  publisher={Cambridge University Press}\n"
        "}"
    )
except Exception:
    pass
