"""Likelihood, filter, smoother and EM steps for Markov-switching regressions.

Private to :mod:`statspai.timeseries.mswitch`.

Parameter vector (the scale on which the likelihood is maximised and on
which ``vcov`` is reported; it is the one Stata's ``mswitch`` uses):

``[constants | x coefficients | switching coefficients, state by state |
AR coefficients | log sigma | q]`` with transition probabilities
``p_ij = exp(-q_ij) / (1 + sum_k exp(-q_ik))`` for ``j < K`` and
``p_iK = 1 / (1 + sum_k exp(-q_ik))``.

Every likelihood routine takes a batch of parameter vectors and is written
without ``abs`` / ``max`` on the parameters, so that it can be evaluated at
complex arguments: the score is the complex-step derivative (exact to
rounding) and the Hessian a central difference of exact scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__: List[str] = []

_LOG2PI = float(np.log(2.0 * np.pi))
_CHUNK_BYTES = 48_000_000


@dataclass(frozen=True)
class Spec:
    """Shape of a Markov-switching model."""

    k: int
    model: str
    p: int
    nx: int
    nz: int
    const: str
    sw_ar: bool
    sw_var: bool

    @property
    def n_mu(self) -> int:
        return {"switch": self.k, "common": 1, "none": 0}[self.const]

    @property
    def n_phi(self) -> int:
        return self.p * (self.k if self.sw_ar else 1)

    @property
    def n_sig(self) -> int:
        return self.k if self.sw_var else 1

    @property
    def n_mean(self) -> int:
        return self.n_mu + self.nx + self.k * self.nz + self.n_phi

    @property
    def n_par(self) -> int:
        return self.n_mean + self.n_sig + self.k * (self.k - 1)

    @property
    def p_exp(self) -> int:
        return self.p if self.model == "ar" else 0

    @property
    def n_exp(self) -> int:
        return int(self.k ** (self.p_exp + 1))

    def digits(self) -> np.ndarray:
        """``digits[j, m]`` is the state ``m`` periods back in regime ``j``."""
        idx = np.arange(self.n_exp)
        cols = [
            (idx // self.k ** (self.p_exp - m)) % self.k for m in range(self.p_exp + 1)
        ]
        return np.stack(cols, axis=1)


@dataclass
class Data:
    """Series with ``p`` presample rows in front of the estimation sample."""

    y: np.ndarray
    x: np.ndarray
    z: np.ndarray
    p: int

    @property
    def n(self) -> int:
        return int(len(self.y) - self.p)


def unpack(spec: Spec, theta: np.ndarray) -> Dict[str, np.ndarray]:
    """Split a batch ``theta[B, n_par]`` into per-state arrays."""
    k, b = spec.k, theta.shape[0]
    pos = 0
    if spec.const == "none":
        mu = np.zeros((b, k), dtype=theta.dtype)
    else:
        mu = theta[:, pos : pos + spec.n_mu] + np.zeros((b, k), dtype=theta.dtype)
        pos += spec.n_mu
    a = theta[:, pos : pos + spec.nx]
    pos += spec.nx
    bz = theta[:, pos : pos + k * spec.nz].reshape(b, k, spec.nz)
    pos += k * spec.nz
    phi_raw = theta[:, pos : pos + spec.n_phi]
    phi = phi_raw.reshape(b, k if spec.sw_ar else 1, spec.p) + np.zeros(
        (b, k, spec.p), dtype=theta.dtype
    )
    pos += spec.n_phi
    lnsig = theta[:, pos : pos + spec.n_sig] + np.zeros((b, k), dtype=theta.dtype)
    pos += spec.n_sig
    q = theta[:, pos:].reshape(b, k, k - 1)
    e = np.exp(-q)
    den = 1.0 + e.sum(axis=2, keepdims=True)
    trans = np.concatenate([e / den, 1.0 / den], axis=2)
    return {"mu": mu, "a": a, "b": bz, "phi": phi, "lnsig": lnsig, "P": trans}


def pack(spec: Spec, parts: Dict[str, np.ndarray]) -> np.ndarray:
    """Inverse of :func:`unpack` for one parameter set (no batch axis)."""
    k = spec.k
    out: List[np.ndarray] = []
    if spec.const == "switch":
        out.append(parts["mu"])
    elif spec.const == "common":
        out.append(parts["mu"][:1])
    out.append(parts["a"])
    out.append(parts["b"].ravel())
    out.append((parts["phi"] if spec.sw_ar else parts["phi"][:1]).ravel())
    out.append(parts["lnsig"] if spec.sw_var else parts["lnsig"][:1])
    trans = parts["P"]
    out.append((-np.log(trans[:, : k - 1] / trans[:, k - 1 :])).ravel())
    return np.concatenate([np.asarray(v, dtype=float).ravel() for v in out])


def ergodic(trans: np.ndarray) -> np.ndarray:
    """Stationary distribution of each chain in ``trans[B, K, K]``.

    Solves ``(I - P') pi = 0`` with the last equation replaced by
    ``sum(pi) = 1``; the diagonal of ``I - P'`` is formed from the
    off-diagonal probabilities so that it stays accurate when a state is
    nearly absorbing.
    """
    b, k, _ = trans.shape
    off = trans.transpose(0, 2, 1) * (1.0 - np.eye(k))[None]
    a = -off + np.eye(k)[None] * off.sum(axis=1)[:, None, :]
    a[:, k - 1, :] = 1.0
    rhs = np.zeros((b, k, 1), dtype=trans.dtype)
    rhs[:, k - 1, 0] = 1.0
    return np.linalg.solve(a, rhs)[:, :, 0]


def residuals(spec: Spec, data: Data, parts: Dict[str, np.ndarray]) -> np.ndarray:
    """Residual of each date in each (expanded) regime: ``[B, T, N]``."""
    p = spec.p
    y = data.y
    mean = parts["mu"][:, None, :] + (parts["a"] @ data.x.T)[:, :, None]
    if spec.nz:
        mean = mean + np.einsum("tz,bkz->btk", data.z, parts["b"])
    dev = y[None, :, None] - mean
    n_all = len(y)
    if spec.model == "dr":
        res = dev[:, p:, :]
        for lag in range(1, p + 1):
            ylag = y[p - lag : n_all - lag]
            res = res - parts["phi"][:, None, :, lag - 1] * ylag[None, :, None]
        return np.asarray(res)
    dig = spec.digits()
    res = dev[:, p:, :][:, :, dig[:, 0]]
    for lag in range(1, p + 1):
        lagged = dev[:, p - lag : n_all - lag, :][:, :, dig[:, lag]]
        res = res - parts["phi"][:, dig[:, 0], lag - 1][:, None, :] * lagged
    return np.asarray(res)


def _predict(spec: Spec, trans: np.ndarray, xi: np.ndarray) -> np.ndarray:
    k, pe = spec.k, spec.p_exp
    if pe == 0:
        return np.asarray(np.einsum("bij,bi->bj", trans, xi))
    b = xi.shape[0]
    marg = xi.reshape(b, k**pe, k).sum(axis=2).reshape(b, 1, k, k ** (pe - 1))
    out = trans.transpose(0, 2, 1)[:, :, :, None] * marg
    return np.asarray(out.reshape(b, -1))


def _initial(spec: Spec, trans: np.ndarray) -> np.ndarray:
    pi = ergodic(trans)
    dig = spec.digits()
    pe = spec.p_exp
    out = pi[:, dig[:, pe]]
    for m in range(pe, 0, -1):
        out = out * trans[:, dig[:, m], dig[:, m - 1]]
    return out


def _run_filter(
    spec: Spec, data: Data, theta: np.ndarray, keep: bool
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    with np.errstate(all="ignore"):
        return _filter_pass(spec, data, theta, keep)


def _filter_pass(
    spec: Spec, data: Data, theta: np.ndarray, keep: bool
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    parts = unpack(spec, theta)
    res = residuals(spec, data, parts)
    s0 = spec.digits()[:, 0]
    lnsig = parts["lnsig"][:, None, s0]
    logd = -0.5 * _LOG2PI - lnsig - 0.5 * res * res * np.exp(-2.0 * lnsig)
    shift = logd.real.max(axis=2)
    dens = np.exp(logd - shift[:, :, None])
    b, n, m = dens.shape
    trans = parts["P"]
    pred = _initial(spec, trans)
    xi = pred
    ll = np.empty((b, n), dtype=theta.dtype)
    preds = np.empty((n, m)) if keep else None
    filts = np.empty((n, m)) if keep else None
    for t in range(n):
        if t:
            pred = _predict(spec, trans, xi)
        joint = pred * dens[:, t, :]
        c = joint.sum(axis=1)
        xi = joint / c[:, None]
        ll[:, t] = np.log(c) + shift[:, t]
        if preds is not None and filts is not None:
            preds[t] = pred[0].real
            filts[t] = xi[0].real
    return ll, preds, filts, res


def loglik_obs(spec: Spec, data: Data, theta: np.ndarray) -> np.ndarray:
    """Log-likelihood contributions ``[B, T]`` of a batch of parameters."""
    theta = np.atleast_2d(theta)
    per = max(1, data.n * spec.n_exp * 16 * 4)
    step = max(1, _CHUNK_BYTES // per)
    try:
        out = [
            _run_filter(spec, data, theta[i : i + step], False)[0]
            for i in range(0, theta.shape[0], step)
        ]
    except np.linalg.LinAlgError:  # no ergodic distribution: not a valid point
        return np.full((theta.shape[0], data.n), np.nan, dtype=theta.dtype)
    return np.concatenate(out, axis=0)


def loglik(spec: Spec, data: Data, theta: np.ndarray) -> float:
    """Log likelihood at one parameter vector."""
    return float(loglik_obs(spec, data, theta[None, :]).real.sum())


def scores(spec: Spec, data: Data, theta: np.ndarray) -> np.ndarray:
    """Per-date scores ``[n_par, T]`` at each row of ``theta[B, n_par]``."""
    theta = np.atleast_2d(theta)
    b, n_par = theta.shape
    h = 1e-30
    pert = theta[:, None, :] + 1j * h * np.eye(n_par)[None]
    obs = loglik_obs(spec, data, pert.reshape(b * n_par, n_par))
    return (obs.imag / h).reshape(b, n_par, -1)


def gradient(spec: Spec, data: Data, theta: np.ndarray) -> np.ndarray:
    """Exact (complex-step) gradient of the log likelihood."""
    return np.asarray(scores(spec, data, theta[None, :])[0].sum(axis=1))


def hessian(spec: Spec, data: Data, theta: np.ndarray) -> np.ndarray:
    """Central difference of exact gradients; relative error about 1e-10."""
    n_par = len(theta)
    step = 1e-5 * np.maximum(1.0, np.abs(theta))
    pts = np.concatenate([theta + np.diag(step), theta - np.diag(step)], axis=0)
    grads = scores(spec, data, pts).sum(axis=2)
    hess = (grads[:n_par] - grads[n_par:]) / (2.0 * step[:, None])
    return np.asarray(0.5 * (hess + hess.T))


def smooth(
    spec: Spec, trans: np.ndarray, preds: np.ndarray, filts: np.ndarray
) -> np.ndarray:
    """Kim smoother on the (expanded) chain: ``[T, N]``."""
    k, pe = spec.k, spec.p_exp
    out = np.empty_like(filts)
    out[-1] = filts[-1]
    for t in range(len(filts) - 2, -1, -1):
        ratio = np.divide(
            out[t + 1], preds[t + 1], out=np.zeros(len(out[t])), where=preds[t + 1] > 0
        )
        if pe == 0:
            back = trans @ ratio
        else:
            r3 = ratio.reshape(k, k, k ** (pe - 1))
            back = np.repeat(np.einsum("ij,jim->im", trans, r3).ravel(), k)
        out[t] = filts[t] * back
    return out


def filter_smooth(spec: Spec, data: Data, theta: np.ndarray) -> Dict[str, np.ndarray]:
    """Filtered, predicted and smoothed regime probabilities at ``theta``."""
    ll, preds, filts, res = _run_filter(spec, data, theta[None, :], True)
    assert preds is not None and filts is not None
    trans = unpack(spec, theta[None, :])["P"][0]
    sm = smooth(spec, trans, preds, filts)
    n, k = data.n, spec.k
    return {
        "llobs": ll[0].real,
        "pred_exp": preds,
        "filt_exp": filts,
        "smooth_exp": sm,
        "resid_exp": res[0].real,
        "pred": preds.reshape(n, k, -1).sum(axis=2),
        "filt": filts.reshape(n, k, -1).sum(axis=2),
        "smooth": sm.reshape(n, k, -1).sum(axis=2),
        "P": trans,
    }


def design_dr(spec: Spec, data: Data) -> np.ndarray:
    """Stacked design ``[T, K, n_mean]`` of the dynamic-regression form."""
    p, k, n = spec.p, spec.k, data.n
    d = np.zeros((n, k, spec.n_mean))
    lags = np.column_stack(
        [data.y[p - lag : len(data.y) - lag] for lag in range(1, p + 1)]
        or [np.empty((n, 0))]
    )
    for s in range(k):
        pos = 0
        if spec.const == "switch":
            d[:, s, s] = 1.0
        elif spec.const == "common":
            d[:, s, 0] = 1.0
        pos += spec.n_mu
        d[:, s, pos : pos + spec.nx] = data.x[p:]
        pos += spec.nx
        d[:, s, pos + s * spec.nz : pos + (s + 1) * spec.nz] = data.z[p:]
        pos += k * spec.nz
        off = s * p if spec.sw_ar else 0
        d[:, s, pos + off : pos + off + p] = lags
    return d


def em_dr(
    spec: Spec,
    data: Data,
    theta: np.ndarray,
    maxiter: int,
    tol: float,
    var_floor: float,
) -> Tuple[np.ndarray, float]:
    """EM iterations on the dynamic-regression form; returns theta and ll.

    The transition update ignores the term of the ergodic initial
    distribution, and with a common coefficient next to state variances the
    regression step is one conditional maximisation, so this is a
    generalised EM used only to reach the neighbourhood of a maximum.
    """
    k = spec.k
    design = design_dr(spec, data)
    yv = data.y[spec.p :]
    ll_old = -np.inf
    ll = -np.inf
    for _ in range(maxiter):
        fs = filter_smooth(spec, data, theta)
        ll = float(fs["llobs"].sum())
        if not np.isfinite(ll) or abs(ll - ll_old) < tol:
            break
        ll_old = ll
        g, trans = fs["smooth"], fs["P"]
        ratio = np.divide(
            g[1:], fs["pred"][1:], out=np.zeros_like(g[1:]), where=fs["pred"][1:] > 0
        )
        joint = fs["filt"][:-1, :, None] * trans[None] * ratio[:, None, :]
        new_p = joint.sum(axis=0)
        new_p = np.clip(new_p / new_p.sum(axis=1, keepdims=True), 1e-6, None)
        new_p = new_p / new_p.sum(axis=1, keepdims=True)
        parts = {key: v[0] for key, v in unpack(spec, theta[None, :]).items()}
        w = g * np.exp(-2.0 * parts["lnsig"])[None, :]
        xtx = np.einsum("tk,tkm,tkn->mn", w, design, design)
        xty = np.einsum("tk,tkm,t->m", w, design, yv)
        beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]
        res2 = (yv[:, None] - design @ beta) ** 2
        if spec.sw_var:
            s2 = (g * res2).sum(axis=0) / np.maximum(g.sum(axis=0), 1e-12)
        else:
            s2 = np.full(k, (g * res2).sum() / len(yv))
        s2 = np.maximum(s2, var_floor)
        tail = np.concatenate(
            [
                0.5 * np.log(s2[: spec.n_sig]),
                (-np.log(new_p[:, : k - 1] / new_p[:, k - 1 :])).ravel(),
            ]
        )
        theta = np.concatenate([beta, tail])
    return theta, ll


_Q_EDGE = 15.0


def free_mask(spec: Spec, theta: np.ndarray) -> np.ndarray:
    """Parameters not pinned at a transition probability of 0 or 1."""
    mask = np.ones(len(theta), dtype=bool)
    nq = spec.k * (spec.k - 1)
    mask[len(theta) - nq :] = np.abs(theta[len(theta) - nq :]) < _Q_EDGE
    return mask


def start_values(
    spec: Spec, data: Data, rng: Optional[np.random.Generator]
) -> np.ndarray:
    """Starting values on the dynamic-regression form.

    ``rng=None`` gives the deterministic start (regimes split at the
    quantiles of the pooled residuals); otherwise a random perturbation.
    """
    k, p = spec.k, spec.p
    dr = Spec(k, "dr", p, spec.nx, spec.nz, spec.const, spec.sw_ar, spec.sw_var)
    design = design_dr(dr, data)
    yv = data.y[p:]
    pooled = design.mean(axis=1) if dr.n_mean else np.zeros((len(yv), 0))
    coef = np.linalg.lstsq(pooled, yv, rcond=None)[0] if dr.n_mean else np.zeros(0)
    res = yv - pooled @ coef if dr.n_mean else yv.copy()
    sd = float(np.std(res)) or 1.0
    beta = coef.copy()
    grid = np.linspace(-1.0, 1.0, k)
    draw = (lambda: grid) if rng is None else (lambda: rng.normal(size=k))
    pos = 0
    if spec.const == "switch":
        groups = np.array_split(np.sort(res), k)
        centre = np.array([g.mean() for g in groups])
        beta[:k] = coef[:k] + (centre if rng is None else sd * rng.normal(size=k))
    pos += spec.n_mu + spec.nx
    zsd = np.maximum(data.z[p:].std(axis=0), 1e-12) if spec.nz else np.zeros(0)
    for j in range(spec.nz):
        beta[pos + j : pos + k * spec.nz : spec.nz] += sd / zsd[j] * draw()
    pos += k * spec.nz
    if spec.sw_ar:
        for lag in range(p):
            beta[pos + lag : pos + k * p : p] += 0.2 * draw()
    if spec.sw_var:
        scale = (
            np.linspace(0.5, 1.5, k) if rng is None else np.exp(rng.normal(0, 0.5, k))
        )
        lnsig = np.log(sd * scale)
    else:
        lnsig = np.array([np.log(0.7 * sd)])
    diag = np.full(k, 0.9) if rng is None else rng.uniform(0.5, 0.98, size=k)
    trans = np.empty((k, k))
    for i in range(k):
        trans[i] = (1.0 - diag[i]) / (k - 1)
        trans[i, i] = diag[i]
    q = -np.log(trans[:, : k - 1] / trans[:, k - 1 :])
    return np.concatenate([beta, lnsig, q.ravel()])


def dr_to_model(spec: Spec, theta: np.ndarray) -> np.ndarray:
    """Turn dynamic-regression intercepts into the means of ``model='ar'``."""
    if spec.model != "ar" or spec.const == "none" or spec.p == 0:
        return theta
    parts = {key: v[0] for key, v in unpack(spec, theta[None, :]).items()}
    gain = 1.0 - parts["phi"].sum(axis=1)
    parts["mu"] = np.where(np.abs(gain) > 0.05, parts["mu"] / gain, parts["mu"])
    return pack(spec, parts)


def maximise(
    spec: Spec,
    data: Data,
    theta: np.ndarray,
    maxiter: int,
    tol: float,
    lnsig_floor: float,
) -> Tuple[np.ndarray, float, bool, int]:
    """Quasi-Newton, then Newton steps on the exact score.

    Converged when the Newton decrement ``g' (-H)^{-1} g`` is below
    ``tol`` (the criterion Stata calls ``nrtolerance``).
    """
    from scipy.optimize import minimize

    n_par = len(theta)
    nq = spec.k * (spec.k - 1)
    sig = slice(spec.n_mean, spec.n_mean + spec.n_sig)
    bounds: List[Tuple[Optional[float], Optional[float]]] = [(None, None)] * n_par
    for i in range(sig.start, sig.stop):
        bounds[i] = (lnsig_floor, None)
    for i in range(n_par - nq, n_par):
        bounds[i] = (-2 * _Q_EDGE, 2 * _Q_EDGE)

    def negll(v: np.ndarray) -> Tuple[float, np.ndarray]:
        obs = loglik_obs(spec, data, v[None, :] + 1j * 1e-30 * np.eye(n_par))
        value = float(obs[0].real.sum())
        if not np.isfinite(value):
            return 1e300, np.zeros(n_par)
        return -value, -(obs.imag.sum(axis=1) / 1e-30)

    opt = minimize(
        negll,
        theta,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-7},
    )
    theta = np.asarray(opt.x, dtype=float)
    ll = loglik(spec, data, theta)
    n_iter = int(opt.nit)
    converged = False
    for _ in range(50):
        free = free_mask(spec, theta)
        free[sig] &= theta[sig] > lnsig_floor + 1e-8
        grad = gradient(spec, data, theta)[free]
        hess = hessian(spec, data, theta)[np.ix_(free, free)]
        try:
            if not (np.isfinite(hess).all() and np.isfinite(grad).all()):
                break
            np.linalg.cholesky(-hess)
        except np.linalg.LinAlgError:
            break
        step = np.linalg.solve(-hess, grad)
        if float(grad @ step) < tol:
            converged = True
            break
        scale = 1.0
        moved = False
        while scale > 1e-4:
            trial = theta.copy()
            trial[free] += scale * step
            ll_trial = loglik(spec, data, trial)
            if np.isfinite(ll_trial) and ll_trial >= ll - 1e-12 * abs(ll):
                theta, ll, moved = trial, ll_trial, True
                break
            scale *= 0.5
        n_iter += 1
        if not moved:
            break
    return theta, ll, converged, n_iter


def order_key(spec: Spec, parts: Dict[str, np.ndarray]) -> Tuple[np.ndarray, str]:
    """The quantity whose increasing order labels the states."""
    if spec.const == "switch":
        return parts["mu"], "increasing constant"
    if spec.nz:
        return parts["b"][:, 0], "increasing first switching coefficient"
    if spec.sw_ar:
        return parts["phi"][:, 0], "increasing first autoregressive coefficient"
    return parts["lnsig"], "increasing standard deviation"


def reorder(spec: Spec, theta: np.ndarray) -> Tuple[np.ndarray, str]:
    """Relabel the states so that the ordering key increases."""
    parts = {key: v[0] for key, v in unpack(spec, theta[None, :]).items()}
    key, rule = order_key(spec, parts)
    perm = np.argsort(key, kind="stable")
    if np.array_equal(perm, np.arange(spec.k)):
        return theta, rule
    for name in ("mu", "b", "phi", "lnsig"):
        parts[name] = parts[name][perm]
    trans = np.clip(parts["P"][perm][:, perm], 1e-300, None)
    parts["P"] = trans
    return pack(spec, parts), rule


def covariance(
    spec: Spec, data: Data, theta: np.ndarray, vce: str
) -> Tuple[np.ndarray, List[str]]:
    """Covariance on the estimation scale and notes on what is missing."""
    notes: List[str] = []
    free = free_mask(spec, theta)
    n_par = len(theta)
    vcov = np.full((n_par, n_par), np.nan)
    hess = hessian(spec, data, theta)[np.ix_(free, free)]
    try:
        np.linalg.cholesky(-hess)
    except np.linalg.LinAlgError:
        notes.append(
            "the Hessian is not negative definite at the estimates; standard "
            "errors are not available"
        )
        return vcov, notes
    inv = np.linalg.inv(-hess)
    if vce == "robust":
        sc = scores(spec, data, theta[None, :])[0][free]
        n = sc.shape[1]
        inv = inv @ (sc @ sc.T) @ inv * (n / (n - 1.0))
    vcov[np.ix_(free, free)] = inv
    if not free.all():
        notes.append(
            f"{int((~free).sum())} transition probabilit(ies) are at 0 or 1; "
            "they are held fixed and have no standard error"
        )
    return vcov, notes
