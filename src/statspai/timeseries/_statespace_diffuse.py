"""Exact diffuse initialisation of the Kalman filter and smoother.

Shared by :mod:`statspai.timeseries.statespace`. The initial covariance is
``P0 = P0_* + kappa P0_inf`` and the recursions are the limit as ``kappa``
goes to infinity: the predicted covariance is carried as the pair
``(P_*, P_inf)`` until ``P_inf`` has been used up by the data.

While ``P_inf`` is not zero the elements of ``Y_t`` are brought in one at
a time. A measurement-error covariance that is not diagonal is first
rotated by its eigenvectors, which leaves the likelihood and the state
moments at the end of the date unchanged. For one element with loading row
``z``, prediction error ``v``, ``F_inf = z P_inf z'`` and
``F_* = z P_* z' + r``:

* ``F_inf > 0``: the element is absorbed by the diffuse part. With
  ``K0 = P_inf z' / F_inf`` and
  ``K1 = P_* z' / F_inf - K0 F_* / F_inf``, the state moves by ``K0 v``,
  ``P_inf`` loses ``K0 K0' F_inf`` (its rank falls by one) and ``P_*``
  becomes ``P_* + K0 K0' F_* - K0 z P_* - P_* z' K0'``. The element adds
  ``-0.5 [log(2 pi) + log F_inf]`` to the diffuse log-likelihood.
* ``F_inf = 0``: the usual update with ``P_*``.

Once ``P_inf`` is zero the ordinary filter takes over from the filtered
moments. The smoother runs the ordinary backward recursion down to that
date and continues with the expansion of ``r`` and ``N`` in powers of
``1 / kappa`` (``r0, r1``; ``N0, N1, N2``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..exceptions import MethodIncompatibility
from . import _statespace_core as core

_LOG_2PI = float(np.log(2.0 * np.pi))
_TOL = 1e-8

# one scalar observation of the diffuse period:
# (z, v, F_inf, F_*, P_inf z', P_* z', absorbed by the diffuse part?)
Step = Tuple[np.ndarray, float, float, float, np.ndarray, np.ndarray, bool]


def initial_exact(
    F: np.ndarray,
    Q: np.ndarray,
    x0: Optional[Any],
    P0: Optional[Any],
    diffuse: Optional[Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, ``P0_*`` and ``P0_inf`` of the state before the first date."""
    m = F.shape[1]
    mean = np.zeros(m) if x0 is None else np.array(x0, dtype=float).reshape(-1)
    if mean.shape != (m,):
        raise MethodIncompatibility(f"x0 must have {m} entries.")
    if diffuse is None:
        flag = np.ones(m, dtype=bool)
    else:
        raw = np.asarray(diffuse)
        if raw.shape != (m,) or raw.dtype.kind not in "bi":
            raise MethodIncompatibility(
                f"diffuse must hold {m} booleans, one per state.",
                recovery_hint="For example diffuse=[True, False].",
            )
        flag = raw.astype(bool)
    keep = ~flag
    star = np.zeros((m, m))
    if P0 is not None:
        cov = np.array(P0, dtype=float)
        if cov.ndim == 0 and m == 1:
            cov = cov.reshape(1, 1)
        if cov.shape != (m, m) or not np.allclose(cov, cov.T, atol=1e-12):
            raise MethodIncompatibility(f"P0 must be a symmetric {m} x {m} matrix.")
        star[np.ix_(keep, keep)] = 0.5 * (cov + cov.T)[np.ix_(keep, keep)]
    elif keep.any():
        if F.shape[0] > 1 or Q.shape[0] > 1:
            raise MethodIncompatibility(
                "The stationary covariance of the states that are not "
                "diffuse is defined for constant F and Q.",
                recovery_hint="Pass P0; its block for those states is used.",
            )
        if np.any(F[0][np.ix_(keep, flag)] != 0.0):
            raise MethodIncompatibility(
                "A state that is not marked diffuse depends on a diffuse "
                "state, so it has no stationary distribution.",
                recovery_hint="Mark it diffuse as well, or pass P0.",
            )
        sub = F[0][np.ix_(keep, keep)]
        radius = float(np.max(np.abs(np.linalg.eigvals(sub))))
        if radius >= 1.0 - 1e-9:
            raise core.NotStationary(
                "The states that are not marked diffuse need a stable "
                f"transition block; its largest modulus is {radius:.6g}.",
                recovery_hint="Mark them diffuse, or pass P0.",
            )
        star[np.ix_(keep, keep)] = core.stationary_cov(sub, Q[0][np.ix_(keep, keep)])
    return mean, star, np.diag(flag.astype(float))


def _rows(
    yt: np.ndarray, obs: np.ndarray, At: np.ndarray, Gt: np.ndarray, Rt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed elements of one date as uncorrelated scalar observations."""
    idx = np.flatnonzero(obs)
    z = Gt[idx]
    w = yt[idx] - At[idx]
    r = Rt[np.ix_(idx, idx)]
    d = np.diag(r).copy()
    if idx.size > 1 and np.any(r - np.diag(d) != 0.0):
        d, U = np.linalg.eigh(r)
        z = U.T @ z
        w = U.T @ w
    return z, w, d


def filter_exact(
    y: np.ndarray,
    mask: np.ndarray,
    A: np.ndarray,
    G: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    Pstar0: np.ndarray,
    Pinf0: np.ndarray,
) -> Dict[str, Any]:
    """Filter the dates on which ``P_inf`` is not zero.

    Returns the moments of those dates, the scalar steps the smoother
    needs, and ``d``, the number of dates handled; the ordinary filter
    continues at date ``d`` from ``xf[d - 1]`` and ``Pf[d - 1]``.
    """
    T, n = y.shape
    m = x0.shape[0]
    x = x0.copy()
    Ps = Pstar0.copy()
    Pi = Pinf0.copy()
    xp: List[np.ndarray] = []
    Pp: List[np.ndarray] = []
    Ppi: List[np.ndarray] = []
    xf: List[np.ndarray] = []
    Pf: List[np.ndarray] = []
    Pfi: List[np.ndarray] = []
    steps: List[List[Step]] = []
    ll: List[float] = []
    absorbed = 0
    d = T
    for t in range(T):
        Ft = F[t if F.shape[0] > 1 else 0]
        x_new = Ft @ x
        Ps_new = Ft @ Ps @ Ft.T + Q[t if Q.shape[0] > 1 else 0]
        Pi_new = Ft @ Pi @ Ft.T
        if np.linalg.matrix_rank(Pi_new) == 0:
            d = t
            break
        x = x_new
        Ps = 0.5 * (Ps_new + Ps_new.T)
        Pi = 0.5 * (Pi_new + Pi_new.T)
        rank = int(np.linalg.matrix_rank(Pi))
        xp.append(x.copy())
        Pp.append(Ps.copy())
        Ppi.append(Pi.copy())
        todo: List[Step] = []
        lt = 0.0
        if mask[t].any():
            Z, w, dvar = _rows(
                y[t],
                mask[t],
                A[t if A.shape[0] > 1 else 0],
                G[t if G.shape[0] > 1 else 0],
                R[t if R.shape[0] > 1 else 0],
            )
            for i in range(Z.shape[0]):
                z = Z[i]
                v = float(w[i] - z @ x)
                Mi = Pi @ z
                Ms = Ps @ z
                Finf = float(z @ Mi)
                Fst = float(z @ Ms + dvar[i])
                zz = float(z @ z)
                big = rank > 0 and Finf > _TOL * zz * float(np.max(np.abs(Pi)))
                if big:
                    K0 = Mi / Finf
                    x = x + K0 * v
                    Ps = Ps + np.outer(K0, K0) * Fst
                    Ps = Ps - np.outer(K0, Ms) - np.outer(Ms, K0)
                    Pi = Pi - np.outer(K0, K0) * Finf
                    rank -= 1
                    absorbed += 1
                    if rank == 0:
                        Pi = np.zeros((m, m))
                    lt += -0.5 * (_LOG_2PI + np.log(Finf))
                else:
                    floor = 1e-12 * (zz * float(np.max(np.abs(Ps))) + abs(dvar[i]))
                    if not Fst > floor:
                        raise np.linalg.LinAlgError(
                            "zero prediction-error variance in the diffuse period"
                        )
                    x = x + Ms * (v / Fst)
                    Ps = Ps - np.outer(Ms, Ms) / Fst
                    lt += -0.5 * (_LOG_2PI + np.log(Fst) + v * v / Fst)
                Ps = 0.5 * (Ps + Ps.T)
                Pi = 0.5 * (Pi + Pi.T)
                todo.append((z, v, Finf, Fst, Mi, Ms, big))
        xf.append(x.copy())
        Pf.append(Ps.copy())
        Pfi.append(Pi.copy())
        steps.append(todo)
        ll.append(lt)

    def arr(items: List[np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
        return np.array(items) if items else np.zeros((0,) + shape)

    return {
        "d": d,
        "xp": arr(xp, (m,)),
        "Pp": arr(Pp, (m, m)),
        "Ppi": arr(Ppi, (m, m)),
        "xf": arr(xf, (m,)),
        "Pf": arr(Pf, (m, m)),
        "Pfi": arr(Pfi, (m, m)),
        "ll": np.array(ll, dtype=float),
        "steps": steps,
        "absorbed": absorbed,
    }


def smooth_exact(
    head: Dict[str, Any], F: np.ndarray, r: np.ndarray, N: np.ndarray, T: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Smoothed moments of the diffuse dates.

    ``r`` and ``N`` are the ordinary smoother's quantities at the first
    date after the diffuse period (zero when there is none): the smoothed
    state there is ``x_pred + P_pred r``. ``T`` is the sample length.
    """
    d = int(head["d"])
    m = F.shape[1]
    xs = np.zeros((d, m))
    Vs = np.zeros((d, m, m))
    eye = np.eye(m)
    r0 = r.copy()
    N0 = N.copy()
    r1 = np.zeros(m)
    N1 = np.zeros((m, m))
    N2 = np.zeros((m, m))
    for t in range(d - 1, -1, -1):
        if t + 1 < T:
            Fn = F[(t + 1) if F.shape[0] > 1 else 0]
            r0 = Fn.T @ r0
            r1 = Fn.T @ r1
            N0 = Fn.T @ N0 @ Fn
            N1 = Fn.T @ N1 @ Fn
            N2 = Fn.T @ N2 @ Fn
        for z, v, Finf, Fst, Mi, Ms, big in reversed(head["steps"][t]):
            zz = np.outer(z, z)
            if big:
                K0 = Mi / Finf
                K1 = Ms / Finf - K0 * (Fst / Finf)
                L0 = eye - np.outer(K0, z)
                L1 = -np.outer(K1, z)
                r1 = z * (v / Finf) + L1.T @ r0 + L0.T @ r1
                r0 = L0.T @ r0
                N2 = (
                    -zz * (Fst / Finf**2)
                    + L0.T @ N2 @ L0
                    + L1.T @ N1 @ L0
                    + L0.T @ N1 @ L1
                    + L1.T @ N0 @ L1
                )
                N1 = zz / Finf + L0.T @ N1 @ L0 + L1.T @ N0 @ L0 + L0.T @ N0 @ L1
                N0 = L0.T @ N0 @ L0
            else:
                Lst = eye - np.outer(Ms / Fst, z)
                r0 = z * (v / Fst) + Lst.T @ r0
                r1 = Lst.T @ r1
                N0 = zz / Fst + Lst.T @ N0 @ Lst
                N1 = Lst.T @ N1 @ Lst
                N2 = Lst.T @ N2 @ Lst
        Ps = head["Pp"][t]
        Pi = head["Ppi"][t]
        xs[t] = head["xp"][t] + Ps @ r0 + Pi @ r1
        cross = Pi @ N1 @ Ps
        V = Ps - Ps @ N0 @ Ps - cross - cross.T - Pi @ N2 @ Pi
        Vs[t] = 0.5 * (V + V.T)
    return xs, Vs


def _tail(stack: np.ndarray, d: int) -> np.ndarray:
    return stack[d:] if stack.shape[0] > 1 else stack


def run_exact(
    yv: np.ndarray,
    sysm: Dict[str, np.ndarray],
    m: int,
    start: Tuple[Optional[Any], Optional[Any], Optional[Any]],
    smooth: bool,
    compiled: bool,
) -> Dict[str, Any]:
    """Exact diffuse filter for the first dates, the ordinary one after.

    ``start`` is ``(x0, P0, diffuse)`` as the caller gave them.
    """
    T, n = yv.shape
    x0, P0, P0inf = initial_exact(sysm["F"], sysm["Q"], *start)
    mask = np.isfinite(yv)
    y0 = np.where(mask, yv, 0.0)
    keys = ("A", "G", "F", "Q", "R")
    A, G, F, Q, R = (sysm[k] for k in keys)
    head = filter_exact(y0, mask, A, G, F, Q, R, x0, P0, P0inf)
    d = int(head["d"])
    names = ("xp", "Pp", "xf", "Pf", "v", "S", "Sinv", "K", "e", "ll")
    kf, ks = core.kernels(compiled)
    res: Dict[str, Any] = {}
    if d < T:
        xa = head["xf"][d - 1] if d > 0 else x0
        Pa = head["Pf"][d - 1] if d > 0 else P0
        tail = kf(y0[d:], mask[d:], *(_tail(sysm[k], d) for k in keys), xa, Pa)
        res = dict(zip(names, tail))
    else:
        res = {"ll": np.zeros(0)}
        for key in ("xp", "xf"):
            res[key] = np.zeros((0, m))
        for key in ("Pp", "Pf"):
            res[key] = np.zeros((0, m, m))
        res["v"] = res["e"] = np.zeros((0, n))
        res["S"] = np.zeros((0, n, n))
    out: Dict[str, Any] = {
        key: np.concatenate([head[key], res[key]]) for key in ("xp", "Pp", "xf", "Pf")
    }
    out["ll"] = np.concatenate([head["ll"], res["ll"]])
    zeros = np.zeros((T - d, m, m))
    out["Ppi"] = np.concatenate([head["Ppi"], zeros])
    out["Pfi"] = np.concatenate([head["Pfi"], zeros])
    # diffuse dates: the joint prediction error and the finite part of its
    # covariance; there is nothing to standardise it by
    v = np.zeros((T, n))
    S = np.zeros((T, n, n))
    e = np.full((T, n), np.nan)
    for t in range(d):
        Gt = sysm["G"][t if sysm["G"].shape[0] > 1 else 0]
        At = sysm["A"][t if sysm["A"].shape[0] > 1 else 0]
        Rt = sysm["R"][t if sysm["R"].shape[0] > 1 else 0]
        v[t] = np.where(mask[t], y0[t] - At - Gt @ head["xp"][t], 0.0)
        S[t] = Gt @ head["Pp"][t] @ Gt.T + Rt
    v[d:], S[d:], e[d:] = res["v"], res["S"], res["e"]
    out.update(v=v, S=S, e=e, sys=sysm, mask=mask, x0=x0, P0=P0, m=m)
    out.update(rule="exact", P0inf=P0inf, d=d, absorbed=int(head["absorbed"]))
    if smooth:
        r, N = np.zeros(m), np.zeros((m, m))
        xs = np.zeros((0, m))
        Ps = np.zeros((0, m, m))
        if d < T:
            xs, Ps, r, N = ks(
                res["xp"],
                res["Pp"],
                res["v"],
                res["Sinv"],
                res["K"],
                _tail(sysm["G"], d),
                mask[d:],
                _tail(sysm["F"], d),
            )
        hx, hP = smooth_exact(head, sysm["F"], r, N, T)
        out["xs"] = np.concatenate([hx, xs])
        out["Ps"] = np.concatenate([hP, Ps])
    return out
