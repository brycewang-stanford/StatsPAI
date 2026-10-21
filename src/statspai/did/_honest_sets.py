"""Restriction families of Honest DiD beyond plain relative magnitudes.

Rambachan and Roth [@rambachan2023more] bound the parallel-trends violation
``delta`` by a polyhedron (or a finite union of polyhedra) and invert the
Andrews-Roth-Pakes conditional tests [@andrews2023inference] over a grid of
the target ``theta = l' tau_post``. Their R package ``HonestDiD`` 0.2.8
(MIT licence; notice in THIRD_PARTY_NOTICES.md) offers, besides
smoothness ``Delta^SD(M)`` and relative magnitudes ``Delta^RM(Mbar)``:

* sign restrictions (``biasDirection``): ``delta_post >= 0`` or ``<= 0``;
* shape restrictions (``monotonicityDirection``): ``delta`` increasing or
  decreasing in event time, through the normalised reference period;
* relative magnitudes of the *second* differences, ``Delta^SDRM(Mbar)``
  ("deviation from linear trend"): post-period curvature at most ``Mbar``
  times the largest pre-period curvature;

each combinable with the base restriction (``Delta^SDPB``, ``Delta^SDI``,
``Delta^RMNB``, ``Delta^SDRMD`` ...). This module builds the moment
matrices exactly as ``HonestDiD`` does (``.create_A_*`` / ``.create_d_*``),
lays out its default ``theta`` grids (from the identified set at
``beta = 0`` for the smoothness family, ``+/- 20`` standard deviations for
the relative-magnitude family) and hands them to the grid inversion of
:mod:`statspai.did._arp`, so a correct port accepts the same grid points.

Departures, each documented where it happens:

* With one pre-period, ``HonestDiD``'s ``.create_A_M`` loops over
  ``1:(numPrePeriods - 1)``, which in R is ``c(1, 0)``; the first iteration
  writes ``(1, -1)`` into the row that should read ``delta_{-2} <= 0`` and
  so imposes ``delta_{-2} <= delta_0`` instead, a weaker restriction. The
  matrix here is the intended one.
* The conditional-FLCI hybrid's ``vbar`` is a least-squares solution rather
  than a CVXR solve; the test depends on it only through ``A' vbar``, which
  every minimiser shares.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import optimize

from ..exceptions import ConvergenceFailure, MethodIncompatibility
from ._arp import create_a_rm
from ._arp_sets import union_confidence_set

__all__ = [
    "create_a_sd",
    "create_a_bias",
    "create_a_monotone",
    "create_a_sdrm",
    "smoothness_pieces",
    "rm_family_pieces",
    "sd_family_confidence_set",
    "rm_family_confidence_set",
]


# ---------------------------------------------------------------------------
#  Moment matrices (columns: delta without the normalised reference period)
# ---------------------------------------------------------------------------


def create_a_sd(n_pre: int, n_post: int) -> np.ndarray:
    """``A`` of ``Delta^SD(M)``: ``|second difference| <= M`` (``d = M``)."""
    k = n_pre + n_post
    a_tilde = np.zeros((k - 1, k + 1))
    for r in range(k - 1):
        a_tilde[r, r : r + 3] = (1.0, -2.0, 1.0)
    a_tilde = np.delete(a_tilde, n_pre, axis=1)
    return np.vstack([a_tilde, -a_tilde])


def create_a_bias(n_pre: int, n_post: int, direction: str) -> np.ndarray:
    """``A`` of the sign restriction: ``delta_post >= 0`` (``'positive'``)
    or ``<= 0`` (``'negative'``), ``d = 0`` (``HonestDiD`` ``.create_A_B``)."""
    k = n_pre + n_post
    a = -np.eye(k)[n_pre:]
    return -a if direction == "negative" else a


def create_a_monotone(n_pre: int, n_post: int, direction: str) -> np.ndarray:
    """``A`` of the shape restriction (``HonestDiD`` ``.create_A_M`` with
    ``postPeriodMomentsOnly = FALSE``), ``d = 0``: ``delta`` increasing
    (``'increasing'``) or decreasing through the reference period ``0``.

    One row per adjacent pair of event times: ``delta_r - delta_{r+1} <= 0``
    before the reference period, ``delta_{-2} <= 0`` (the last pre-period
    against the reference), ``-delta_0 <= 0`` and then
    ``delta_{t-1} - delta_t <= 0`` after it; negated for ``'decreasing'``.
    """
    k = n_pre + n_post
    a = np.zeros((k, k))
    for r in range(n_pre - 1):
        a[r, r : r + 2] = (1.0, -1.0)
    a[n_pre - 1, n_pre - 1] = 1.0
    if n_post > 0:
        a[n_pre, n_pre] = -1.0
        for r in range(n_pre + 1, k):
            a[r, r - 1 : r + 1] = (1.0, -1.0)
    return -a if direction == "decreasing" else a


def create_a_sdrm(
    n_pre: int, n_post: int, m_bar: float, s: int, max_positive: bool
) -> np.ndarray:
    """``A`` of ``Delta^SDRM_{s,sign}(Mbar)`` (``HonestDiD`` ``.create_A_SDRM``),
    ``d = 0``.

    ``|second difference_t| <= +/-`` (the pre-period second difference ending
    at ``s``, ``s = -(n_pre - 2), ..., 0``) before treatment and ``<= Mbar``
    times it after.
    """
    k = n_pre + n_post
    a_tilde = np.zeros((k - 1, k + 1))
    for r in range(k - 1):
        a_tilde[r, r : r + 3] = (1.0, -2.0, 1.0)
    v = np.zeros(k + 1)
    # R: v_max_dif[(numPre + 1 + s - 2):(numPre + 1 + s)] <- c(1, -2, 1).
    v[n_pre + s - 2 : n_pre + s + 1] = (1.0, -2.0, 1.0)
    if not max_positive:
        v = -v
    a_ub = np.vstack([np.tile(v, (n_pre - 1, 1)), np.tile(m_bar * v, (n_post, 1))])
    a = np.vstack([a_tilde - a_ub, -a_tilde - a_ub])
    a = a[np.einsum("ij,ij->i", a, a) > 1e-10]
    return np.delete(a, n_pre, axis=1)


def _extra(n_pre: int, n_post: int, bias: Optional[str], mono: Optional[str]):
    if bias is not None:
        return create_a_bias(n_pre, n_post, bias)
    if mono is not None:
        return create_a_monotone(n_pre, n_post, mono)
    return None


def smoothness_pieces(
    n_pre: int,
    n_post: int,
    m: float,
    bias: Optional[str] = None,
    mono: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(A, d)`` of ``Delta^SD(M)`` intersected with a sign or shape
    restriction (``Delta^SDPB`` / ``SDNB`` / ``SDI`` / ``SDD``)."""
    a = create_a_sd(n_pre, n_post)
    d = np.full(a.shape[0], float(m))
    extra = _extra(n_pre, n_post, bias, mono)
    if extra is not None:
        a = np.vstack([a, extra])
        d = np.concatenate([d, np.zeros(extra.shape[0])])
    return a, d


def rm_family_pieces(
    n_pre: int,
    n_post: int,
    m_bar: float,
    base: str = "rm",
    bias: Optional[str] = None,
    mono: Optional[str] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """The polyhedral pieces whose union is ``Delta^RM`` / ``Delta^SDRM``
    (``base='rm'`` / ``'sdrm'``), each intersected with a sign or shape
    restriction; HonestDiD's order (``s`` ascending, ``+`` before ``-``)."""
    extra = _extra(n_pre, n_post, bias, mono)
    if base == "sdrm":
        s_range = range(-(n_pre - 2), 1)
        build = create_a_sdrm
    else:
        s_range = range(-(n_pre - 1), 1)
        build = create_a_rm
    out = []
    for s in s_range:
        for positive in (True, False):
            a = build(n_pre, n_post, float(m_bar), s, positive)
            if extra is not None:
                a = np.vstack([a, extra])
            out.append((a, np.zeros(a.shape[0])))
    return out


# ---------------------------------------------------------------------------
#  Identified set at beta = 0 (HonestDiD's default grid for Delta^SD*)
# ---------------------------------------------------------------------------


def _identified_set_at_zero(
    a: np.ndarray, d: np.ndarray, n_pre: int, l_vec: np.ndarray
) -> Tuple[float, float]:
    """``(id.lb, id.ub)`` of ``.compute_IDset_DeltaSD*`` at ``trueBeta = 0``:
    ``-max`` and ``-min`` of ``l' delta_post`` over ``A delta <= d``,
    ``delta_pre = 0``."""
    k = a.shape[1]
    f = np.concatenate([np.zeros(n_pre), l_vec])
    a_eq = np.hstack([np.eye(n_pre), np.zeros((n_pre, k - n_pre))])
    out = []
    for sign in (1.0, -1.0):  # min l'delta, then max l'delta
        res = optimize.linprog(
            sign * f,
            A_ub=a,
            b_ub=d,
            A_eq=a_eq,
            b_eq=np.zeros(n_pre),
            bounds=[(None, None)] * k,
            method="highs",
        )
        out.append(res)
    if out[0].status != 0 and out[1].status != 0:
        # HonestDiD: "Solver did not find an optimum", identified set {0}.
        return 0.0, 0.0
    if out[0].status != 0 or out[1].status != 0:
        raise ConvergenceFailure(
            "honest_did: the identified-set linear program that lays out the "
            "default grid did not solve on one side.",
            recovery_hint="Pass grid_lb and grid_ub explicitly.",
            diagnostics={"status": [int(out[0].status), int(out[1].status)]},
        )
    lo_val, hi_val = float(out[0].fun), -float(out[1].fun)
    return -hi_val, -lo_val


# ---------------------------------------------------------------------------
#  Confidence sets
# ---------------------------------------------------------------------------


def _sd_theta(sigma: np.ndarray, n_pre: int, l_vec: np.ndarray) -> float:
    post = sigma[n_pre:, n_pre:]
    return float(np.sqrt(l_vec @ post @ l_vec))


def sd_family_confidence_set(
    betahat: np.ndarray,
    sigma: np.ndarray,
    n_pre: int,
    n_post: int,
    m: float,
    l_vec: np.ndarray,
    alpha: float = 0.05,
    hybrid: str = "FLCI",
    bias: Optional[str] = None,
    mono: Optional[str] = None,
    grid_points: int = 1000,
    grid_lb: Optional[float] = None,
    grid_ub: Optional[float] = None,
    max_expand: int = 0,
    seed: int = 0,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Conditional / hybrid confidence set under ``Delta^SD(M)`` (with an
    optional sign or shape restriction): HonestDiD's
    ``computeConditionalCS_DeltaSD`` / ``SDB`` / ``SDM``.

    ``hybrid`` is ``"ARP"`` (``method = "Conditional"``), ``"LF"``
    (``"C-LF"``) or ``"FLCI"`` (``"C-F"``). The C-F set lies inside the
    level-``alpha/10`` FLCI, which is its grid, so it is never extended; the
    others start from the identified set at ``beta = 0`` widened by
    ``+/- 20`` standard deviations of the target and are extended at an
    accepted end up to ``max_expand`` times; when that grid accepts nothing
    because the estimate lies outside it, the same span is moved around the
    estimate (HonestDiD reports an empty set there).
    """
    from ._flci import flci_delta_sd

    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    l_vec = np.asarray(l_vec, dtype=float).ravel()
    kappa = alpha / 10.0
    a, d = smoothness_pieces(n_pre, n_post, m, bias=bias, mono=mono)
    flci = None
    if hybrid == "FLCI":
        res = flci_delta_sd(
            betahat, sigma, n_pre, n_post, float(m), l_post=l_vec, alpha=kappa
        )
        flci_l = np.concatenate([res.pre_period_weights, l_vec])
        flci = (flci_l, float(res.half_length))
        centre = float(flci_l @ betahat)
        lb = centre - flci[1] if grid_lb is None else float(grid_lb)
        ub = centre + flci[1] if grid_ub is None else float(grid_ub)
        expand_lo = expand_hi = False
        recentre = None
    else:
        lb = grid_lb
        ub = grid_ub
        recentre = None
        if grid_lb is None or grid_ub is None:
            id_lb, id_ub = _identified_set_at_zero(a, d, n_pre, l_vec)
            if bias == "negative" and n_post > 1:
                # HonestDiD flips the identified set for a negative bias in
                # the several-post-period branch only (a grid choice; the
                # grid extension below recovers any set it cuts off).
                id_lb, id_ub = -id_ub, -id_lb
            sd = _sd_theta(sigma, n_pre, l_vec)
            lb = id_lb - 20.0 * sd if grid_lb is None else float(grid_lb)
            ub = id_ub + 20.0 * sd if grid_ub is None else float(grid_ub)
            if grid_lb is None and grid_ub is None:
                # HonestDiD centres this grid on the identified set at
                # beta = 0. An estimate further out than its half-width plus
                # 20 sd lies outside it, nothing is accepted and the set would
                # be reported as empty (R returns NA); test the same span
                # around the estimate instead, as the relative-magnitude
                # family does.
                recentre = max(abs(id_lb), abs(id_ub)) + 20.0 * sd
        expand_lo, expand_hi = grid_lb is None, grid_ub is None
    return union_confidence_set(
        [(a, d)],
        betahat,
        sigma,
        n_pre,
        n_post,
        l_vec,
        alpha,
        hybrid,
        float(lb),
        float(ub),
        grid_points,
        kappa=kappa,
        seed=seed,
        flci=flci,
        expand_lo=expand_lo,
        expand_hi=expand_hi,
        max_expand=max_expand,
        recentre_sd=recentre,
    )


def rm_family_confidence_set(
    betahat: np.ndarray,
    sigma: np.ndarray,
    n_pre: int,
    n_post: int,
    m_bar: float,
    l_vec: np.ndarray,
    alpha: float = 0.05,
    hybrid: str = "LF",
    base: str = "rm",
    bias: Optional[str] = None,
    mono: Optional[str] = None,
    grid_points: int = 1000,
    grid_lb: Optional[float] = None,
    grid_ub: Optional[float] = None,
    max_expand: int = 0,
    seed: int = 0,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Conditional / C-LF confidence set under ``Delta^RM`` or ``Delta^SDRM``
    (``base``), with an optional sign or shape restriction: HonestDiD's
    ``computeConditionalCS_DeltaRM`` / ``RMB`` / ``RMM`` / ``SDRM`` /
    ``SDRMB`` / ``SDRMM``. Default grid ``+/- 20`` standard deviations of the
    target, extended at an accepted end it was not given."""
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    l_vec = np.asarray(l_vec, dtype=float).ravel()
    if base == "sdrm" and n_pre < 2:
        raise MethodIncompatibility(
            "The smoothness relative-magnitudes restriction (Delta^SDRM) needs "
            "at least two pre-treatment coefficients: its bound is the largest "
            "pre-period second difference.",
            recovery_hint="Use method='relative_magnitude', or an event study "
            "with more leads.",
            diagnostics={"n_pre": n_pre},
        )
    sd = _sd_theta(sigma, n_pre, l_vec)
    lb = -20.0 * sd if grid_lb is None else float(grid_lb)
    ub = 20.0 * sd if grid_ub is None else float(grid_ub)
    pieces = rm_family_pieces(n_pre, n_post, m_bar, base=base, bias=bias, mono=mono)
    return union_confidence_set(
        pieces,
        betahat,
        sigma,
        n_pre,
        n_post,
        l_vec,
        alpha,
        hybrid,
        lb,
        ub,
        grid_points,
        seed=seed,
        nonuis_post_rows_only=True,
        expand_lo=grid_lb is None,
        expand_hi=grid_ub is None,
        max_expand=max_expand,
        recentre_sd=(20.0 * sd if grid_lb is None and grid_ub is None else None),
    )


# ---------------------------------------------------------------------------
#  Sensitivity tables (sp.honest_did's native path for these restrictions)
# ---------------------------------------------------------------------------

#: HonestDiD's name of the restriction set, by (family, sign, shape).
_DELTA_NAME = {
    ("smoothness", None, None): "DeltaSD",
    ("smoothness", "positive", None): "DeltaSDPB",
    ("smoothness", "negative", None): "DeltaSDNB",
    ("smoothness", None, "increasing"): "DeltaSDI",
    ("smoothness", None, "decreasing"): "DeltaSDD",
    ("rm", None, None): "DeltaRM",
    ("rm", "positive", None): "DeltaRMPB",
    ("rm", "negative", None): "DeltaRMNB",
    ("rm", None, "increasing"): "DeltaRMI",
    ("rm", None, "decreasing"): "DeltaRMD",
    ("sdrm", None, None): "DeltaSDRM",
    ("sdrm", "positive", None): "DeltaSDRMPB",
    ("sdrm", "negative", None): "DeltaSDRMNB",
    ("sdrm", None, "increasing"): "DeltaSDRMI",
    ("sdrm", None, "decreasing"): "DeltaSDRMD",
}

_INTERVAL_OF_HYBRID = {"ARP": "arp_conditional", "LF": "arp_c_lf", "FLCI": "arp_c_f"}


def delta_name(family: str, bias: Optional[str], mono: Optional[str]) -> str:
    """HonestDiD's label for a restriction set (``'DeltaSDRMPB'`` ...)."""
    return _DELTA_NAME[(family, bias, mono)]


def restricted_table(
    inputs: tuple,
    m_grid,
    alpha: float,
    hybrid: str,
    family: str,
    bias: Optional[str],
    mono: Optional[str],
    grid_lb: Optional[float],
    grid_ub: Optional[float],
    grid_points: int,
    grid_expand: bool,
    seed: int = 0,
):
    """Sensitivity table over ``m_grid`` for one restriction set.

    ``inputs`` is ``(betahat, sigma, n_pre, n_post, l_post)`` in HonestDiD
    order; ``family`` is ``'smoothness'``, ``'rm'`` or ``'sdrm'``. Returns
    the ``DataFrame`` and the grid values at which the set reached an
    unextended end (``open_at``) and those whose grid was extended.
    """
    import pandas as pd

    b, s, n_pre, n_post, l_post = inputs
    rows = []
    open_at: List[float] = []
    extended_at: List[float] = []
    max_expand = 10 if grid_expand else 0
    for m in m_grid:
        if family == "smoothness":
            lo, hi, grid, accept = sd_family_confidence_set(
                b,
                s,
                n_pre,
                n_post,
                float(m),
                l_post,
                alpha=alpha,
                hybrid=hybrid,
                bias=bias,
                mono=mono,
                grid_points=grid_points,
                grid_lb=grid_lb,
                grid_ub=grid_ub,
                max_expand=max_expand,
                seed=seed,
            )
        else:
            lo, hi, grid, accept = rm_family_confidence_set(
                b,
                s,
                n_pre,
                n_post,
                float(m),
                l_post,
                alpha=alpha,
                hybrid=hybrid,
                base=family,
                bias=bias,
                mono=mono,
                grid_points=grid_points,
                grid_lb=grid_lb,
                grid_ub=grid_ub,
                max_expand=max_expand,
                seed=seed,
            )
        if grid.size > grid_points:
            extended_at.append(float(m))
        # The C-F set lives inside its first-stage FLCI, which is its grid:
        # reaching an end of that grid is not an open set.
        if hybrid != "FLCI" and accept.size and (accept[0] == 1 or accept[-1] == 1):
            open_at.append(float(m))
        rows.append(
            {
                "M": float(m),
                "ci_lower": lo,
                "ci_upper": hi,
                "rejects_zero": bool(np.isfinite(lo) and not (lo <= 0 <= hi)),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["interval"] = _INTERVAL_OF_HYBRID[hybrid]
    out.attrs["delta"] = delta_name(family, bias, mono)
    out.attrs["grid_extended_at"] = extended_at
    out.attrs["open_at"] = open_at
    return out
