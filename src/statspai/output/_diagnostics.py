"""Auto-extract publication-table diagnostic rows from result objects.

Top-tier journal regression tables conventionally include rows that are not
coefficients — fixed-effect indicators, cluster-SE labels, IV first-stage F,
DiD pre-trend p-values, RD bandwidth and kernel — that
:func:`statspai.regtable` historically required users to type by hand via
``add_rows={...}``. This module reads the metadata that StatsPAI estimators
already attach to their result objects (``model_info`` /
``data_info`` / ``diagnostics``) and turns it into a list of
``(row_label, [cell_per_model, ...])`` pairs that the renderer can drop
directly above the summary-stats block.

The extraction layer is intentionally tolerant: every probe is wrapped in a
``try``/``except`` and silently returns an empty cell when the metadata is
missing. This means ``regtable(..., diagnostics="auto")`` works on a mixed
list of OLS / IV / DiD / RD results without per-model branching by the
caller.

Public API
----------
:func:`extract_diagnostic_rows`
    Inspect every result and return an ``OrderedDict`` mapping row label to
    one cell per model (string), suitable for :class:`regtable.RegtableResult`'s
    ``add_rows`` slot.
:func:`extract_fe_cluster_indicators`
    Subset of the above limited to FE / cluster-SE indicator rows (used
    when the caller wants those without the IV/DiD/RD specifics).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result-object adapters
# ---------------------------------------------------------------------------

def _is_causal(result) -> bool:
    """Match :class:`statspai.core.results.CausalResult` duck-style."""
    return (
        hasattr(result, "estimand")
        and hasattr(result, "estimate")
        and hasattr(result, "se")
        and hasattr(result, "method")
    )


def _is_econometric(result) -> bool:
    """Match :class:`statspai.core.results.EconometricResults` duck-style."""
    return (
        hasattr(result, "params")
        and hasattr(result, "std_errors")
        and not _is_causal(result)
    )


def _model_info(result) -> Dict[str, Any]:
    return getattr(result, "model_info", {}) or {}


def _data_info(result) -> Dict[str, Any]:
    return getattr(result, "data_info", {}) or {}


def _diagnostics(result) -> Dict[str, Any]:
    return getattr(result, "diagnostics", {}) or {}


# ---------------------------------------------------------------------------
# Cell formatters
# ---------------------------------------------------------------------------

def _yes_no_cell(present: bool) -> str:
    return "Yes" if present else "No"


def _fmt_num(val, fmt: str = "%.2f") -> str:
    """Format a number, returning ``""`` for None/NaN/non-numeric input."""
    if val is None:
        return ""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(f):
        return ""
    return fmt % f


def _fmt_int(val) -> str:
    if val is None:
        return ""
    try:
        return f"{int(round(float(val))):,}"
    except (TypeError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# Per-row extractors
# ---------------------------------------------------------------------------

def _fe_cell(result) -> str:
    """Return ``"Yes"`` if the model absorbs at least one fixed effect.

    Looks for the canonical ``model_info['fixed_effects']`` field that the
    pyfixest adapter, ``feols`` wrapper, and HDFE pathway populate. The
    shape is forgiving — a non-empty string, list, tuple, or dict counts.
    """
    if _is_causal(result):
        return ""  # Causal estimators rarely report FE in this form
    mi = _model_info(result)
    fe = mi.get("fixed_effects") or mi.get("absorbed_fe") or mi.get("fe")
    if fe in (None, "", "None", "none"):
        return "No"
    if isinstance(fe, (list, tuple, set, dict)):
        return "Yes" if len(fe) > 0 else "No"
    return "Yes"


def _cluster_cell(result) -> str:
    """Return the cluster variable name (or ``"Yes"`` / ``"No"``).

    A common Top-5 convention is to label the row ``"Cluster SE"`` and put
    the variable name (e.g. ``"firm"``) in each column. We follow that.
    Reads from the same ``model_info`` keys for econometric and causal
    results — the metadata shape is identical.
    """
    mi = _model_info(result)
    cl = mi.get("cluster") or mi.get("cluster_var")
    if cl in (None, "", "None", "none", False):
        return "No"
    if isinstance(cl, (list, tuple, set)):
        return ", ".join(str(c) for c in cl) if cl else "No"
    return str(cl)


def _se_type_cell(result) -> str:
    """Return a short label for the SE type (HC0/HC3/Cluster/etc.)."""
    mi = _model_info(result)
    if _is_causal(result):
        return ""
    cluster = mi.get("cluster")
    robust = mi.get("robust") or mi.get("vcov_type")
    if cluster:
        return "Clustered"
    if not robust or robust in ("nonrobust", "iid", "constant"):
        return "Homoskedastic"
    return str(robust).upper()


# ---- IV diagnostics --------------------------------------------------------

def _iv_first_stage_F_cell(result) -> str:
    """Return the first-stage F-stat for IV models, or ``""`` if not IV."""
    diag = _diagnostics(result)
    # pyfixest / fixed F naming
    for key in (
        "Olea-Pflueger effective F",
        "OP effective F",
        "KP rk Wald F",
        "F (first stage)",
    ):
        if key in diag:
            return _fmt_num(diag[key], "%.2f")
    # Per-endog naming used by sp.regress(method='2sls')
    for key in diag:
        if isinstance(key, str) and key.startswith("First-stage F (") and not key.endswith("p-value)"):
            return _fmt_num(diag[key], "%.2f")
    return ""


def _iv_kp_F_cell(result) -> str:
    diag = _diagnostics(result)
    val = diag.get("KP rk Wald F")
    return _fmt_num(val, "%.2f") if val is not None else ""


def _iv_hansen_J_p_cell(result) -> str:
    """Return Hansen-J / Sargan p-value (overid test) when applicable."""
    diag = _diagnostics(result)
    for key in ("Hansen J p-value", "Sargan p-value", "Hansen-J p-value"):
        if key in diag:
            return _fmt_num(diag[key], "%.3f")
    return ""


def _is_iv_result(result) -> bool:
    """Detect IV: result has any IV-specific diagnostic key."""
    if _is_causal(result):
        return False
    diag = _diagnostics(result)
    iv_keys = (
        "Olea-Pflueger effective F",
        "OP effective F",
        "KP rk Wald F",
        "F (first stage)",
        "Sargan statistic",
        "Hansen J statistic",
        "Hansen J p-value",
    )
    if any(k in diag for k in iv_keys):
        return True
    return any(
        isinstance(k, str) and k.startswith("First-stage F (")
        for k in diag
    )


# ---- DiD diagnostics -------------------------------------------------------

def _is_did_result(result) -> bool:
    """Detect a DiD/event-study result via ``method`` only.

    The earlier version also matched ``estimand in {"ATT", "ATET"}`` but
    those estimands are produced by AIPW / DML / matching / TMLE too, which
    would trigger spurious DiD-only diagnostic probes (pre-trend p, treated
    group count). Method-string detection is narrower and safe.
    """
    if not _is_causal(result):
        return False
    method = (getattr(result, "method", "") or "").lower()
    return "did" in method or "diff-in-diff" in method or "staggered" in method


def _did_pretrend_p_cell(result) -> str:
    if not _is_did_result(result):
        return ""
    mi = _model_info(result)
    pt = mi.get("pretrend_test") or mi.get("pre_trend_test")
    if pt is None:
        return ""
    if isinstance(pt, dict):
        for key in ("pvalue", "p_value", "p"):
            if key in pt:
                return _fmt_num(pt[key], "%.3f")
        return ""
    return _fmt_num(pt, "%.3f")


def _did_n_groups_cell(result) -> str:
    if not _is_did_result(result):
        return ""
    mi = _model_info(result)
    n = mi.get("n_groups") or mi.get("n_treated_groups")
    return _fmt_int(n)


# ---- RD diagnostics --------------------------------------------------------

def _is_rd_result(result) -> bool:
    if not _is_causal(result):
        return False
    method = (getattr(result, "method", "") or "").lower()
    if "rd" in method or "discontinuity" in method:
        return True
    mi = _model_info(result)
    return "bandwidth_h" in mi or "bandwidth" in mi or "kernel" in mi


def _rd_bandwidth_cell(result) -> str:
    if not _is_rd_result(result):
        return ""
    mi = _model_info(result)
    bw = mi.get("bandwidth_h")
    if bw is None:
        bw = mi.get("bandwidth")
    if bw is None:
        return ""
    if isinstance(bw, (tuple, list)) and len(bw) == 2:
        return f"{_fmt_num(bw[0], '%.3f')} / {_fmt_num(bw[1], '%.3f')}"
    if isinstance(bw, dict):
        if "left" in bw and "right" in bw:
            return f"{_fmt_num(bw['left'], '%.3f')} / {_fmt_num(bw['right'], '%.3f')}"
        for k in ("h", "value"):
            if k in bw:
                return _fmt_num(bw[k], "%.3f")
        return ""
    return _fmt_num(bw, "%.3f")


def _rd_kernel_cell(result) -> str:
    if not _is_rd_result(result):
        return ""
    mi = _model_info(result)
    k = mi.get("kernel")
    if k is None:
        return ""
    return str(k).capitalize()


def _rd_polyorder_cell(result) -> str:
    if not _is_rd_result(result):
        return ""
    mi = _model_info(result)
    p = mi.get("polynomial_p")
    if p is None:
        p = mi.get("p_order") or mi.get("p")
    return _fmt_int(p) if p is not None else ""


# ---------------------------------------------------------------------------
# Public extractors
# ---------------------------------------------------------------------------

def extract_fe_cluster_indicators(
    results: Sequence[Any],
) -> "OrderedDict[str, List[str]]":
    """Return ``{row_label: [cell_per_model, ...]}`` for FE & cluster rows.

    Rows are emitted only when at least one model produces a non-empty cell;
    rows that would be empty across every column are dropped. This avoids
    ``Fixed Effects | No | No | No`` rows in tables where nobody absorbs
    any FE.
    """
    rows: "OrderedDict[str, List[str]]" = OrderedDict()
    fe_cells = [_fe_cell(r) for r in results]
    cluster_cells = [_cluster_cell(r) for r in results]
    if any(c == "Yes" for c in fe_cells):
        rows["Fixed Effects"] = fe_cells
    if any(c not in ("", "No") for c in cluster_cells):
        rows["Cluster SE"] = cluster_cells
    return rows


def extract_diagnostic_rows(
    results: Sequence[Any],
    *,
    include_fe_cluster: bool = True,
    include_iv: bool = True,
    include_did: bool = True,
    include_rd: bool = True,
) -> "OrderedDict[str, List[str]]":
    """Build the dict of auto-diagnostic rows for a list of results.

    Parameters
    ----------
    results : sequence
        Model results (one per column) — any mix of ``EconometricResults``
        and ``CausalResult`` is fine.
    include_fe_cluster, include_iv, include_did, include_rd : bool
        Disable specific row families. Default is to enable all four.

    Returns
    -------
    OrderedDict
        Maps row label to a list of strings (one per model). Rows where
        every cell is empty are dropped, so tables stay clean for OLS-only
        bundles.
    """
    rows: "OrderedDict[str, List[str]]" = OrderedDict()

    # FE / cluster indicators come first — every paper has them.
    if include_fe_cluster:
        rows.update(extract_fe_cluster_indicators(results))

    # IV-specific diagnostics
    if include_iv and any(_is_iv_result(r) for r in results):
        fs = [_iv_first_stage_F_cell(r) for r in results]
        if any(c for c in fs):
            rows["First-stage F"] = fs
        kp = [_iv_kp_F_cell(r) for r in results]
        # Only show KP F when distinct from the first-stage F we already showed
        if any(c for c in kp) and kp != fs:
            rows["KP rk Wald F"] = kp
        hj = [_iv_hansen_J_p_cell(r) for r in results]
        if any(c for c in hj):
            rows["Hansen J p-value"] = hj

    # DiD-specific
    if include_did and any(_is_did_result(r) for r in results):
        pt = [_did_pretrend_p_cell(r) for r in results]
        if any(c for c in pt):
            rows["Pre-trend p-value"] = pt
        ng = [_did_n_groups_cell(r) for r in results]
        if any(c for c in ng):
            rows["Treated groups"] = ng

    # RD-specific
    if include_rd and any(_is_rd_result(r) for r in results):
        bw = [_rd_bandwidth_cell(r) for r in results]
        if any(c for c in bw):
            rows["Bandwidth"] = bw
        kn = [_rd_kernel_cell(r) for r in results]
        if any(c for c in kn):
            rows["Kernel"] = kn
        po = [_rd_polyorder_cell(r) for r in results]
        if any(c for c in po):
            rows["Polynomial order"] = po

    return rows
