"""Routing of the explicit-design local projections.

``sp.local_projections`` hands its arguments here first; the IV, state-
dependent, panel and explicit-lag designs are fitted by ``_lp_linear``,
``_lp_state`` and ``_lp_panel``. ``None`` means none of their arguments was
given and the original direct / Cholesky paths run unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import MethodIncompatibility


def extended_design(**kw: Any) -> Any:
    """Route the IV, state-dependent, panel and explicit-lag designs.

    Returns ``None`` when none of their arguments is set, so the original
    two paths run unchanged.
    """
    from ..output._lineage import attach_provenance

    cholesky = kw["cholesky"]
    panel = kw["id"] is not None or kw["time"] is not None
    linear = (
        kw["lags"] is not None
        or bool(kw["instruments"])
        or kw["vce"] is not None
        or kw["trend"] != 0
        or kw["common_sample"]
        or kw["dfk"] is not None
    )
    if not (panel or kw["state"] is not None or linear):
        return None
    if kw["horizons"] < 0:
        raise MethodIncompatibility(
            f"horizons must be non-negative, got {kw['horizons']!r}.",
            recovery_hint="Pass the last horizon H; horizons 0..H are fitted.",
        )
    if panel:
        if kw["id"] is None or kw["time"] is None:
            raise MethodIncompatibility(
                "sp.local_projections: a panel projection needs both id= and " "time=.",
                recovery_hint="Pass the unit column as id= and the period "
                "column as time=.",
            )
        if cholesky or kw["state"] is not None:
            raise MethodIncompatibility(
                "sp.local_projections: panel projections take an observed "
                "shock only (no Cholesky or state-dependent form).",
                recovery_hint="Drop identification= / state=, or fit one "
                "unit's series without id= / time=.",
            )
        from ._lp_panel import lp_panel

        result = lp_panel(
            data=kw["data"],
            outcome=kw["outcome"],
            shock=kw["shock"],
            controls=kw["controls"],
            horizons=kw["horizons"],
            id=kw["id"],
            time=kw["time"],
            lags=kw["lags"],
            endog_order=kw["endog_order"],
            instruments=kw["instruments"],
            panel_effect=kw["panel_effect"],
            diff_shock=kw["diff_shock"],
            cumulative=kw["cumulative"],
            vce=kw["vce"],
            nw_lags=kw["nw_lags"],
            alpha=kw["alpha"],
        )
    elif kw["state"] is not None:
        if kw["instruments"]:
            raise MethodIncompatibility(
                "sp.local_projections: instruments= and state= together are "
                "not supported (lpirfs::lp_nl_iv has no two-stage form).",
                recovery_hint="Fit the reduced form with the instrument as "
                "the shock, or split the sample by regime.",
            )
        from ._lp_core import resolve_vce
        from ._lp_state import lp_state

        result = lp_state(
            data=kw["data"],
            outcome=kw["outcome"],
            shock=kw["shock"],
            controls=kw["controls"],
            horizons=kw["horizons"],
            lags=kw["lags"],
            endog_order=kw["endog_order"],
            state=kw["state"],
            state_transform=kw["state_transform"],
            gamma=kw["gamma"],
            hp_lambda=kw["hp_lambda"],
            lag_state=kw["lag_state"],
            trend=kw["trend"],
            cumulative=kw["cumulative"],
            nw_lags=kw["nw_lags"],
            vce=resolve_vce(kw["vce"]),
            alpha=kw["alpha"],
            identification="cholesky" if cholesky else "direct",
        )
    else:
        if cholesky:
            raise MethodIncompatibility(
                "sp.local_projections: identification='lpirfs_cholesky' takes "
                "one lag and Newey-West bands (lpirfs::lp_lin, "
                "lags_endog_lin=1); lags=, instruments=, vce=, trend= and "
                "common_sample= belong to the observed-shock design.",
                recovery_hint="Drop those arguments, or use "
                "identification='direct' with the shock as a column.",
            )
        from ._lp_core import resolve_vce
        from ._lp_linear import lp_linear

        lags = kw["lags"]
        if lags is None:
            lags = 1 if kw["auto_lag"] else 0
        result = lp_linear(
            data=kw["data"],
            outcome=kw["outcome"],
            shock=kw["shock"],
            controls=kw["controls"],
            horizons=kw["horizons"],
            lags=lags,
            endog_order=kw["endog_order"],
            instruments=kw["instruments"],
            trend=kw["trend"],
            cumulative=kw["cumulative"],
            nw_lags=kw["nw_lags"],
            vce=resolve_vce(kw["vce"]),
            dfk=kw["dfk"],
            alpha=kw["alpha"],
            common_sample=kw["common_sample"],
        )
    params = {
        k: v for k, v in kw.items() if k != "data" and not isinstance(v, np.ndarray)
    }
    attach_provenance(
        result,
        function="sp.timeseries.local_projections",
        params=params,
        data=kw["data"],
        overwrite=False,
    )
    return result
