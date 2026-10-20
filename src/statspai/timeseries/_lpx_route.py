"""Which local-projection design a call to ``sp.local_projections`` asks for.

The original two paths (an observed shock with ``auto_lag`` controls, and
``lpirfs``' one-lag Cholesky projection) run when none of the newer
arguments is given; :func:`route` then returns ``None``. Otherwise it
checks the arguments of the requested design (single-series explicit or
IV, state-dependent, panel) and returns the fitted result. Arguments that
belong to another design are refused rather than ignored.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .._input_validation import check_columns
from . import _lpx_common as C

SERIES_VCE = ("hac", "robust", "ols")


def _names(value: Optional[Sequence[str]], what: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    out = list(value)
    if len(out) != len(set(out)):
        raise C.refuse(f"{what} lists a column twice.", f"Pass each {what} once.")
    return out


def route(
    data: pd.DataFrame,
    *,
    outcome: str,
    shock: str,
    controls: List[str],
    horizons: int,
    nw_lags: Optional[int],
    alpha: float,
    cumulative: bool,
    auto_lag: bool,
    identification: str,
    endog_order: Optional[List[str]],
    lags: Any,
    instruments: Optional[List[str]],
    vce: Optional[str],
    dfk: Optional[bool],
    trend: int,
    common_sample: bool,
    state: Optional[str],
    state_transform: str,
    gamma: Optional[float],
    hp_lambda: Optional[float],
    lag_state: bool,
    id: Optional[str],
    time: Optional[str],
    panel_effect: str,
    diff_shock: bool,
) -> Optional[Any]:
    """The fitted explicit design, or ``None`` for the original paths."""
    instruments = _names(instruments, "instruments")
    panel = id is not None or time is not None
    explicit = (
        lags is not None
        or bool(instruments)
        or vce is not None
        or dfk is not None
        or trend != 0
        or bool(common_sample)
    )
    if state is None:
        stray = {
            "state_transform": state_transform != "logistic",
            "gamma": gamma is not None,
            "hp_lambda": hp_lambda is not None,
            "lag_state": not lag_state,
        }
        given = [k for k, v in stray.items() if v]
        if given:
            raise C.refuse(
                f"{', '.join(given)} shape a state-dependent projection, "
                "but no state= was given.",
                "Pass state='<column>' or drop those arguments.",
            )
    if not panel and (panel_effect != "individual" or diff_shock):
        raise C.refuse(
            "panel_effect= and diff_shock= apply to panel projections, but "
            "no id= / time= was given.",
            "Pass id= and time=, or drop those arguments.",
        )
    if not (panel or state is not None or explicit):
        return None

    if isinstance(horizons, (bool, np.bool_)) or not isinstance(
        horizons, (int, np.integer)
    ):
        raise C.refuse(
            f"horizons={horizons!r} is not an integer.",
            "Pass the last horizon H; horizons 0..H are fitted.",
        )
    if horizons < 0:
        raise C.refuse(
            f"horizons={horizons} is negative.",
            "Pass the last horizon H >= 0; horizons 0..H are fitted.",
        )
    if trend not in (0, 1, 2) or isinstance(trend, (bool, np.bool_)):
        raise C.refuse(
            f"trend={trend!r} is not 0, 1 or 2.",
            "0 = no trend, 1 = linear trend, 2 = linear and quadratic trend.",
        )
    if shock in instruments:
        raise C.refuse(
            "the shock is listed among its own instruments.",
            "instruments= are the excluded instruments of the endogenous "
            "shock, e.g. shock='g', instruments=['news'].",
        )
    cholesky = identification.lower().replace("-", "_") in {
        "lpirfs_cholesky",
        "cholesky",
    }
    order = _names(endog_order, "endog_order") or [outcome, shock]
    check_columns(
        data,
        {
            "endog_order": order,
            "instruments": instruments,
            **({"state": state} if state is not None else {}),
            **({"id": id} if id is not None else {}),
            **({"time": time} if time is not None else {}),
        },
        function=C.FN,
    )
    common: Dict[str, Any] = dict(
        outcome=outcome,
        shock=shock,
        controls=controls,
        horizons=int(horizons),
        nw_lags=nw_lags,
        alpha=alpha,
        cumulative=bool(cumulative),
    )

    if panel:
        return _panel(data, common, order, lags, instruments, vce, dfk, trend,
                      common_sample, state, cholesky, id, time, panel_effect,
                      diff_shock)  # fmt: skip

    lag_orders = C.lag_list((1 if auto_lag else 0) if lags is None else lags)
    if vce is None:
        vce = "hac"
    C.check_choice(
        "vce",
        vce,
        SERIES_VCE,
        "Use vce='hac' (Newey-West), 'robust' (HC0) or 'ols' "
        "(homoskedastic); 'cluster' and 'driscoll_kraay' need id= and time=.",
    )
    C.check_nw_lags(nw_lags, "the lpirfs rule h + 1")

    if state is not None:
        if instruments:
            raise C.refuse(
                "instruments= and state= together are not supported (the "
                "state-dependent projection takes an observed shock).",
                "Fit the reduced form with the instrument as the shock, or "
                "split the sample by regime.",
            )
        weight_kw = dict(
            transform=state_transform,
            gamma=gamma,
            hp_lambda=hp_lambda,
            lag_state=bool(lag_state),
        )
        if cholesky:
            return _state_cholesky(
                data, common, order, lags, auto_lag, vce, dfk, trend,
                common_sample, state, weight_kw,
            )  # fmt: skip
        from ._lpx_state import fit_state_direct

        return fit_state_direct(
            data,
            **common,
            lag_orders=lag_orders,
            lag_variables=order,
            vce=vce,
            dfk=dfk,
            trend=trend,
            common_sample=bool(common_sample),
            state=state,
            weight_kw=weight_kw,
        )

    if cholesky:
        raise C.refuse(
            "identification='lpirfs_cholesky' is the one-lag projection of "
            "lpirfs::lp_lin with Newey-West bands; lags=, instruments=, "
            "vce=, dfk=, trend= and common_sample= belong to the "
            "observed-shock design.",
            "Drop those arguments, or use identification='direct' with the "
            "shock as a column.",
        )
    from ._lpx_linear import fit_linear

    return fit_linear(
        data,
        **common,
        lag_orders=lag_orders,
        lag_variables=order,
        instruments=instruments,
        vce=vce,
        dfk=dfk,
        trend=trend,
        common_sample=bool(common_sample),
    )


def _state_cholesky(
    data: pd.DataFrame,
    common: Dict[str, Any],
    order: List[str],
    lags: Any,
    auto_lag: bool,
    vce: str,
    dfk: Optional[bool],
    trend: int,
    common_sample: bool,
    state: str,
    weight_kw: Dict[str, Any],
) -> Any:
    from ._lpx_state import fit_state_cholesky

    outcome, shock = common["outcome"], common["shock"]
    if order.count(outcome) != 1 or order.count(shock) != 1 or outcome == shock:
        raise C.refuse(
            "endog_order must list outcome and shock once each (it is the "
            "Cholesky order of the VAR).",
            "e.g. endog_order=['gdp', 'rate'].",
        )
    for name, given in (
        ("cumulative=True", common["cumulative"]),
        ("controls=", bool(common["controls"])),
        ("common_sample=True", bool(common_sample)),
    ):
        if given:
            raise C.refuse(
                f"{name} is not available for the Cholesky state-dependent "
                "projection (lpirfs::lp_nl has no such option).",
                "Use identification='direct' with an observed shock.",
            )
    orders = C.lag_list((1 if auto_lag else 0) if lags is None else lags)
    if not orders or orders != list(range(1, len(orders) + 1)):
        raise C.refuse(
            f"the Cholesky projection needs lags 1..p with p >= 1 (got " f"{orders}).",
            "Pass lags=p.",
        )
    return fit_state_cholesky(
        data,
        outcome=outcome,
        shock=shock,
        order=order,
        horizons=common["horizons"],
        nw_lags=common["nw_lags"],
        alpha=common["alpha"],
        p=len(orders),
        vce=vce,
        dfk=dfk,
        trend=trend,
        state=state,
        weight_kw=weight_kw,
    )


def _panel(
    data: pd.DataFrame,
    common: Dict[str, Any],
    order: List[str],
    lags: Any,
    instruments: List[str],
    vce: Optional[str],
    dfk: Optional[bool],
    trend: int,
    common_sample: bool,
    state: Optional[str],
    cholesky: bool,
    id: Optional[str],
    time: Optional[str],
    panel_effect: str,
    diff_shock: bool,
) -> Any:
    from ._lpx_panel import EFFECTS, PANEL_VCE, fit_panel

    if id is None or time is None:
        raise C.refuse(
            "a panel projection needs both id= and time=.",
            "Pass the unit column as id= and the period column as time=.",
        )
    if state is not None or cholesky:
        raise C.refuse(
            "panel projections take an observed shock only (no Cholesky or "
            "state-dependent form).",
            "Drop state= / identification=, or fit one unit as a series.",
        )
    for name, given in (
        ("dfk=", dfk is not None),
        ("trend=", trend != 0),
        ("common_sample=True", bool(common_sample)),
    ):
        if given:
            raise C.refuse(
                f"{name} is not available for panel projections.",
                "Drop it; the fixed effects (panel_effect=) take the place "
                "of a trend.",
            )
    C.check_choice(
        "panel_effect",
        panel_effect,
        EFFECTS,
        "'individual' (unit effects), 'time', 'twoways' or 'none' (pooled).",
    )
    vce = "cluster" if vce is None else vce
    C.check_choice(
        "vce",
        vce,
        PANEL_VCE,
        "Use vce='cluster' (by unit), 'driscoll_kraay' or 'ols' for panel "
        "projections.",
    )
    C.check_nw_lags(common["nw_lags"], "floor(T^(1/4)) periods")
    if common["nw_lags"] is not None and vce != "driscoll_kraay":
        raise C.refuse(
            "nw_lags= sets the Driscoll-Kraay lag; it has no role with "
            f"vce={vce!r}.",
            "Drop nw_lags=, or use vce='driscoll_kraay'.",
        )
    return fit_panel(
        data,
        **common,
        lag_orders=C.lag_list(0 if lags is None else lags),
        lag_variables=order,
        instruments=instruments,
        vce=vce,
        id=id,
        time=time,
        panel_effect=panel_effect,
        diff_shock=bool(diff_shock),
    )
