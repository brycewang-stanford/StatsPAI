"""Several treatments in one partially linear DML fit, with their joint covariance.

DoubleML's convention (``DoubleMLData`` with several ``d_cols``,
``use_other_treat_as_covariate=True``): for each treatment ``d_j`` a PLR is
fitted with the other treatments added to the controls, all on the *same*
cross-fitting split. The coefficients and their standard errors are then
exactly the single-treatment fits; what the joint fit adds is the covariance
across treatments, from the stacked orthogonal scores:

    Sigma_jk = mean(psi_j * psi_k) / (J_j * J_k) / n,
    psi_j = (y_tilde_j - theta_j d_tilde_j) d_tilde_j,   J_j = -mean(d_tilde_j^2),

which is what a joint Wald test, ``lincom`` of two treatment effects, or a
simultaneous band needs. It is the covariance DoubleML's ``psi`` arrays
imply (``tests/reference_parity/test_dml_multi_treatment_parity.py``).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..core.results import EconometricResults
from ..exceptions import MethodIncompatibility


def dml_multi_treatment(
    data: pd.DataFrame,
    y: str,
    treats: List[str],
    covariates: List[str],
    kwargs: Dict[str, Any],
) -> EconometricResults:
    """Joint PLR over ``treats``; see the module docstring."""
    from .plr import DoubleMLPLR

    if len(set(treats)) != len(treats):
        raise MethodIncompatibility("dml: treatment columns must be distinct.")
    model = str(kwargs.pop("model", "plr")).lower()
    if model != "plr":
        raise MethodIncompatibility(
            f"dml: several treatments are supported for model='plr' only "
            f"(got model={model!r}); the IRM / IIVM scores need one binary "
            "treatment.",
            recovery_hint="Fit one sp.dml per treatment, or use model='plr'.",
        )
    unsupported = [
        k
        for k in ("instrument", "cluster", "sample_weight", "external_predictions")
        if kwargs.get(k) is not None
    ]
    if kwargs.get("n_rep", 1) != 1:
        unsupported.append("n_rep>1")
    if kwargs.get("store_oof"):
        unsupported.append("store_oof")
    if kwargs.get("score") not in (None, "partialling out"):
        unsupported.append(f"score={kwargs.get('score')!r}")
    if unsupported:
        raise MethodIncompatibility(
            f"dml with several treatments does not support {unsupported}: the "
            "joint covariance is built from one cross-fitting split of the "
            "partialling-out score.",
            recovery_hint="Drop those options, or fit one sp.dml per treatment.",
            diagnostics={"unsupported": unsupported},
        )
    alpha = float(kwargs.get("alpha", 0.05))
    thetas, ses, psis, Js = [], [], [], []
    per_treat: Dict[str, Any] = {}
    n = None
    for j, d in enumerate(treats):
        controls = list(covariates) + [t for t in treats if t != d]
        est = DoubleMLPLR(
            data=data,
            y=y,
            treat=d,
            covariates=controls,
            ml_g=kwargs.get("ml_g"),
            ml_m=kwargs.get("ml_m"),
            n_folds=int(kwargs.get("n_folds", 5)),
            n_rep=1,
            alpha=alpha,
            random_state=int(kwargs.get("random_state", 42)),
            fold_indices=kwargs.get("fold_indices"),
        )
        res = est.fit()
        resid = getattr(est, "_last_rep_residuals", None)
        if resid is None:
            raise MethodIncompatibility(  # pragma: no cover - internal contract
                "dml: the PLR fit did not expose its residuals."
            )
        yt = np.asarray(resid["y_resid"], dtype=float)
        dt = np.asarray(resid["d_resid"], dtype=float)
        theta = float(res.estimate)
        psi = (yt - theta * dt) * dt
        J = -float(np.mean(dt**2))
        n = len(dt)
        thetas.append(theta)
        ses.append(float(res.se))
        psis.append(psi)
        Js.append(J)
        per_treat[d] = res
    P = np.column_stack(psis)
    Jv = np.asarray(Js)
    V = (P.T @ P) / n / np.outer(Jv, Jv) / n
    # The diagonal is the single-treatment variance by construction.
    params = pd.Series(thetas, index=list(treats))
    std_errors = pd.Series(np.sqrt(np.diag(V)), index=list(treats))
    model_info = {
        "model_type": "DML-PLR (several treatments)",
        "method": "Double/debiased ML, partialling out, joint score covariance",
        "dml_model": "plr",
        "score": "partialling out",
        "n_folds": int(kwargs.get("n_folds", 5)),
        "n_rep": 1,
        "treatments": list(treats),
        "alpha": alpha,
        "other_treatments_as_controls": True,
    }
    data_info = {
        "nobs": int(n or 0),
        "var_cov": V,
        "var_names": list(treats),
        "inference": "z",
        "dependent_var": y,
    }
    out = EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info=model_info,
        data_info=data_info,
        diagnostics={},
    )
    out.per_treatment = per_treat  # type: ignore[attr-defined]
    return out


__all__ = ["dml_multi_treatment"]
