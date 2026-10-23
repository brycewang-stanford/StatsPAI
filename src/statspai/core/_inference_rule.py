"""Which inference rule a fitted regression used, read off the result.

Two programs can agree on every coefficient and standard error and still
print different t, F and p values: one refers the statistics to t(N - K),
the other to z, or to t(G - 1) under clustering; one scales a cluster
variance by G/(G-1), the other does not. An external audit against Stata
18 (2026-10-23) found that every "coefficients match, p-values do not"
case it met on 1.40.1 was a convention of this kind, legitimate but
invisible in the output. ``inference_rule`` states it, so that
``summary()`` and the agent payload carry the rule next to the numbers.

It only reads what the estimator recorded in ``model_info`` /
``data_info``; it never recomputes anything.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

#: Logit / probit / Poisson label their HC1 variance with this prefix.
_ML_HC1 = "HC1 (sandwich, N/(N-K))"


def _df_value(df: float) -> Any:
    return int(df) if float(df).is_integer() else float(df)


def _vcov_label(mi: Dict[str, Any]) -> str:
    for key in ("se_type", "vcov_type"):
        if isinstance(mi.get(key), str) and mi[key]:
            return str(mi[key])
    if mi.get("gmm_vcov"):
        # GMM: robust= / cluster= pick the weight matrix, not the variance
        kind = "sandwich" if mi["gmm_vcov"] == "sandwich" else "efficient"
        weight = mi.get("gmm_wmatrix", "?")
        label = f"GMM {kind} variance, {weight} weight matrix"
        if mi.get("cluster"):
            g = mi.get("n_clusters")
            label += f" (cluster {mi['cluster']}" + (f", G={g})" if g else ")")
        return label
    cluster = mi.get("cluster")
    if cluster:
        names = cluster if isinstance(cluster, str) else ", ".join(map(str, cluster))
        g = mi.get("n_clusters")
        if isinstance(g, (list, tuple)):
            g_text = ", G=" + "/".join(str(v) for v in g)
        elif g is not None:
            g_text = f", G={g}"
        else:
            g_text = ""
        return f"cluster ({names}){g_text}"
    robust = str(mi.get("robust", "nonrobust")).lower()
    if robust == "nonrobust":
        return "classical (iid)"
    if robust == "hac":
        lags = mi.get("hac_lags")
        return "HAC (Newey-West" + (f", {lags} lags)" if lags is not None else ")")
    return robust.upper()


def inference_rule(result: Any) -> Dict[str, Any]:
    """The variance, reference distribution and finite-sample convention.

    Keys: ``vcov`` (label), ``distribution`` (``'t'`` or ``'z'``), ``df``
    (``None`` under z), ``joint_test`` (``'F(q, df)'`` or ``'chi2(q)'``),
    ``n_obs``; when recorded, ``n_clusters``, ``small`` (IV),
    ``ssc`` (panel), ``n_rows`` / ``n_groups`` (panel), and ``notes``.
    """
    from ..postestimation._covariance import inference_df

    mi = getattr(result, "model_info", None) or {}
    di = getattr(result, "data_info", None) or {}
    df = inference_df(result, stores=("data_info",))
    z = not np.isfinite(df)
    rule: Dict[str, Any] = {
        "vcov": _vcov_label(mi),
        "distribution": "z" if z else "t",
        "df": None if z else _df_value(df),
        "joint_test": "chi2(q)" if z else f"F(q, {_df_value(df)})",
        "n_obs": di.get("nobs"),
    }
    if mi.get("n_clusters") is not None:
        rule["n_clusters"] = mi["n_clusters"]
    for key in ("n_rows", "n_groups"):
        if di.get(key) is not None:
            rule[key] = di[key]
    notes: List[str] = []
    if "small" in mi:
        rule["small"] = bool(mi["small"])
        if mi["small"]:
            notes.append(
                "small=True: the conventions of Stata's `ivregress ..., small` "
                "(N - K divisor, t / F). Stata's ivregress default is "
                "large-sample (z / chi2): sp.iv(..., small=False)."
            )
        else:
            notes.append(
                "small=False: the large-sample statistics of Stata's "
                "`ivregress` without `small` (N divisor, z / chi2)."
            )
    if mi.get("ssc") is not None:
        rule["ssc"] = mi["ssc"]
        if mi.get("ssc_reference"):
            notes.append(str(mi["ssc_reference"]))
    se_type = mi.get("se_type")
    if isinstance(se_type, str) and se_type.startswith(_ML_HC1):
        notes.append(
            "HC1 here scales by N/(N-K); Stata's `vce(robust)` after logit / "
            "probit / poisson scales by N/(N-1), which is robust='robust'."
        )
    if notes:
        rule["notes"] = notes
    return rule


def inference_lines(rule: Dict[str, Any], width: int = 80) -> List[str]:
    """``summary()`` lines for :func:`inference_rule`."""
    import textwrap

    dist = "z" if rule["distribution"] == "z" else f"t({rule['df']})"
    lines = [
        f"Std. errors: {rule['vcov']}",
        f"Inference: {dist}; joint tests {rule['joint_test']}",
    ]
    if "n_rows" in rule and "n_groups" in rule:
        lines.append(
            f"Sample: {rule['n_rows']} observations in {rule['n_groups']} groups "
            f"(N used for degrees of freedom: {rule['n_obs']})"
        )
    for note in rule.get("notes", ()):
        lines.extend(
            textwrap.wrap(
                note, width=width, initial_indent="Note: ", subsequent_indent="      "
            )
        )
    return lines
