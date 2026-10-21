"""Validation-scope map for ``sp.did_2x2``.

Same contract as :mod:`statspai.validation_scope`: each row lists the
configurations its artifact runs and the outputs it compares. The
``crit_dist`` dimension is the distribution behind the ATT's p-value and
interval: t(N - K) (``'t_resid'``, the default before 1.41 also under
clustering), t(G - 1) (``'t_cluster'``, ``parity='stata'`` / ``'r'`` or
``crit_dist='t'`` with ``cluster=``), t on the survey design's degrees of
freedom (``'t_design'``, ``design=``) or the wild cluster bootstrap
(``vce='wild'``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def register(add: Callable[[Any], None], vs: Any) -> None:
    _Row, _Scope, _vals, _mi, _set = (
        vs._Row,
        vs._Scope,
        vs._vals,
        vs._mi,
        vs._set,
    )
    RP = "tests/reference_parity/"
    EST_SE = ("estimate", "se")
    EST_SE_CI = ("estimate", "se", "interval")

    def cfg(**dims: Any) -> Dict[str, Any]:
        return {k: _vals(*v) for k, v in dims.items()}

    def x_did2x2(r: Any) -> Dict[str, Optional[str]]:
        mi = _mi(r)
        if "conventions" not in mi:
            # engine='bayes' returns sp.bayes_did's result: not this map.
            return {}
        if mi.get("survey_design") is not None:
            vce = "survey"
        elif mi.get("inference") is not None:
            vce = "wild"
        elif mi.get("cluster") is not None:
            vce = "cluster"
        else:
            vce = "hc1" if mi.get("robust_se") else "classical"
        detail = getattr(r, "detail", None)
        n_terms = None if detail is None else len(detail)
        conv = mi["conventions"]
        crit: Optional[str]
        if conv.get("crit_dist") == "bootstrap":
            crit = "bootstrap"
        elif vce == "survey":
            crit = "t_design"
        elif n_terms is None or getattr(r, "n_obs", None) is None:
            crit = None
        else:
            # t(N - K) is the residual degrees of freedom of the regression
            # (constant, treat, time, interaction and the covariates).
            resid = int(r.n_obs) - n_terms
            crit = "t_resid" if conv.get("crit_df") == resid else "t_cluster"
        return {
            "vce": vce,
            "weights": _set(mi.get("weights")),
            "covariates": None if n_terms is None else _set(n_terms > 4),
            "crit_dist": crit,
        }

    add(
        _Scope(
            "did_2x2",
            {
                "vce": ("classical", "hc1", "cluster", "wild", "survey"),
                "weights": ("none", "set"),
                "covariates": ("none", "set"),
                "crit_dist": ("t_resid", "t_cluster", "t_design", "bootstrap"),
            },
            x_did2x2,
            (
                _Row(
                    "T2",
                    RP + "test_did2x2_ddd_weighted_robust_parity.py",
                    cfg(
                        vce=["hc1"],
                        weights=["set"],
                        covariates=["none"],
                        crit_dist=["t_resid"],
                    ),
                    EST_SE,
                    "analytic weights with robust=True: the ATT and its HC1 SE "
                    "vs Stata 18 regress [aw=w], robust, to 1e-12",
                    "sp.did_2x2(",
                ),
                _Row(
                    "T2",
                    RP + "test_survey_did_parity.py",
                    cfg(
                        vce=["survey"],
                        weights=["none"],
                        covariates=["none"],
                        crit_dist=["t_design"],
                    ),
                    EST_SE,
                    "design= (strata, clusters, survey weights): the ATT and "
                    "its linearization SE vs R survey::svyglm(y ~ treat * "
                    "post) on the long design",
                    "sp.did_2x2(",
                ),
                _Row(
                    "T2",
                    RP + "test_survey_replicate_parity.py",
                    cfg(
                        vce=["survey"],
                        weights=["none"],
                        covariates=["none"],
                        crit_dist=["t_design"],
                    ),
                    EST_SE,
                    "a replicate-weight design=: the ATT and its replicate SE "
                    "vs R survey::svrepdesign + svyglm, to 1e-10",
                    "sp.did_2x2(",
                ),
                _Row(
                    "T2",
                    vs._ES_CRIT,
                    cfg(
                        vce=["cluster"],
                        weights=["none", "set"],
                        covariates=["none"],
                        crit_dist=["t_cluster"],
                    ),
                    EST_SE_CI,
                    "parity='stata' / 'r' on the castle-doctrine 2006 cohort "
                    "against the never treated, clustered by state, "
                    "unweighted and weighted: the ATT's SE, p-value and "
                    "confidence limits vs Stata regress, cluster() and R "
                    "fixest (t(G-1)), to 1e-10",
                    "sp.did_2x2(",
                ),
                _Row(
                    "T2",
                    vs._ES_CRIT,
                    cfg(
                        vce=["hc1"],
                        weights=["none"],
                        covariates=["none"],
                        crit_dist=["t_resid"],
                    ),
                    EST_SE_CI,
                    "the same 2x2 with robust SEs: SE, p-value and confidence "
                    "limits vs Stata regress, robust and R fixest (t(N-K), "
                    "which is also the default), to 1e-10",
                    "sp.did_2x2(",
                ),
            ),
            invariant={
                "crit_dist": (
                    EST_SE,
                    "the critical value turns the SE into a p-value and an "
                    "interval; the ATT and its SE are computed before it is "
                    "chosen (asserted equal across parity= / crit_dist= in "
                    "tests/reference_parity/test_es_crit_default.py; "
                    "crit_dist='t' there is the same fit as parity='stata')",
                ),
            },
            note="vce='hc1' is robust=True (the default) without cluster=; "
            "'survey' is design=, whose weights live in the design, so "
            "weights reads 'none'. crit_dist 't_resid' under cluster= is the "
            "pre-1.41 default, which neither Stata nor fixest uses; its "
            "intervals have no reference row. engine='bayes' returns "
            "sp.bayes_did's result and reads as unchecked.",
        )
    )
