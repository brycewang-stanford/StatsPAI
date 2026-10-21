"""Runnable card examples for the most used entry points.

The 2026-10-19 audit found that the ``example`` on the agent cards of the
70 entry points the card audit covers (``scripts/agent_card_audit.py``
``AUDITED``) mostly assumed a ``df`` with columns that no dataset has,
and several used legacy argument spellings (``sp.psm(d=, X=)``). An agent
that copies such an example gets a ``KeyError`` before it learns anything.

Each example below runs as-is after ``import statspai as sp``: the data
come from a bundled dataset (``sp.cps_wage``, ``sp.datasets.*``, real
extracts where one fits the design) or a ``sp.dgp_*`` simulator, and the
keywords are the function's own parameter names, the names its schema
lists (aliases such as ``id=`` work too, but an agent reads the schema).
``tests/test_card_examples.py`` executes every one of them, so an example
that stops running fails the suite instead of misleading an agent.

The registry applies these last (after docstring harvesting), overriding
whatever example the card had.
"""

from __future__ import annotations

from typing import Dict

#: ``{function name: example code}``; each binds the result to ``r``.
RUNNABLE_EXAMPLES: Dict[str, str] = {
    "regress": (
        "df = sp.cps_wage()\n"
        'r = sp.regress("log_wage ~ education + experience", data=df, vce="hc1")\n'
    ),
    "ivreg": (
        "df = sp.datasets.card_1995()\n"
        "r = sp.ivreg(\n"
        '    "lwage ~ (educ ~ nearc4 + nearc2) + exper + black", data=df,\n'
        '    vce="hc1",\n'
        ")\n"
    ),
    "iv": (
        "df = sp.datasets.card_1995()\n"
        "r = sp.iv(\n"
        '    "lwage ~ (educ ~ nearc4 + nearc2) + exper + black", data=df,\n'
        '    method="liml",\n'
        ")\n"
    ),
    "logit": (
        "df = sp.cps_wage()\n"
        'r = sp.logit("union ~ education + experience", data=df)\n'
    ),
    "probit": (
        "df = sp.cps_wage()\n"
        'r = sp.probit("union ~ education + experience", data=df)\n'
    ),
    "poisson": (
        "df = sp.datasets.castle_doctrine()\n"
        'df["n_homicide"] = (df.homicide * df.population / 1e5).round().astype(int)\n'
        "r = sp.poisson(\n"
        '    "n_homicide ~ post + unemployrt", data=df, cluster="state",\n'
        ")\n"
    ),
    "qreg": (
        "df = sp.cps_wage()\n"
        'r = sp.qreg(df, "log_wage ~ education + experience", quantile=0.9)\n'
    ),
    "feols": (
        "df = sp.datasets.castle_doctrine()\n"
        "r = sp.feols(\n"
        '    "l_homicide ~ post | sid + year", data=df, vcov={"CRV1": "sid"},\n'
        ")\n"
    ),
    "panel": (
        "df = sp.dgp_panel(seed=0)\n"
        'r = sp.panel(df, "y ~ x", id="unit", time="time", method="fe")\n'
    ),
    "did": (
        "df = sp.dgp_did(seed=0)\n"
        'r = sp.did(df, y="y", treat="treated", time="time", id="unit")\n'
    ),
    "callaway_santanna": (
        "df = sp.datasets.castle_doctrine()\n"
        "r = sp.callaway_santanna(\n"
        '    df, y="l_homicide", g="effyear", t="year", i="sid",\n'
        ")\n"
    ),
    "sun_abraham": (
        "df = sp.datasets.castle_doctrine()\n"
        'r = sp.sun_abraham(df, y="l_homicide", g="effyear", t="year", i="sid")\n'
    ),
    "did_imputation": (
        "df = sp.datasets.castle_doctrine()\n"
        "r = sp.did_imputation(\n"
        '    df, y="l_homicide", group="sid", time="year",\n'
        '    first_treat="effyear",\n'
        ")\n"
    ),
    "gardner_did": (
        "df = sp.datasets.castle_doctrine()\n"
        "r = sp.gardner_did(\n"
        '    df, y="l_homicide", group="sid", time="year",\n'
        '    first_treat="effyear", event_study=True,\n'
        ")\n"
    ),
    "etwfe": (
        "df = sp.datasets.castle_doctrine()\n"
        "r = sp.etwfe(\n"
        '    df, y="l_homicide", group="sid", time="year",\n'
        '    first_treat="effyear",\n'
        ")\n"
    ),
    "stacked_did": (
        "df = sp.datasets.castle_doctrine()\n"
        "r = sp.stacked_did(\n"
        '    df, y="l_homicide", group="sid", time="year",\n'
        '    first_treat="effyear", window=(-3, 3),\n'
        ")\n"
    ),
    "event_study": (
        "df = sp.datasets.castle_doctrine()\n"
        "r = sp.event_study(\n"
        '    df, y="l_homicide", treat_time="effyear", time="year", unit="sid",\n'
        "    window=(-3, 3),\n"
        ")\n"
    ),
    "bacon_decomposition": (
        "df = sp.datasets.castle_doctrine()\n"
        'df["d"] = (df.year >= df.effyear).astype(int)\n'
        "r = sp.bacon_decomposition(\n"
        '    df, y="l_homicide", treat="d", time="year", id="sid",\n'
        ")\n"
    ),
    "rdrobust": (
        "df = sp.datasets.lee_2008_senate()\n"
        'r = sp.rdrobust(df, y="y", x="x", c=0)\n'
    ),
    "rddensity": (
        "df = sp.datasets.lee_2008_senate()\n" 'r = sp.rddensity(df, x="x", c=0)\n'
    ),
    "synth": (
        "df = sp.datasets.california_prop99()\n"
        "r = sp.synth(\n"
        '    df, outcome="cigsale", unit="state", time="year",\n'
        '    treated_unit="California", treatment_time=1989,\n'
        ")\n"
    ),
    "sdid": (
        "df = sp.datasets.california_prop99()\n"
        "r = sp.sdid(\n"
        '    df, y="cigsale", unit="state", time="year", treat="treated",\n'
        "    n_reps=50, seed=0,\n"
        ")\n"
    ),
    "dml": (
        "df = sp.dgp_observational(seed=0)\n"
        "r = sp.dml(\n"
        '    df, y="y", treat="treatment", covariates=["x1", "x2"], model="plr",\n'
        ")\n"
    ),
    "metalearner": (
        "df = sp.dgp_rct(heterogeneous=True, seed=0)\n"
        "r = sp.metalearner(\n"
        '    df, y="y", treat="treatment", covariates=["x1", "x2", "x3"],\n'
        '    learner="x",\n'
        ")\n"
    ),
    "causal_forest": (
        "df = sp.dgp_rct(n=400, heterogeneous=True, seed=0)\n"
        "r = sp.causal_forest(\n"
        '    "y ~ treatment | x1 + x2 + x3", data=df, n_trees=100, seed=0,\n'
        ")\n"
    ),
    "psm": (
        "df = sp.datasets.nsw_lalonde()\n"
        "r = sp.psm(\n"
        '    df, y="re78", treat="treat",\n'
        '    covariates=["age", "educ", "re74", "re75"],\n'
        ")\n"
    ),
    "match": (
        "df = sp.datasets.nsw_lalonde()\n"
        "r = sp.match(\n"
        '    df, y="re78", treat="treat",\n'
        '    covariates=["age", "educ", "re74", "re75"],\n'
        ")\n"
    ),
    "ipw": (
        "df = sp.dgp_observational(seed=0)\n"
        "r = sp.ipw(\n"
        '    df, y="y", treat="treatment", covariates=["x1", "x2"],\n'
        '    estimand="ATT", seed=0,\n'
        ")\n"
    ),
    "aipw": (
        "df = sp.dgp_observational(seed=0)\n"
        "r = sp.aipw(\n"
        '    df, y="y", treat="treatment", covariates=["x1", "x2"], seed=0,\n'
        ")\n"
    ),
    "tmle": (
        "df = sp.dgp_observational(seed=0)\n"
        "r = sp.tmle(\n"
        '    df, y="y", treat="treatment", covariates=["x1", "x2"],\n'
        "    random_state=0,\n"
        ")\n"
    ),
    "nbreg": (
        "df = sp.datasets.castle_doctrine()\n"
        'df["n_homicide"] = (df.homicide * df.population / 1e5).round().astype(int)\n'
        'r = sp.nbreg("n_homicide ~ post + unemployrt", data=df, cluster="state")\n'
    ),
    "glm": (
        "df = sp.cps_wage()\n"
        'r = sp.glm("union ~ education + experience", data=df, family="binomial")\n'
    ),
    "ppmlhdfe": (
        "df = sp.datasets.castle_doctrine()\n"
        "r = sp.ppmlhdfe(\n"
        '    "homicide ~ post + unemployrt | state + year", data=df,\n'
        '    cluster="state",\n'
        ")\n"
    ),
    "hdfe_ols": (
        "df = sp.datasets.castle_doctrine()\n"
        "r = sp.hdfe_ols(\n"
        '    "l_homicide ~ post | sid + year", data=df, cluster="sid",\n'
        ")\n"
    ),
    "ologit": (
        "df = sp.cps_wage()\n"
        'df["edu_level"] = (df.education >= 12).astype(int) + (df.education >= 16)\n'
        'r = sp.ologit("edu_level ~ experience + female", data=df)\n'
    ),
    "mlogit": (
        "df = sp.cps_wage()\n"
        'df["edu_level"] = (df.education >= 12).astype(int) + (df.education >= 16)\n'
        'r = sp.mlogit("edu_level ~ experience + female", data=df, base=0)\n'
    ),
    "tobit": (
        "df = sp.datasets.nsw_dw()\n"
        'r = sp.tobit(df, y="re78", x=["treat", "age", "education"], ll=0)\n'
    ),
    "cloglog": (
        "df = sp.cps_wage()\n"
        'r = sp.cloglog("union ~ education + experience", data=df)\n'
    ),
    "fepois": (
        "df = sp.datasets.castle_doctrine()\n"
        "r = sp.fepois(\n"
        '    "homicide ~ post | state + year", data=df, vcov={"CRV1": "state"},\n'
        ")\n"
    ),
    "feglm": (
        "df = sp.cps_wage()\n"
        "r = sp.feglm(\n"
        '    "union ~ education + experience | female", data=df, family="logit",\n'
        ")\n"
    ),
    "heckman": (
        "df = sp.datasets.nsw_dw()\n"
        'df["employed"] = (df.re78 > 0).astype(int)\n'
        "r = sp.heckman(\n"
        '    df, y="re78", x=["education", "age"], select="employed",\n'
        '    z=["married", "nodegree"],\n'
        ")\n"
    ),
    "mediate": (
        "df = sp.cps_wage()\n"
        "r = sp.mediate(\n"
        '    df, y="log_wage", treat="union", mediator="tenure",\n'
        '    covariates=["education"], n_boot=100, seed=0,\n'
        ")\n"
    ),
    "oaxaca": (
        "df = sp.cps_wage()\n"
        "r = sp.oaxaca(\n"
        '    df, y="log_wage", group="female",\n'
        '    x=["education", "experience", "tenure"],\n'
        ")\n"
    ),
    "ebalance": (
        "df = sp.datasets.nsw_lalonde()\n"
        "r = sp.ebalance(\n"
        '    df, y="re78", treat="treat",\n'
        '    covariates=["age", "educ", "re74", "re75"],\n'
        ")\n"
    ),
    "cbps": (
        "df = sp.datasets.nsw_lalonde()\n"
        "r = sp.cbps(\n"
        '    df, y="re78", treat="treat",\n'
        '    covariates=["age", "educ", "re74", "re75"], estimand="ATT",\n'
        "    n_bootstrap=10, seed=0,\n"
        ")\n"
    ),
    "lee_bounds": (
        "df = sp.datasets.nsw_dw()\n"
        'df["employed"] = (df.re78 > 0).astype(int)\n'
        "r = sp.lee_bounds(\n"
        '    df, y="re78", treat="treat", selection="employed", n_bootstrap=50,\n'
        ")\n"
    ),
    "rdrandinf": (
        "df = sp.datasets.lee_2008_senate()\n"
        "r = sp.rdrandinf(\n"
        '    df, y="y", x="x", c=0, wl=-2.5, wr=2.5, n_perms=200, seed=0,\n'
        ")\n"
    ),
    "rd_honest": (
        "df = sp.dgp_rd(seed=0)\n" 'r = sp.rd_honest(df, y="y", x="x", c=0, M=1.0)\n'
    ),
    "wooldridge_did": (
        "df = sp.datasets.castle_doctrine()\n"
        "r = sp.wooldridge_did(\n"
        '    df, y="l_homicide", group="sid", time="year",\n'
        '    first_treat="effyear",\n'
        ")\n"
    ),
    "lp_did": (
        "df = sp.dgp_did(staggered=True, seed=0)\n"
        "r = sp.lp_did(\n"
        '    df, y="y", unit="unit", time="time", treatment="treated",\n'
        "    horizons=(-3, 3),\n"
        ")\n"
    ),
    "did_multiplegt_dyn": (
        "df = sp.dgp_did(staggered=True, seed=0)\n"
        "r = sp.did_multiplegt_dyn(\n"
        '    df, y="y", group="unit", time="time", treatment="treated",\n'
        "    placebo=2, dynamic=2,\n"
        ")\n"
    ),
    "fect": (
        "df = sp.dgp_did(staggered=True, seed=0)\n"
        "r = sp.fect(\n"
        '    df, y="y", treat="treated", unit="unit", time="time", method="fe",\n'
        ")\n"
    ),
    "interflex": (
        "df = sp.dgp_rct(seed=0)\n"
        'r = sp.interflex(df, y="y", d="treatment", x="x1", estimator="binning")\n'
    ),
    "xtabond": (
        "df = sp.dgp_panel(seed=0)\n"
        'r = sp.xtabond(df, y="y", x=["x"], id="unit", time="time")\n'
    ),
    "sensemakr": (
        "df = sp.cps_wage()\n"
        "r = sp.sensemakr(\n"
        '    df, y="log_wage", treat="union",\n'
        '    controls=["education", "experience"], benchmark=["education"],\n'
        ")\n"
    ),
    "oster_bounds": (
        "df = sp.cps_wage()\n"
        "r = sp.oster_bounds(\n"
        '    df, y="log_wage", treat="union",\n'
        '    controls=["education", "experience"], delta=1.0,\n'
        ")\n"
    ),
    "rdbwselect": (
        "df = sp.datasets.lee_2008_senate()\n"
        'r = sp.rdbwselect(df, y="y", x="x", c=0)\n'
    ),
    "rd_discrete": (
        "df = sp.dgp_rd(seed=0)\n"
        'df["xd"] = df.x.round(1)\n'
        'r = sp.rd_discrete(df, y="y", x="xd", c=0)\n'
    ),
    "mixed": (
        "df = sp.dgp_panel(seed=0)\n"
        'r = sp.mixed(df, y="y", x_fixed=["x"], group="unit")\n'
    ),
    "liml": (
        "df = sp.datasets.card_1995()\n"
        "r = sp.liml(\n"
        '    data=df, y="lwage", x_endog=["educ"], x_exog=["exper", "black"],\n'
        '    z=["nearc4", "nearc2"],\n'
        ")\n"
    ),
    "aggte": (
        "df = sp.datasets.castle_doctrine()\n"
        "cs = sp.callaway_santanna(\n"
        '    df, y="l_homicide", g="effyear", t="year", i="sid",\n'
        ")\n"
        'r = sp.aggte(cs, type="dynamic")\n'
    ),
    "honest_did": (
        "df = sp.datasets.castle_doctrine()\n"
        "cs = sp.callaway_santanna(\n"
        '    df, y="l_homicide", g="effyear", t="year", i="sid",\n'
        ")\n"
        'r = sp.honest_did(sp.aggte(cs, type="dynamic"), e=0)\n'
    ),
    "drdid": (
        'df = sp.datasets.castle_doctrine().query("year in (2005, 2006)")\n'
        "df = df[df.effyear.isna() | (df.effyear == 2006)].copy()\n"
        'df["post"] = (df.year == 2006).astype(int)\n'
        'df["d"] = df.effyear.notna().astype(int)\n'
        'r = sp.drdid(df, y="l_homicide", treat="d", time="post", id="sid")\n'
    ),
    "did_multiplegt": (
        "df = sp.dgp_did(staggered=True, seed=0)\n"
        "r = sp.did_multiplegt(\n"
        '    df, y="y", group="unit", time="time", treatment="treated",\n'
        "    n_boot=20, seed=0,\n"
        ")\n"
    ),
    "gsynth": (
        "df = sp.dgp_synth(seed=0)\n"
        "r = sp.gsynth(\n"
        '    df, "y", "unit", "time", treat="treated", n_factors=1,\n'
        '    inference="none",\n'
        ")\n"
    ),
    "psmatch2": (
        "df = sp.datasets.nsw_lalonde()\n"
        "r = sp.psmatch2(\n"
        '    df, treat="treat", y="re78",\n'
        '    covariates=["age", "educ", "re74", "re75"],\n'
        ")\n"
    ),
    "xtnbreg": (
        "df = sp.datasets.castle_doctrine()\n"
        'df["n_homicide"] = (df.homicide * df.population / 1e5).round().astype(int)\n'
        "r = sp.xtnbreg(\n"
        '    "n_homicide ~ post + unemployrt", data=df, entity="state",\n'
        '    model="fe",\n'
        ")\n"
    ),
    "spec_curve": (
        "df = sp.cps_wage()\n"
        "r = sp.spec_curve(\n"
        '    df, y="log_wage", x="union",\n'
        '    controls=[[], ["education"], ["education", "experience"]],\n'
        ")\n"
    ),
    "iv_diag": (
        "df = sp.datasets.card_1995()\n"
        "r = sp.iv.iv_diag(\n"
        '    df, y="lwage", endog="educ", instruments=["nearc4", "nearc2"],\n'
        '    exog=["exper", "black"], n_boot=100, random_state=0,\n'
        ")\n"
    ),
    "g_computation": (
        "df = sp.dgp_observational(seed=0)\n"
        "r = sp.g_computation(\n"
        '    df, y="y", treat="treatment", covariates=["x1", "x2"], n_boot=50,\n'
        "    seed=0,\n"
        ")\n"
    ),
}

# Entry points added on 2026-10-19/20 (kept in their own module for size).
from ._card_examples_new import NEW_EXAMPLES as _NEW_EXAMPLES  # noqa: E402

RUNNABLE_EXAMPLES.update(_NEW_EXAMPLES)

# Entry points that gained parameters in that batch.
from ._card_examples_changed import CHANGED_EXAMPLES as _CHANGED  # noqa: E402

RUNNABLE_EXAMPLES.update(_CHANGED)

__all__ = ["RUNNABLE_EXAMPLES"]
