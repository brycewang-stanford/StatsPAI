"""Scaling sweep: how the wall-clock time of each estimator grows with n.

Runs about 120 public estimators on synthetic data at three sizes and
prints, for each, the time at every size and the empirical exponent
between the two largest sizes that ran (time ~ n^exp). An exponent well
above 1 on a method that should be linear, or seconds at the smallest
size, is how the per-observation Python loops and n x n allocations
removed in October 2026 were found (Cox at n^1.9, nearest-neighbour
matching at n^1.8, quantile regression at n^1.6, fracreg allocating
n^2 doubles).

    python benchmarks/bench_scaling.py                # everything
    python benchmarks/bench_scaling.py --only did     # names containing 'did'
    python benchmarks/bench_scaling.py --budget 30 --json out.json

A case stops growing once one size takes longer than ``--budget`` seconds
and is abandoned after four times that. Times include whatever the
default call does (bootstrap replications, cross-fitting, placebo
inference), so a slow row is a statement about the default call, not
only about the estimator's kernel. Not part of ``run_all.py`` or the
benchmark ratchet: it takes tens of minutes and is meant to be read.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
import warnings
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

import statspai as sp

warnings.filterwarnings("ignore")


def xs(n, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 5))
    ps = 1 / (1 + np.exp(-(0.5 * X[:, 0] - 0.3 * X[:, 1])))
    d = (r.uniform(size=n) < ps).astype(int)
    z = r.normal(size=n)
    u = r.normal(size=n)
    endo = 0.8 * z + 0.5 * u + r.normal(size=n)
    y = 1 + 2 * d + X @ np.array([1, 0.5, -0.5, 0.2, 0]) + u
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(1, 6)])
    df["d"] = d
    df["y"] = y
    df["z"] = z
    df["endo"] = endo
    df["yiv"] = 1 + 1.5 * endo + X[:, 0] + u
    df["ybin"] = (y > np.median(y)).astype(int)
    df["ycnt"] = r.poisson(np.exp(0.2 + 0.3 * X[:, 0] + 0.2 * d))
    df["ycens"] = np.maximum(y, 0)
    df["cl"] = r.integers(0, max(20, n // 50), size=n)
    df["grp"] = r.integers(0, 2, size=n)
    df["run"] = r.uniform(-1, 1, size=n)
    df["yrd"] = 0.5 * (df["run"] >= 0) + df["run"] + 0.3 * r.normal(size=n)
    df["m"] = 0.5 * d + 0.3 * X[:, 0] + r.normal(size=n)
    df["dur"] = r.exponential(np.exp(-0.3 * d - 0.2 * X[:, 0]))
    df["ev"] = (r.uniform(size=n) < 0.7).astype(int)
    df["sel"] = (0.5 * z + X[:, 0] + r.normal(size=n) > 0).astype(int)
    df["yord"] = pd.cut(y, [-np.inf, 0, 2, 4, np.inf], labels=False)
    return df


def panel(N, T=10, seed=0):
    r = np.random.default_rng(seed)
    i = np.repeat(np.arange(N), T)
    t = np.tile(np.arange(1, T + 1), N)
    g_unit = r.choice([0, 4, 6, 8], size=N, p=[0.4, 0.2, 0.2, 0.2])
    g = g_unit[i]
    a = r.normal(size=N)[i]
    lam = r.normal(size=T)[t - 1]
    x1 = r.normal(size=N * T)
    treated = ((g > 0) & (t >= g)).astype(int)
    y = a + lam + 0.5 * x1 + treated * (1 + 0.2 * (t - g)) + r.normal(size=N * T)
    df = pd.DataFrame({"id": i, "t": t, "g": g, "x1": x1, "y": y, "treated": treated})
    df["rel"] = np.where(g > 0, t - g, np.nan)
    df["tt"] = np.where(g > 0, g, np.nan)
    df["state"] = df["id"] % max(10, N // 20)
    return df


def synthpanel(J, T=40, seed=0):
    r = np.random.default_rng(seed)
    f = np.cumsum(r.normal(size=(T, 2)), axis=0)
    L = r.normal(size=(J, 2))
    Y = L @ f.T + r.normal(scale=0.3, size=(J, T))
    Y[0, 30:] += 2
    return pd.DataFrame(
        {
            "unit": np.repeat(np.arange(J), T),
            "time": np.tile(np.arange(T), J),
            "y": Y.ravel(),
        }
    )


X5 = ["x1", "x2", "x3", "x4", "x5"]
F = "y ~ d + x1 + x2 + x3 + x4 + x5"
C: Dict[str, Tuple[str, Callable[[pd.DataFrame], Any]]] = {}


def case(name: str, kind: str) -> Callable[[Callable], Callable]:
    def deco(f: Callable) -> Callable:
        C[name] = (kind, f)
        return f

    return deco


@case("regress_iid", "xs")
def _(d):
    return sp.regress(F, d)


@case("regress_hc1", "xs")
def _(d):
    return sp.regress(F, d, robust="hc1")


@case("regress_cluster", "xs")
def _(d):
    return sp.regress(F, d, cluster="cl")


@case("regress_factor", "xs")
def _(d):
    return sp.regress("y ~ d + x1 + C(cl)", d)


@case("logit", "xs")
def _(d):
    return sp.logit("ybin ~ d + x1 + x2 + x3", d)


@case("logit_cluster_me", "xs")
def _(d):
    return sp.logit(
        "ybin ~ d + x1 + x2 + x3", d, cluster="cl", marginal_effects="average"
    )


@case("probit", "xs")
def _(d):
    return sp.probit("ybin ~ d + x1 + x2 + x3", d)


@case("poisson_robust", "xs")
def _(d):
    return sp.poisson("ycnt ~ d + x1 + x2", d, robust="hc1")


@case("nbreg", "xs")
def _(d):
    return sp.nbreg("ycnt ~ d + x1 + x2", d)


@case("glm_binom", "xs")
def _(d):
    return sp.glm("ybin ~ d + x1 + x2", d, family="binomial")


@case("ologit", "xs")
def _(d):
    return sp.ologit("yord ~ d + x1 + x2", d)


@case("mlogit", "xs")
def _(d):
    return sp.mlogit("yord ~ d + x1 + x2", d)


@case("ivreg_robust", "xs")
def _(d):
    return sp.ivreg("yiv ~ x1 + (endo ~ z)", d, robust="hc1")


@case("iv_default", "xs")
def _(d):
    return sp.iv("yiv ~ x1 + (endo ~ z)", d)


@case("qreg", "xs")
def _(d):
    return sp.qreg(d, "y ~ d + x1 + x2")


@case("tobit", "xs")
def _(d):
    return sp.tobit(d, formula="ycens ~ d + x1 + x2")


@case("heckman", "xs")
def _(d):
    return sp.heckman(d, y="y", x=["d", "x1"], select="sel", z=["z", "x1"])


@case("rdrobust", "xs")
def _(d):
    return sp.rdrobust(d, "yrd", "run")


@case("rd_honest", "xs")
def _(d):
    return sp.rd_honest(d, "yrd", "run")


@case("rddensity", "xs")
def _(d):
    return sp.rddensity(d, "run")


@case("dml_plr", "xs")
def _(d):
    return sp.dml(d, y="y", treat="d", covariates=X5)


@case("dml_irm", "xs")
def _(d):
    return sp.dml(d, y="y", treat="d", covariates=X5, model="irm")


@case("match_nn", "xs")
def _(d):
    return sp.match(d, y="y", treat="d", covariates=X5)


@case("psmatch2", "xs")
def _(d):
    return sp.psmatch2(d, treat="d", covariates=X5, y="y")


@case("ipw_bootstrap", "xs")
def _(d):
    return sp.ipw(d, "y", "d", X5)


@case("ipw_sandwich", "xs")
def _(d):
    return sp.ipw(d, "y", "d", X5, se_method="sandwich")


@case("aipw", "xs")
def _(d):
    return sp.aipw(d, "y", "d", X5)


@case("ebalance", "xs")
def _(d):
    return sp.ebalance(d, "y", "d", X5)


@case("cbps", "xs")
def _(d):
    return sp.cbps(d, "y", "d", X5)


@case("overlap_weights", "xs")
def _(d):
    return sp.overlap_weights(d, "y", "d", X5)


@case("g_computation", "xs")
def _(d):
    return sp.g_computation(d, "y", "d", X5)


@case("tmle", "xs")
def _(d):
    return sp.tmle(d, "y", "d", X5)


@case("metalearner_dr", "xs")
def _(d):
    return sp.metalearner(d, "y", "d", X5)


@case("mediate", "xs")
def _(d):
    return sp.mediate(d, "y", "d", "m", ["x1"])


@case("causal_forest", "xs")
def _(d):
    return sp.causal_forest(Y=d["y"].values, T=d["d"].values, X=d[X5].values)


@case("cox", "xs")
def _(d):
    return sp.cox(data=d, duration="dur", event="ev", x=["d", "x1"])


@case("cox_cluster", "xs")
def _(d):
    return sp.cox(data=d, duration="dur", event="ev", x=["d", "x1"], cluster="cl")


@case("kaplan_meier", "xs")
def _(d):
    return sp.kaplan_meier(d, "dur", "ev", group="d", conf_type="log-log")


@case("oaxaca", "xs")
def _(d):
    return sp.oaxaca(d, "y", "grp", ["x1", "x2", "x3"])


@case("sensemakr", "xs")
def _(d):
    return sp.sensemakr(d, "y", "d", X5, benchmark=["x1"])


@case("wild_cluster_bootstrap", "xs")
def _(d):
    return sp.wild_cluster_bootstrap(d, "y", ["d", "x1"], "cl", test_var="d", seed=1)


@case("balance_table", "xs")
def _(d):
    return sp.balance_table(d, "d", X5)


@case("glmnet_cv", "xs")
def _(d):
    return sp.glmnet(d, "y", X5 + ["d", "z", "endo", "run", "m"])


@case("mixed", "xs")
def _(d):
    return sp.mixed(d, "y", ["d", "x1"], "cl")


@case("bootstrap_mean_200", "xs")
def _(d):
    return sp.bootstrap(d, lambda q: q["y"].mean(), n_boot=200, seed=1)


@case("winsor", "xs")
def _(d):
    return sp.winsor(d, ["y", "x1", "x2"])


@case("esttab_3", "xs")
def _(d):
    r = sp.regress(F, d.iloc[:2000])
    return sp.esttab(r, r, r)


@case("feols_twfe_cluster", "panel")
def _(d):
    return sp.feols("y ~ treated + x1 | id + t", d, vcov={"CRV1": "id"})


@case("hdfe_ols", "panel")
def _(d):
    return sp.hdfe_ols("y ~ treated + x1 | id + t", d, cluster="id")


@case("panel_fe", "panel")
def _(d):
    return sp.panel(d, "y ~ treated + x1", entity="id", time="t", method="fe")


@case("panel_re", "panel")
def _(d):
    return sp.panel(d, "y ~ treated + x1", entity="id", time="t", method="re")


@case("did_auto", "panel")
def _(d):
    return sp.did(d, "y", "g", "t", id="id")


@case("callaway_santanna", "panel")
def _(d):
    return sp.callaway_santanna(d, "y", "g", "t", "id")


@case("cs_covariates", "panel")
def _(d):
    return sp.callaway_santanna(d, "y", "g", "t", "id", x=["x1"])


@case("sun_abraham", "panel")
def _(d):
    return sp.sun_abraham(d, "y", "g", "t", "id")


@case("did_imputation", "panel")
def _(d):
    return sp.did_imputation(d, "y", "id", "t", "g")


@case("etwfe", "panel")
def _(d):
    return sp.etwfe(d, "y", "id", "t", "g")


@case("gardner_did", "panel")
def _(d):
    return sp.gardner_did(d, "y", "id", "t", "g")


@case("stacked_did", "panel")
def _(d):
    return sp.stacked_did(d, "y", "id", "t", "g", window=(-3, 3))


@case("wooldridge_did", "panel")
def _(d):
    return sp.wooldridge_did(d, "y", "id", "t", "g")


@case("bacon_decomposition", "panel")
def _(d):
    return sp.bacon_decomposition(d, "y", "treated", "t", "id")


@case("twfe_decomposition", "panel")
def _(d):
    return sp.twfe_decomposition(d, "y", "id", "t", "g")


@case("event_study", "panel")
def _(d):
    return sp.event_study(d, "y", "tt", "t", "id", window=(-3, 3))


@case("lp_did", "panel")
def _(d):
    return sp.lp_did(d, "y", "id", "t", "treated")


@case("did_multiplegt_dyn", "panel")
def _(d):
    return sp.did_multiplegt_dyn(
        d, "y", group="id", time="t", treatment="treated", seed=1
    )


@case("honest_did", "panel")
def _(d):
    r = sp.callaway_santanna(d, "y", "g", "t", "id")
    return sp.honest_did(r)


@case("xtabond", "panel")
def _(d):
    return sp.xtabond(d, "y", ["x1"], id="id", time="t")


@case("fect", "panel")
def _(d):
    return sp.fect(d, "y", "treated", "id", "t")


@case("mc_panel", "panel")
def _(d):
    return sp.mc_panel(d, "y", "id", "t", "treated")


@case("uniform_bands", "panel")
def _(d):
    r = sp.callaway_santanna(d, "y", "g", "t", "id")
    return sp.uniform_bands(r)


@case("synth_classic", "synth")
def _(d):
    return sp.synth(d, "y", "unit", "time", 0, 30)


@case("sdid", "synth")
def _(d):
    return sp.sdid(d, "y", "unit", "time", 0, 30)


@case("gsynth", "synth")
def _(d):
    return sp.gsynth(d, "y", "unit", "time", 0, 30, seed=1)


@case("synth_compare", "synth")
def _(d):
    return sp.synth_compare(d, "y", "unit", "time", 0, 30, seed=1)


def _geo(d):
    r = np.random.default_rng(1)
    d = d.copy()
    d["lat"] = r.uniform(30, 45, len(d))
    d["lon"] = r.uniform(-120, -80, len(d))
    return d


@case("conley", "xs")
def _(d):
    d = _geo(d)
    return sp.conley(sp.regress("y ~ d + x1", d), d, "lat", "lon", 50.0)


@case("regress_conley", "xs")
def _(d):
    d = _geo(d)
    return sp.regress(
        "y ~ d + x1", d, conley_lat="lat", conley_lon="lon", conley_cutoff=50.0
    )


@case("knn_weights+moran", "xs")
def _(d):
    r = np.random.default_rng(1)
    w = sp.knn_weights(r.uniform(size=(len(d), 2)), 5)
    return sp.moran(d["y"].values, w, permutations=99, seed=1)


@case("garch", "xs")
def _(d):
    return sp.garch(d["y"].values[:50000])


@case("var2", "xs")
def _(d):
    return sp.var(d[["y", "x1", "x2"]], lags=2)


@case("local_projections", "xs")
def _(d):
    return sp.local_projections(d.iloc[:50000], "y", "x1", horizons=8)


@case("unitroot_adf", "xs")
def _(d):
    return sp.unitroot(d["y"].cumsum().values[:50000])


@case("rifreg", "xs")
def _(d):
    return sp.rifreg("y ~ d + x1 + x2", d)


@case("machado_mata", "xs")
def _(d):
    return sp.machado_mata(d, "y", "grp", ["x1", "x2"])


@case("lee_bounds", "xs")
def _(d):
    return sp.lee_bounds(d, "y", "d", "sel")


@case("manski_bounds", "xs")
def _(d):
    return sp.manski_bounds(d, "ybin", "d")


@case("romano_wolf", "xs")
def _(d):
    return sp.romano_wolf(
        d, ["y", "yiv", "m"], "d", controls=["x1"], n_boot=200, seed=1
    )


@case("westfall_young", "xs")
def _(d):
    return sp.westfall_young(d, ["y", "yiv", "m"], "d", n_perms=500, seed=1)


@case("frontier", "xs")
def _(d):
    return sp.frontier(d, "y", ["x1", "x2"])


@case("survreg", "xs")
def _(d):
    return sp.survreg(data=d, duration="dur", event="ev", x=["d", "x1"])


@case("cuminc", "xs")
def _(d):
    d = d.copy()
    d["ev2"] = d["ev"] * (1 + (d["x1"] > 0))
    return sp.cuminc(d, "dur", "ev2", group="d")


@case("melogit", "xs")
def _(d):
    return sp.melogit(d, "ybin", ["d", "x1"], "cl")


@case("rdplot", "xs")
def _(d):
    import matplotlib

    matplotlib.use("Agg")
    r = sp.rdplot(d, "yrd", "run")
    import matplotlib.pyplot as plt

    plt.close("all")
    return r


@case("rdrandinf", "xs")
def _(d):
    return sp.rdrandinf(d, "yrd", "run", wl=-0.1, wr=0.1)


@case("dose_response", "xs")
def _(d):
    return sp.dose_response(d, "y", "m", ["x1", "x2"])


@case("bunching", "xs")
def _(d):
    return sp.bunching(d, "run", 0.0)


@case("policy_tree", "xs")
def _(d):
    return sp.policy_tree(d, "y", treat="d", covariates=["x1", "x2", "x3"], depth=2)


@case("conformal_cate", "xs")
def _(d):
    return sp.conformal_cate(d, "y", "d", X5)


@case("qte", "xs")
def _(d):
    return sp.qte(d, "y", "d", controls=["x1"])


@case("kdensity", "xs")
def _(d):
    return sp.kdensity(d, "y")


@case("lpoly", "xs")
def _(d):
    return sp.lpoly(d, "y", "x1")


@case("binscatter", "xs")
def _(d):
    r = sp.binscatter(d, "y", "x1", controls=["x2"])
    import matplotlib.pyplot as plt

    plt.close("all")
    return r


@case("liml", "xs")
def _(d):
    return sp.liml("yiv ~ x1 + (endo ~ z)", d)


@case("ivqreg", "xs")
def _(d):
    return sp.ivqreg(d, "yiv", "endo", "z", exog=["x1"])


@case("ivprobit", "xs")
def _(d):
    return sp.ivprobit(d, "ybin", ["x1"], "endo", "z")


@case("ppmlhdfe", "xs")
def _(d):
    return sp.ppmlhdfe("ycnt ~ d + x1", d, absorb="cl")


@case("biprobit", "xs")
def _(d):
    return sp.biprobit(d, "ybin", "sel", ["x1", "z"])


@case("zinb", "xs")
def _(d):
    return sp.zinb("ycnt ~ d + x1", d)


@case("hurdle", "xs")
def _(d):
    return sp.hurdle("ycnt ~ d + x1", d)


@case("truncreg", "xs")
def _(d):
    return sp.truncreg(d[d["y"] > 0], "y", ["d", "x1"], ll=0)


@case("fracreg", "xs")
def _(d):
    d = d.copy()
    d["fr"] = 1 / (1 + np.exp(-d["y"] / 3))
    return sp.fracreg(d, "fr", ["d", "x1"])


@case("betareg", "xs")
def _(d):
    d = d.copy()
    d["fr"] = 1 / (1 + np.exp(-d["y"] / 3))
    return sp.betareg(d, "fr", ["d", "x1"])


@case("robreg_mm", "xs")
def _(d):
    return sp.robreg("y ~ d + x1 + x2", d)


@case("gee_exch", "xs")
def _(d):
    return sp.gee("ybin ~ d + x1", d, id="cl", family="binomial", corstr="exchangeable")


@case("spec_curve", "xs")
def _(d):
    return sp.spec_curve(
        d, "y", "d", controls=[["x1"], ["x1", "x2"], ["x1", "x2", "x3"], X5]
    )


@case("oster_bounds", "xs")
def _(d):
    return sp.oster_bounds(d, "y", "d", X5)


@case("lasso_iv", "xs")
def _(d):
    return sp.lasso_iv(d, "yiv", ["endo"], ["x1"], ["z", "x2", "x3", "x4"])


@case("its", "xs")
def _(d):
    return sp.its(d.iloc[:20000], "y", intervention=1000)


@case("causal_impact", "xs")
def _(d):
    q = d.iloc[:2000].copy()
    q["tt"] = np.arange(len(q))
    return sp.causal_impact(q, "y", "tt", 1500, covariates=["x1"])


@case("stl", "xs")
def _(d):
    return sp.stl(d["y"].values[:50000], 12)


@case("tsfilter_hp", "xs")
def _(d):
    return sp.tsfilter(d["y"].values[:50000])


@case("ardl", "xs")
def _(d):
    return sp.ardl(d.iloc[:50000], "y", "x1", lags=2)


@case("did_multiplegt", "panel")
def _(d):
    return sp.did_multiplegt(d, "y", "id", "t", "treated", seed=1)


@case("augsynth", "synth")
def _(d):
    return sp.augsynth(d, "y", "unit", "time", 0, 30)


@case("conformal_synth", "synth")
def _(d):
    return sp.conformal_synth(d, "y", "unit", "time", 0, 30)


SIZES = {
    "xs": [2000, 20000, 200000],
    "panel": [200, 2000, 20000],  # units; 10 periods each
    "synth": [20, 50, 100],  # donor pool; 40 periods
}
MAKE = {"xs": xs, "panel": panel, "synth": synthpanel}


class _Timeout(Exception):
    pass


def _alarm(*_: Any) -> None:
    raise _Timeout()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="substring of the case name")
    ap.add_argument("--budget", type=float, default=30.0, help="seconds per size")
    ap.add_argument("--json", default=None, help="write the timings to this file")
    args = ap.parse_args()

    signal.signal(signal.SIGALRM, _alarm)
    data: Dict[Tuple[str, int], pd.DataFrame] = {}
    results: Dict[str, List[Dict[str, Any]]] = {}
    print(f"StatsPAI {sp.__version__}")
    for name, (kind, fn) in C.items():
        if args.only and args.only not in name:
            continue
        row: List[Dict[str, Any]] = []
        for i, n in enumerate(SIZES[kind]):
            if (kind, n) not in data:
                data[(kind, n)] = MAKE[kind](n)
            if i == 0:
                # untimed first call: lazy imports and JIT compilation
                signal.alarm(int(args.budget * 4))
                try:
                    fn(data[(kind, n)])
                except Exception:  # reported by the timed call below
                    pass
                signal.alarm(0)
            err = None
            signal.alarm(int(args.budget * 4))
            start = time.perf_counter()
            try:
                fn(data[(kind, n)])
            except _Timeout:
                err = "timeout"
            except Exception as exc:  # a case that cannot run is reported
                err = f"{type(exc).__name__}: {str(exc)[:80]}"
            signal.alarm(0)
            elapsed = time.perf_counter() - start
            row.append({"n": n, "seconds": round(elapsed, 4), "error": err})
            if err or elapsed > args.budget:
                break
        results[name] = row
        ok = [r for r in row if not r["error"]]
        expo = ""
        if len(ok) >= 2 and ok[-2]["seconds"] > 0.002:
            ratio = np.log(ok[-1]["seconds"] / ok[-2]["seconds"])
            expo = f"exp={ratio / np.log(ok[-1]['n'] / ok[-2]['n']):.2f}"
        cells = "  ".join(
            f"{r['n']}:{r['seconds']:.3f}" + (f"[{r['error']}]" if r["error"] else "")
            for r in row
        )
        print(f"{name:26s} {cells} {expo}", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"statspai": sp.__version__, "cases": results}, fh, indent=1)


if __name__ == "__main__":
    main()
