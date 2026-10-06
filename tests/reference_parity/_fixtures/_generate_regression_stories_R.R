# Reference values for tests/reference_parity/test_regression_stories_r_parity.py
#
# Writes regression_stories_R.json next to this file. The synthetic data are
# generated here and stored in the JSON, so the Python side reads exactly the
# numbers R used.
#
# Requires: loo, rstanarm, arm, retrodesign, jsonlite (rstanarm and arm are
# GPL; they are run, never read).
#   Rscript tests/reference_parity/_fixtures/_generate_regression_stories_R.R
suppressMessages({
  library(loo); library(rstanarm); library(arm); library(retrodesign); library(jsonlite)
})
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
out = list(versions = list(R = R.version.string, loo = as.character(packageVersion("loo")),
  rstanarm = as.character(packageVersion("rstanarm")), arm = as.character(packageVersion("arm")),
  retrodesign = as.character(packageVersion("retrodesign"))))

## ---- PSIS-LOO, WAIC, comparison -------------------------------------
## The log-likelihood matrices follow a closed-form recipe (no random
## numbers), repeated in the test, so they are not stored.
loglik = function(n, S, outliers) {
  i = 1:n; x = qnorm((i - 0.5) / n); y = 1 + 2 * x + 0.8 * sin(3 * i)
  if (outliers) { y[3] = y[3] + 8; y[n - 2] = y[n - 2] - 6 }
  s = 1:S
  u = function(m) ((s * m) %% S + 0.5) / S
  a = 1 + 0.2 * qnorm(u(7919)); b = 2 + 0.2 * qnorm(u(104729)); sig = 0.8 * exp(0.15 * qnorm(u(1299709)))
  ll = sapply(i, function(j) dnorm(y[j], a + b * x[j], sig, log = TRUE))
  ll2 = sapply(i, function(j) dnorm(y[j], a, 2.2 * sig, log = TRUE))
  list(ll = ll, ll2 = ll2)
}
one = function(ll, r_eff) {
  l = suppressWarnings(loo(ll, r_eff = r_eff, save_psis = TRUE))
  k = l$diagnostics$pareto_k; ne = l$diagnostics$n_eff
  list(estimates = unname(l$estimates), elpd = unname(l$pointwise[, "elpd_loo"]),
    mcse = unname(l$pointwise[, "mcse_elpd_loo"]), p = unname(l$pointwise[, "p_loo"]),
    pareto_k = k, n_eff = ifelse(is.na(ne), -1, ne),
    log_weights_first = weights(l$psis_object, log = TRUE)[, 1],
    mcse_total = { v = mcse_loo(l); if (is.na(v)) -1 else v })
}
out$loo = list()
for (cfg in list(list("clean", 20, 400, FALSE), list("outlier", 20, 400, TRUE), list("long", 6, 2300, FALSE))) {
  m = loglik(cfg[[2]], cfg[[3]], cfg[[4]]); n = cfg[[2]]
  r_eff = if (cfg[[1]] == "long") seq(0.4, 0.95, length.out = n) else rep(1, n)
  w = suppressWarnings(waic(m$ll))
  la = suppressWarnings(loo(m$ll, r_eff = r_eff)); lb = suppressWarnings(loo(m$ll2, r_eff = r_eff))
  cmp = loo_compare(la, lb)
  out$loo[[cfg[[1]]]] = list(n = n, S = cfg[[3]], outliers = cfg[[4]], r_eff = r_eff,
    a = one(m$ll, r_eff), b = one(m$ll2, r_eff),
    waic_estimates = unname(w$estimates), waic_elpd = unname(w$pointwise[, "elpd_waic"]),
    waic_p = unname(w$pointwise[, "p_waic"]),
    compare = unname(unclass(cmp)[, 1:2]), compare_rows = rownames(cmp),
    ll_checksum = c(sum(m$ll), sum(m$ll2), m$ll[7, 3]))
}

## ---- data for the regression helpers --------------------------------
set.seed(20261007)
N = 300
d = data.frame(x = round(runif(N, -4, 4), 6), z = round(rnorm(N), 6))
d$g = rep(c("a", "b", "c"), N / 3)
d$y = rbinom(N, 1, plogis(0.3 + 0.8 * d$x - 0.5 * d$z)); d$y[1:6] = 1 - d$y[1:6]
d$cnt = rpois(N, exp(0.2 + 0.3 * d$x)) * rbinom(N, 1, 0.7) * 2
d$yc = round(1 + 0.5 * d$x - 0.7 * d$z + 0.6 * (d$g == "b") + rnorm(N), 6)
d$da = as.numeric(d$g == "a"); d$db = as.numeric(d$g == "b"); d$dc = as.numeric(d$g == "c")
d$trials = 5 + (1:N) %% 7; d$succ = rbinom(N, d$trials, plogis(-0.2 + 0.4 * d$x))
out$data = d
tight = glm.control(epsilon = 1e-11, maxit = 500)
co = function(f) { s = coef(summary(f)); list(names = rownames(s), est = unname(s[, 1]), se = unname(s[, 2])) }

## robit link (Student-t), as a user-defined glm link
robit = function(df, s = 1) structure(list(linkfun = function(mu) s * qt(mu, df),
  linkinv = function(eta) pt(eta / s, df), mu.eta = function(eta) dt(eta / s, df) / s,
  valideta = function(eta) TRUE, name = "robit"), class = "link-glm")
out$robit4 = co(glm(y ~ x + z, binomial(link = robit(4)), d, control = tight))
out$robit7 = co(glm(y ~ x + z, binomial(link = robit(7)), d, control = tight))
out$robit4_unit = co(glm(y ~ x + z, binomial(link = robit(4, sqrt(2 / 4))), d, control = tight))
## quasi families
fq = glm(cnt ~ x + z, quasipoisson, d, control = tight); out$quasipoisson = c(co(fq), dispersion = summary(fq)$dispersion)
fb = glm(y ~ x + z, quasibinomial, d, control = tight); out$quasibinomial = c(co(fb), dispersion = summary(fb)$dispersion)
## grouped binomial
fg = glm(cbind(succ, trials - succ) ~ x + z, binomial, d, control = tight)
out$cbind = c(co(fg), deviance = deviance(fg))
## logical outcome
out$logical = co(glm((yc > 1) ~ x + z, binomial, d, control = tight))
## aliased regressors: R reports NA for the later member of a dependent set
al = function(f) { cf = coef(f); s = coef(summary(f)); list(names = names(cf), aliased = unname(is.na(cf)), est = unname(s[, 1]), se = unname(s[, 2])) }
out$alias_logit = al(glm(y ~ x + da + db + dc, binomial, d, control = tight))
out$alias_poisson = al(glm(cnt ~ x + z + I(x + 2 * z), poisson, d, control = tight))
out$alias_lm_dot = al(lm(yc ~ . - g - y - cnt - trials - succ, d))

## arm: binned residuals, rescaling, standardized refit
f = glm(y ~ x + z, binomial, d, control = tight); p = fitted(f); r = d$y - p
out$binned = list(default = unname(binned.resids(p, r)$binned), n12 = unname(binned.resids(p, r, nclass = 12)$binned),
  by_x = unname(binned.resids(d$x, r, nclass = 10)$binned))
tx = c(d$x[1:50], d$x[1:10]); tr = c(r[1:50], r[11:20])
out$binned$ties_x = tx; out$binned$ties_r = tr; out$binned$ties = unname(binned.resids(tx, tr, nclass = 7)$binned)
out$rescale = list(x = rescale(d$x), y_center = rescale(d$y), y_full = rescale(d$y, "full"),
  y_01 = rescale(d$y + 3, "0/1"), y_half = rescale(d$y, "-0.5,0.5"))
df = d; df$g = factor(df$g)
s = standardize(lm(yc ~ x + z + y + g + x:y, df)); out$standardize = list(names = names(coef(s)), est = unname(coef(s)))

## design analysis
pts = list(c(0.1, 3.28), c(2, 8.1), c(0.5, 1), c(2.8, 1), c(0.01, 1))
out$retro = list(effect = sapply(pts, `[`, 1), se = sapply(pts, `[`, 2),
  closed = t(sapply(pts, function(v) unlist(retro_design_closed_form(v[1], v[2])))))
## with 20 degrees of freedom the package's power and sign error are noncentral t
set.seed(1); rd = retrodesign(0.5, 1, df = 20, n.sims = 2e6)
out$retro$df20_power = rd$power; out$retro$df20_type_s = rd$type_s

## rstanarm: default prior scales, Bayesian R-squared
dd = d[1:120, ]
f1 = stan_glm(yc ~ x + z + da + x:z, data = dd, refresh = 0, chains = 2, iter = 400, seed = 1)
ps = prior_summary(f1)
out$prior_gaussian = list(n = 120, coef_scale = ps$prior$adjusted_scale, intercept_location = ps$prior_intercept$location,
  intercept_scale = ps$prior_intercept$adjusted_scale, sigma_rate = 1 / ps$prior_aux$adjusted_scale, names = names(coef(f1))[-1])
f2 = stan_glm(y ~ x + z + da, family = binomial, data = dd, refresh = 0, chains = 2, iter = 400, seed = 1)
ps2 = prior_summary(f2)
out$prior_binomial = list(coef_scale = ps2$prior$adjusted_scale, intercept_scale = ps2$prior_intercept$scale)
keep = seq(1, 400, by = 8)
mu1 = posterior_epred(f1)[keep, ]; r1 = bayes_R2(f1)[keep]
out$r2_gaussian = list(mu = mu1, sigma = as.matrix(f1)[keep, "sigma"], r2 = r1)
mu2 = posterior_epred(f2)[keep, ]; r2 = bayes_R2(f2)[keep]
out$r2_binomial = list(mu = mu2, r2 = r2)

writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, dataframe = "columns"), file.path(here, "regression_stories_R.json"))
cat("wrote", file.path(here, "regression_stories_R.json"), "\n")
