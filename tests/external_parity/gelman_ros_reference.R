# ---------------------------------------------------------------------------
# Answer key for tests/external_parity/test_gelman_ros_examples.py
#
# Gelman, Hill and Vehtari, "Regression and Other Stories" (2020), the
# examples that "Active Statistics" (Gelman and Vehtari 2024) builds on:
# https://github.com/avehtari/ROS-Examples. This script reruns a selection of
# them and writes what R reports. Neither the programs nor the data are
# redistributed.
#
# Requires: R with rstanarm, loo, survey, MASS, AER, jsonlite.
# Run:      STATSPAI_ROS_DIR=/path/to/ROS-Examples \
#               Rscript tests/external_parity/gelman_ros_reference.R
#
# It writes data/gelman_ros_R.json next to itself. The Bayesian fits are long
# chains (4 x 5,000 after warm-up) so that their Monte Carlo error is small
# next to the tolerance of the comparison.
# ---------------------------------------------------------------------------
suppressMessages({ library(rstanarm); library(loo); library(survey); library(MASS); library(AER); library(jsonlite) })
root = Sys.getenv("STATSPAI_ROS_DIR")
stopifnot(nzchar(root), dir.exists(root))
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
options(mc.cores = 4)
out = list(versions = list(R = R.version.string, rstanarm = as.character(packageVersion("rstanarm")),
  loo = as.character(packageVersion("loo")), survey = as.character(packageVersion("survey"))))
tight = glm.control(epsilon = 1e-12, maxit = 200)
co = function(f) { s = coef(summary(f)); list(names = rownames(s), est = unname(s[, 1]), se = unname(s[, 2])) }
bayes = function(f, y = NULL) {
  d = as.matrix(f)
  l = suppressWarnings(loo(f))
  r = list(names = colnames(d), median = unname(apply(d, 2, median)), mad_sd = unname(apply(d, 2, mad)),
    mean = unname(colMeans(d)), sd = unname(apply(d, 2, sd)),
    elpd_loo = l$estimates["elpd_loo", "Estimate"], se_elpd_loo = l$estimates["elpd_loo", "SE"],
    p_loo = l$estimates["p_loo", "Estimate"], max_k = max(l$diagnostics$pareto_k))
  if (family(f)$family %in% c("gaussian", "binomial")) { r$bayes_r2_median = median(bayes_R2(f)) }
  if (family(f)$family == "gaussian") { set.seed(1); r$loo_r2_mean = mean(suppressWarnings(loo_R2(f))) }
  r
}
long = function(...) stan_glm(..., refresh = 0, chains = 4, iter = 6000, warmup = 1000, seed = 20261007)

## chapter 7: elections and the economy
hibbs = read.table(file.path(root, "ElectionsEconomy/data/hibbs.dat"), header = TRUE)
out$hibbs = list(lm = co(lm(vote ~ growth, hibbs)), stan = bayes(long(vote ~ growth, data = hibbs)))

## chapters 10-11: children's test scores
kidiq = read.csv(file.path(root, "KidIQ/data/kidiq.csv"))
f3 = long(kid_score ~ mom_hs + mom_iq, data = kidiq)
f1 = long(kid_score ~ mom_hs, data = kidiq)
cmp = loo_compare(loo(f3), loo(f1))
out$kidiq = list(stan = bayes(f3), stan_hs = bayes(f1), elpd_diff = cmp[2, "elpd_diff"], se_diff = cmp[2, "se_diff"])

## chapters 13-14: arsenic in wells
wells = read.csv(file.path(root, "Arsenic/data/wells.csv"))
out$wells = list(glm = co(glm(switch ~ dist100 + arsenic + educ4, binomial, wells, control = tight)),
  stan = bayes(long(switch ~ dist100 + arsenic, family = binomial(link = "logit"), data = wells)))
b = coef(glm(switch ~ dist100 + arsenic + educ4, binomial, wells, control = tight))
apc = function(hi, lo, v) { a = wells; a[[v]] = hi; z = wells; z[[v]] = lo
  mean(plogis(cbind(1, a$dist100, a$arsenic, a$educ4) %*% b) - plogis(cbind(1, z$dist100, z$arsenic, z$educ4) %*% b)) }
out$wells$apc = list(dist100 = apc(1, 0, "dist100"), arsenic = apc(1.0, 0.5, "arsenic"), educ4 = apc(3, 0, "educ4"))

## chapter 15: roaches (counts with exposure), golf (grouped binomial), earnings
roaches = read.csv(file.path(root, "Roaches/data/roaches.csv")); roaches$roach100 = roaches$roach1 / 100
## (the offset is passed outside the long() wrapper: an expression in `...`
## cannot be evaluated in the model frame)
off = log(roaches$exposure2)
rnb = stan_glm(y ~ roach100 + treatment + senior, family = neg_binomial_2, offset = off, data = roaches,
  refresh = 0, chains = 4, iter = 6000, warmup = 1000, seed = 20261007)
rpo = stan_glm(y ~ roach100 + treatment + senior, family = poisson, offset = off, data = roaches,
  refresh = 0, chains = 4, iter = 6000, warmup = 1000, seed = 20261007)
out$roaches = list(
  negbin = bayes(rnb),
  poisson = bayes(rpo),
  quasipoisson = co(glm(y ~ roach100 + treatment + senior, quasipoisson, roaches, offset = log(exposure2), control = tight)),
  prop_zero = mean(roaches$y == 0))
golf = read.table(file.path(root, "Golf/data/golf.txt"), header = TRUE, skip = 2)
out$golf = co(glm(cbind(y, n - y) ~ x, binomial, golf, control = tight))
earnings = read.csv(file.path(root, "Earnings/data/earnings.csv"))
out$earnings = list(positive = co(glm((earn > 0) ~ height + male, binomial, earnings, control = tight)),
  log = co(lm(log(earn) ~ height + male + height:male, earnings, subset = earn > 0)))

## chapter 17: poststratification
poll = read.csv(file.path(root, "Poststrat/data/poll.csv"))
fp = long(vote ~ factor(pid), family = binomial(link = "logit"), data = poll)
cells = data.frame(pid = c("Republican", "Democrat", "Independent")); N = c(0.33, 0.36, 0.31)
ps = posterior_epred(fp, newdata = cells) %*% N / sum(N)
out$poststrat = list(mean = mean(ps), sd = sd(ps), raw = mean(poll$vote, na.rm = TRUE))

## chapter 20: child care, propensity scores with redundant indicators
cc2 = read.csv(file.path(root, "Childcare/data/cc2.csv"))
covs = c("bw", "preterm", "dayskidh", "sex", "first", "age", "black", "hispanic", "white", "b.marr", "lths", "hs", "ltcoll", "college", "work.dur", "prenatal", "momage")
## glm.fit detects aliased columns with tolerance min(1e-7, epsilon / 1000):
## at epsilon = 1e-12 the redundant indicators are no longer dropped and the
## fit diverges, so this model keeps a looser stopping rule
pf = glm(reformulate(covs, "treat"), binomial, cc2, control = glm.control(epsilon = 1e-10, maxit = 100))
cf = coef(pf); s = coef(summary(pf))
cc2$w = ifelse(cc2$treat == 1, 1, fitted(pf) / (1 - fitted(pf)))
sv = svyglm(reformulate(c("treat", covs), "ppvtr.36"), design = svydesign(ids = ~1, weights = ~w, data = cc2))
out$childcare = list(covs = covs, aliased = names(cf)[is.na(cf)], names = rownames(s), est = unname(s[, 1]), se = unname(s[, 2]),
  svy_treat = unname(coef(sv)["treat"]), svy_treat_se = unname(SE(sv)["treat"]))

## chapter 21: Sesame Street, instrumental variables
sesame = read.csv(file.path(root, "Sesame/data/sesame.csv"))
out$sesame = co(ivreg(postlet ~ watched | encouraged, data = sesame))

dir.create(file.path(here, "data"), showWarnings = FALSE)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), file.path(here, "data", "gelman_ros_R.json"))
cat("wrote", file.path(here, "data", "gelman_ros_R.json"), "\n")
