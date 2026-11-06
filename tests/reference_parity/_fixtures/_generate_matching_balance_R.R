# ---------------------------------------------------------------------------
# Reference numbers for tests/reference_parity/test_matching_balance_r_parity.py
#
# Propensity scores, trimming, weights, effective sample size, energy
# distance, balance tables and the L1 imbalance, each from the R package
# that defines it, on one simulated frame (matching_balance_data.csv, written
# and read back here, so both sides read the same CSV bytes).
#
# Packages (all GPL, run as black boxes: outputs only, no source read):
#   propensity, halfmoon, WeightIt, cobalt, lmw from the default library;
#   cem 1.1.31 from a private library. cem lists tcltk (for its GUI) under
#   Depends, and tcltk does not load on an R without X11, so the package is
#   installed from the CRAN tarball with that one manifest entry removed;
#   no R code of cem is changed (a tcltk call would fail loudly):
#     L=tests/reference_parity/_fixtures/_rlib_matching; mkdir -p $L
#     curl -sSLO https://cloud.r-project.org/src/contrib/cem_1.1.31.tar.gz
#     tar xzf cem_1.1.31.tar.gz
#     sed -i '' 's/^Depends: tcltk, lattice/Depends: lattice/' cem/DESCRIPTION
#     sed -i '' '/import("tcltk")/d' cem/NAMESPACE
#     R CMD INSTALL --no-test-load -l $L cem
# Run (from the repository root):
#   R_LIBS_USER=tests/reference_parity/_fixtures/_rlib_matching \
#     Rscript tests/reference_parity/_fixtures/_generate_matching_balance_R.R
# ---------------------------------------------------------------------------
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages({
  library(propensity); library(halfmoon); library(WeightIt); library(cobalt)
  library(cem); library(lmw); library(jsonlite)
})
here <- "tests/reference_parity/_fixtures"
set.seed(20261105)
n <- 500
x1 <- round(rnorm(n), 2)            # rounded: ties for KS and the L1 cells
x2 <- round(runif(n, -2, 2), 3)
x3 <- rpois(n, 2)                   # few distinct values
b <- rbinom(n, 1, 0.4)
e <- plogis(-0.4 + 1.4 * x1 - 0.6 * x2 + 0.15 * x3 + 0.7 * b)
t <- rbinom(n, 1, e)
dose <- round(1 + 0.6 * x1 - 0.4 * x2 + 0.5 * b + rnorm(n), 4)
d <- data.frame(t = t, x1 = x1, x2 = x2, x3 = x3, b = b, dose = dose)
write.csv(d, file.path(here, "matching_balance_data.csv"), row.names = FALSE)
d <- read.csv(file.path(here, "matching_balance_data.csv"))

pkgs <- c("propensity", "halfmoon", "WeightIt", "cobalt", "cem", "lmw")
out <- list(`_versions` = c(list(R = R.version.string),
                            lapply(setNames(pkgs, pkgs), function(p) as.character(packageVersion(p)))))
X <- d[, c("x1", "x2", "x3", "b")]
num <- function(v) as.numeric(unclass(v))

## propensity scores ------------------------------------------------------------
ctl <- glm.control(epsilon = 1e-14, maxit = 100)
ps <- list(
  logit = unname(fitted(glm(t ~ x1 + x2 + x3 + b, data = d, family = binomial("logit"), control = ctl))),
  probit = unname(fitted(glm(t ~ x1 + x2 + x3 + b, data = d, family = binomial("probit"), control = ctl))))
out$ps <- ps

## trimming (propensity::ps_trim) ----------------------------------------------
trim <- function(p, ...) {
  tr <- ps_trim(p, ...)
  list(kept = which(!is_unit_trimmed(tr)), cutoff = ps_trim_meta(tr)$cutoff)
}
out$trim <- list(
  crump_logit = trim(ps$logit, method = "adaptive"),
  crump_probit = trim(ps$probit, method = "adaptive"),
  fixed_logit = trim(ps$logit, method = "ps", lower = 0.1, upper = 0.9))

## weights (propensity::wt_*) ---------------------------------------------------
p <- ps$logit
W <- list(ATE = wt_ate(p, d$t), ATT = wt_att(p, d$t), ATC = wt_atu(p, d$t),
          ATM = wt_atm(p, d$t), ATO = wt_ato(p, d$t))
out$weights <- lapply(W, num)
out$weights_probit_ATE <- num(wt_ate(ps$probit, d$t))
out$weights_stabilized <- num(wt_ate(p, d$t, stabilize = TRUE))
tps <- ps_trunc(p, method = "ps", lower = 0.05, upper = 0.9)
tq <- ps_trunc(p, method = "pctl", lower = 0.02, upper = 0.98)
out$weights_truncated <- list(
  ps_ATE = num(wt_ate(tps, d$t)), ps_ATT = num(wt_att(tps, d$t)),
  pctl_ATE = num(wt_ate(tq, d$t)), pctl_ATO = num(wt_ato(tq, d$t)))
# continuous exposure: a normal exposure model with one residual SD
om <- lm(dose ~ x1 + x2 + b, data = d)
sig <- summary(om)$sigma
out$continuous <- list(
  mu = unname(fitted(om)), sigma = sig,
  w = num(wt_ate(unname(fitted(om)), d$dose, .sigma = rep(sig, n), exposure_type = "continuous")),
  w_stab = num(wt_ate(unname(fitted(om)), d$dose, .sigma = rep(sig, n), exposure_type = "continuous",
                      stabilize = TRUE)),
  # WeightIt's generalized propensity score weights for the same normal
  # exposure model (its continuous weights are always stabilized)
  weightit_glm = unname(weightit(dose ~ x1 + x2 + b, data = d, method = "glm")$weights))

## effective sample size (WeightIt::ESS and halfmoon::ess) ----------------------
out$ess <- lapply(W, function(w) c(weightit = ESS(num(w)), halfmoon = ess(num(w))))
out$ess_by_group <- lapply(W, function(w) as.list(tapply(num(w), d$t, ESS)))

## energy distance (halfmoon::bal_energy) -----------------------------------------
out$energy <- list(
  raw = bal_energy(X, d$t),
  ATE = bal_energy(X, d$t, .weights = num(W$ATE)),
  ATT = bal_energy(X, d$t, .weights = num(W$ATT)),
  ATO = bal_energy(X, d$t, .weights = num(W$ATO)),
  continuous_only_ATE = bal_energy(X[, c("x1", "x2")], d$t, .weights = num(W$ATE)))

## balance (cobalt) -------------------------------------------------------------------
# bal.tab with cobalt's defaults for a binary 0/1 covariate (binary = "std":
# p(1 - p) variances) and col_w_* with bin.vars = FALSE, which treats b as a
# number (the sample variance), i.e. the convention of sp.balance_diagnostics.
M <- as.matrix(X)
nb <- rep(FALSE, ncol(M))
bal <- function(w) {
  bt <- bal.tab(X, treat = d$t, weights = w, s.d.denom = "pooled", stats = c("m", "v", "ks"),
                un = TRUE, binary = "std", continuous = "std")$Balance
  bt$var <- rownames(bt)
  tr <- d$t == 1
  sd_w <- function(g) col_w_sd(M[g, , drop = FALSE], weights = w[g], bin.vars = nb)
  list(
    baltab = bt[, c("var", "Diff.Un", "Diff.Adj", "V.Ratio.Un", "V.Ratio.Adj", "KS.Un", "KS.Adj")],
    smd_raw = col_w_smd(M, d$t, s.d.denom = "pooled", bin.vars = nb),
    smd_unweighted_denom = col_w_smd(M, d$t, weights = w, s.d.denom = "pooled", bin.vars = nb),
    # pooled weighted group SDs: the default sd_denom = "weighted"
    smd_weighted_denom = col_w_smd(M, d$t, weights = w, bin.vars = nb,
                                   s.d.denom = sqrt((sd_w(tr)^2 + sd_w(!tr)^2) / 2)),
    vr = col_w_vr(M, d$t, weights = w, bin.vars = nb),
    ks = col_w_ks(M, d$t, weights = w, bin.vars = nb),
    mean_w_t = col_w_mean(M[tr, ], weights = w[tr]),
    mean_w_c = col_w_mean(M[!tr, ], weights = w[!tr]))
}
out$balance <- lapply(W[c("ATE", "ATT", "ATO")], function(w) bal(num(w)))

# sp.bal_tab options not in weightit_cobalt_R.json: s.d.denom overrides and
# raw continuous differences, on the binary treatment with ATE weights
Wb <- weightit(t ~ x1 + x2 + x3 + b, data = d, method = "glm", estimand = "ATE")
bt2 <- function(...) {
  b <- bal.tab(Wb, un = TRUE, stats = c("m", "v", "ks"), ...)
  r <- as.data.frame(b$Balance); r$var <- rownames(r); r[r$var != "prop.score", ]
}
out$bal_tab <- list(
  w = unname(Wb$weights),
  sd_all = bt2(s.d.denom = "all"),
  sd_treated = bt2(s.d.denom = "treated"),
  sd_control = bt2(s.d.denom = "control"),
  cont_raw = bt2(continuous = "raw"),
  bin_std_all = bt2(binary = "std", s.d.denom = "all"))

## L1 imbalance (cem::imbalance) ----------------------------------------------------
# The coarsening of sp.l1_imbalance: a covariate with more than k distinct
# values is cut at k + 1 equal-width cutpoints between its minimum and its
# maximum; one with at most k distinct values is passed as a factor, so it
# is not coarsened.
cov <- c("x1", "x2", "x3", "b")
l1 <- function(k, w = NULL, vars = cov) {
  dl <- d[, c("t", vars), drop = FALSE]
  cut_vars <- vars[sapply(vars, function(v) length(unique(d[[v]])) > k)]
  for (v in setdiff(vars, cut_vars)) dl[[v]] <- factor(dl[[v]])
  br <- lapply(setNames(cut_vars, cut_vars),
               function(v) seq(min(d[[v]]), max(d[[v]]), length.out = k + 1))
  if (is.null(w)) imbalance(dl$t, dl, drop = "t", breaks = br)$L1$L1
  else imbalance(dl$t, dl, drop = "t", breaks = br, weights = w)$L1$L1
}
k0 <- ceiling(log2(n) + 1)
out$l1 <- list(
  k_sturges = k0,
  sturges = l1(k0), k5 = l1(5), k12 = l1(12), k3 = l1(3),
  sturges_ATE = l1(k0, num(W$ATE)), k5_ATO = l1(5, num(W$ATO)),
  only_x1 = l1(k0, vars = "x1"), x1_x3_k4 = l1(4, vars = c("x1", "x3")))

## implied regression weights (lmw) ---------------------------------------------------
out$lmw <- list(
  uri = lmw(~ t + x1 + x2 + x3 + b, data = d, treat = "t")$weights,
  mri_ate = lmw(~ t * (x1 + x2 + x3 + b), data = d, treat = "t", estimand = "ATE")$weights,
  mri_att = lmw(~ t * (x1 + x2 + x3 + b), data = d, treat = "t", estimand = "ATT")$weights,
  mri_atc = lmw(~ t * (x1 + x2 + x3 + b), data = d, treat = "t", estimand = "ATC")$weights,
  uri_x1x2 = lmw(~ t + x1 + x2, data = d, treat = "t")$weights)

## logit standard errors at a converged maximum (for the sp.pscore test) -----------
# Stata's pscore reports the logit's standard errors at its own stopping
# point; these are glm's, with the deviance converged to 1e-14.
ll <- read.csv(file.path(here, "matching_lalonde.csv"))
lse <- function(f) {
  m <- glm(f, data = ll, family = binomial("logit"), control = ctl)
  # vcov(glm) uses the IRLS weights of its last iteration, evaluated before
  # the final update; se_info is the inverse information at the fitted values
  X <- model.matrix(m); p <- fitted(m)
  list(b = as.list(coef(m)), se = as.list(sqrt(diag(vcov(m)))),
       se_info = as.list(sqrt(diag(solve(crossprod(X * sqrt(p * (1 - p))))))))
}
out$lalonde_logit <- list(
  x8 = lse(treat ~ age + educ + black + hispan + married + nodegree + re74 + re75),
  x5 = lse(treat ~ age + educ + black + married + re74))

write_json(out, file.path(here, "matching_balance_R.json"), digits = NA, auto_unbox = TRUE,
           null = "null")
