# ---------------------------------------------------------------------------
# Reference numbers for tests/reference_parity/
#   test_barrett_causal_inference_in_r_parity.py
#
# The propensity-score workflow of Barrett, D'Agostino McGowan and Gerke,
# "Causal Inference in R" (https://www.r-causal.org), run with the packages
# the book uses on data simulated here. Both sides read the CSV this script
# writes.
#
# Requires: propensity (0.1.0), halfmoon (0.2.0), tipr (1.0.2), cobalt,
#           lmw, marginaleffects, dagitty, pROC, jsonlite.
# Run:      Rscript tests/reference_parity/_fixtures/_generate_barrett_causal_inference_in_r.R
#           (from the repository root)
# ---------------------------------------------------------------------------
suppressMessages({
  library(propensity); library(halfmoon); library(tipr); library(cobalt)
  library(lmw); library(marginaleffects); library(dagitty); library(jsonlite)
})
options(tipr.verbose = FALSE)
here <- "tests/reference_parity/_fixtures"
set.seed(20261005)
n <- 500
x1 <- round(rnorm(n), 3)
x2 <- round(runif(n, -2, 2), 3)
b <- rbinom(n, 1, 0.4)
e <- plogis(-0.3 + 1.1 * x1 - 0.6 * x2 + 0.8 * b)
t <- rbinom(n, 1, e)
y <- round(1 + 2 * t + x1 + 0.5 * x2^2 + b + t * x1 + rnorm(n), 4)
yb <- rbinom(n, 1, plogis(-0.5 + 0.7 * t + 0.6 * x1 - 0.4 * b))
d <- data.frame(y = y, yb = yb, t = t, x1 = x1, x2 = x2, b = b)
write.csv(d, file.path(here, "barrett_ps_workflow.csv"), row.names = FALSE)
d <- read.csv(file.path(here, "barrett_ps_workflow.csv"))

out <- list()
ctl <- glm.control(epsilon = 1e-14, maxit = 100)
pm <- glm(t ~ x1 + x2 + b, data = d, family = binomial(), control = ctl)
ps <- unname(fitted(pm))
out$ps <- ps

## weights, effective sample size ------------------------------------------
W <- list(ATE = wt_ate(ps, d$t), ATT = wt_att(ps, d$t), ATC = wt_atu(ps, d$t),
          ATM = wt_atm(ps, d$t), ATO = wt_ato(ps, d$t))
out$weights <- lapply(W, as.numeric)
out$weights_stabilized <- as.numeric(wt_ate(ps, d$t, stabilize = TRUE))
out$ess <- lapply(W, function(w) ess(as.numeric(w)))
out$ess_by_group <- as.list(tapply(as.numeric(W$ATE), d$t, ess))

## trimming and truncation ---------------------------------------------------
tr <- ps_trim(ps, method = "adaptive")
out$crump <- list(cutoff = ps_trim_meta(tr)$cutoff, trimmed = which(is_unit_trimmed(tr)))
tc <- ps_trunc(ps, method = "pctl", lower = 0.05, upper = 0.95)
out$trunc_weights <- as.numeric(wt_ate(tc, d$t))

## IPW with the M-estimation variance ---------------------------------------
for (est in c("ATE", "ATT", "ATM", "ATO")) {
  d$w <- as.numeric(W[[est]])
  om <- lm(y ~ t, data = d, weights = w)
  r <- as.data.frame(ipw(pm, om, .data = d, estimand = tolower(est)))
  out$ipw[[est]] <- c(estimate = r$estimate, se = r$std.err)
}
d$w <- as.numeric(W$ATC)
out$ipw$ATC <- c(estimate = unname(coef(lm(y ~ t, data = d, weights = w))[2]))

## balance --------------------------------------------------------------------
X <- d[, c("x1", "x2", "b")]
w <- as.numeric(W$ATE)
out$cobalt <- list(
  unweighted_denominator = bal.tab(X, treat = d$t, weights = w, s.d.denom = "pooled",
    stats = c("m", "v", "ks"), un = TRUE, binary = "std", continuous = "std")$Balance[
    , c("Diff.Un", "Diff.Adj", "V.Ratio.Un", "V.Ratio.Adj", "KS.Un", "KS.Adj")])
out$cobalt$rows <- rownames(out$cobalt$unweighted_denominator)
out$halfmoon <- list(
  vr_x1 = bal_vr(d$x1, d$t, .weights = w), vr_x2 = bal_vr(d$x2, d$t, .weights = w),
  ks_x1 = bal_ks(d$x1, d$t, .weights = w),
  energy_raw = bal_energy(X, d$t), energy_weighted = bal_energy(X, d$t, .weights = w))
d$tf <- factor(d$t); d$ps <- ps; d$w <- w
a <- check_model_auc(d, .exposure = tf, .fitted = ps, .weights = w)
out$auc <- as.list(setNames(a$auc, a$method))
# halfmoon's area is not the Mann-Whitney probability (it is 3e-4 lower
# here, with no tied scores); the rank statistic and pROC are the reference.
out$auc$wilcoxon <- unname(wilcox.test(ps[d$t == 1], ps[d$t == 0])$statistic) / (sum(d$t == 1) * sum(d$t == 0))
out$auc$proc <- as.numeric(pROC::auc(d$t, ps, quiet = TRUE))

## implied regression weights ---------------------------------------------------
out$lmw <- list(
  uri = lmw(~ t + x1 + x2 + b, data = d, treat = "t")$weights,
  mri_ate = lmw(~ t * (x1 + x2 + b), data = d, treat = "t", estimand = "ATE")$weights,
  mri_att = lmw(~ t * (x1 + x2 + b), data = d, treat = "t", estimand = "ATT")$weights,
  ols = unname(coef(lm(y ~ t + x1 + x2 + b, data = d))["t"]))

## g-computation -------------------------------------------------------------------
gl <- glm(yb ~ t + x1 + x2 + b, data = d, family = binomial(), control = ctl)
ac <- function(m, nd = NULL, ...) {
  a <- if (is.null(nd)) avg_comparisons(m, variables = "t", ...) else avg_comparisons(m, variables = "t", newdata = nd, ...)
  c(estimate = a$estimate, se = a$std.error, lo = a$conf.low, hi = a$conf.high)
}
out$gcomp_logit <- list(
  rd = ac(gl), att = ac(gl, subset(d, t == 1)), atc = ac(gl, subset(d, t == 0)),
  lnrr = ac(gl, comparison = "lnratioavg"), lnor = ac(gl, comparison = "lnoravg"),
  rr = ac(gl, comparison = "lnratioavg", transform = exp),
  or = ac(gl, comparison = "lnoravg", transform = exp))
ol <- lm(y ~ t * x1 + x2 + b, data = d)
out$gcomp_ols <- list(ate = ac(ol), att = ac(ol, subset(d, t == 1)), atc = ac(ol, subset(d, t == 0)))

## unmeasured confounder ----------------------------------------------------------
f <- function(x) as.list(as.data.frame(x)[1, ])
out$tipr <- list(
  adjust_coef = f(adjust_coef(6.58, exposure_confounder_effect = -0.17, confounder_outcome_effect = -2.3)),
  adjust_coef_binary = f(adjust_coef_with_binary(-12.5, exposed_confounder_prev = 0.26, unexposed_confounder_prev = 0.05, confounder_outcome_effect = -10)),
  adjust_rr = f(adjust_rr(1.5, exposure_confounder_effect = 0.5, confounder_outcome_effect = 1.8)),
  adjust_rr_binary = f(adjust_rr_with_binary(1.5, exposed_confounder_prev = 0.4, unexposed_confounder_prev = 0.1, confounder_outcome_effect = 1.8)),
  adjust_or_common = f(adjust_or(1.5, exposure_confounder_effect = 0.5, confounder_outcome_effect = 1.8, or_correction = TRUE)),
  adjust_hr_common = f(adjust_hr(1.5, exposure_confounder_effect = 0.5, confounder_outcome_effect = 1.8, hr_correction = TRUE)),
  adjust_hr_binary_common = f(adjust_hr_with_binary(1.5, 0.4, 0.1, 1.8, hr_correction = TRUE)),
  tip_coef_d = f(tip_coef(6.58, confounder_outcome_effect = -2.3)),
  tip_coef_g = f(tip_coef(-10.2, exposure_confounder_effect = 4)),
  tip_coef_n = f(tip_coef(6.58, exposure_confounder_effect = -0.5, confounder_outcome_effect = -2.3)),
  tip_rr_d = f(tip_with_continuous(1.2, confounder_outcome_effect = 2.5)),
  tip_rr_n = f(tip_with_continuous(1.5, 0.3, 1.3)),
  tip_bin_g = f(tip_with_binary(1.2, exposed_confounder_prev = 0.5, unexposed_confounder_prev = 0.1)),
  tip_bin_p1 = f(tip_with_binary(1.2, unexposed_confounder_prev = 0.1, confounder_outcome_effect = 2.5)),
  tip_bin_p0 = f(tip_with_binary(1.2, exposed_confounder_prev = 0.5, confounder_outcome_effect = 2.5)),
  tip_bin_n = f(tip_with_binary(1.5, 0.4, 0.1, 1.3)),
  tip_bin_protective = f(tip_with_binary(0.8, unexposed_confounder_prev = 0.1, confounder_outcome_effect = 0.5)),
  tip_hr_common = f(tip_hr(1.5, confounder_outcome_effect = 2, hr_correction = TRUE)),
  tip_or_binary_common = f(tip_or_with_binary(1.5, exposed_confounder_prev = 0.5, unexposed_confounder_prev = 0.1, or_correction = TRUE)))

## DAGs --------------------------------------------------------------------------------
dags <- list(
  book = "dag { emm -> wait ; close -> wait ; season -> wait ; temp -> wait ; temp -> emm }",
  two_sizes = "dag { W -> X ; P -> W ; Q -> W ; P -> Y ; Q -> Y ; X -> Y }",
  chain = "dag { A -> B ; B -> C ; C -> D ; A -> D ; E -> D ; B -> E }")
ends <- list(book = c("emm", "wait"), two_sizes = c("X", "Y"), chain = c("A", "D"))
for (k in names(dags)) {
  g <- dagitty(dags[[k]])
  sets <- function(type) lapply(adjustmentSets(g, ends[[k]][1], ends[[k]][2], type = type), function(z) sort(as.character(z)))
  cp <- edges(equivalenceClass(g))
  out$dag[[k]] <- list(
    minimal = unname(sets("minimal")), all = unname(sets("all")),
    n_equivalent = length(equivalentDAGs(g)),
    undirected = lapply(which(cp$e == "--"), function(i) sort(c(as.character(cp$v[i]), as.character(cp$w[i])))))
}
write_json(out, file.path(here, "barrett_causal_inference_in_r_R.json"), digits = NA, auto_unbox = TRUE)
cat("written\n")
