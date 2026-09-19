#!/usr/bin/env Rscript
# R references for tests/reference_parity/test_survival_epi_R_parity.py.
#
# Inputs: _fixtures/survival_epi_*.csv (written by
# _fixtures/_generate_survival_epi_data.py). Output:
# _fixtures/survival_epi_R.json, full precision (17 significant digits).
#
# Regenerate (from the repository root):
#   python tests/reference_parity/_fixtures/_generate_survival_epi_data.py
#   Rscript tests/reference_parity/_generate_survival_epi_R.R
#
# Conventions pinned here
# -----------------------
# * cmprsk::cuminc returns each CIF as a step function with the corners
#   duplicated; the jump points are the even entries after the leading 0.
#   `var` is Gray's asymptotic variance. Tests are unstratified.
# * cmprsk::crr: `invinf` is the inverse information (model-based), `var`
#   the Fine-Gray sandwich (with the censoring-estimation term).
# * survival::coxph frailty(): fits at FIXED theta with sparse = FALSE (the
#   full penalised information, which is what Stata stcox, shared() uses)
#   and a tight inner convergence. The theta maximiser is found by
#   optimize() on the integrated likelihood (history c.loglik) with a
#   1e-12 tolerance; the default method = "em" result is stored only for
#   the record (it stops on a 1e-5 relative likelihood change).
# * epitools::ageadjust.direct: Poisson variance, Fay-Feuer gamma CI.
#   ageadjust.indirect: log-normal SIR CI.
# * DescTools::BreslowDayTest with the Mantel-Haenszel common OR;
#   correct = TRUE adds Tarone's correction.
# * pROC::roc(levels = c(0, 1), direction = "<"): AUC, DeLong variance,
#   DeLong CI (truncated to [0, 1]).
# * epiR::epi.tests on the table [[tp, fp], [fn, tn]].
# * bw.nrd0; bw.SJ at its defaults (nb = 1000 binned pair counts, uniroot
#   tol = 0.1 * lower) and at nb = 1e7, tol = 1e-14.
# * stats::power.prop.test(strict = TRUE): pooled-null two-sample test,
#   both tails.
suppressPackageStartupMessages({
  library(jsonlite)
  library(survival)
  library(cmprsk)
  library(epitools)
  library(DescTools)
  library(pROC)
  library(epiR)
})

fx <- function(f) file.path("tests", "reference_parity", "_fixtures", f)
out <- list()

# ---- competing risks ---------------------------------------------------
cr <- read.csv(fx("survival_epi_cr.csv"))
jumps <- function(obj) {
  i <- seq(3, length(obj$time) - 1, by = 2)
  list(time = obj$time[i], est = obj$est[i], var = obj$var[i])
}
ci <- cuminc(cr$time, cr$status)
out$cuminc <- list(cause1 = jumps(ci[["1 1"]]), cause2 = jumps(ci[["1 2"]]))
ci_g <- cuminc(cr$time, cr$status, cr$grp)
out$cuminc_grp <- list(
  g0_cause1 = jumps(ci_g[["0 1"]]), g1_cause1 = jumps(ci_g[["1 1"]]),
  tests = list(stat = unname(ci_g$Tests[, "stat"]), pv = unname(ci_g$Tests[, "pv"]))
)
ci_rho <- cuminc(cr$time, cr$status, cr$grp, rho = 1)
out$gray_rho1 <- list(stat = unname(ci_rho$Tests[, "stat"]),
                      pv = unname(ci_rho$Tests[, "pv"]))
ci_g3 <- cuminc(cr$time, cr$status, cr$g3)
out$gray_g3 <- list(stat = unname(ci_g3$Tests[, "stat"]),
                    pv = unname(ci_g3$Tests[, "pv"]),
                    df = unname(ci_g3$Tests[, "df"]))

crr_out <- function(failcode) {
  f <- crr(cr$time, cr$status, cbind(x1 = cr$x1, x2 = cr$x2, grp = cr$grp),
           failcode = failcode, gtol = 1e-12, maxiter = 100)
  list(coef = unname(f$coef), se_robust = unname(sqrt(diag(f$var))),
       se_model = unname(sqrt(diag(f$invinf))), loglik = f$loglik,
       converged = f$converged)
}
out$crr_cause1 <- crr_out(1)
out$crr_cause2 <- crr_out(2)

# Cox on the cause-1 data without competing events: Breslow and Efron ties
# (sp.cox(ties="breslow") used to run Efron; the tied times make them differ).
cr1 <- cr[cr$status != 2, ]
cox_out <- function(ties) {
  f <- coxph(Surv(time, status) ~ x1 + x2, data = cr1, ties = ties,
             control = coxph.control(eps = 1e-11))
  list(coef = unname(coef(f)), se = unname(sqrt(diag(vcov(f)))),
       loglik = f$loglik[2], loglik0 = f$loglik[1])
}
out$cox_breslow <- cox_out("breslow")
out$cox_efron <- cox_out("efron")

# ---- shared gamma frailty ----------------------------------------------
fr <- read.csv(fx("survival_epi_frailty.csv"))
ctl <- coxph.control(eps = 1e-11, iter.max = 200)
fixed_fit <- function(th) {
  g <- coxph(Surv(time, event) ~ x1 + x2 +
               frailty(cid, theta = th, sparse = FALSE),
             data = fr, control = ctl)
  list(theta = th, coef = unname(g$coefficients[1:2]),
       se = unname(sqrt(diag(g$var))[1:2]),
       log_frailty = unname(g$coefficients[-(1:2)]),
       loglik_partial = g$loglik[2],
       c_loglik = g$history[[1]]$c.loglik)
}
out$frailty_fixed_050 <- fixed_fit(0.5)
out$frailty_fixed_024 <- fixed_fit(0.24)
ilik <- function(th) fixed_fit(th)$c_loglik
opt <- optimize(ilik, c(0.01, 3), maximum = TRUE, tol = 1e-12)
out$frailty_ml <- fixed_fit(opt$maximum)
em <- coxph(Surv(time, event) ~ x1 + x2 + frailty(cid, distribution = "gamma"),
            data = fr)
out$frailty_em_default <- list(theta = em$history[[1]]$theta,
                               coef = unname(em$coefficients),
                               c_loglik = em$history[[1]]$c.loglik)
out$cox_loglik <- coxph(Surv(time, event) ~ x1 + x2, data = fr,
                        control = ctl)$loglik[2]

# ---- rate standardisation ----------------------------------------------
st <- read.csv(fx("survival_epi_std.csv"))
dd <- ageadjust.direct(count = st$events, pop = st$pop, stdpop = st$std_pop)
out$direct <- as.list(dd)
ii <- ageadjust.indirect(count = st$events, pop = st$pop,
                         stdcount = st$ref_events, stdpop = st$ref_pop)
out$indirect <- as.list(ii$sir)

# ---- Breslow-Day -------------------------------------------------------
bd <- read.csv(fx("survival_epi_bd.csv"))
arr <- array(0, dim = c(2, 2, nrow(bd)))
for (k in seq_len(nrow(bd))) {
  arr[, , k] <- matrix(c(bd$a[k], bd$c[k], bd$b[k], bd$d[k]), 2, 2)
}
b0 <- BreslowDayTest(arr)
b1 <- BreslowDayTest(arr, correct = TRUE)
out$breslow_day <- list(stat = unname(b0$statistic), p = b0$p.value,
                        stat_tarone = unname(b1$statistic), p_tarone = b1$p.value,
                        or_mh = unname(mantelhaen.test(arr)$estimate))

# ---- ROC / AUC ---------------------------------------------------------
rd <- read.csv(fx("survival_epi_roc.csv"))
roc_out <- function(s) {
  r <- roc(rd$y, s, levels = c(0, 1), direction = "<", quiet = TRUE)
  cc <- ci.auc(r, method = "delong")
  list(auc = as.numeric(r$auc), var_delong = var(r, method = "delong"),
       ci_lo = cc[1], ci_hi = cc[3])
}
out$roc <- roc_out(rd$score)
out$roc_tied <- roc_out(rd$score_tied)

# ---- diagnostic tests --------------------------------------------------
dg <- read.csv(fx("survival_epi_diag.csv"))
out$diag <- lapply(seq_len(nrow(dg)), function(k) {
  tab <- as.table(matrix(c(dg$tp[k], dg$fp[k], dg$fn[k], dg$tn[k]), 2, 2,
                         byrow = TRUE))
  res <- list()
  for (m in c("wilson", "exact")) {
    det <- epi.tests(tab, method = m)$detail
    g <- function(s) as.numeric(det[det$statistic == s, c("est", "lower", "upper")])
    res[[m]] <- list(se = g("se"), sp = g("sp"), pv_pos = g("pv.pos"),
                     pv_neg = g("pv.neg"), lr_pos = g("lr.pos")[1],
                     lr_neg = g("lr.neg")[1], tp = g("tp")[1])
  }
  res
})

# ---- bandwidths --------------------------------------------------------
kd <- read.csv(fx("survival_epi_kd.csv"))
out$bw <- list(nrd0 = bw.nrd0(kd$x), sj_default = bw.SJ(kd$x),
               sj_nb1e7 = bw.SJ(kd$x, nb = 1e7L, tol = 1e-14))

# ---- two-proportion power (case-control, 1:1) --------------------------
p0 <- 0.3
p1 <- 2 * p0 / (1 + p0)
out$power_prop <- list(
  n200 = power.prop.test(n = 200, p1 = p0, p2 = p1, strict = TRUE)$power,
  n60 = power.prop.test(n = 60, p1 = p0, p2 = p1, strict = TRUE)$power,
  n200_one = power.prop.test(n = 200, p1 = p0, p2 = p1, strict = TRUE,
                             alternative = "one.sided")$power
)

out$provenance <- list(
  generated_by = "tests/reference_parity/_generate_survival_epi_R.R",
  r_version = R.version.string,
  packages = list(
    survival = as.character(packageVersion("survival")),
    cmprsk = as.character(packageVersion("cmprsk")),
    epitools = as.character(packageVersion("epitools")),
    DescTools = as.character(packageVersion("DescTools")),
    pROC = as.character(packageVersion("pROC")),
    epiR = as.character(packageVersion("epiR"))
  )
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), pretty = TRUE),
           fx("survival_epi_R.json"))
cat("wrote survival_epi_R.json\n")
