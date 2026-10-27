# Cross-check of sp.cox(tvc=, texp=) against survival::coxph(tt = ).
# Run from tests/reference_parity/_fixtures:
#   Rscript _generate_cox_tvc_R.R
# coxph's tt() evaluates g(t) at each event time on internally split
# data, the same quantity as Stata's tvc(). survival is LGPL; it is used
# as a black box (documented interface and outputs only).
suppressPackageStartupMessages({
  library(survival)
  library(jsonlite)
})
cont <- read.csv("cox_tvc_cont.csv")
ties <- read.csv("cox_tvc_ties.csv")

grab <- function(fit) {
  b <- coef(fit)
  list(
    b = unname(b),
    names = names(b),
    se = unname(sqrt(diag(vcov(fit)))),
    se_model = unname(sqrt(diag(fit$naive.var %||% vcov(fit)))),
    loglik = fit$loglik[2]
  )
}
`%||%` <- function(a, b) if (is.null(a)) b else a

out <- list(version = as.character(packageVersion("survival")))
lin <- function(x, t, ...) x * t
lg <- function(x, t, ...) x * log(t)

out$c1_breslow_t <- grab(coxph(Surv(time, died) ~ x1 + race + tt(race),
  data = cont, tt = lin, ties = "breslow"))
out$c2_efron_ln <- grab(coxph(Surv(time, died) ~ x1 + race + tt(race),
  data = cont, tt = lg, ties = "efron"))
out$t1_efron_t <- grab(coxph(Surv(time, died) ~ x1 + race + tt(race),
  data = ties, tt = lin, ties = "efron"))
out$t2_breslow_ln_robust <- grab(coxph(
  Surv(time, died) ~ x1 + race + tt(race) + tt(x1),
  data = ties, tt = list(lg, lg), ties = "breslow", cluster = id))
out$t3_efron_strata_cluster <- grab(coxph(
  Surv(time, died) ~ x1 + race + tt(race) + strata(grp),
  data = ties, tt = lin, ties = "efron", cluster = cl))
out$t6_efron_entry_ln <- grab(coxph(Surv(entry, time, died) ~ x1 + race + tt(race),
  data = ties, tt = lg, ties = "efron"))

writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
  "cox_tvc_R.json")
