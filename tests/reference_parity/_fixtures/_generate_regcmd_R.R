# R reference numbers for tests/reference_parity/test_regcmd_stata_r_parity.py,
# on the same CSV bytes as _generate_regcmd_Stata.do.
#
#   intreg      survival::survreg(Surv(lo, hi, type = "interval2"), "gaussian"),
#               observed information
#   heckprobit  sampleSelection::selection(method = "ml") with a binary
#               outcome, maximised by BHHH (its default) and by Newton-Raphson
#   gologit     VGAM::vglm(cumulative(reverse = TRUE)) -- coefficients only:
#               VGAM's covariance is the expected information (Fisher
#               scoring), gologit2's the observed one
#   nl          stats::nls with the analytic gradient from deriv(), and
#               sandwich::sandwich / vcovCL on the fit
#
# The packages are GPL / LGPL and are run as black boxes (outputs only).
#
#   Rscript tests/reference_parity/_fixtures/_generate_regcmd_R.R
# (run from the repository root)
suppressPackageStartupMessages({
  library(jsonlite)
  library(survival)
  library(sampleSelection)
  library(VGAM)
  library(sandwich)
})

fx <- "tests/reference_parity/_fixtures"
cs <- read.csv(file.path(fx, "regcmd_cs.csv"))
out <- list(
  R = R.version.string,
  packages = lapply(
    c(survival = "survival", sampleSelection = "sampleSelection",
      VGAM = "VGAM", sandwich = "sandwich", jsonlite = "jsonlite"),
    function(p) as.character(packageVersion(p))
  )
)
pack <- function(b, V) {
  list(names = names(b), b = unname(b), se = unname(sqrt(diag(V))))
}

# ------------------------------------------------------------------ intreg
ctl <- survreg.control(rel.tolerance = 1e-13, iter.max = 500)
fml <- Surv(ilo, ihi, type = "interval2") ~ x1 + x2 + x3
# Observed information only. survreg's robust = TRUE / cluster() variance is
# built from its dfbeta residuals, which for these interval-censored rows do
# not reproduce the closed-form score sandwich that Stata's vce(robust) and
# StatsPAI report (x1: 0.0538 against 0.0497); see the test module.
fits <- list(
  intreg_oim = survreg(fml, data = cs, dist = "gaussian", control = ctl)
)
for (k in names(fits)) {
  m <- fits[[k]]
  b <- c(coef(m), "Log(scale)" = log(m$scale))
  out[[k]] <- c(pack(b, vcov(m)), list(N = nrow(m$y), ll = m$loglik[2]))
}

# -------------------------------------------------------------- heckprobit
# A logical outcome selects the binary-outcome (bivariate probit) model.
# sampleSelection maximises it by BHHH, so the default covariance is the
# inverse outer product of the scores; maxMethod = "NR" gives the observed
# information, which is what heckprobit reports.
cs$ybh_l <- cs$ybh == 1
cs$s_l <- cs$s == 1
for (meth in c("BHHH", "NR")) {
  hp <- selection(s_l ~ z + x1, ybh_l ~ x1 + x2 + x3, data = cs,
                  method = "ml", maxMethod = meth, iterlim = 500,
                  tol = 1e-14, reltol = 1e-16)
  out[[paste0("heckprobit_", tolower(meth))]] <- c(
    pack(coef(hp), vcov(hp)),
    list(N = nrow(cs), ll = as.numeric(logLik(hp)),
         method = maxLik::maximType(hp))
  )
}

# ----------------------------------------------------------------- gologit
cs$yo_f <- ordered(cs$yo)
vctl <- vglm.control(epsilon = 1e-12, maxit = 200)
vg <- function(fam) {
  m <- vglm(yo_f ~ x1 + x2 + x3, family = fam, data = cs, control = vctl)
  list(names = names(coef(m)), b = unname(coef(m)),
       ll = as.numeric(logLik(m)), N = nrow(cs))
}
out$gologit_npl <- vg(cumulative(parallel = FALSE, reverse = TRUE))
out$gologit_pl <- vg(cumulative(parallel = TRUE, reverse = TRUE))
out$gologit_npl_x1 <- vg(cumulative(parallel = TRUE ~ x2 + x3 - 1, reverse = TRUE))
out$gologit_pl_x2x3 <- out$gologit_npl_x1
out$gologit_probit_npl <- vg(
  cumulative(link = "probitlink", parallel = FALSE, reverse = TRUE)
)
out$gologit_probit_npl_x1 <- vg(
  cumulative(link = "probitlink", parallel = TRUE ~ x2 + x3 - 1, reverse = TRUE)
)

# ---------------------------------------------------------------------- nl
tb <- read.csv(file.path(fx, "textbook_cs.csv"))
tb$xp <- exp(tb$x1 / 2)
tb$ynl <- 2 + 3 * tb$xp^0.5 + tb$z2 / 4
f_nl <- deriv(~ a + b * xp^c, c("a", "b", "c"), function(a, b, c, xp) {})
m <- nls(ynl ~ f_nl(a, b, c, xp), data = tb, start = list(a = 1, b = 1, c = 1),
         control = nls.control(tol = 1e-9, maxiter = 1000,
                             minFactor = 1e-12))
stopifnot(m$convInfo$isConv)
rss <- sum(residuals(m)^2)
nl_out <- function(V) c(pack(coef(m), V), list(N = nrow(tb), rss = rss))
out$nl_gnr <- nl_out(vcov(m))
# vcovHC() needs a model matrix, which an nls fit does not have; sandwich()
# is HC0 from estfun() / bread() on the analytic gradient, and HC1 is HC0
# times n / (n - k), as vcovHC defines it.
k_nl <- length(coef(m))
out$nl_robust <- nl_out(sandwich(m) * nrow(tb) / (nrow(tb) - k_nl))
out$nl_cluster <- nl_out(vcovCL(m, cluster = tb$g, type = "HC1"))

writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           file.path(fx, "regcmd_R.json"))
