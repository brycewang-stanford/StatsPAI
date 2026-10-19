# Reference values for test_feglm_biascorr_parity.py: alpaca::biasCorr
# (analytical bias correction of Fernandez-Val and Weidner 2016), L = 0,
# panel.structure = "classic". alpaca is GPL and used as a black box: only
# its output is read. dev.tol = 1e-14 so alpaca's own stopping rule (1e-8
# by default, which leaves the slopes ~4e-5 off) does not dominate.
# Run with R_LIBS_USER pointing at a private library holding alpaca.
suppressPackageStartupMessages({
  library(alpaca)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
d <- read.csv(file.path(here, "feglm_biascorr_data.csv"))
num <- function(x) unname(as.numeric(x))
se <- function(m, ...) num(sqrt(diag(vcov(m, ...))))
one <- function(f, fam) {
  m <- alpaca::feglm(as.formula(paste(f, "| cl")), d, binomial(fam),
                     control = feglmControl(dev.tol = 1e-14))
  b <- biasCorr(m)
  list(coef_uncorr = num(coef(m)), coef = num(coef(b)), nobs = num(m$nobs[["nobs"]]),
       se_hessian = se(b), se_sandwich = se(b, type = "sandwich"),
       se_cluster = se(b, type = "clustered", cluster = ~cl))
}
out <- list(
  versions = list(alpaca = as.character(packageVersion("alpaca")), R = R.version.string),
  logit_it = one("y ~ x1 + x2 | i + t", "logit"),
  logit_i = one("y ~ x1 + x2 | i", "logit"),
  probit_it = one("yp ~ x1 + x2 | i + t", "probit"),
  probit_i = one("yp ~ x1 + x2 | i", "probit")
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           file.path(here, "feglm_biascorr_R.json"))
