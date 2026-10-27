# Secondary reference for sp.fmm: R flexmix (EM), started from the simulated
# class and run to a tight EM tolerance, on two of the CSVs used for Stata.
#   gauss2_plain : FLXMRglm(family = "gaussian"), constant class shares
#   pois2_lcprob : FLXMRglm(family = "poisson"), concomitant FLXPmultinom(~ z)
# flexmix reports the class standard deviation (ML, divisor = posterior
# count); the test squares it. flexmix has no observed-information standard
# errors by default, so only the log-likelihood, coefficients, shares and
# the multinomial-logit coefficients are compared.
#
# Run from the repository root:
#   Rscript tests/reference_parity/_fixtures/_generate_fmm_flexmix_R.R
suppressPackageStartupMessages(library(flexmix))
dir <- "tests/reference_parity/_fixtures"
ctrl <- list(iter.max = 100000, tolerance = 1e-13, minprior = 0)
out <- list()

d <- read.csv(file.path(dir, "fmm_gauss2.csv"))
m <- flexmix(y ~ x1 + x2, data = d, k = 2, cluster = d$cls,
             model = FLXMRglm(family = "gaussian"), control = ctrl)
out$gauss2_plain <- list(
  ll = as.numeric(logLik(m)),
  coef = lapply(1:2, function(j) as.list(parameters(m, component = j)[, 1])),
  prior = as.numeric(prior(m)),
  iter = m@iter
)

d <- read.csv(file.path(dir, "fmm_pois2.csv"))
m <- flexmix(y ~ x1 + x2, data = d, k = 2, cluster = d$cls,
             model = FLXMRglm(family = "poisson"),
             concomitant = FLXPmultinom(~ z), control = ctrl)
out$pois2_lcprob <- list(
  ll = as.numeric(logLik(m)),
  coef = lapply(1:2, function(j) as.list(parameters(m, component = j)[, 1])),
  concomitant = as.list(as.data.frame(parameters(m, which = "concomitant"))),
  iter = m@iter
)
out$versions <- list(R = R.version.string,
                     flexmix = as.character(packageVersion("flexmix")))
writeLines(jsonlite::toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE),
           file.path(dir, "fmm_flexmix_R.json"))
