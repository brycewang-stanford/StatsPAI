# Reference for tests/reference_parity/test_feols_iv_native_fixest_parity.py.
# fixest 2SLS with absorbed firm and year effects, singletons kept
# (fixef.rm = "none", sp.feols's default), default small-sample
# corrections. ivwald1 is the first-stage Wald statistic under the
# model's variance. Run from this directory:
#   Rscript _generate_feols_iv_native_R.R
suppressPackageStartupMessages({library(fixest); library(jsonlite)})
d <- read.csv("feols_iv_native_data.csv")
fmls <- list(
  just = y ~ x2 | firm + year | x1 ~ z,
  over = y ~ x2 | firm + year | x1 ~ z + z2,
  two = y ~ x2 | firm + year | x1 + x3 ~ z + z2)
vcovs <- list(iid = "iid", hetero = "hetero", cr1 = ~g, twoway = ~g + year)
out <- list(r_version = R.version.string,
            fixest_version = as.character(packageVersion("fixest")))
for (f in names(fmls)) for (v in names(vcovs)) for (wt in c("none", "w")) {
  m <- if (wt == "none") feols(fmls[[f]], d, vcov = vcovs[[v]], fixef.rm = "none")
       else feols(fmls[[f]], d, vcov = vcovs[[v]], weights = ~w, fixef.rm = "none")
  cf <- coef(m); names(cf) <- sub("^fit_", "", names(cf))
  s <- se(m); names(s) <- sub("^fit_", "", names(s))
  cell <- list(coef = as.list(cf), se = as.list(s), nobs = nobs(m))
  if (f != "two" && wt == "none") cell$ivwald1 <- fitstat(m, ~ivwald1)[[1]]$stat
  out[[paste(f, v, wt, sep = "_")]] <- cell
}
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), "feols_iv_native_R.json")
