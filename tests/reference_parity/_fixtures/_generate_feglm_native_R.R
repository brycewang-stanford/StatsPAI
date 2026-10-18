# Reference for tests/reference_parity/test_feglm_native_fixest_parity.py.
# fixest::feglm with its default conventions (perfect-fit observations
# dropped, Fisher-scoring variance, default small-sample correction) and
# tight tolerances, so the reference is the optimum rather than fixest's
# default stopping point. Run from this directory:
#   Rscript _generate_feglm_native_R.R
suppressPackageStartupMessages({library(fixest); library(jsonlite)})
d <- read.csv("feglm_native_data.csv")
vcovs <- list(iid = "iid", hetero = "hetero", cr1 = ~g)
out <- list(r_version = R.version.string,
            fixest_version = as.character(packageVersion("fixest")))
for (fam in c("logit", "probit")) for (fe in c("one", "multi"))
  for (v in names(vcovs)) for (wt in c("none", "w")) {
    rhs <- if (fe == "one") "firm" else "firm + year"
    f <- as.formula(paste("d ~ x1 + x2 |", rhs))
    m <- if (wt == "none")
      feglm(f, d, family = fam, vcov = vcovs[[v]], glm.tol = 1e-10, fixef.tol = 1e-10)
    else
      feglm(f, d, family = fam, vcov = vcovs[[v]], weights = ~w,
            glm.tol = 1e-10, fixef.tol = 1e-10)
    out[[paste(fam, fe, v, wt, sep = "_")]] <- list(
      coef = unname(coef(m)[c("x1", "x2")]),
      se = unname(se(m)[c("x1", "x2")]),
      pvalue = unname(pvalue(m)[c("x1", "x2")]),
      nobs = nobs(m))
  }
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), "feglm_native_R.json")
