# Reference for tests/reference_parity/test_feols_grid_fixest_parity.py.
# R 4.5, fixest (default small-sample conventions). Run from this directory:
#   Rscript _generate_feols_grid_R.R
suppressPackageStartupMessages({library(fixest); library(jsonlite)})
d <- read.csv("feols_grid_data.csv")
fes <- list(none = "y ~ x1 + x2", one = "y ~ x1 + x2 | firm",
            multi = "y ~ x1 + x2 | firm + year")
vcovs <- list(iid = "iid", hetero = "hetero", cr1 = ~g, twoway = ~g + h)
out <- list(r_version = R.version.string,
            fixest_version = as.character(packageVersion("fixest")))
for (fe in names(fes)) for (v in names(vcovs)) for (wt in c("none", "w")) {
  f <- as.formula(fes[[fe]])
  m <- if (wt == "none") feols(f, d, vcov = vcovs[[v]])
       else feols(f, d, vcov = vcovs[[v]], weights = ~w)
  out[[paste(fe, v, wt, sep = "_")]] <- list(
    coef = unname(coef(m)[c("x1", "x2")]),
    se = unname(se(m)[c("x1", "x2")]))
}
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), "feols_grid_R.json")
