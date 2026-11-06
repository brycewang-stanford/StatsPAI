# R reference for the within and between panel IV cases of
# tests/test_xtivreg_stata.py (2026-11-05): plm's model = "within" and
# model = "between" with instruments, on the CSVs that _data.py writes.
# plm's random-effects IV uses other variance-component formulas than
# Stata's xtivreg, re, so it is not part of this reference.
# From this directory: python _data.py . && Rscript r_plm_xtivreg.R
# (writes r_plm_xtivreg.json; the CSVs are not kept)
suppressMessages({library(plm); library(jsonlite)})
out <- list(about = list(R = R.version.string,
                         plm = as.character(packageVersion("plm"))))
dump <- function(fit, V = vcov(fit)) {
  b <- coef(fit)
  names(b)[names(b) == "(Intercept)"] <- "_cons"
  lapply(setNames(seq_along(b), names(b)), function(j)
    c(unname(b[j]), sqrt(unname(V[j, j]))))
}
one <- pdata.frame(read.csv("panel.csv"), index = c("id", "t"))
two <- pdata.frame(read.csv("two_endog.csv"), index = c("id", "t"))
fe <- plm(y ~ d + x1 | z1 + z2 + x1, data = one, model = "within")
out$fe <- dump(fe)
out$fe_robust <- dump(fe, vcovHC(fe, method = "arellano", type = "sss"))
out$be <- dump(plm(y ~ d + x1 + w | z1 + z2 + x1 + w, data = one, model = "between"))
fe2 <- plm(y ~ d1 + d2 + x1 | z1 + z2 + z3 + x1, data = two, model = "within")
out$two_fe <- dump(fe2)
out$two_fe_robust <- dump(fe2, vcovHC(fe2, method = "arellano", type = "sss"))
out$two_be <- dump(plm(y ~ d1 + d2 + x1 + w | z1 + z2 + z3 + x1 + w,
                       data = two, model = "between"))
writeLines(toJSON(out, digits = I(17), auto_unbox = TRUE, pretty = TRUE),
           "r_plm_xtivreg.json")
