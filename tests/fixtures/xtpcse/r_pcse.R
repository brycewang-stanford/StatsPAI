# R reference for the OLS panel-corrected standard errors of
# tests/test_xtpcse_stata.py (2026-11-05): pcse::pcse on lm(), casewise and
# pairwise, on the CSVs that _data.py writes. pcse has no AR(1), hetonly or
# independent variant, so only correlation(independent) is covered here.
# From this directory: python _data.py . && Rscript r_pcse.R
# (writes r_pcse.json; the CSVs are not kept)
suppressMessages({library(pcse); library(jsonlite)})
out <- list(about = list(R = R.version.string,
                         pcse = as.character(packageVersion("pcse"))))
dump <- function(d, pairwise) {
  m <- lm(y ~ x1 + x2, data = d)
  p <- pcse(m, groupN = d$id, groupT = d$t, pairwise = pairwise)
  b <- coef(m)
  names(b)[names(b) == "(Intercept)"] <- "_cons"
  se <- unname(p$pcse)
  lapply(setNames(seq_along(b), names(b)), function(j) c(unname(b[j]), se[j]))
}
bal <- read.csv("tscs.csv")
unb <- read.csv("tscs_unbal.csv")
out$tscs_default <- dump(bal, FALSE)
out$tscs_unbal_default <- dump(unb, FALSE)
out$tscs_unbal_pairwise <- dump(unb, TRUE)
writeLines(toJSON(out, digits = I(17), auto_unbox = TRUE, pretty = TRUE),
           "r_pcse.json")
