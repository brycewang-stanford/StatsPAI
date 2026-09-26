# R reference for tests/reference_parity/test_ldv_design_stata_parity.py
# (sp.qreg(vce="nid") against quantreg::summary.rq(se = "nid")).
#
# Reads the same bytes as the Stata side: _fixtures/ldv_design_data.csv,
# written by _fixtures/_generate_ldv_design_stata.do.
#
# Run from tests/reference_parity/:  Rscript _generate_qreg_nid_R.R

suppressMessages({
  library(quantreg)
  library(jsonlite)
})

d <- read.csv("_fixtures/ldv_design_data.csv")
out <- list()
for (q in c(25, 50, 75)) {
  fit <- rq(yq ~ x1 + x2, tau = q / 100, data = d)
  s <- summary(fit, se = "nid")
  out[[paste0("nid_", q)]] <- list(
    terms = names(coef(fit)),
    b = unname(coef(fit)),
    se = unname(s$coefficients[, 2]),
    p = unname(s$coefficients[, 4]),
    rdf = s$rdf
  )
}
out[["_meta"]] <- list(
  r_version = R.version.string,
  quantreg = as.character(packageVersion("quantreg")),
  generated = format(Sys.Date())
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "_fixtures/qreg_nid_R.json")
cat("wrote _fixtures/qreg_nid_R.json\n")
