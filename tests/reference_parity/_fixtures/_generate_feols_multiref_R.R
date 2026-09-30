# fixest's i(rel, ref = c(-1, -5)) -- two reference levels -- with unit and
# period effects, clustered by unit, for sp.feols("... i(rel, ref=[-1, -5])").
#   Rscript tests/reference_parity/_fixtures/_generate_feols_multiref_R.R
suppressMessages(library(fixest))
suppressMessages(library(jsonlite))
d <- read.csv("tests/reference_parity/_fixtures/feols_multiref.csv")
m <- feols(y ~ i(rel, ref = c(-1, -5)) | u + t, data = d, cluster = ~u)
out <- list(
  fixest_version = as.character(packageVersion("fixest")),
  names = names(coef(m)),
  b = unname(coef(m)),
  se = unname(se(m))
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE),
           "tests/reference_parity/_fixtures/feols_multiref_R.json")
