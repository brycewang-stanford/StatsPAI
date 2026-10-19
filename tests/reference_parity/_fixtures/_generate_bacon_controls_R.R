# Reference numbers for sp.bacon_decomposition(covariates=) from R
# bacondecomp 0.1.1 (MIT).
#
#   python tests/reference_parity/_fixtures/_generate_bacon_controls_data.py
#   Rscript tests/reference_parity/_fixtures/_generate_bacon_controls_R.R
suppressMessages({
  library(bacondecomp)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
d <- read.csv(file.path(here, "bacon_controls.csv"))
one <- function(f) {
  r <- bacon(f, data = d, id_var = "id", time_var = "t", quietly = TRUE)
  twfe <- unname(lm(update(f, . ~ . + factor(t) + factor(id)), data = d)$coefficients["d"])
  list(
    Omega = r$Omega,
    beta_hat_w = r$beta_hat_w,
    twfe = twfe,
    two_by_twos = r$two_by_twos
  )
}
out <- list(
  meta = list(bacondecomp = as.character(packageVersion("bacondecomp")),
              R = R.version.string),
  x1 = one(y ~ d + x1),
  x1x2 = one(y ~ d + x1 + x2)
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, dataframe = "rows"),
           file.path(here, "bacon_controls_R.json"))
