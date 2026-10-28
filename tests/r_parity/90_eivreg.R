# StatsPAI errors-in-variables regression parity (R side) -- Module 90.
#
# Reads data/90_eivreg.csv and runs eivtools::eivreg with known
# reliabilities, unweighted and with analytic weights.
#
# eivtools (J.R. Lockwood, GPL >= 2) was written to reproduce Stata's
# eivreg; it was archived on CRAN on 2025-03-24 (maintainer address
# undeliverable), so it is installed from the CRAN archive into a private
# library -- see tests/r_parity/R_ENVIRONMENT.md, "Module 90". Its DESCRIPTION
# lists R2jags, which needs the JAGS system library and is used only by
# lr_ancova(); the private install drops that one dependency and leaves
# every .R file byte-identical (eivreg.R SHA-256 0e14b0cc...526559dd4c).
# Point the harness at the library with STATSPAI_EIV_LIB.
#
# The point estimates and the M-estimation sandwich are the same
# quantities as Stata's (module 90's Python docstring); df_adj = FALSE is
# the default, i.e. no small-sample factor, as in Stata.
#
# Tolerance: rel < 1e-6.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

.eiv_lib <- Sys.getenv("STATSPAI_EIV_LIB")
if (nzchar(.eiv_lib)) .libPaths(c(.eiv_lib, .libPaths()))
if (!requireNamespace("eivtools", quietly = TRUE)) {
  stop(paste(
    "eivtools is not installed (archived on CRAN 2025-03-24). Install",
    "0.1-8 from the CRAN archive into a private library as described in",
    "tests/r_parity/R_ENVIRONMENT.md and point STATSPAI_EIV_LIB at it."
  ))
}
stopifnot(as.character(utils::packageVersion("eivtools")) == "0.1.8")
suppressPackageStartupMessages(library(eivtools))

MODULE <- "90_eivreg"

df <- read_csv_strict(MODULE)

REL <- c(x1 = 0.8, x2 = 0.7)
TERMS <- c(x1 = "x1", x2 = "x2", z = "z", cons = "(Intercept)")

rows <- list()
for (design in c("unw", "aw")) {
  fit <- if (design == "unw") {
    eivtools::eivreg(y ~ x1 + x2 + z, data = df, reliability = REL)
  } else {
    eivtools::eivreg(y ~ x1 + x2 + z, data = df, weights = w, reliability = REL)
  }
  b <- coef(fit)
  se <- sqrt(diag(fit$vcov))
  for (nm in names(TERMS)) {
    term <- TERMS[[nm]]
    rows[[length(rows) + 1L]] <- parity_row(
      module    = MODULE,
      statistic = sprintf("%s_b_%s", design, nm),
      estimate  = unname(b[term]),
      se        = unname(se[term]),
      n         = nrow(df)
    )
  }
}

write_results(
  MODULE, rows,
  extra = list(
    reliability = as.list(REL),
    n_rows = length(rows),
    eivtools_version = as.character(utils::packageVersion("eivtools")),
    eivtools_source = paste(
      "CRAN archive eivtools_0.1-8.tar.gz (SHA-256",
      "64de6a6dfd387a0ee3981be36f4759cadcbd28ffb5a8c5d9a920ec6628ced57a);",
      "R2jags dropped from Depends / NAMESPACE, R sources unchanged"
    )
  )
)
