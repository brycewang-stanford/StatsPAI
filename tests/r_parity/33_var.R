# StatsPAI VAR parity (R side) -- Module 33.
#
# Reads data/33_var.csv and runs vars::VAR(p=2, type="const").
# Tolerance: rel < 1e-3 on each coefficient.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(vars)
})

MODULE <- "33_var"

df <- read_csv_strict(MODULE)
y_mat <- as.matrix(df[, c("y1", "y2")])

fit <- vars::VAR(y_mat, p = 2L, type = "const")

# vars::VAR returns one lm-style fit per equation in fit$varresult.
rows <- list()
for (eq in c("y1", "y2")) {
  m <- fit$varresult[[eq]]
  co <- coef(m)
  se <- sqrt(diag(vcov(m)))
  for (nm in names(co)) {
    # vars names lags as "y1.l1", "y2.l1", "y1.l2", "y2.l2", "const";
    # remap to sp's naming convention.
    py_name <- nm
    if (nm == "const") py_name <- "_cons"
    else if (grepl("\\.l(\\d+)$", nm)) {
      lag <- sub(".*\\.l(\\d+)$", "\\1", nm)
      var <- sub("\\.l\\d+$", "", nm)
      py_name <- paste0("L", lag, ".", var)
    }
    rows[[length(rows) + 1L]] <- parity_row(
      module = MODULE,
      statistic = paste0("eq_", eq, "__", py_name),
      estimate = unname(co[nm]),
      se = unname(se[nm]),
      n = nrow(df) - 2L
    )
  }
}

# Log-likelihood
rows[[length(rows) + 1L]] <- parity_row(
  MODULE, "logLik",
  estimate = as.numeric(logLik(fit)), n = nrow(df) - 2L
)

write_results(MODULE, rows, extra = list(p = 2, type = "const"))
