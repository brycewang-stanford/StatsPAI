# StatsPAI multinomial logit parity (R side) -- Module 44.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else getwd()
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({ library(nnet) })

MODULE <- "44_mlogit"
df <- read_csv_strict(MODULE)
df$y <- factor(df$y)

fit <- nnet::multinom(
  y ~ x1 + x2,
  data = df,
  trace = FALSE,
  maxit = 1000,
  reltol = 1e-12
)

co <- summary(fit)$coefficients     # rows: classes 1, 2; cols: intercept, x1, x2
se <- summary(fit)$standard.errors

rows <- list()
for (cls in c("1", "2")) {
  for (term in c("(Intercept)", "x1", "x2")) {
    label <- if (term == "(Intercept)") "intercept" else term
    rows[[length(rows) + 1L]] <- parity_row(
      MODULE, paste0("class", cls, "_", label),
      estimate = unname(co[cls, term]),
      se = unname(se[cls, term]),
      n = nrow(df))
  }
}

write_results(MODULE, rows, extra = list(base_class = 0,
                                          package = "nnet::multinom",
                                          optim_reltol = 1e-12,
                                          optim_maxit = 1000))
