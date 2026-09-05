# StatsPAI PPML+HDFE three-way FE parity (R side) -- Module 47.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else getwd()
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({ library(fixest) })

# Rows are named beta_<x>__fixestK: fixest::fepois's heteroskedasticity-
# robust vcov applies N/(N-K) with K = slopes + absorbed FE levels
# (minus one per additional FE dimension); the Python side emits the
# same-convention rows under this suffix (ssc = "fixest") and keeps the
# Stata ppmlhdfe N/(N-1) rows unsuffixed for the Stata side.
MODULE <- "47_ppmlhdfe_3fe"
df <- read_csv_strict(MODULE)

fit <- fixest::fepois(
  trade ~ log_dist + contig | origin + dest + year,
  data = df,
  vcov = "hetero"
)
co <- coef(fit); se <- sqrt(diag(vcov(fit)))

rows <- list(
  parity_row(MODULE, "beta_log_dist__fixestK", estimate = unname(co["log_dist"]),
             se = unname(se["log_dist"]), n = nrow(df)),
  parity_row(MODULE, "beta_contig__fixestK",   estimate = unname(co["contig"]),
             se = unname(se["contig"]), n = nrow(df))
)
write_results(MODULE, rows, extra = list(fe = "origin + dest + year",
                                          vcov = "HC1",
                                          package = "fixest::fepois"))
