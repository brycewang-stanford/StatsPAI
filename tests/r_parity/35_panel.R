# StatsPAI panel FE / RE / Hausman parity (R side) -- Module 35.
#
# Reads data/35_panel.csv and runs plm::plm + plm::phtest.
# Tolerance: rel < 1e-3 on FE coefficients.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(plm)
})

MODULE <- "35_panel"

df <- read_csv_strict(MODULE)
pdf_obj <- plm::pdata.frame(df, index = c("unit", "year"))

fit_fe <- plm::plm(y ~ x1 + x2, data = pdf_obj, model = "within")
fit_re <- plm::plm(y ~ x1 + x2, data = pdf_obj, model = "random")
haus   <- plm::phtest(fit_fe, fit_re)

co_fe <- coef(fit_fe)
se_fe <- sqrt(diag(vcov(fit_fe)))
co_re <- coef(fit_re)
se_re <- sqrt(diag(vcov(fit_re)))

rows <- list(
  parity_row(MODULE, "fe_beta_x1",
             estimate = unname(co_fe["x1"]),
             se = unname(se_fe["x1"]), n = nrow(df)),
  parity_row(MODULE, "fe_beta_x2",
             estimate = unname(co_fe["x2"]),
             se = unname(se_fe["x2"]), n = nrow(df)),
  parity_row(MODULE, "re_beta_x1",
             estimate = unname(co_re["x1"]),
             se = unname(se_re["x1"]), n = nrow(df)),
  parity_row(MODULE, "re_beta_x2",
             estimate = unname(co_re["x2"]),
             se = unname(se_re["x2"]), n = nrow(df)),
  parity_row(MODULE, "hausman_chi2",
             estimate = unname(haus$statistic), n = nrow(df)),
  parity_row(MODULE, "hausman_pvalue",
             estimate = unname(haus$p.value), n = nrow(df))
)

write_results(MODULE, rows, extra = list(method_fe = "within",
                                          method_re = "random"))
