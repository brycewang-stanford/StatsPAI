# StatsPAI CR2 / CR3 cluster-robust SE parity (R side) -- Module 53.
#
# Reads data/53_cr2.csv and runs lm + clubSandwich::vcovCR with
# cluster = countyreal, types CR2 and CR3.
#
# CR2 (Bell-McCaffrey (I - H_gg)^{-1/2}) and analytic CR3
# ((I - H_gg)^{-1}) are both strict headline rows and should match
# StatsPAI to machine precision.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(clubSandwich)
})

MODULE  <- "53_cr2"
FORMULA <- lemp ~ treat + year

df <- read_csv_strict(MODULE)

fit <- lm(FORMULA, data = df)
se2 <- sqrt(diag(vcovCR(fit, cluster = df$countyreal, type = "CR2")))
se3 <- sqrt(diag(vcovCR(fit, cluster = df$countyreal, type = "CR3")))
betas <- coef(fit)

rows <- list()
for (name in names(betas)) {
  beta <- unname(betas[name])
  rows[[length(rows) + 1L]] <- parity_row(
    module = MODULE, statistic = paste0("cr2_", name),
    estimate = beta, se = unname(se2[name]), n = nrow(df)
  )
  rows[[length(rows) + 1L]] <- parity_row(
    module = MODULE, statistic = paste0("cr3_", name),
    estimate = beta, se = unname(se3[name]), n = nrow(df)
  )
}

write_results(MODULE, rows,
              extra = list(formula = deparse(FORMULA),
                           vcov = "CR2 (Bell-McCaffrey) + CR3 (clubSandwich analytic)",
                           cluster_var = "countyreal"))
