# StatsPAI HC2 / HC3 heteroskedasticity-robust SE parity (R side) -- Module 55.
#
# Reads data/55_hc2_hc3.csv and runs lm + sandwich::vcovHC types HC2 and HC3.
# These are the MacKinnon-White (1985) small-sample heteroskedasticity-robust
# variants; sp.regress(robust="hc2"/"hc3") implements the same adjustments,
# so each row must match to machine precision (module 01 covers HC1).

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(sandwich)
})

MODULE  <- "55_hc2_hc3"
FORMULA <- lemp ~ treat + year

df <- read_csv_strict(MODULE)

fit <- lm(FORMULA, data = df)
betas <- coef(fit)

rows <- list()
for (kind in c("hc2", "hc3")) {
  se <- sqrt(diag(vcovHC(fit, type = toupper(kind))))
  for (name in names(betas)) {
    beta <- unname(betas[name])
    s    <- unname(se[name])
    rows[[length(rows) + 1L]] <- parity_row(
      module    = MODULE,
      statistic = paste0(kind, "_", name),
      estimate  = beta,
      se        = s,
      ci_lo     = beta - qnorm(0.975) * s,
      ci_hi     = beta + qnorm(0.975) * s,
      n         = nrow(df)
    )
  }
}

write_results(MODULE, rows,
              extra = list(formula = deparse(FORMULA),
                           vcov = "HC2 + HC3 (sandwich::vcovHC)"))
