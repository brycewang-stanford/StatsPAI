# StatsPAI HDFE 2-way parity (R side) -- Module 03.
#
# Reads data/03_hdfe.csv and runs fixest::feols with absorbed firm+year
# fixed effects and IID SE. Tolerance: rel < 1e-6.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(fixest)
})

MODULE  <- "03_hdfe"
FORMULA <- y ~ x1 + x2 | firm + year

df <- read_csv_strict(MODULE)

# fixest expects categorical variables for | firm + year absorption.
df$firm <- as.factor(df$firm)
df$year <- as.factor(df$year)

fit <- fixest::feols(FORMULA, data = df, vcov = "iid")

co <- coef(fit)
se <- sqrt(diag(vcov(fit, vcov = "iid")))

rows <- list()
for (name in c("x1", "x2")) {
  beta <- unname(co[name])
  s    <- unname(se[name])
  rows[[length(rows) + 1L]] <- parity_row(
    module    = MODULE,
    statistic = paste0("beta_", name),
    estimate  = beta,
    se        = s,
    ci_lo     = beta - qnorm(0.975) * s,
    ci_hi     = beta + qnorm(0.975) * s,
    n         = nrow(df)
  )
}

write_results(MODULE, rows, extra = list(formula = deparse(FORMULA), vcov = "iid"))
