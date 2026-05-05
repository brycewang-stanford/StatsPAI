# StatsPAI HDFE cluster-SE parity (R side) -- Module 15.
#
# Reads data/15_hdfe_cluster.csv and runs fixest::feols with cluster
# = firm. Tolerance: rel < 1e-3 (CR1 small-sample correction
# convention may differ slightly).

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

MODULE  <- "15_hdfe_cluster"
FORMULA <- y ~ x1 + x2 | firm + year

df <- read_csv_strict(MODULE)
df$firm <- as.factor(df$firm)
df$year <- as.factor(df$year)

fit <- fixest::feols(FORMULA, data = df, cluster = ~ firm)

co <- coef(fit)
se <- sqrt(diag(vcov(fit)))

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

write_results(MODULE, rows,
              extra = list(formula = deparse(FORMULA),
                           vcov = "cluster",
                           cluster_var = "firm"))
