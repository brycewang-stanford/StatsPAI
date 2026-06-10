# StatsPAI zero-inflated Poisson parity (R side) -- Module 63.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else getwd()
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({ library(pscl) })

MODULE <- "63_zip"
df <- read_csv_strict(MODULE)

fit <- zeroinfl(y ~ x1 + x2 | z, data = df, dist = "poisson", link = "logit",
                control = zeroinfl.control(reltol = 1e-14, maxit = 10000))
co <- coef(fit); se <- sqrt(diag(vcov(fit)))

lab <- c("count_(Intercept)" = "beta_count_intercept",
         "count_x1" = "beta_count_x1",
         "count_x2" = "beta_count_x2",
         "zero_(Intercept)" = "beta_inflate_intercept",
         "zero_z" = "beta_inflate_z")
rows <- lapply(names(lab), function(k) {
  parity_row(MODULE, unname(lab[k]), estimate = unname(co[k]), se = unname(se[k]), n = nrow(df))
})
write_results(MODULE, rows,
              extra = list(package = "pscl::zeroinfl",
                           count_dist = "poisson", inflate_link = "logit"))
