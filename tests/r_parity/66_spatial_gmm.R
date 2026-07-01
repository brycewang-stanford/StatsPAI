# StatsPAI spatial GMM (SAR-2SLS / SEM-GMM) parity — Module 66 (R side).
#
# Reads data/66_spatial_gmm.csv, rebuilds the identical row-standardised
# 12x12 rook contiguity from the grid coordinates, and runs the spatialreg
# moment-based references:
#   * SAR 2SLS : spatialreg::stsls(y ~ x1 + x2, W2X = FALSE)  [instruments X, WX]
#   * SEM GMM  : spatialreg::GMerrorsar(y ~ x1 + x2)          [Kelejian-Prucha 1999]
# stsls rows carry the sig2n_k (n-k) SEs; GMerrorsar rows are emitted
# point-only to match the Python side (coefficient-SE conventions differ).

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(spatialreg)
  library(spdep)
})

MODULE <- "66_spatial_gmm"

df <- read_csv_strict(MODULE)
n <- nrow(df)

dr <- abs(outer(df$grid_row, df$grid_row, "-"))
dc <- abs(outer(df$grid_col, df$grid_col, "-"))
Wd <- (dr + dc) == 1
storage.mode(Wd) <- "double"
lw <- spdep::mat2listw(Wd, style = "W")

# SAR spatial two-stage least squares. coef() = c(Rho, (Intercept), x1, x2).
s2sls <- spatialreg::stsls(y ~ x1 + x2, data = df, listw = lw, W2X = FALSE)
s_co <- coef(s2sls)
s_se <- sqrt(diag(s2sls$var))

# SEM generalized-moments error model.
gm <- spatialreg::GMerrorsar(y ~ x1 + x2, data = df, listw = lw)
gm_co <- coef(gm)  # (Intercept), x1, x2

rows <- list(
  # --- SAR 2SLS (point + n-k SE) ---
  parity_row(MODULE, "sar_gmm_const",
             estimate = unname(s_co["(Intercept)"]),
             se = unname(s_se["(Intercept)"]), n = n),
  parity_row(MODULE, "sar_gmm_x1",
             estimate = unname(s_co["x1"]), se = unname(s_se["x1"]), n = n),
  parity_row(MODULE, "sar_gmm_x2",
             estimate = unname(s_co["x2"]), se = unname(s_se["x2"]), n = n),
  parity_row(MODULE, "sar_gmm_rho",
             estimate = unname(s_co["Rho"]), se = unname(s_se["Rho"]), n = n),
  # --- SEM GMM (point-only) ---
  parity_row(MODULE, "sem_gmm_const",
             estimate = unname(gm_co["(Intercept)"]), n = n),
  parity_row(MODULE, "sem_gmm_x1",
             estimate = unname(gm_co["x1"]), n = n),
  parity_row(MODULE, "sem_gmm_x2",
             estimate = unname(gm_co["x2"]), n = n),
  parity_row(MODULE, "sem_gmm_lambda",
             estimate = unname(gm$lambda), n = n)
)

write_results(MODULE, rows, extra = list(
  sar_gmm = "spatialreg::stsls(W2X=FALSE)",
  sem_gmm = "spatialreg::GMerrorsar",
  weights = "rook contiguity, row-standardised (style='W')"
))
