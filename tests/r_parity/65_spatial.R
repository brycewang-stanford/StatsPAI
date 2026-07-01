# StatsPAI spatial ML (SAR / SEM / SDM) parity — Module 65 (R side).
#
# Reads data/65_spatial.csv (written by 65_spatial.py), rebuilds the
# identical 12x12 rook-contiguity weights from the grid coordinates,
# row-standardises them (spdep style = "W"), and runs the canonical
# spatialreg maximum-likelihood references:
#   * SAR : spatialreg::lagsarlm(y ~ x1 + x2)
#   * SEM : spatialreg::errorsarlm(y ~ x1 + x2)
#   * SDM : spatialreg::lagsarlm(y ~ x1 + x2, Durbin = TRUE)
# Emits every coefficient with its asymptotic SE so compare.py grades
# both the point estimates and the full-information standard errors.

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
  library(Matrix)
})

MODULE <- "65_spatial"

df <- read_csv_strict(MODULE)
n <- nrow(df)

# Rebuild the rook contiguity from the integer grid coordinates so the
# R weights are byte-identical to the Python side, then row-standardise.
dr <- abs(outer(df$grid_row, df$grid_row, "-"))
dc <- abs(outer(df$grid_col, df$grid_col, "-"))
Wd <- (dr + dc) == 1
storage.mode(Wd) <- "double"
lw <- spdep::mat2listw(Wd, style = "W")

sar <- spatialreg::lagsarlm(y ~ x1 + x2, data = df, listw = lw)
sem <- spatialreg::errorsarlm(y ~ x1 + x2, data = df, listw = lw)
sdm <- spatialreg::lagsarlm(y ~ x1 + x2, data = df, listw = lw, Durbin = TRUE)

sar_co <- summary(sar)$Coef  # rows: (Intercept), x1, x2 (Estimate, Std. Error)
sem_co <- summary(sem)$Coef
sdm_co <- summary(sdm)$Coef  # rows: (Intercept), x1, x2, lag.x1, lag.x2

rows <- list(
  # --- SAR ---
  parity_row(MODULE, "sar_const",
             estimate = sar_co["(Intercept)", "Estimate"],
             se = sar_co["(Intercept)", "Std. Error"], n = n),
  parity_row(MODULE, "sar_x1",
             estimate = sar_co["x1", "Estimate"],
             se = sar_co["x1", "Std. Error"], n = n),
  parity_row(MODULE, "sar_x2",
             estimate = sar_co["x2", "Estimate"],
             se = sar_co["x2", "Std. Error"], n = n),
  parity_row(MODULE, "sar_rho",
             estimate = unname(sar$rho), se = unname(sar$rho.se), n = n),
  # --- SEM ---
  parity_row(MODULE, "sem_const",
             estimate = sem_co["(Intercept)", "Estimate"],
             se = sem_co["(Intercept)", "Std. Error"], n = n),
  parity_row(MODULE, "sem_x1",
             estimate = sem_co["x1", "Estimate"],
             se = sem_co["x1", "Std. Error"], n = n),
  parity_row(MODULE, "sem_x2",
             estimate = sem_co["x2", "Estimate"],
             se = sem_co["x2", "Std. Error"], n = n),
  parity_row(MODULE, "sem_lambda",
             estimate = unname(sem$lambda), se = unname(sem$lambda.se), n = n),
  # --- SDM ---
  parity_row(MODULE, "sdm_const",
             estimate = sdm_co["(Intercept)", "Estimate"],
             se = sdm_co["(Intercept)", "Std. Error"], n = n),
  parity_row(MODULE, "sdm_x1",
             estimate = sdm_co["x1", "Estimate"],
             se = sdm_co["x1", "Std. Error"], n = n),
  parity_row(MODULE, "sdm_x2",
             estimate = sdm_co["x2", "Estimate"],
             se = sdm_co["x2", "Std. Error"], n = n),
  parity_row(MODULE, "sdm_W_x1",
             estimate = sdm_co["lag.x1", "Estimate"],
             se = sdm_co["lag.x1", "Std. Error"], n = n),
  parity_row(MODULE, "sdm_W_x2",
             estimate = sdm_co["lag.x2", "Estimate"],
             se = sdm_co["lag.x2", "Std. Error"], n = n),
  parity_row(MODULE, "sdm_rho",
             estimate = unname(sdm$rho), se = unname(sdm$rho.se), n = n)
)

write_results(MODULE, rows, extra = list(
  sar = "spatialreg::lagsarlm",
  sem = "spatialreg::errorsarlm",
  sdm = "spatialreg::lagsarlm(Durbin=TRUE)",
  weights = "rook contiguity, row-standardised (style='W')"
))
