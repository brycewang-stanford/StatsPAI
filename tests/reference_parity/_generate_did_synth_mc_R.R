# Reference values for tests/reference_parity/test_did_synth_mc_parity.py
#
#   sp.mc_panel / sp.mc_synth (nuclear-norm matrix completion, fixed lambda)
#     vs  MCPanel::mcnnm_fit      (Athey, Bayati, Doudchenko, Imbens & Khosravi;
#                                  github.com/susanathey/MCPanel)
#     and fect::fect(method = "mc", CV = FALSE)   (gsynth(estimator = "mc")
#                                  1.4.0 is a thin wrapper that calls this)
#
# Regenerate (from the repository root):
#   python  tests/reference_parity/_generate_did_synth_mc_data.py   # panel
#   Rscript tests/reference_parity/_generate_did_synth_mc_R.R
#
# MCPanel install note: the GitHub HEAD bundles a 2017 copy of Eigen in src/Eigen
# that no longer compiles with current clang. Install it with that directory
# removed so RcppEigen's Eigen is used; nothing else is touched:
#   git clone https://github.com/susanathey/MCPanel && rm -rf MCPanel/src/Eigen
#   R CMD INSTALL MCPanel
# The SHA recorded below is read from that checkout (MCPANEL_SHA env var).
#
# Lambda scales (one minimiser, three parameterisations; see
# src/statspai/matrix_completion/_core.py):
#   StatsPAI theta (threshold on 1/2 ||P_O(Y-F)||^2 + theta ||L||_*)
#   MCPanel  lambda_L = 2 theta / |O|
#   fect     lambda   = theta / (N T)
# Convergence: both references stop on a loose relative criterion by default
# (MCPanel rel_tol 1e-5 on the objective, fect tol 1e-3 on the fit), which
# leaves ~1e-5 relative error in the ATT on this panel. The fixture must hold
# the minimiser, not a stopping-rule artefact, so:
#   * MCPanel runs with rel_tol = 0, i.e. exactly `niter` coordinate-descent
#     sweeps at every lambda of its warm-start path (its stopping test is
#     `0 <= rel_improvement < rel_tol`, never true at 0). Stopping on the
#     objective at 1e-15 instead leaves ~1e-9 in the ATT (objective change is
#     quadratic in the parameter error); 2000 vs 20000 sweeps agree to 1e-15.
#   * fect runs with tol = 1e-15 on the relative change of the fit.

suppressMessages({
  library(MCPanel)
  library(fect)
  library(gsynth)
  library(jsonlite)
})

script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) "tests/reference_parity"
)
fixtures <- file.path(script_dir, "_fixtures")
d <- read.csv(file.path(fixtures, "did_synth_mc_panel.csv"))
d <- d[order(d$unit, d$time), ]
units <- sort(unique(d$unit))
times <- sort(unique(d$time))
N <- length(units)
TT <- length(times)
Y <- matrix(d$y, nrow = N, ncol = TT, byrow = TRUE)
D <- matrix(d$d, nrow = N, ncol = TT, byrow = TRUE)

THETAS <- c(8, 20)
MC_NITER <- 3000L
MC_RTOL <- 0
FECT_MAXIT <- 200000L
FECT_TOL <- 1e-15

mcpanel_case <- function(mask, treated, theta, u, v) {
  nobs <- sum(mask)
  lam <- 2 * theta / nobs
  fit <- mcnnm_fit(Y, mask, lambda_L = lam, to_estimate_u = u,
                   to_estimate_v = v, niter = MC_NITER, rel_tol = MC_RTOL)
  Fhat <- fit$L + outer(fit$u, rep(1, TT)) + outer(rep(1, N), fit$v)
  list(
    theta = theta,
    lambda_L = lam,
    n_obs = nobs,
    att = mean((Y - Fhat)[treated]),
    fit = Fhat,
    L = fit$L,
    L_singular_values = svd(fit$L)$d
  )
}

fect_case <- function(dd, theta) {
  lam <- theta / (N * TT)
  f <- fect(y ~ d, data = dd, index = c("unit", "time"), method = "mc",
            force = "two-way", CV = FALSE, lambda = lam, se = FALSE,
            tol = FECT_TOL, max.iteration = FECT_MAXIT, min.T0 = 1, parallel = FALSE)
  # Y.ct is (T x N); transpose to the N x T layout used everywhere else.
  list(theta = theta, lambda = lam, att = f$att.avg,
       fit = t(f$Y.ct), eff_sum = sum(f$eff * f$D.dat, na.rm = TRUE))
}

out <- list()

# ------------------------------------------------------ sp.mc_panel design --
mask_p <- (D == 0) * 1
treated_p <- D == 1
out$panel <- list()
for (th in THETAS) {
  key <- paste0("theta_", th)
  out$panel[[key]] <- list(
    twoway = mcpanel_case(mask_p, treated_p, th, 1, 1),
    none = mcpanel_case(mask_p, treated_p, th, 0, 0),
    unit = mcpanel_case(mask_p, treated_p, th, 1, 0),
    time = mcpanel_case(mask_p, treated_p, th, 0, 1),
    fect_twoway = fect_case(d, th)
  )
}
# gsynth(estimator = "mc") delegates to fect::fect; record one point to show it.
g <- gsynth(y ~ d, data = d, index = c("unit", "time"), estimator = "mc",
            force = "two-way", CV = FALSE, lambda = THETAS[1] / (N * TT),
            se = FALSE, tol = FECT_TOL, min.T0 = 1, parallel = FALSE)
out$panel$gsynth_theta_8_att <- g$att.avg

# ------------------------------------------------------ sp.mc_synth design --
# Treated unit 40 from period 21: only its post cells are masked.
SYN_UNIT <- 40
SYN_T0 <- 21
mask_s <- matrix(1, N, TT)
mask_s[which(units == SYN_UNIT), times >= SYN_T0] <- 0
treated_s <- mask_s == 0
ds <- d
ds$d <- as.integer(ds$unit == SYN_UNIT & ds$time >= SYN_T0)
out$synth <- list(treated_unit = SYN_UNIT, treatment_time = SYN_T0)
for (th in THETAS) {
  key <- paste0("theta_", th)
  out$synth[[key]] <- list(
    twoway = mcpanel_case(mask_s, treated_s, th, 1, 1),
    none = mcpanel_case(mask_s, treated_s, th, 0, 0),
    fect_twoway = fect_case(ds, th)
  )
}

out$meta <- list(
  r_version = R.version.string,
  MCPanel = as.character(packageVersion("MCPanel")),
  MCPanel_sha = Sys.getenv("MCPANEL_SHA", "unknown"),
  MCPanel_build_note = "src/Eigen removed before R CMD INSTALL (RcppEigen headers used)",
  RcppEigen = as.character(packageVersion("RcppEigen")),
  fect = as.character(packageVersion("fect")),
  gsynth = as.character(packageVersion("gsynth")),
  MCPanel_niter = MC_NITER,
  MCPanel_rel_tol = MC_RTOL,
  fect_tol = FECT_TOL,
  fect_max_iteration = FECT_MAXIT,
  N = N, T = TT,
  thetas = THETAS
)

writeLines(
  toJSON(out, digits = I(17), auto_unbox = TRUE, matrix = "rowmajor"),
  file.path(fixtures, "did_synth_mc_R.json")
)
cat("wrote", file.path(fixtures, "did_synth_mc_R.json"), "\n")
