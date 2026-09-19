#!/usr/bin/env Rscript
# Ground truth for tests/reference_parity/test_rd_iv_R_parity.py (IV block
# of the rd_iv parity family).
#
# Writes, under tests/reference_parity/_fixtures/:
#   rd_iv_ivw.csv    synthetic weak-IV design (n = 800, 40 clusters, one
#                    endogenous regressor, three instruments, two controls,
#                    heteroskedastic and clustered errors); seed below
#   rd_iv_rueda.csv  ivDiag::rueda (Rueda 2017 vote-buying application
#                    shipped with ivDiag), written verbatim
#   rd_iv_R.json     reference values at full double precision
#
# Re-run only when the contract changes:
#   Rscript tests/reference_parity/_generate_rd_iv_R.R
#
# References exercised
#   ivDiag 1.0.6  tF()      -- LMMP (2022) tF critical value table
#                 eff_F()   -- Olea-Pflueger effective F (lfe::felm vcov)
#                 ivDiag()  -- 2SLS / OLS analytic SEs, first-stage F's
#   ivmodel 1.9.1 CLR()     -- Moreira CLR statistic (closed form)
#                 AR.test() -- homoskedastic Anderson-Rubin F
#   quantreg      rq()      -- the Chernozhukov-Hansen inverse-QR estimate,
#                              computed as the root of the instrument's QR
#                              coefficient (writes rd_iv_ivqr.csv). IVQR
#                              0.1.0, the dedicated package, stops with
#                              "the condition has length > 1" on R 4.5.
#
# Every ivDiag call is made with prec = 16: ivDiag rounds its outputs to
# `prec` digits (default 4), and ivDiag() itself calls eff_F() and tF()
# without forwarding `prec`, so the tF it reports is evaluated at an F
# rounded to 4 decimals. The fixture therefore re-evaluates tF() at the
# full-precision effective F rather than storing ivDiag()$tF.

suppressMessages({
  library(ivDiag)
  library(ivmodel)
  library(quantreg)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("--file=", "", grep("--file=", args, value = TRUE)))
FIX <- file.path(here, "_fixtures")
if (!dir.exists(FIX)) FIX <- "tests/reference_parity/_fixtures"

# ---- data ------------------------------------------------------------------
set.seed(20260918)
n <- 800; G <- 40
cl <- rep(1:G, each = n / G)
x1 <- rnorm(n) + 0.3 * rnorm(G)[cl]
x2 <- runif(n)
z1 <- rnorm(n) + 0.5 * x1
z2 <- rnorm(n)
z3 <- rbinom(n, 1, 0.4)
u <- rnorm(n) * (1 + 0.5 * abs(x2)) + 0.4 * rnorm(G)[cl]
v <- 0.6 * u + rnorm(n) * (0.5 + abs(z2) * 0.5)
d <- 0.12 * z1 + 0.08 * z2 + 0.15 * z3 + 0.5 * x1 - 0.3 * x2 + v
y <- 1 + 0.5 * d + 0.3 * x1 + 0.2 * x2 + u
ivw <- data.frame(y, d, z1, z2, z3, x1, x2, cl)
write.csv(ivw, file.path(FIX, "rd_iv_ivw.csv"), row.names = FALSE)
ivw <- read.csv(file.path(FIX, "rd_iv_ivw.csv"))  # the committed bytes

write.csv(rueda, file.path(FIX, "rd_iv_rueda.csv"), row.names = FALSE)
rue <- read.csv(file.path(FIX, "rd_iv_rueda.csv"))

# ---- tF table ----------------------------------------------------------------
F_grid <- c(4, 4.2, 4.5, 5, 6, 7.3, 8, 9, 10, 12.5, 15, 16.38, 20, 25, 30,
            40, 50, 64, 75, 90, 100, 104.7, 106.09, 110, 250, 1e4)
tF_grid <- sapply(F_grid, function(f) unname(tF(1, 1, f, prec = 16)["cF"]))

# ---- designs ---------------------------------------------------------------
designs <- list(
  rueda_hc = list(data = rue, Y = "e_vote_buying", D = "lm_pob_mesa",
                  Z = "lz_pob_mesa_f", X = c("lpopulation", "lpotencial"),
                  cl = NULL),
  rueda_cl = list(data = rue, Y = "e_vote_buying", D = "lm_pob_mesa",
                  Z = "lz_pob_mesa_f", X = c("lpopulation", "lpotencial"),
                  cl = "muni_code"),
  ivw1_hc = list(data = ivw, Y = "y", D = "d", Z = "z1",
                 X = c("x1", "x2"), cl = NULL),
  ivw1_cl = list(data = ivw, Y = "y", D = "d", Z = "z1",
                 X = c("x1", "x2"), cl = "cl"),
  ivw3_hc = list(data = ivw, Y = "y", D = "d", Z = c("z1", "z2", "z3"),
                 X = c("x1", "x2"), cl = NULL),
  ivw3_cl = list(data = ivw, Y = "y", D = "d", Z = c("z1", "z2", "z3"),
                 X = c("x1", "x2"), cl = "cl")
)

ivdiag_out <- list()
for (nm in names(designs)) {
  s <- designs[[nm]]
  g <- ivDiag(s$data, Y = s$Y, D = s$D, Z = s$Z, controls = s$X, cl = s$cl,
              bootstrap = FALSE, run.AR = FALSE, prec = 16, parallel = FALSE)
  effF <- eff_F(s$data, Y = s$Y, D = s$D, Z = s$Z, controls = s$X,
                cl = s$cl, prec = 16)
  rec <- list(
    beta_2sls = unname(g$est_2sls[1, "Coef"]),
    se_2sls = unname(g$est_2sls[1, "SE"]),
    beta_ols = unname(g$est_ols[1, "Coef"]),
    se_ols = unname(g$est_ols[1, "SE"]),
    F_standard = unname(g$F_stat["F.standard"]),
    F_robust = unname(g$F_stat["F.robust"]),
    F_cluster = if (is.null(s$cl)) NULL else unname(g$F_stat["F.cluster"]),
    eff_F = effF,
    n_instruments = length(s$Z)
  )
  if (length(s$Z) == 1) {
    t <- tF(g$est_2sls[1, "Coef"], g$est_2sls[1, "SE"], effF, prec = 16)
    rec$tF_cF <- unname(t["cF"])
    rec$tF_ci <- unname(t[c("CI2.5%", "CI97.5%")])
  }
  ivdiag_out[[nm]] <- rec
}

# ---- ivmodel CLR / AR (homoskedastic, k = 3) --------------------------------
m <- ivmodel(Y = ivw$y, D = ivw$d, Z = as.matrix(ivw[, c("z1", "z2", "z3")]),
             X = as.matrix(ivw[, c("x1", "x2")]))
clr0 <- CLR(m, beta0 = 0)
clr1 <- CLR(m, beta0 = 0.5)
ar0 <- AR.test(m, beta0 = 0)

# ---- IV quantile regression: root of the inverse-QR equation --------------
set.seed(5)
nq <- 1500
zq <- rnorm(nq); x1q <- rnorm(nq); uq <- rnorm(nq)
dq <- 0.8 * zq + 0.3 * x1q + 0.7 * uq + rnorm(nq, 0, 0.5)
yq <- 1 + 1.0 * dq + 0.5 * x1q + 0.9 * uq + rnorm(nq, 0, 0.5)
write.csv(data.frame(y = yq, d = dq, z = zq, x1 = x1q),
          file.path(FIX, "rd_iv_ivqr.csv"), row.names = FALSE)
ivq <- read.csv(file.path(FIX, "rd_iv_ivqr.csv"))
bz <- function(a, tau) coef(rq(I(y - a * d) ~ z + x1, tau = tau, data = ivq))["z"]
ivqr_out <- list()
for (tau in c(0.25, 0.5, 0.75)) {
  g <- seq(0.9, 1.2, by = 0.0005)
  bv <- sapply(g, bz, tau = tau)
  ch <- which(diff(sign(bv)) != 0)
  root <- uniroot(function(a) bz(a, tau), c(g[ch[1]], g[ch[1] + 1]),
                  tol = 1e-13)$root
  ivqr_out[[as.character(tau)]] <- list(root = root, n_sign_changes = length(ch))
}

ref <- list(
  meta = list(
    R_version = R.version.string,
    ivDiag_version = as.character(packageVersion("ivDiag")),
    ivmodel_version = as.character(packageVersion("ivmodel")),
    lfe_version = as.character(packageVersion("lfe")),
    quantreg_version = as.character(packageVersion("quantreg")),
    seed = 20260918L,
    generated = "deterministic; re-run only on contract change"
  ),
  tF_grid = list(F = F_grid, cF = tF_grid),
  ivdiag = ivdiag_out,
  ivqr_root = ivqr_out,
  ivmodel_ivw3 = list(
    clr_stat_b0 = as.numeric(clr0$test.stat),
    clr_p_b0 = as.numeric(clr0$p.value),
    clr_stat_b05 = as.numeric(clr1$test.stat),
    clr_p_b05 = as.numeric(clr1$p.value),
    clr_ci = as.numeric(clr0$ci),
    ar_F_b0 = as.numeric(ar0$Fstat)
  )
)
writeLines(toJSON(ref, digits = I(17), auto_unbox = TRUE, pretty = TRUE,
                  null = "null"),
           file.path(FIX, "rd_iv_R.json"))
cat("wrote rd_iv_ivw.csv, rd_iv_rueda.csv, rd_iv_ivqr.csv, rd_iv_R.json\n")
