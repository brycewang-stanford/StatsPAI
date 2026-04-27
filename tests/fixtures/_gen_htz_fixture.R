# Regenerate clubSandwich HTZ fixture for tests/test_fast_htz.py.
# DO NOT run in CI — this is a developer-side script. Outputs are
# committed to git; tests read them directly.
#
# Usage:
#   cd tests/fixtures && Rscript _gen_htz_fixture.R
#
# Requires: R >= 4.0, clubSandwich, jsonlite, data.table.

suppressMessages({
  library(clubSandwich)
  library(jsonlite)
  library(data.table)
})

set.seed(20260427L)

# ---------------------------------------------------------------------------
# Helper: simulate a clustered OLS panel.
# ---------------------------------------------------------------------------

simulate_panel <- function(G, m, beta = c(0.30, -0.20), unbalanced = FALSE) {
  rows <- list()
  for (g in seq_len(G)) {
    n_g <- if (unbalanced) sample(3:50, 1L) else m
    x1 <- rnorm(n_g)
    x2 <- rnorm(n_g)
    u_g <- rnorm(1L, sd = 0.5)
    eps <- rnorm(n_g) + u_g
    y <- beta[1] * x1 + beta[2] * x2 + eps
    rows[[g]] <- data.table(g = g, x1 = x1, x2 = x2, y = y)
  }
  rbindlist(rows)
}

# ---------------------------------------------------------------------------
# Build three panels: q=1 / q=2 / q=3-unbalanced.
# x1 + x2 + intercept ⇒ k = 3 ⇒ supports up to q = 3.
# ---------------------------------------------------------------------------

panels <- list(
  list(name = "q1",         G = 15, m = 20, q = 1, unbalanced = FALSE),
  list(name = "q2",         G = 25, m = 15, q = 2, unbalanced = FALSE),
  list(name = "q3_unbal",   G = 50, m = NA, q = 3, unbalanced = TRUE)
)

results <- list()

for (cfg in panels) {
  set.seed(20260427L + nchar(cfg$name))   # distinct but deterministic
  d <- simulate_panel(cfg$G, cfg$m, unbalanced = cfg$unbalanced)
  csv_path <- paste0("htz_panel_", cfg$name, ".csv")
  fwrite(d, csv_path)

  fit <- lm(y ~ x1 + x2, data = d)         # k = 3 (intercept + x1 + x2)
  V_CR2 <- vcovCR(fit, cluster = d$g, type = "CR2")

  # Restriction matrix: q rows × k=3 cols. Test the *non-intercept* coefs.
  if (cfg$q == 1) {
    R <- matrix(c(0, 1, 0), nrow = 1)        # H0: β_x1 = 0
  } else if (cfg$q == 2) {
    R <- rbind(c(0, 1, 0), c(0, 0, 1))      # H0: β_x1 = β_x2 = 0
  } else if (cfg$q == 3) {
    # q=3 with k=3 ⇒ joint test of all 3 coefs (intercept + x1 + x2).
    R <- diag(3)
  }

  wt <- Wald_test(fit, constraints = R, vcov = V_CR2, test = "HTZ")

  # wt is a 1-row data.frame: $Fstat, $df_num, $df_denom, $p_val.
  # Note: wt$df_denom = nu_Z - q + 1 (the F-test denom DOF), NOT the
  # moment-matching DOF nu_Z itself. We store nu_Z under `eta` and
  # df_denom under `df_denom` so Python tests can compare the right
  # quantity (Python's WaldTestResult.eta is nu_Z, per spec §2.1).
  V_R_mat <- as.matrix(R %*% V_CR2 %*% t(R))
  q_int <- as.integer(wt$df_num)

  # Re-derive nu_Z from clubSandwich internals (cleanest, no float drift):
  GH <- clubSandwich:::get_GH(fit, V_CR2)
  GH$G <- lapply(GH$G, function(s) R %*% s)
  GH$H <- clubSandwich:::array_multiply(R, GH$H)
  P_array <- clubSandwich:::get_P_array(GH = GH, all_terms = TRUE)
  Omega <- apply(P_array, 1:2, function(x) sum(diag(x)))
  Omega_nsqrt <- clubSandwich:::matrix_power(Omega, -1/2)
  Var_mat <- clubSandwich:::total_variance_mat(P_array, Omega_nsqrt, q = q_int)
  nu_Z <- q_int * (q_int + 1) / sum(Var_mat)

  # Q (raw Wald) = F · q · df_denom / 1, since F = delta · Q / q with
  # delta = df_denom / nu_Z, so Q = F · q · nu_Z / df_denom.
  Q_val <- wt$Fstat * q_int * nu_Z / wt$df_denom

  results[[cfg$name]] <- list(
    panel = cfg$name,
    csv = csv_path,
    G = cfg$G,
    q = q_int,
    R = unname(as.matrix(R)),
    beta = unname(coef(fit)),
    eta = nu_Z,                       # moment-matching DOF (matches spec)
    df_denom = wt$df_denom,           # = nu_Z - q + 1 (R's Wald_test field)
    F_stat = wt$Fstat,
    p_value = wt$p_val,
    Q = Q_val,
    V_R = unname(V_R_mat)
  )
}

write(toJSON(results, auto_unbox = TRUE, digits = 14, pretty = TRUE),
      "htz_clubsandwich.json")

cat("Generated:\n")
for (cfg in panels) {
  cat("  htz_panel_", cfg$name, ".csv\n", sep = "")
}
cat("  htz_clubsandwich.json\n")

# Pinned versions
cat("\nVersions:\n")
cat("  R: ", R.version$major, ".", R.version$minor, "\n", sep = "")
cat("  clubSandwich: ", as.character(packageVersion("clubSandwich")), "\n", sep = "")
