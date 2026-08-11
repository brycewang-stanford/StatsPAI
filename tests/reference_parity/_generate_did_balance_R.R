## Reference values for sp.did_balance.
##
## The Imbens-Rubin normalized difference is an analytic formula, not a
## package-specific algorithm, so this generator pins it TWICE from R:
##
##   1. cobalt::col_w_smd(..., std = TRUE, s.d.denom = "pooled"), a
##      widely used independent implementation; and
##   2. a direct transcription of the formula
##      (Xbar_T - Xbar_C) / sqrt((S2_T + S2_C)/2),
##      written here so the check does not inherit any single package's
##      convention.
##
## Agreement between (1) and (2) is itself part of the fixture: if they
## disagree, the convention is ambiguous and the Python side should not
## be pinned to either without saying which.
##
## The weighted arm uses the reliability-weight variance correction
##   sum(w (x - xbar)^2) / (sum w - sum(w^2)/sum w),
## which reduces to the usual ddof=1 sample variance at w = 1.

suppressMessages(library(cobalt))

set.seed(20260810)

n_units <- 400
Tt <- 8
base_period <- 3
comparison_period <- 4

rows <- list()
k <- 1
for (i in 1:n_units) {
  g <- sample(c(4, 0), 1)
  x0 <- rnorm(1, 10, 2)
  z0 <- rnorm(1, 50, 8) + if (g > 0) 3 else 0   # imbalanced in levels
  w <- rgamma(1, 2, 1) * 100
  drift <- if (g > 0) -1.2 else -0.2            # imbalanced in changes
  for (t in 1:Tt) {
    shift <- if (t <= base_period) 0 else drift * (t - base_period)
    rows[[k]] <- data.frame(
      i = i, t = t, g = g, w = w,
      x = x0 + shift + rnorm(1, 0, 0.3),
      z = z0 + rnorm(1, 0, 1.0)
    )
    k <- k + 1
  }
}
df <- do.call(rbind, rows)
write.csv(df, "_fixtures/did_balance_panel.csv", row.names = FALSE)

## --- helpers -------------------------------------------------------
wmean <- function(v, w) sum(w * v) / sum(w)

wvar <- function(v, w) {
  mu <- wmean(v, w)
  denom <- sum(w) - sum(w^2) / sum(w)
  sum(w * (v - mu)^2) / denom
}

norm_diff_direct <- function(vt, vc, wt, wc) {
  (wmean(vt, wt) - wmean(vc, wc)) /
    sqrt((wvar(vt, wt) + wvar(vc, wc)) / 2)
}

## --- build both panels ---------------------------------------------
base <- df[df$t == base_period, ]
comp <- df[df$t == comparison_period, ]
stopifnot(identical(base$i, comp$i))

treated <- base$g == 4
control <- base$g == 0

out <- list()
for (cov in c("x", "z")) {
  levels_v <- base[[cov]]
  changes_v <- comp[[cov]] - base[[cov]]
  for (panel in c("levels", "changes")) {
    v <- if (panel == "levels") levels_v else changes_v
    for (wtd in c(FALSE, TRUE)) {
      ww <- if (wtd) base$w else rep(1, nrow(base))

      direct <- norm_diff_direct(v[treated], v[control], ww[treated], ww[control])

      # cobalt: treat vs control, pooled SD denominator.
      cob <- cobalt::col_w_smd(
        mat = matrix(v, ncol = 1, dimnames = list(NULL, cov)),
        treat = as.numeric(treated),
        weights = ww,
        std = TRUE,
        s.d.denom = "pooled"
      )

      out[[length(out) + 1]] <- data.frame(
        covariate = cov,
        panel = panel,
        weighted = wtd,
        mean_treated = wmean(v[treated], ww[treated]),
        mean_comparison = wmean(v[control], ww[control]),
        norm_diff_direct = as.numeric(direct),
        norm_diff_cobalt = as.numeric(cob)
      )
    }
  }
}
res <- do.call(rbind, out)
res$direct_vs_cobalt_rel <- abs(res$norm_diff_direct - res$norm_diff_cobalt) /
  pmax(abs(res$norm_diff_direct), 1e-12)
write.csv(res, "_fixtures/did_balance_reference.csv", row.names = FALSE)

print(res, digits = 10)
cat("\nn treated:", sum(treated), " n control:", sum(control), "\n")
cat("base period:", base_period, " comparison period:", comparison_period, "\n")
cat("cobalt version:", as.character(packageVersion("cobalt")), "\n")
cat("max |direct - cobalt| rel:", max(res$direct_vs_cobalt_rel), "\n")
