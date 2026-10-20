# Black-box outputs of R sensemakr::robustness_value (GPL-3; run, not read).
# Fixture for the clean-room rewrite of statspai.diagnostics.sensemakr
# ._robustness_value (docs/dev/cleanroom/sensemakr_mice_helpers.md).
# Run from this directory: Rscript _generate_sensemakr_rv_blackbox_R.R
suppressMessages(library(sensemakr))
g <- expand.grid(t = c(-4.18445, 0.3, 1.5, 1.9, 2.5, 4.18445, 12, 40),
                 dof = c(5, 30, 783), q = c(0.5, 1, 2),
                 alpha = c(1, 0.05, 0.2), invert = c(FALSE, TRUE))
g$rv <- mapply(function(t, d, q, a, inv)
  as.numeric(robustness_value(t, dof = d, q = q, alpha = a, invert = inv)),
  g$t, g$dof, g$q, g$alpha, g$invert)
write.csv(g, "sensemakr_rv_blackbox_R.csv", row.names = FALSE)
cat("sensemakr", as.character(packageVersion("sensemakr")), "\n")
