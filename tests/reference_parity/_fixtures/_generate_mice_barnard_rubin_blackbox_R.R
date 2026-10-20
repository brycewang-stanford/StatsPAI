# Black-box outputs of the Barnard-Rubin df used by R mice::pool
# (mice 3.19.0, GPL >= 2; the function is called, its source is not read).
# Fixture for the clean-room rewrite of statspai.imputation.mice
# ._barnard_rubin_df (docs/dev/cleanroom/sensemakr_mice_helpers.md).
# Only lambda = (1 + 1/m) b / t in [0, 1] is reachable from real data.
suppressMessages(library(mice))
br <- get("barnard.rubin", asNamespace("mice"))
g <- expand.grid(m = c(2, 5, 20), b = c(0, 0.01, 0.3, 0.5), t = c(1, 2),
                 dfcom = c(3, 50, 1e7))
g$df <- mapply(function(m, b, t, d) br(m, b, t, d), g$m, g$b, g$t, g$dfcom)
write.csv(g, "mice_barnard_rubin_blackbox_R.csv", row.names = FALSE)
cat("mice", as.character(packageVersion("mice")), "\n")
