#!/usr/bin/env Rscript
# clubSandwich 0.6.2 (GPL-3) run as a black box -- public API only
# (vcovCR, Wald_test); no clubSandwich source was read. For every design of
# _generate_htz_blackbox_data.py and every restriction in
# htz_blackbox_tests.json: the CR2 covariance of R beta and the HTZ Wald
# test (Fstat, df_num, df_denom, p_val; eta = df_denom + q - 1).
#   python tests/reference_parity/_fixtures/_generate_htz_blackbox_data.py
#   Rscript tests/reference_parity/_fixtures/_generate_htz_blackbox_R.R
suppressMessages({library(clubSandwich); library(jsonlite)})
here <- "tests/reference_parity/_fixtures"
dd <- read.csv(file.path(here, "htz_blackbox_designs.csv"))
tests <- fromJSON(file.path(here, "htz_blackbox_tests.json"), simplifyVector = FALSE)
out <- list(meta = list(R = R.version.string,
                        clubSandwich = as.character(packageVersion("clubSandwich"))))
for (d in names(tests)) {
  D <- dd[dd$design == as.integer(d), ]
  D <- D[, colSums(is.na(D)) == 0]
  rhs <- setdiff(names(D), c("design", "g", "y"))
  fit <- lm(as.formula(paste("y ~", paste(rhs, collapse = " + "))), data = D)
  V <- vcovCR(fit, cluster = D$g, type = "CR2")
  res <- list()
  for (cs in tests[[d]]) {
    R <- do.call(rbind, lapply(cs$R, unlist))
    r <- unlist(cs$r)
    wt <- Wald_test(fit, constraints = R, vcov = V, test = "HTZ", null_constant = r)
    q <- nrow(R)
    res[[length(res) + 1]] <- list(
      R = R, r = r, q = q,
      V_R = unname(as.matrix(R %*% V %*% t(R))),
      F_stat = wt$Fstat, df_num = wt$df_num, df_denom = wt$df_denom,
      eta = wt$df_denom + q - 1, p_value = wt$p_val)
  }
  out[[paste0("d", d)]] <- list(beta = unname(coef(fit)), tests = res)
}
writeLines(toJSON(out, digits = I(17), auto_unbox = TRUE, matrix = "rowmajor"),
           file.path(here, "htz_blackbox_R.json"))
cat("wrote htz_blackbox_R.json\n")
