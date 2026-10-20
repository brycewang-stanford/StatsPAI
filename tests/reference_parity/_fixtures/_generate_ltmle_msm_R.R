# Reference values for sp.ltmle_msm vs R ltmle::ltmleMSM 1.3-0
# (tests/reference_parity/test_ltmle_msm_parity.py). ltmle is GPL-2: used
# as a black box (outputs only). Run from this folder:
#   Rscript _generate_ltmle_msm_R.R
# Data: teffects_wide.csv / teffects_wide_cens.csv (two periods). Regimes:
# the four static (A0, A1) combinations; working model Y ~ cumA (number of
# treated periods), SL.library = NULL (glm), default gbounds,
# variance.method = "ic", one final outcome node.
suppressPackageStartupMessages({ library(ltmle); library(jsonlite) })
wide <- read.csv("teffects_wide.csv"); widec <- read.csv("teffects_wide_cens.csv")
n <- nrow(wide)
combos <- rbind(c(0, 0), c(1, 0), c(0, 1), c(1, 1))
mk <- function(cmb) {
  r <- array(NA, dim = c(n, 2, nrow(cmb)))
  for (j in seq_len(nrow(cmb))) { r[, 1, j] <- cmb[j, 1]; r[, 2, j] <- cmb[j, 2] }
  r
}
smk <- function(cmb) array(rowSums(cmb), dim = c(nrow(cmb), 1, 1),
                           dimnames = list(NULL, "cumA", NULL))
sw <- 0.4 + (seq_len(n) %% 5) * 0.3
run <- function(data, Ynode, Cnodes = NULL, msm = "Y ~ cumA", w = "empirical",
                regimes = mk(combos), sm = smk(combos), ...) {
  r <- suppressMessages(ltmleMSM(data, Anodes = c("A0", "A1"), Cnodes = Cnodes,
         Lnodes = "L1", Ynodes = Ynode, regimes = regimes, summary.measures = sm,
         working.msm = msm, msm.weights = w, SL.library = NULL,
         estimate.time = FALSE, variance.method = "ic", ...))
  s <- summary(r)
  list(beta = unname(r$beta), se = unname(s$cmat[, 2]),
       beta_iptw = unname(r$beta.iptw))
}
d <- wide[, c("v", "L0", "A0", "L1", "A1", "Yb")]
dy <- wide[, c("v", "L0", "A0", "L1", "A1", "Y")]
dc <- widec[, c("v", "L0", "A0", "C0", "L1", "A1", "C1", "Yb")]
dc$C0 <- BinaryToCensoring(is.uncensored = dc$C0)
dc$C1 <- BinaryToCensoring(is.uncensored = dc$C1)
# dynamic: regime 2 treats at time 2 only when L1 > 0
rdyn <- mk(combos); rdyn[, 2, 2] <- as.numeric(wide$L1 > 0)
# a duplicated regime (the fifth repeats (1, 1))
cdup <- rbind(combos, c(1, 1))
out <- list(
  static = run(d, "Yb"),
  equal_weights = run(d, "Yb", w = NULL),
  baseline_term = run(d, "Yb", msm = "Y ~ cumA + v"),
  censored = run(dc, "Yb", Cnodes = c("C0", "C1")),
  continuous = run(dy, "Y"),
  observation_weights = run(d, "Yb", observation.weights = sw),
  dynamic = run(d, "Yb", regimes = rdyn),
  user_weights = run(d, "Yb", w = matrix(c(1, 2, 3, 4), 4, 1)),
  duplicate_regime = run(d, "Yb", regimes = mk(cdup), sm = smk(cdup)),
  versions = list(R = R.version.string, ltmle = as.character(packageVersion("ltmle"))))
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), na = "null", pretty = TRUE),
           "ltmle_msm_R.json")
