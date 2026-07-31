#!/usr/bin/env Rscript
# =====================================================================
#  Fast (se = FALSE) reference values for the Firpo (2007) QTE / QTT.
#  Companion to _generate_qte_R.R, which additionally bootstraps SEs and
#  therefore takes ~an hour; this one runs in seconds and carries the
#  point estimates the parity suite actually anchors on.
#
#  Writes qte_firpo_R.json.
#
#  ---------------------------------------------------------------
#  WHY THIS FILE RECORDS THE *ARM-LEVEL* QUANTILES, NOT JUST THE QTE
#
#  R's `qte` package gets each reweighted marginal quantile from
#  BMisc::weighted_quantile, which is
#
#      optimize(weighted.checkfun, lower = min(y), upper = max(y), ...)
#
#  i.e. a golden-section search with tolerance .Machine$double.eps^0.25.
#  The check function is piecewise linear, so between order statistics it
#  has PLATEAUS on which every point is a minimiser; golden section returns
#  an arbitrary interior point of the plateau, plus optimiser tolerance.
#  On lalonde.exp (range ~[0, 60307]) that tolerance alone is worth several
#  units, and observed plateau ambiguity reaches ~800.
#
#  So "match R's number to 1e-6" is NOT a well-posed target: R's number is
#  not a well-defined functional of the data. What IS well-posed is the
#  objective. We therefore export q1 and q0 per arm so the Python test can
#  evaluate the SAME weighted check function at R's solution and at ours
#  and assert ours is no worse. See test_firpo_qte_parity.py.
#
#  Environment: R 4.5.2 / qte 1.3.1 / BMisc (as installed with qte)
# =====================================================================

suppressPackageStartupMessages({
  library(qte)
  library(BMisc)
  library(jsonlite)
})

.args <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .args[grep("^--file=", .args)])
OUT_DIR <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

PROBS <- seq(0.05, 0.95, 0.05)
XF <- ~ age + education + black + hispanic + married + nodegree

data(lalonde)
out <- list()

# ---------------------------------------------------------------------
#  Reproduce compute.ci.qte's weighting exactly, and record BOTH the
#  final QTE and the two arm-level quantile vectors it differenced.
# ---------------------------------------------------------------------
firpo_arms <- function(dat, estimand = c("qte", "qtet"), use_x = FALSE) {
  estimand <- match.arg(estimand)
  y <- dat$re78
  D <- dat$treat
  if (!use_x) {
    p <- rep(mean(D), length(D))
  } else {
    p <- fitted(glm(treat ~ age + education + black + hispanic + married +
                      nodegree, data = dat, family = binomial(link = "logit")))
  }
  if (estimand == "qte") {
    w1 <- D / p
    w0 <- (1 - D) / (1 - p)
  } else {
    w1 <- D
    w0 <- (1 - D) * p / (1 - p)
  }
  q1 <- BMisc::getWeightedQuantiles(PROBS, y, w1, norm = TRUE)
  q0 <- BMisc::getWeightedQuantiles(PROBS, y, w0, norm = TRUE)
  list(q1 = as.numeric(q1), q0 = as.numeric(q0),
       qte = as.numeric(q1 - q0),
       pscore = as.numeric(p))
}

for (ds in c("exp", "psid")) {
  dat <- if (ds == "exp") lalonde.exp else lalonde.psid
  for (est in c("qte", "qtet")) {
    for (ux in c(FALSE, TRUE)) {
      key <- paste0(est, "_", ds, if (ux) "_cov" else "_nocov")
      r <- firpo_arms(dat, est, ux)
      out[[key]] <- list(q1 = r$q1, q0 = r$q0, qte = r$qte,
                         dataset = ds, estimand = est, covariates = ux,
                         pscore_min = min(r$pscore), pscore_max = max(r$pscore))
      cat("[firpo-fixture] ok", key, "\n")
    }
  }
}

# Package-level call, for the record (no SE => fast, deterministic).
out[["pkg_ci_qte_exp"]] <- as.numeric(
  ci.qte(re78 ~ treat, data = lalonde.exp, probs = PROBS, se = FALSE)$qte)
out[["pkg_ci_qtet_exp"]] <- as.numeric(
  ci.qtet(re78 ~ treat, data = lalonde.exp, probs = PROBS, se = FALSE)$qte)
out[["pkg_ci_qte_psid_cov"]] <- as.numeric(
  ci.qte(re78 ~ treat, xformla = XF, data = lalonde.psid,
         probs = PROBS, se = FALSE)$qte)

out[["_meta"]] <- list(
  generated_by = "_generate_qte_firpo_R.R",
  r_version = R.version.string,
  qte_version = as.character(packageVersion("qte")),
  probs = PROBS,
  note = paste("Arm-level quantiles come from BMisc::getWeightedQuantiles,",
               "which minimises the weighted check function with",
               "stats::optimize. Point values carry golden-section plateau",
               "ambiguity; the objective value does not.")
)

write(toJSON(out, digits = 12, auto_unbox = TRUE, null = "null"),
      file.path(OUT_DIR, "qte_firpo_R.json"))
cat("[firpo-fixture] wrote qte_firpo_R.json\n")
