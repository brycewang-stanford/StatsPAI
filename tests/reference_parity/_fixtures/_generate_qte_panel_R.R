#!/usr/bin/env Rscript
# =====================================================================
#  Panel reference values: QDiD / CiC / MDiD / ddid2 / panel.qtet.
#
#  Point estimates only (se = FALSE).  The bootstrap variants of these
#  calls were measured at >2.5 hours on lalonde.psid.panel (2675 units x
#  3 periods, 100 iters x 8 specs), which is not a workable fixture.
#  Point estimates are DETERMINISTIC and are what the parity suite
#  anchors on; SEs are validated separately by Monte-Carlo coverage on
#  the Python side, which does not require R.
#
#  TRAP (do not "fix" by swapping the arguments): t is the POST period
#  and tmin1 the PRE period, given as VALUES of the tname column. The
#  panel is 1974 / 1975 / 1978, so t = 1978, tmin1 = 1975, tmin2 = 1974.
#  Swapping them does not error -- it silently returns a sign-flipped
#  estimate.
#
#  Environment: R 4.5.2 / qte 1.3.1
# =====================================================================
suppressPackageStartupMessages({library(qte); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

PROBS <- seq(0.05, 0.95, 0.05)
XF <- ~ age + education + black + hispanic + married + nodegree
data(lalonde)
out <- list()

for (spec in list(
  list(k = "qdid", fn = QDiD), list(k = "cic", fn = CiC), list(k = "mdid", fn = MDiD)
)) for (ux in c(FALSE, TRUE)) {
  key <- paste0(spec$k, if (ux) "_cov" else "_nocov")
  r <- try(spec$fn(re ~ treat, xformla = if (ux) XF else NULL,
                   t = 1978, tmin1 = 1975, tname = "year",
                   data = lalonde.psid.panel, panel = TRUE, idname = "id",
                   probs = PROBS, se = FALSE), silent = TRUE)
  if (inherits(r, "try-error")) { cat("FAILED", key, "\n"); next }
  out[[key]] <- list(probs = PROBS, qte = as.numeric(r$qte),
                     ate = if (!is.null(r$ate)) as.numeric(r$ate) else NULL,
                     covariates = ux)
  cat("ok", key, "\n")
}

for (m in c("qr", "pscore")) {
  key <- paste0("panel_qtet_", m)
  r <- try(panel.qtet(re ~ treat, xformla = XF, t = 1978, tmin1 = 1975,
                      tmin2 = 1974, tname = "year", data = lalonde.psid.panel,
                      idname = "id", probs = PROBS, method = m, se = FALSE),
           silent = TRUE)
  if (inherits(r, "try-error")) { cat("FAILED", key, ":", as.character(r), "\n"); next }
  out[[key]] <- list(probs = PROBS, qte = as.numeric(r$qte),
                     ate = if (!is.null(r$ate)) as.numeric(r$ate) else NULL,
                     method = m)
  cat("ok", key, "\n")
}

out[["_meta"]] <- list(generated_by = "_generate_qte_panel_R.R",
                       r_version = R.version.string,
                       qte_version = as.character(packageVersion("qte")),
                       probs = PROBS, t = 1978, tmin1 = 1975, tmin2 = 1974,
                       se = FALSE)
write(toJSON(out, digits = 12, auto_unbox = TRUE, null = "null"),
      file.path(OUT, "qte_panel_R.json"))
cat("wrote qte_panel_R.json\n")
