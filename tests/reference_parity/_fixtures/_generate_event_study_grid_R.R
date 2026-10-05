#!/usr/bin/env Rscript
# =====================================================================
#  Reference values for sp.event_study across its common options.
#
#  Reads event_study_grid.csv (written by
#  _generate_event_study_grid_data.py), writes event_study_grid_R.json.
#
#  Grid: adoption (single date, staggered) x window ((-4,4), (-3,5)) x
#  reference period (-1, -2) x covariate (none, x) x weights (none, w)
#  = 32 specifications, each a fixest::feols two-way fixed-effects
#  regression on event-time dummies, clustered on the unit.
#
#  Relative time outside the window is binned into the endpoints, as
#  sp.event_study documents. Never-treated units are parked at the
#  reference level and interacted with the treated indicator, so they
#  stay in the sample with no dummy of their own (see the note in
#  tests/r_parity/85_twfe_event_study.R).
#
#  Environment: R 4.5.2 / fixest 0.14
# =====================================================================
suppressPackageStartupMessages({library(fixest); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."
d0 <- read.csv(file.path(OUT, "event_study_grid.csv"))

out <- list()
for (adoption in c("single", "staggered"))
  for (win in list(c(-4, 4), c(-3, 5)))
    for (ref in c(-1, -2))
      for (cov in c("none", "x"))
        for (wt in c("none", "w")) {
          d <- if (adoption == "single") d0[d0$g %in% c(0, 7), ] else d0
          d$treat <- as.integer(d$g > 0)
          rel <- ifelse(d$g > 0, d$time - d$g, NA_integer_)
          rel <- pmin(pmax(rel, win[1]), win[2])
          d$rel_f <- ifelse(is.na(rel), ref, rel)
          fml <- as.formula(paste0(
            "y ~ i(rel_f, treat, ref = ", ref, ")",
            if (cov == "x") " + x" else "", " | unit + time"))
          fit <- if (wt == "w") {
            feols(fml, data = d, cluster = ~unit, weights = ~w)
          } else {
            feols(fml, data = d, cluster = ~unit)
          }
          ct <- summary(fit)$coeftable
          coefs <- list()
          for (nm in rownames(ct)) {
            k <- suppressWarnings(as.integer(sub("^.*::(-?[0-9]+).*$", "\\1", nm)))
            if (is.na(k) || !grepl("rel_f", nm)) next
            coefs[[as.character(k)]] <- list(
              estimate = unname(ct[nm, "Estimate"]),
              se = unname(ct[nm, "Std. Error"]))
          }
          key <- paste(adoption, paste0("w", win[1], "_", win[2]),
                       paste0("ref", ref), paste0("cov_", cov),
                       paste0("wt_", wt), sep = "|")
          out[[key]] <- list(
            adoption = adoption, window = win, ref = ref,
            covariates = cov, weights = wt, n = nobs(fit), coefs = coefs,
            x = if (cov == "x") list(estimate = unname(ct["x", "Estimate"]),
                                     se = unname(ct["x", "Std. Error"])) else NULL)
        }
out[["_meta"]] <- list(
  generated_by = "_generate_event_study_grid_R.R",
  r_version = R.version.string,
  fixest_version = as.character(packageVersion("fixest")),
  n_specs = length(out))
write(toJSON(out, digits = 15, auto_unbox = TRUE, null = "null"),
      file.path(OUT, "event_study_grid_R.json"))
cat("[event-study-grid] wrote", length(out) - 1, "specs\n")
