# sp.did_multiplegt dynamic / placebo / controls conventions vs R
# DIDmultiplegt 0.1.4 (archived; the 2.x mode="old" path returns NaN, see
# tests/r_parity/81_didm.R). Run with STATSPAI_DIDM_LIB pointing at the
# private library that holds 0.1.4:
#
#   STATSPAI_DIDM_LIB=~/.statspai-rlib-didm014 Rscript _generate_dcdh_dynamic_R.R
#
# Data: dcdh_dynamic_panel.csv (see _generate_dcdh_dynamic_data.py).
# Writes dcdh_dynamic_R.csv: tag,name,value, with the tags of the Stata
# script (layout, treatment, controls, balance) and mode "r":
# placebo = 2, dynamic = 3, brep = 0.

.lib <- Sys.getenv("STATSPAI_DIDM_LIB")
if (nzchar(.lib)) .libPaths(c(.lib, .libPaths()))
suppressPackageStartupMessages(library(DIDmultiplegt))
if (utils::compareVersion(as.character(packageVersion("DIDmultiplegt")), "2.0.0") >= 0)
  stop("DIDmultiplegt 0.1.4 is required (STATSPAI_DIDM_LIB)")

.args <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", grep("^--file=", .args, value = TRUE)[1])
here <- dirname(normalizePath(.f))
raw <- read.csv(file.path(here, "dcdh_dynamic_panel.csv"))

out <- character()
put <- function(tag, name, value) {
  out <<- c(out, sprintf("%s,%s,%.17g", tag, name, as.numeric(value)))
}

for (layout in c("indiv", "cell")) {
  for (bal in c("bal", "unb")) {
    df <- raw
    if (bal == "unb") df <- df[df$keep == 1, ]
    if (layout == "cell") {
      df <- aggregate(cbind(y, yb, dc, db, x) ~ g + t, data = df, FUN = mean)
    }
    for (ctrl in c("none", "x")) {
      cv <- if (ctrl == "x") "x" else c()
      for (spec in list(c("db", "yb"), c("dc", "y"))) {
        tag <- sprintf("%s_%s_%s_%s_r", layout, spec[1], ctrl, bal)
        res <- suppressWarnings(did_multiplegt(
          df = df, Y = spec[2], G = "g", T = "t", D = spec[1],
          controls = cv, placebo = 2, dynamic = 3, brep = 0))
        put(tag, "effect_0", res$effect)
        for (l in 1:3) put(tag, sprintf("dynamic_%d", l), res[[sprintf("dynamic_%d", l)]])
        for (l in 1:2) put(tag, sprintf("placebo_%d", l), res[[sprintf("placebo_%d", l)]])
      }
    }
  }
}
writeLines(c("tag,name,value", out), file.path(here, "dcdh_dynamic_R.csv"))
cat("R", as.character(packageVersion("DIDmultiplegt")), "\n")
