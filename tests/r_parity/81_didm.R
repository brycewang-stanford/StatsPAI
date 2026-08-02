# StatsPAI dCDH 2020 DID_M parity (R side) -- Module 81.
#
# Runs DIDmultiplegt::did_multiplegt on the panel dumped by 81_didm.py.
#
# ⚠️ Version matters. The CRAN package's 2.x rewrite routes the classic
# estimator through mode="old", and that path returns NaN even on the
# package's own bundled `wagepan_mgt` example. The archived 0.1.4 is the
# last release where it works:
#
#   install.packages("assertthat")
#   install.packages(
#     "https://cran.r-project.org/src/contrib/Archive/DIDmultiplegt/DIDmultiplegt_0.1.4.tar.gz",
#     repos = NULL, type = "source", lib = <a private library>)
#
# Point it at that library with STATSPAI_DIDM_LIB, otherwise this falls back
# to the default library and will fail loudly rather than emit 2.x's NaN.
#
# Standard errors are not emitted: this runs brep = 0.
#
# Tolerance: rel < 1e-6.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

.didm_lib <- Sys.getenv("STATSPAI_DIDM_LIB")
if (nzchar(.didm_lib)) .libPaths(c(.didm_lib, .libPaths()))
suppressPackageStartupMessages(library(DIDmultiplegt))

.ver <- as.character(utils::packageVersion("DIDmultiplegt"))
if (utils::compareVersion(.ver, "2.0.0") >= 0) {
  stop(sprintf(
    paste0("DIDmultiplegt %s is installed; its mode='old' path returns NaN ",
           "even on the package's own example. Install 0.1.4 into a private ",
           "library and point STATSPAI_DIDM_LIB at it."), .ver))
}

MODULE <- "81_didm"
df <- as.data.frame(read_csv_strict(MODULE))

res <- suppressWarnings(
  did_multiplegt(df = df, Y = "y", G = "id", T = "t", D = "d",
                 placebo = 1, dynamic = 1, brep = 0)
)

rows <- list()
emit <- function(nm, value) {
  rows[[length(rows) + 1]] <<- parity_row(
    module = MODULE, statistic = nm, estimate = as.numeric(value),
    se = NA_real_, n = nrow(df)
  )
}
emit("effect", res$effect)
emit("dynamic_1", res$dynamic_1)
emit("placebo_1", res$placebo_1)

write_results(MODULE, rows, extra = list(
  reference = "DIDmultiplegt::did_multiplegt (archived 0.1.4)",
  DIDmultiplegt = .ver
))
