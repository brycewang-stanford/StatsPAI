# ---------------------------------------------------------------------------
# Reference for tests/reference_parity/test_mmhc_bnlearn_parity.py
#
# bnlearn on the 24 data sets of _generate_mmhc_data.py: the skeletons of
# mmpc() and si.hiton.pc() (tests "cor" and "mi", alpha 0.05), and the arcs
# and BIC of mmhc() and of rsmax2(restrict = "si.hiton.pc", maximize = "hc").
#
# Requires: bnlearn (5.2.1), jsonlite.
# Run:      python tests/reference_parity/_fixtures/_generate_mmhc_data.py <dir>
#           Rscript tests/reference_parity/_fixtures/_generate_mmhc_bnlearn.R <dir>
#           (from the repository root; <dir> is any scratch directory)
# ---------------------------------------------------------------------------
suppressMessages({library(bnlearn); library(jsonlite)})
a <- commandArgs(TRUE)[1]
und <- function(m) { x <- arcs(m); x <- x[x[, 1] < x[, 2], , drop = FALSE]
  lapply(seq_len(nrow(x)), function(i) unname(x[i, ])) }
dir_ <- function(m) { x <- arcs(m); lapply(seq_len(nrow(x)), function(i) unname(x[i, ])) }
cases <- list()
for (f in sort(list.files(a, pattern = "^d[0-9]+\\.csv$"))) {
  d <- read.csv(file.path(a, f), stringsAsFactors = TRUE)
  h <- mmhc(d); h2 <- rsmax2(d, restrict = "si.hiton.pc", maximize = "hc")
  cases[[sub("\\.csv$", "", f)]] <- list(
    n = nrow(d), p = ncol(d),
    mmpc = und(suppressWarnings(mmpc(d))), hiton = und(suppressWarnings(si.hiton.pc(d))),
    mmhc_arcs = dir_(h), mmhc_score = score(h, d),
    hiton_hc_arcs = dir_(h2), hiton_hc_score = score(h2, d))
}
write_json(list(bnlearn = as.character(packageVersion("bnlearn")), R = R.version.string, cases = cases),
           "tests/reference_parity/_fixtures/mmhc_bnlearn_R.json", digits = NA, auto_unbox = TRUE)
