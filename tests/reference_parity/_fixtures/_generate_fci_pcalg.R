# ---------------------------------------------------------------------------
# Reference PAGs for tests/reference_parity/test_fci_pcalg_parity.py
#
# pcalg::fci() and pcalg::rfci() with the Gaussian conditional independence
# test on (a) the 24 data sets of pc_pcalg_data.csv, which have no latent
# variables but small samples, and (b) the 12 data sets of
# fci_latent_data.csv, which have hidden common causes. A PAG is stored row
# by row as pcalg's amat: entry [i, j] is the mark at the j end of the edge
# between i and j (0 none, 1 circle, 2 arrowhead, 3 tail).
#
# Requires: pcalg (2.7-12; Bioconductor graph and RBGL), jsonlite.
# Run:      Rscript tests/reference_parity/_fixtures/_generate_fci_pcalg.R
#           (from the repository root)
# ---------------------------------------------------------------------------
suppressMessages({
  library(pcalg); library(jsonlite)
})
here <- "tests/reference_parity/_fixtures"
alphas <- c(0.01, 0.05, 0.2)
run <- function(file) {
  dat <- read.csv(file.path(here, file))
  out <- list()
  for (id in sort(unique(dat$case))) {
    d <- dat[dat$case == id, -1]
    d <- d[, colSums(is.na(d)) == 0, drop = FALSE]
    alpha <- alphas[id %% 3 + 1]
    ss <- list(C = cor(d), n = nrow(d))
    f <- fci(ss, indepTest = gaussCItest, alpha = alpha, labels = colnames(d))
    r <- rfci(ss, indepTest = gaussCItest, alpha = alpha, labels = colnames(d))
    out[[length(out) + 1]] <- list(
      case = id, p = ncol(d), n = nrow(d), alpha = alpha,
      fci = as.integer(t(f@amat)),
      rfci_skeleton = as.integer(t((r@amat != 0) * 1L)))
  }
  out
}
write_json(
  list(pcalg = as.character(packageVersion("pcalg")), R = R.version.string,
       no_latent = run("pc_pcalg_data.csv"),
       latent = run("fci_latent_data.csv")),
  file.path(here, "fci_pcalg_R.json"), auto_unbox = TRUE)
