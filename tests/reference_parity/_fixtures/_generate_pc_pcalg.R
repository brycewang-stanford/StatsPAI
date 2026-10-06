# ---------------------------------------------------------------------------
# Reference graphs for tests/reference_parity/test_pc_pcalg_parity.py
#
# pcalg::pc() with the Gaussian conditional independence test on the 24 data
# sets written by _generate_pc_pcalg_data.py, for both skeleton methods and
# three significance levels. The CPDAG is stored row by row as a 0/1 matrix:
# entry [i, j] = 1 means i -> j, or i -- j when [j, i] is 1 as well.
#
# Requires: pcalg (2.7-12; needs the Bioconductor packages graph and RBGL),
#           jsonlite.
# Run:      Rscript tests/reference_parity/_fixtures/_generate_pc_pcalg.R
#           (from the repository root)
# ---------------------------------------------------------------------------
suppressMessages({
  library(pcalg); library(jsonlite)
})
here <- "tests/reference_parity/_fixtures"
dat <- read.csv(file.path(here, "pc_pcalg_data.csv"))
alphas <- c(0.01, 0.05, 0.2)
cases <- list()
for (id in sort(unique(dat$case))) {
  d <- dat[dat$case == id, -1]
  d <- d[, colSums(is.na(d)) == 0, drop = FALSE]
  alpha <- alphas[id %% 3 + 1]
  res <- list(case = id, p = ncol(d), n = nrow(d), alpha = alpha)
  for (m in c("stable", "original")) {
    fit <- pc(list(C = cor(d), n = nrow(d)), indepTest = gaussCItest,
              alpha = alpha, labels = colnames(d), skel.method = m)
    res[[m]] <- as.integer(t(as(fit@graph, "matrix")))
  }
  cases[[length(cases) + 1]] <- res
}
write_json(
  list(pcalg = as.character(packageVersion("pcalg")), R = R.version.string,
       cases = cases),
  file.path(here, "pc_pcalg_R.json"), auto_unbox = TRUE)
