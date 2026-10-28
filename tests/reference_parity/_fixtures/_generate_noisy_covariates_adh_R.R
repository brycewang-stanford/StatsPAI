# Golden values for tests/reference_parity/test_noisy_covariates_authors_parity.py:
# the authors' own estimator (rrr_iv, src/wt/*.R of the Agarwal-Singh
# replication package, Zenodo record 19866751) on the covariate matrices
# written by _generate_noisy_covariates_adh_data.py.
#
# Usage: Rscript _generate_noisy_covariates_adh_R.R <replication_package dir> <work dir> <out.json>
args <- commandArgs(trailingOnly = TRUE)
pkg <- args[1]; work <- args[2]; out <- args[3]
suppressPackageStartupMessages(library(dplyr))
for (f in c("census/specifications.R", "wt/primitives.R", "wt/stage1_PCR.R",
            "wt/stage1.R", "wt/stage2.R"))
  source(file.path(pkg, "src", f))
# scripts/00_setup_simulations.R
RNGkind(kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
df <- foreign::read.dta(file.path(pkg, "data", "workfile_china.dta"))
dd <- get_data(df)
Y <- dd[[1]]; D <- dd[[3]]; Z <- dd[[4]]; W <- dd[[5]]
n <- nrow(Y); p <- 31
cfg <- jsonlite::fromJSON(file.path(work, "configs.json"))
# rrr_iv draws its folds with split(sample(n, n), as.factor(1:5)) after
# set.seed(1), as in scripts/03_run_census.R; record them for Python.
set.seed(1); perm <- sample(n, n)
fold <- integer(n); fold[perm] <- ((seq_len(n) - 1) %% 5) + 1
res <- list()
for (name in cfg$configs) {
  X <- matrix(readBin(file.path(work, paste0(name, ".bin")), "double", n * p,
                      endian = "little"), n, p)
  set.seed(1)
  r <- rrr_iv(Y = Y, T = D, Z = Z, X, b1, 3, 3, 0, k = 5, W = W)
  res[[name]] <- list(estimate = sprintf("%.17g", r[3]), se = sprintf("%.17g", r[4]))
}
info <- list(
  reference = "Agarwal and Singh (2026) replication package, Zenodo 10.5281/zenodo.19866751, src/wt/stage2.R::rrr_iv (alpha_estimator = gamma_estimator = 3, k = 5, dictionary b1, 5 folds)",
  r_version = R.version.string,
  generated = format(Sys.Date()),
  fold = fold,
  results = res
)
writeLines(jsonlite::toJSON(info, auto_unbox = TRUE, pretty = TRUE, digits = NA), out)
