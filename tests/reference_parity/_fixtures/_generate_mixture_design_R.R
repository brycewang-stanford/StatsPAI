# ---------------------------------------------------------------------------
# Reference designs for tests/reference_parity/test_mixture_design_r_parity.py
#
# mixexp::SLD (simplex lattice) and mixexp::SCD (simplex centroid). mixexp
# is GPL and archived on CRAN; it was run as a black box, not read.
#
# Requires: mixexp 1.2.7 (CRAN archive; needs a Fortran compiler), jsonlite.
# Install:  R CMD INSTALL -l <private lib> mixexp_1.2.7.tar.gz
#           (here tests/reference_parity/_fixtures/_rlib_parity-families/,
#           gitignored; FC / F77 pointed at Homebrew gfortran through
#           R_MAKEVARS_USER).
# Run:      Rscript tests/reference_parity/_fixtures/_generate_mixture_design_R.R
#           (from the repository root)
# ---------------------------------------------------------------------------
lib <- file.path("tests", "reference_parity", "_fixtures", "_rlib_parity-families")
if (dir.exists(lib)) .libPaths(c(normalizePath(lib), .libPaths()))
suppressMessages({ library(mixexp); library(jsonlite) })

out <- list(lattice = list(), centroid = list())
for (q in 3:5) for (m in 1:4) {
  d <- as.matrix(SLD(q, m))
  out$lattice[[sprintf("%d_%d", q, m)]] <- unname(d)
}
for (q in 3:6) {
  d <- as.matrix(SCD(q))
  out$centroid[[as.character(q)]] <- unname(d)
}
out$versions <- list(
  R = R.version.string,
  mixexp = as.character(packageVersion("mixexp"))
)
writeLines(
  toJSON(out, digits = NA, auto_unbox = TRUE, matrix = "rowmajor"),
  file.path("tests", "reference_parity", "_fixtures", "mixture_design_R.json")
)
