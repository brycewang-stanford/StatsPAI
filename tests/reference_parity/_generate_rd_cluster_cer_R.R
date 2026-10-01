# Reference values for tests/reference_parity/test_rd_cluster_cer_parity.py.
#
# Clustered CER bandwidths and the rdrobust fit that uses them, from
# rdrobust 4.0.0 on the committed fixture. Stata 18's rdbwselect gives the
# same h_cerrd (0.23049687) on the same bytes.
#
#   Rscript tests/reference_parity/_generate_rd_cluster_cer_R.R
suppressMessages(library(rdrobust))
cat("rdrobust", as.character(packageVersion("rdrobust")), "\n")
d <- read.csv("tests/reference_parity/_fixtures/rd_cluster_cer.csv")
for (bw in c("cerrd", "certwo", "cersum", "mserd")) {
  b <- rdbwselect(d$y, d$x, c = 0.1, covs = d$z, cluster = d$g, bwselect = bw)
  cat(bw, sprintf("%.10f", b$bws), "\n")
}
b <- rdbwselect(d$y, d$x, c = 0.1, bwselect = "cerrd")
cat("cerrd_nocluster", sprintf("%.10f", b$bws), "\n")
r <- rdrobust(d$y, d$x, c = 0.1, covs = d$z, cluster = d$g, bwselect = "cerrd")
cat("rdrobust_cerrd coef", sprintf("%.10f", r$coef), "se", sprintf("%.10f", r$se), "\n")
