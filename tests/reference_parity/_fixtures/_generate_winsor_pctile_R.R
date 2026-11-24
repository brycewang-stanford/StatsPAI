# Percentile reference for tests/reference_parity/test_winsor_pctile_parity.py
# Run from this folder:  Rscript _generate_winsor_pctile_R.R
# The R counterpart of _generate_winsor_pctile_Stata.do: quantile(type = 2)
# on x = 1..n for n = 2..120 at the same percentiles, with the probability
# formed as p / 100, written to winsor_pctile_R.csv. Then farr::winsorize()
# and farr::truncate() on 1..100 with explicit p_low / p_high at two pairs
# where Stata and R differ, written to winsor_pctile_farr_R.csv.
suppressMessages(library(farr))
pcts <- c(0.5, 2.5, 97.5, 99.5, 1:99)
grid <- do.call(rbind, lapply(2:120, function(n) data.frame(
  n = n, p = pcts,
  pctile = unname(quantile(as.numeric(1:n), pcts / 100, type = 2)))))
write.csv(grid, "winsor_pctile_R.csv", row.names = FALSE)
x <- as.numeric(1:100)
write.csv(data.frame(
  x = x,
  x_a = winsorize(x, p_low = 7 / 100, p_high = 93 / 100),
  x_b = winsorize(x, p_low = 29 / 100, p_high = 57 / 100),
  x_c = truncate(x, p_low = 7 / 100, p_high = 93 / 100)),
  "winsor_pctile_farr_R.csv", row.names = FALSE, na = "")
