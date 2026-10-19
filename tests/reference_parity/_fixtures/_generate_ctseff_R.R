# Screen reference for sp.dose_response(method='dr') vs R npcausal::ctseff
# (GPL; installed from GitHub ehkennedy/npcausal 0.1.0 into a private
# library and used as a black box, outputs only). Checked in
# tests/test_dose_response_dr.py (shape screen, not parity).
# Run from this folder with npcausal on .libPaths():
#   Rscript _generate_ctseff_R.R
# Data ctseff_data.csv: set.seed(7), n = 400, x1, x2 ~ N(0, 1),
# a = 0.5 x1 - 0.3 x2 + N(0, 1), y = a + 0.5 a^2 + x1 + x2 + N(0, 0.25),
# all rounded to 6 decimals.
suppressMessages({library(npcausal); library(jsonlite)})
d <- read.csv("ctseff_data.csv"); x <- as.matrix(d[, c("x1", "x2")])
set.seed(1)
r <- ctseff(d$y, d$a, x, bw.seq = seq(.2, 2, length.out = 20), sl.lib = c("SL.glm"))
write_json(list(res = r$res, bw = r$bw.risk,
                versions = list(R = R.version.string,
                                npcausal = as.character(packageVersion("npcausal")))),
           "ctseff_R.json", digits = NA)
