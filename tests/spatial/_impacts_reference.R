# Reference numbers for tests/spatial/test_impacts_interval_oct2026.py.
# R 4.5.2, spdep 1.4.2, spatialreg 1.4.3 (run as a black box).
# Inputs: W1.csv (binary ring, weights 1 at distances 1 and 2, n = 100),
# W2raw.csv (same ring with weights 0.5 at distance 2, n = 120), d2.csv and
# d3.csv written from the test's data builders with float_format "%.17g".
suppressMessages({library(spdep); library(spatialreg)})
W1 <- as.matrix(read.csv("W1.csv", header = FALSE)); lw1 <- mat2listw(W1, style = "B")
W2 <- as.matrix(read.csv("W2raw.csv", header = FALSE)); lw2 <- mat2listw(W2, style = "W")
d2 <- read.csv("d2.csv"); d3 <- read.csv("d3.csv")
f <- lagsarlm(y ~ x1, data = d3, listw = lw1, method = "eigen")
print(c(f$interval, f$rho, f$rho.se), digits = 16)
for (m in list(
  lagsarlm(ysar ~ x1 + x2, data = d2, listw = lw2, method = "eigen"),
  errorsarlm(ysem ~ x1 + x2, data = d2, listw = lw2, method = "eigen"),
  lagsarlm(ysdm ~ x1 + x2, data = d2, listw = lw2, method = "eigen", Durbin = TRUE))) {
  print(c(m$interval, coef(m), logLik(m)), digits = 16)
}
fs <- lagsarlm(ysar ~ x1 + x2, data = d2, listw = lw2, method = "eigen")
for (s in 1:3) {
  set.seed(s)
  im <- impacts(fs, listw = lw2, R = 20000)
  print(im$res, digits = 16); print(summary(im, zstats = TRUE)$semat)
}
slx <- lmSLX(yslx ~ x1 + x2, data = d2, listw = lw2)
sdem <- errorsarlm(yslx ~ x1 + x2, data = d2, listw = lw2, method = "eigen", Durbin = TRUE)
gns <- sacsarlm(yslx ~ x1 + x2, data = d2, listw = lw2, method = "eigen", Durbin = TRUE)
for (m in list(slx, sdem)) {
  s <- summary(impacts(m)); print(coef(m), digits = 17); print(s$semat, digits = 15)
}
print(impacts(gns, listw = lw2)$res, digits = 15)
