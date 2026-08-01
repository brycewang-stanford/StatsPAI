# Reference values for sp.rdmc vs R rdmulti 2.0.0 (Cattaneo, Titiunik,
# Vazquez-Bare & Keele).
#
# A synthetic multi-cutoff design with DIFFERENT effects at each cutoff, so
# the fixture can tell a per-cutoff estimator from one that pools
# everything: if all three effects were equal, an implementation that
# ignored the cutoff assignment would still pass.
library(rdmulti); library(jsonlite)
set.seed(20260802)

n <- 6000
cutoffs <- c(-10, 0, 15)
cvar <- sample(cutoffs, n, replace = TRUE)
x <- runif(n, -40, 40)
tau <- c(2.0, 5.0, -3.0)          # deliberately different, incl. a sign flip
te <- tau[match(cvar, cutoffs)]
y <- 0.3 * x + te * (x >= cvar) + rnorm(n, 0, 2)

d <- data.frame(y = y, x = x, cvar = cvar)
write.csv(d, "rdmulti_design.csv", row.names = FALSE)

r <- rdmc(Y = d$y, X = d$x, C = d$cvar, plot = FALSE, verbose = FALSE)

# B/V/Coefs carry one column per cutoff, then "weighted", then "pooled".
k <- length(cutoffs)
out <- list(
  cutoffs   = cutoffs,
  # Verified empirically against sp.rdrobust, NOT assumed from the names:
  # rdmc's B/V are the BIAS-CORRECTED coefficient and its VARIANCE (V is a
  # variance, not an SE -- r$CI half-widths are 1.96*sqrt(V)), while Coefs
  # holds the CONVENTIONAL point estimate. The display table pairs the
  # conventional estimate with the robust CI, as rdrobust does.
  coefs     = unname(as.numeric(r$Coefs))[1:k],     # conventional
  coefs_rb  = unname(as.numeric(r$B))[1:k],         # bias-corrected
  var_rb    = unname(as.numeric(r$V))[1:k],         # variance of the above
  pvalues   = unname(as.numeric(r$Pv))[1:k],
  h_left    = unname(as.numeric(r$H[1, ]))[1:k],
  h_right   = unname(as.numeric(r$H[2, ]))[1:k],
  Nh_left   = unname(as.numeric(r$Nh[1, ]))[1:k],
  Nh_right  = unname(as.numeric(r$Nh[2, ]))[1:k],
  weights   = unname(as.numeric(r$W)),
  weighted_coef    = unname(as.numeric(r$Coefs))[k + 1],   # conventional
  weighted_coef_rb = unname(as.numeric(r$B))[k + 1],       # bias-corrected
  weighted_var_rb  = unname(as.numeric(r$V))[k + 1],
  pooled_coef   = unname(as.numeric(r$tau)),
  pooled_se_rb  = unname(as.numeric(r$se.rb)),
  pooled_h      = unname(as.numeric(r$hl)),
  true_tau  = tau
)
out[["_meta"]] <- list(n = n, rdmulti_version = as.character(packageVersion("rdmulti")))
write_json(out, "rdmulti_R.json", auto_unbox = TRUE, digits = 15, pretty = TRUE)
cat("wrote rdmulti_R.json\n")
