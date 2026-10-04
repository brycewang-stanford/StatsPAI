# Reference values for sp.rdms at several boundary points with the xnorm
# pooled row, against R rdmulti 2.0 (rdms). The design is the two-score
# one of Cattaneo, Idrobo & Titiunik (2024, Section 5): treatment requires
# both scores to clear zero, so the boundary is the two half-lines from the
# origin and xnorm is the perpendicular distance to it.
library(rdmulti); library(jsonlite)
set.seed(20261004)
n <- 4000
x1 <- round(runif(n, -50, 80), 3); x2 <- round(runif(n, -60, 90), 3)
tr <- as.numeric(x1 >= 0 & x2 >= 0)
y <- 1 + 0.004 * x1 - 0.003 * x2 + tr * (0.6 + 0.004 * x1) + rnorm(n, 0, 0.5)
y <- round(y, 6)
a1 <- abs(x1); a2 <- abs(x2)
xn <- pmin(a1, a2)
xn <- ifelse(x1 <= 0 & x2 >= 0, a1, xn)
xn <- ifelse(x1 >= 0 & x2 <= 0, a2, xn)
xn <- ifelse(x1 <= 0 & x2 <= 0, sqrt(x1^2 + x2^2), xn)
xn <- xn * (2 * tr - 1)
write.csv(data.frame(y = y, x1 = x1, x2 = x2, tr = tr, xnorm = xn),
          "rdms_points_design.csv", row.names = FALSE)
d <- read.csv("rdms_points_design.csv")
sink("/dev/null")
r <- rdms(d$y, d$x1, c(0, 30, 0), d$x2, d$tr, c(0, 0, 40), xnorm = d$xnorm)
sink()
k <- 3
out <- list(
  cutoff1 = c(0, 30, 0), cutoff2 = c(0, 0, 40),
  coefs = unname(as.numeric(r$Coefs))[1:k], coefs_rb = unname(as.numeric(r$B))[1:k],
  var_rb = unname(as.numeric(r$V))[1:k], pvalues = unname(as.numeric(r$Pv))[1:k],
  ci_lower = unname(r$CI[1, 1:k]), ci_upper = unname(r$CI[2, 1:k]),
  h = unname(as.numeric(r$H[1, 1:k])),
  Nh = unname(as.numeric(r$Nh[1, 1:k] + r$Nh[2, 1:k])),
  pooled_coef = unname(as.numeric(r$Coefs))[k + 1],
  pooled_coef_rb = unname(as.numeric(r$B))[k + 1],
  pooled_var_rb = unname(as.numeric(r$V))[k + 1],
  pooled_ci = unname(r$CI[, k + 1]),
  `_meta` = list(n = n, rdmulti_version = as.character(packageVersion("rdmulti")))
)
write_json(out, "rdms_points_R.json", auto_unbox = TRUE, digits = 15, pretty = TRUE)
cat("wrote rdms_points_R.json\n")
