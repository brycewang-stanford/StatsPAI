# Reference for sp.series: least squares on a polynomial or a cubic
# regression spline in base R, with sandwich for the robust and clustered
# variances. The fitted function and its standard error do not depend on
# how the basis is written, so lm() on poly(u, k, raw = TRUE) and on
# splines::bs() is a reference for a fit computed on other columns.
#
# Recorded on series_data.csv, formula y ~ z with the function of x:
#  * leave-one-out cross-validation criterion sum((e / (1 - h))^2) for
#    polynomial degrees 1..8 and for 0..6 quantile knots;
#  * at degree 5 and at 3 knots: the fitted function with z at its mean on
#    11 grid points, its standard error under the classical, HC1, HC2, HC3
#    and clustered (Stata small-sample factor) variances, the coefficient
#    of z with its standard errors, and for the polynomial the first
#    derivative of the function with its HC1 standard error.
#
#   Rscript tests/reference_parity/_fixtures/_generate_series_R.R
suppressMessages({
  library(sandwich)
  library(splines)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
if (length(here) == 0) here <- "."
d <- read.csv(file.path(here, "series_data.csv"))
d$u <- (d$x - mean(d$x)) / sd(d$x)
grid <- seq(min(d$x), max(d$x), length.out = 11)
ug <- (grid - mean(d$x)) / sd(d$x)
cv <- function(fit) sum((residuals(fit) / (1 - hatvalues(fit)))^2)
knots_of <- function(k) {
  if (k == 0) return(NULL)
  unname(quantile(d$u, (1:k) / (k + 1)))
}
spline_fit <- function(k) {
  lm(y ~ z + bs(u, knots = knots_of(k), degree = 3,
                Boundary.knots = range(d$u)), data = d)
}
vcovs <- function(fit) list(
  ols = vcov(fit), hc1 = vcovHC(fit, type = "HC1"),
  hc2 = vcovHC(fit, type = "HC2"), hc3 = vcovHC(fit, type = "HC3"),
  cluster = vcovCL(fit, cluster = d$g, type = "HC1", cadjust = TRUE)
)
tabulate_fit <- function(fit, design) {
  # design: model matrix at the grid, z at its mean
  V <- vcovs(fit)
  out <- list(fit = as.numeric(design %*% coef(fit)),
              z_coef = unname(coef(fit)["z"]))
  for (nm in names(V)) {
    out[[paste0("se_", nm)]] <- sqrt(rowSums((design %*% V[[nm]]) * design))
    out[[paste0("z_se_", nm)]] <- sqrt(V[[nm]]["z", "z"])
  }
  out
}

poly_cv <- sapply(1:8, function(k) cv(lm(y ~ z + poly(u, k, raw = TRUE), data = d)))
spline_cv <- sapply(0:6, function(k) cv(spline_fit(k)))

pf <- lm(y ~ z + poly(u, 5, raw = TRUE), data = d)
Pg <- cbind(1, mean(d$z), outer(ug, 1:5, "^"))
poly_out <- tabulate_fit(pf, Pg)
# derivative with respect to x: d/dx u^j = j u^(j-1) / sd(x)
Dg <- cbind(0, 0, t(sapply(ug, function(v) (1:5) * v^(0:4))) / sd(d$x))
poly_out$derivative <- as.numeric(Dg %*% coef(pf))
Vh <- vcovHC(pf, type = "HC1")
poly_out$derivative_se_hc1 <- sqrt(rowSums((Dg %*% Vh) * Dg))

sf <- spline_fit(3)
Bg <- bs(ug, knots = knots_of(3), degree = 3, Boundary.knots = range(d$u))
Sg <- cbind(1, mean(d$z), Bg)
spline_out <- tabulate_fit(sf, Sg)
spline_out$knots_x <- mean(d$x) + sd(d$x) * knots_of(3)

res <- list(
  versions = list(R = R.version.string,
                  sandwich = as.character(packageVersion("sandwich"))),
  grid = grid, poly_cv = poly_cv, spline_cv = spline_cv,
  poly_cv_order = which.min(poly_cv), spline_cv_order = which.min(spline_cv) - 1,
  polynomial = poly_out, spline = spline_out
)
writeLines(toJSON(res, auto_unbox = TRUE, digits = NA),
           file.path(here, "series_R.json"))
cat("cv orders:", res$poly_cv_order, res$spline_cv_order, "\n")
