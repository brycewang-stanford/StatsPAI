# ---------------------------------------------------------------------------
# Reference for tests/reference_parity/test_path_analysis_lavaan_parity.py
#
# lavaan::sem() on observed-variable path models: two parallel mediators with
# labelled paths and defined effects, a residual covariance, a covariate, and
# an over-identified chain. Normal-theory ML with expected information
# (lavaan's default) and the Satorra-Bentler robust standard errors and
# scaled test (estimator = "MLM").
#
# Requires: lavaan (0.6-19), jsonlite.
# Run:      Rscript tests/reference_parity/_fixtures/_generate_path_analysis_lavaan.R
#           (from the repository root; rewrites the CSV as well)
# ---------------------------------------------------------------------------
suppressMessages({library(lavaan); library(jsonlite)})
here <- "tests/reference_parity/_fixtures"
set.seed(20261006)
n <- 400
x  <- round(rnorm(n), 3)
w  <- round(rnorm(n), 3)
u  <- rnorm(n)                                   # shared shock of the mediators
m1 <- round(0.5 * x + 0.3 * w + 0.6 * u + rnorm(n), 3)
m2 <- round(-0.4 * x + 0.5 * u + rt(n, 5), 3)    # heavy tails: robust SEs differ
y  <- round(0.3 * x + 0.6 * m1 - 0.5 * m2 + 0.2 * w + rnorm(n) * (1 + 0.5 * abs(x)), 3)
z  <- round(0.7 * y + rnorm(n), 3)
xw <- round(x * w, 6)
d <- data.frame(x, w, m1, m2, y, z, xw)
write.csv(d, file.path(here, "path_analysis.csv"), row.names = FALSE)
d <- read.csv(file.path(here, "path_analysis.csv"))

models <- list(
  mediation = '
    m1 ~ a1*x + w
    m2 ~ a2*x
    y  ~ c*x + b1*m1 + b2*m2 + w
    m1 ~~ m2
    ind1  := a1*b1
    ind2  := a2*b2
    total := c + a1*b1 + a2*b2
    prop  := (a1*b1 + a2*b2) / (c + a1*b1 + a2*b2)
  ',
  chain = '
    m1 ~ x
    y  ~ m1
    z  ~ y
  ',
  constrained = '
    m1 ~ a*x + w
    m2 ~ a*x              # the same label: the two paths are set equal
    y  ~ 0.5*m1 + m2 + xw # a fixed coefficient and a product term
    twice := 2*a
  ',
  saturated = '
    m1 ~ x + w
    y  ~ x + w + m1
  '
)
fm <- c("chisq", "df", "pvalue", "baseline.chisq", "baseline.df", "cfi", "tli",
        "rmsea", "rmsea.ci.lower", "rmsea.ci.upper", "srmr", "logl", "aic", "bic",
        "npar")
tab <- function(fit) {
  pe <- as.data.frame(unclass(parameterEstimates(fit, standardized = TRUE)))
  if (is.null(pe$label)) pe$label <- ""
  pe[, c("lhs", "op", "rhs", "label", "est", "se", "z", "pvalue", "ci.lower",
         "ci.upper", "std.all")]
}
out <- list(lavaan = as.character(packageVersion("lavaan")), R = R.version.string,
            n = n, cases = list())
for (nm in names(models)) {
  ml  <- sem(models[[nm]], data = d)
  mlm <- sem(models[[nm]], data = d, estimator = "MLM")
  out$cases[[nm]] <- list(
    model = models[[nm]],
    ml = tab(ml), ml_fit = as.list(fitMeasures(ml, fm)),
    mlm = tab(mlm),
    mlm_fit = as.list(fitMeasures(mlm, c("chisq.scaled", "df.scaled", "pvalue.scaled",
                                          "chisq.scaling.factor"))),
    implied = unclass(fitted(ml)$cov), implied_names = colnames(fitted(ml)$cov))
}
write_json(out, file.path(here, "path_analysis_lavaan_R.json"), digits = NA,
           auto_unbox = TRUE, na = "null")
