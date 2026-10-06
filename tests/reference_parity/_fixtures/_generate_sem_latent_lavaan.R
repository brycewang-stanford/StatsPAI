# ---------------------------------------------------------------------------
# Reference for tests/reference_parity/test_sem_latent_lavaan_parity.py
#
# lavaan models with latent variables: a two-factor CFA, the same with
# standardised latents (std.lv), a structural model with latent and observed
# predictors, a model with a cross-loading and a residual covariance, a CFA
# with a mean structure, and a linear growth model. ML (expected information)
# and, for the covariance-only models, estimator = "MLM".
#
# Requires: lavaan (0.6-21), jsonlite.
# Run:      Rscript tests/reference_parity/_fixtures/_generate_sem_latent_lavaan.R
#           (from the repository root; rewrites the CSV as well)
# ---------------------------------------------------------------------------
suppressMessages({library(lavaan); library(jsonlite)})
here <- "tests/reference_parity/_fixtures"
set.seed(20261008)
n <- 500
x  <- round(rnorm(n), 3)
f1 <- 0.5 * x + rnorm(n)
f2 <- 0.6 * f1 + 0.3 * x + rnorm(n, sd = 0.8)
ind <- function(f, l, s = 0.7) round(l * f + rnorm(n, sd = s) * (1 + 0.3 * abs(f)), 3)
d <- data.frame(x = x,
  a1 = ind(f1, 1.0), a2 = ind(f1, 0.8), a3 = ind(f1, 1.2),
  b1 = ind(f2, 1.0), b2 = ind(f2, 0.7), b3 = ind(f2, 0.9))
d$a3 <- round(d$a3 + 0.3 * f2, 3)                      # a cross-loading
d$y  <- round(0.5 * f2 + 0.2 * x + rt(n, 6), 3)
# four repeated measures with a random intercept and slope
gi <- 2 + rnorm(n); gs <- 0.5 + 0.3 * gi * 0 + rnorm(n, sd = 0.4)
for (t in 0:3) d[[paste0("t", t + 1)]] <- round(gi + gs * t + rnorm(n, sd = 0.6), 3)
write.csv(d, file.path(here, "sem_latent.csv"), row.names = FALSE)
d <- read.csv(file.path(here, "sem_latent.csv"))

models <- list(
  cfa = list(m = 'f1 =~ a1 + a2 + a3
                  f2 =~ b1 + b2 + b3', args = list()),
  cfa_stdlv = list(m = 'f1 =~ a1 + a2 + a3
                        f2 =~ b1 + b2 + b3', args = list(std.lv = TRUE)),
  sem = list(m = 'f1 =~ a1 + a2 + a3
                  f2 =~ b1 + b2 + b3
                  f2 ~ g*f1 + x
                  f1 ~ x
                  y  ~ h*f2 + x
                  ind := g*h', args = list()),
  cross = list(m = 'f1 =~ a1 + a2 + a3
                    f2 =~ b1 + b2 + b3 + a3
                    a1 ~~ b1', args = list()),
  cfa_means = list(m = 'f1 =~ a1 + a2 + a3
                        f2 =~ b1 + b2 + b3', args = list(meanstructure = TRUE)),
  growth = list(m = 'i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4
                     s =~ 0*t1 + 1*t2 + 2*t3 + 3*t4', args = list(growth = TRUE)),
  # two terminal outcomes: sem() lets their disturbances covary
  two_outcomes = list(m = 'a1 ~ x
                           b1 ~ x', args = list()),
  latent_and_outcome = list(m = 'f1 =~ a1 + a2 + a3
                                 f2 =~ b1 + b2 + b3
                                 f2 ~ f1
                                 y ~ f1', args = list()),
  # equal loadings, a freed first loading with the variance fixed instead
  constrained = list(m = 'f1 =~ NA*a1 + l*a1 + l*a2 + a3
                          f2 =~ b1 + b2 + b3
                          f1 ~~ 1*f1
                          b1 ~ 0.1*1', args = list())
)
fm <- c("chisq", "df", "pvalue", "baseline.chisq", "baseline.df", "cfi", "tli",
        "rmsea", "rmsea.ci.lower", "rmsea.ci.upper", "srmr", "logl", "aic", "bic", "npar")
tab <- function(fit) {
  pe <- as.data.frame(unclass(parameterEstimates(fit, standardized = TRUE)))
  if (is.null(pe$label)) pe$label <- ""
  pe[, c("lhs", "op", "rhs", "label", "est", "se", "z", "pvalue", "std.all")]
}
out <- list(lavaan = as.character(packageVersion("lavaan")), R = R.version.string, cases = list())
for (nm in names(models)) {
  spec <- models[[nm]]
  fun <- if (isTRUE(spec$args$growth)) growth else sem
  a <- spec$args; a$growth <- NULL
  ml <- do.call(fun, c(list(model = spec$m, data = d), a))
  res <- list(model = spec$m, args = spec$args, ml = tab(ml), ml_fit = as.list(fitMeasures(ml, fm)))
  mlm <- do.call(fun, c(list(model = spec$m, data = d, estimator = "MLM"), a))
  res$mlm <- tab(mlm)
  res$mlm_fit <- as.list(fitMeasures(mlm, c("chisq.scaled", "pvalue.scaled", "chisq.scaling.factor")))
  if (nm %in% c("cfa", "growth", "cfa_means")) {
    fs <- lavPredict(ml)
    res$factor_scores <- lapply(as.data.frame(fs[1:8, , drop = FALSE]), as.numeric)
  }
  out$cases[[nm]] <- res
}
write_json(out, file.path(here, "sem_latent_lavaan_R.json"), digits = NA, auto_unbox = TRUE, na = "null")
