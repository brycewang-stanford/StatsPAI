# ---------------------------------------------------------------------------
# Reference for tests/reference_parity/test_sem_fiml_lavaan_parity.py
#
# lavaan with missing = "ml" (full-information maximum likelihood, observed
# information) on sem_latent.csv with values removed: a path model, a CFA, a
# structural model with an exogenous covariate, and a growth curve.
#
# Requires: lavaan (0.6-21), jsonlite.
# Run:      Rscript tests/reference_parity/_fixtures/_generate_sem_fiml_lavaan.R
#           (from the repository root; rewrites sem_missing.csv as well)
# ---------------------------------------------------------------------------
suppressMessages({library(lavaan); library(jsonlite)})
here <- Sys.getenv("SEM_FIX", "tests/reference_parity/_fixtures")
d <- read.csv("tests/reference_parity/_fixtures/sem_latent.csv")
set.seed(20261009)
n <- nrow(d)
# missing completely at random in three columns, and at random given x in two
for (v in c("a2", "b3", "t2")) d[[v]][runif(n) < 0.15] <- NA
d$y[runif(n) < plogis(-1.5 + 1.0 * d$x)] <- NA
d$t4[runif(n) < plogis(-1.2 + 0.4 * d$t1)] <- NA
d$b1[d$x > 1.3] <- NA
write.csv(d, file.path(here, "sem_missing.csv"), row.names = FALSE, na = "")
d <- read.csv(file.path(here, "sem_missing.csv"))
models <- list(
  path = list(m = 'a1 ~ p*x
                   y ~ q*a1 + x
                   ind := p*q', growth = FALSE),
  cfa = list(m = 'f1 =~ a1 + a2 + a3
                  f2 =~ b1 + b2 + b3', growth = FALSE),
  sem = list(m = 'f1 =~ a1 + a2 + a3
                  f2 =~ b1 + b2 + b3
                  f2 ~ g*f1 + x
                  f1 ~ x
                  y  ~ h*f2 + x
                  ind := g*h', growth = FALSE),
  growth = list(m = 'i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4
                     s =~ 0*t1 + 1*t2 + 2*t3 + 3*t4', growth = TRUE)
)
fm <- c("chisq", "df", "pvalue", "baseline.chisq", "baseline.df", "cfi", "tli",
        "rmsea", "srmr", "logl", "unrestricted.logl", "aic", "bic", "npar", "ntotal")
out <- list(lavaan = as.character(packageVersion("lavaan")), R = R.version.string, cases = list())
for (nm in names(models)) {
  spec <- models[[nm]]
  fun <- if (spec$growth) growth else sem
  fit <- fun(spec$m, data = d, missing = "ml")
  pe <- as.data.frame(unclass(parameterEstimates(fit, standardized = TRUE)))
  if (is.null(pe$label)) pe$label <- ""
  res <- list(model = spec$m, growth = spec$growth,
              table = pe[, c("lhs", "op", "rhs", "label", "est", "se", "std.all")],
              fit = as.list(fitMeasures(fit, fm)), nobs = lavInspect(fit, "nobs"),
              npatterns = length(lavInspect(fit, "patterns")[, 1]))
  if (nm %in% c("cfa", "growth")) {
    fs <- lavPredict(fit)
    res$factor_scores <- lapply(as.data.frame(fs[1:12, , drop = FALSE]), as.numeric)
  }
  out$cases[[nm]] <- res
}
write_json(out, file.path(here, "sem_fiml_lavaan_R.json"), digits = NA, auto_unbox = TRUE, na = "null")
