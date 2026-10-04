# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_cmtest_micsr_parity.py
#
# Requires: R with micsr (>= 0.1-5) and jsonlite.
# Run:      Rscript _generate_cmtest_micsr.R      (from this directory)
#
# micsr::cmtest is the reference for the conditional moment tests of a probit
# and of a tobit fit (Croissant 2025, Microeconometrics with R, sections on
# the binomial and the censored regression models). micsr is GPL: it is run
# here as a program and only its printed numbers are stored; none of its
# source is reproduced in StatsPAI.
#
# The data are simulated so that both nulls are false to a moderate degree
# (a skewed, heteroskedastic latent error): statistics near zero would pin
# nothing.
# ---------------------------------------------------------------------------
suppressMessages({library(micsr); library(jsonlite)})
set.seed(20261004)
n <- 1200
x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.4); x3 <- runif(n, -1, 1)
e <- (rchisq(n, 6) - 6) / sqrt(12) * exp(0.25 * x1)
ystar <- 0.4 + 0.8 * x1 - 0.6 * x2 + 0.5 * x3 + e
d <- data.frame(yb = as.numeric(ystar > 0), yc = pmax(ystar, 0), x1 = x1, x2 = x2, x3 = x3)
write.csv(format(d, digits = 17), "cmtest_data.csv", row.names = FALSE, quote = FALSE)
d <- read.csv("cmtest_data.csv")

ht <- function(z) list(statistic = unname(z$statistic), df = unname(z$parameter), pvalue = z$p.value)
out <- list()
pb <- binomreg(yb ~ x1 + x2 + x3, d, link = "probit")
out$probit_coef <- unname(coef(pb))
for (t in c("normality", "heterosc"))
    out[[paste0("probit_", t)]] <- ht(cmtest(pb, test = t))
tb <- tobit1(yc ~ x1 + x2 + x3, d)
out$tobit_coef <- unname(coef(tb))
for (t in c("normality", "heterosc", "skewness", "kurtosis")) for (o in c(FALSE, TRUE))
    out[[paste0("tobit_", t, if (o) "_opg" else "")]] <- ht(cmtest(tb, test = t, opg = o))
out$`_meta` <- list(R = R.version.string, micsr = as.character(packageVersion("micsr")))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), "cmtest_micsr.json")
