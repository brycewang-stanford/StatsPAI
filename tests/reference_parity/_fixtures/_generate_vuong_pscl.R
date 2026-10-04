# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_vuong_pscl_parity.py
#
# Requires: R with pscl, MASS and jsonlite.
# Run:      Rscript _generate_vuong_pscl.R      (from this directory)
#
# pscl::vuong only prints its result, so the statistic is rebuilt here from
# the per-observation log-likelihoods of R's own fits, with the formula
# pscl::vuong uses (sample standard deviation with n - 1; corrections
# (k1 - k2) / n for AIC and (k1 - k2) log(n) / (2 n) for BIC, subtracted
# from the mean log-likelihood ratio). The printed output of pscl::vuong is
# stored next to it and the test checks the two agree to the printed digits.
#
# The per-observation log-likelihoods are stored too: they pin the
# `llobs` that StatsPAI results now carry, independently of the test built
# on them.
# ---------------------------------------------------------------------------
suppressMessages({library(pscl); library(MASS); library(jsonlite)})
set.seed(20261004)
n <- 1200
x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.5)
mu <- exp(0.5 + 0.4 * x1 - 0.3 * x2) * rgamma(n, shape = 2.5, rate = 2.5)
zero <- runif(n) < plogis(-1 + 0.6 * x1)
d <- data.frame(y = ifelse(zero, 0L, rpois(n, mu)), x1 = x1, x2 = x2)
write.csv(format(d, digits = 17), "vuong_data.csv", row.names = FALSE, quote = FALSE)
d <- read.csv("vuong_data.csv")

fits <- list(
    poisson = glm(y ~ x1 + x2, family = poisson, data = d, control = glm.control(epsilon = 1e-12)),
    nb2 = glm.nb(y ~ x1 + x2, data = d, control = glm.control(epsilon = 1e-12, maxit = 200)),
    zip = zeroinfl(y ~ x1 + x2, data = d, dist = "poisson", reltol = 1e-14),
    zinb = zeroinfl(y ~ x1 + x2, data = d, dist = "negbin", reltol = 1e-14),
    hurdle = hurdle(y ~ x1 + x2, data = d, dist = "poisson", reltol = 1e-14)
)
llobs <- function(m) {
    if (inherits(m, "negbin")) return(dnbinom(d$y, size = m$theta, mu = fitted(m), log = TRUE))
    if (inherits(m, "glm")) return(dpois(d$y, fitted(m), log = TRUE))
    p <- predict(m, type = "prob")
    log(p[cbind(seq_len(n), d$y + 1)])
}
# Every estimated parameter, the dispersion of a negative binomial included.
# pscl::vuong itself uses length(coef()), which leaves the dispersion out, so
# its printed AIC- and BIC-corrected statistics differ from the ones stored
# here when exactly one model of a pair is negative binomial (zip_nb2 and
# nb2_hurdle). Its raw statistic is the same in every pair.
npar <- function(m) {
    nb <- inherits(m, "negbin") || (inherits(m, "zeroinfl") && m$dist == "negbin")
    length(coef(m)) + nb
}
vu <- function(a, b) {
    m <- llobs(fits[[a]]) - llobs(fits[[b]]); k <- npar(fits[[a]]) - npar(fits[[b]])
    corr <- c(raw = 0, aic = k / n, bic = k * log(n) / (2 * n))
    z <- sapply(corr, function(cc) sum(m - cc) / (sd(m - cc) * sqrt(n)))
    printed <- capture.output(vuong(fits[[a]], fits[[b]]))
    list(z = as.list(z), k1 = npar(fits[[a]]), k2 = npar(fits[[b]]), printed = printed)
}
out <- list()
out$llobs <- lapply(fits, llobs)
out$loglik <- lapply(fits, function(m) as.numeric(logLik(m)))
out$pairs <- list(zip_poisson = vu("zip", "poisson"), zinb_nb2 = vu("zinb", "nb2"),
                  zip_nb2 = vu("zip", "nb2"), hurdle_zip = vu("hurdle", "zip"),
                  nb2_hurdle = vu("nb2", "hurdle"))
out$`_meta` <- list(R = R.version.string, pscl = as.character(packageVersion("pscl")))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), "vuong_pscl.json")
