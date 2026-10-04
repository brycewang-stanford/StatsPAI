# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_nlogit_parity.py
#
# Requires: R with mlogit (>= 1.1), dfidx and jsonlite.
# Run:      Rscript _generate_nlogit_mlogit.R      (from this directory)
#
# Choices are drawn from a nested logit with two nests of two alternatives
# (dissimilarity parameters 0.5 and 0.8), two alternative-varying regressors
# and alternative-specific constants. mlogit::mlogit(nests = ...) is the
# reference, with one dissimilarity parameter per nest and with a common
# one. A Stata reference on the same CSV is in _generate_nlogit_stata.do.
# ---------------------------------------------------------------------------
suppressMessages({library(mlogit); library(dfidx); library(jsonlite)})
set.seed(20261008)
n <- 2500; alts <- c("a1", "a2", "b1", "b2"); nest <- c(a1 = "A", a2 = "A", b1 = "B", b2 = "B")
lam <- c(A = 0.5, B = 0.8); asc <- c(a1 = 0, a2 = 0.3, b1 = -0.2, b2 = 0.4)
d <- expand.grid(alt = alts, id = seq_len(n), stringsAsFactors = FALSE)[, c("id", "alt")]
d$x1 <- rnorm(nrow(d)); d$x2 <- runif(nrow(d), -1, 1)
d$clust <- (d$id - 1) %% 50 + 1
V <- asc[d$alt] + 0.8 * d$x1 - 0.6 * d$x2
d$chosen <- 0L
for (i in seq_len(n)) {
    r <- which(d$id == i); v <- V[r]; k <- nest[d$alt[r]]
    iv <- tapply(exp(v / lam[k]), k, function(z) log(sum(z)))
    pk <- exp(lam[names(iv)] * iv); pk <- pk / sum(pk)
    p <- exp(v / lam[k]) / exp(iv[k]) * pk[k]
    d$chosen[r[sample.int(4, 1, prob = p)]] <- 1L
}
write.csv(format(d, digits = 17), "nlogit_data.csv", row.names = FALSE, quote = FALSE)
d <- read.csv("nlogit_data.csv")
dd <- dfidx(d, idx = c("id", "alt"), choice = "chosen")
nests <- list(A = c("a1", "a2"), B = c("b1", "b2"))
pack <- function(m) list(names = names(coef(m)), b = unname(coef(m)),
                         se = unname(sqrt(diag(vcov(m)))), ll = as.numeric(logLik(m)))
out <- list(
    separate = pack(mlogit(chosen ~ x1 + x2, dd, nests = nests, un.nest.el = FALSE,
                           tol = 1e-14, ftol = 1e-14, steptol = 1e-14, iterlim = 500)),
    common = pack(mlogit(chosen ~ x1 + x2, dd, nests = nests, un.nest.el = TRUE,
                         tol = 1e-14, ftol = 1e-14, steptol = 1e-14, iterlim = 500)),
    no_constants = pack(mlogit(chosen ~ x1 + x2 | 0, dd, nests = nests, un.nest.el = FALSE,
                               tol = 1e-14, ftol = 1e-14, steptol = 1e-14, iterlim = 500)),
    conditional = pack(mlogit(chosen ~ x1 + x2, dd))
)
out$`_meta` <- list(R = R.version.string, mlogit = as.character(packageVersion("mlogit")))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), "nlogit_mlogit.json")
