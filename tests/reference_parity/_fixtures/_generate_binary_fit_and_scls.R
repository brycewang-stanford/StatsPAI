# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_binary_r2_and_scls_parity.py
#
# Requires: R with DescTools, micsr (>= 0.1-5) and jsonlite.
# Run:      Rscript _generate_binary_fit_and_scls.R   (from this directory,
#           after _generate_cmtest_micsr.R has written cmtest_data.csv)
#
# Two things on the data of the conditional-moment fixture.
#
# 1. Goodness-of-fit measures of a probit and a logit, from two packages:
#    DescTools::PseudoR2 (McFadden, Cox-Snell, Nagelkerke, Efron,
#    McKelvey-Zavoina, Tjur) and micsr::rsq (Estrella, and the others again
#    under its own names).
# 2. The coefficients of symmetrically censored least squares from
#    micsr::tobit1(method = "trimmed"). Only the coefficients: the standard
#    errors micsr prints for this estimator are not usable (on the book's
#    charitable data they are 20 to 228 for coefficients of order 1), so the
#    test checks Powell's variance by simulation instead.
# ---------------------------------------------------------------------------
suppressMessages({library(DescTools); library(micsr); library(jsonlite)})
d <- read.csv("cmtest_data.csv")
out <- list()
for (lk in c("probit", "logit")) {
    g <- glm(yb ~ x1 + x2 + x3, family = binomial(link = lk), data = d,
             control = glm.control(epsilon = 1e-13, maxit = 100))
    dt <- PseudoR2(g, which = c("McFadden", "CoxSnell", "Nagelkerke", "Efron",
                                "McKelveyZavoina", "Tjur"))
    m <- binomreg(yb ~ x1 + x2 + x3, d, link = lk)
    ms <- sapply(c("mcfadden", "tjur", "estrella", "mckel_zavo", "rss", "lr"),
                 function(t) as.numeric(rsq(m, type = t)))
    out[[lk]] <- list(desctools = as.list(dt), micsr = as.list(ms))
}
sc <- tobit1(yc ~ x1 + x2 + x3, d, method = "trimmed")
out$scls_coef <- unname(coef(sc))
out$scls_names <- names(coef(sc))
out$`_meta` <- list(R = R.version.string, micsr = as.character(packageVersion("micsr")),
                    DescTools = as.character(packageVersion("DescTools")))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), "binary_fit_and_scls.json")
