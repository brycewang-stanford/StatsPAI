# Reference values for test_marginaleffects_r_parity.py.
# R marginaleffects (GPL-3) is used as a black box: only its output is read.
suppressPackageStartupMessages({
  library(marginaleffects)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
d <- read.csv(file.path(here, "marginaleffects_data.csv"))
d$g <- factor(d$g)

grab <- function(tab, keys = c("term", "contrast")) {
  tab <- as.data.frame(tab)
  keep <- intersect(c(keys, "estimate", "std.error"), names(tab))
  out <- tab[, keep, drop = FALSE]
  for (k in intersect(keys, names(out))) out[[k]] <- as.character(out[[k]])
  out
}

m_lm <- lm(yc ~ x + I(x^2) + b + g, data = d)
# Tight convergence: glm's default epsilon = 1e-8 leaves the covariance
# (taken from the last IRLS weights) about 1e-5 away from the inverse
# information at the reported estimates.
tight <- glm.control(epsilon = 1e-14, maxit = 100)
m_logit <- glm(y ~ x * b + z + g, data = d, family = binomial(), control = tight)
m_probit <- glm(y ~ x + z + b, data = d, family = binomial(link = "probit"),
                control = tight)
m_pois <- glm(cnt ~ x + z + g, data = d, family = poisson(), control = tight)

out <- list(versions = list(
  marginaleffects = as.character(packageVersion("marginaleffects")),
  R = R.version.string
))
models <- list(lm = m_lm, logit = m_logit, probit = m_probit, poisson = m_pois)
for (nm in names(models)) {
  m <- models[[nm]]
  r <- list()
  r$avg_predictions <- grab(avg_predictions(m), character(0))
  r$avg_predictions_by_g <- grab(avg_predictions(m, by = "g"), "g")
  r$avg_comparisons <- grab(avg_comparisons(m))
  r$avg_slopes <- grab(avg_slopes(m))
  r$avg_slopes_eyex <- grab(avg_slopes(m, variables = "x", slope = "eyex"))
  r$avg_slopes_eydx <- grab(avg_slopes(m, variables = "x", slope = "eydx"))
  r$avg_slopes_dyex <- grab(avg_slopes(m, variables = "x", slope = "dyex"))
  r$avg_slopes_by_g <- grab(avg_slopes(m, variables = "x", by = "g"), c("term", "g"))
  r$avg_slopes_wts <- grab(avg_slopes(m, variables = "x", wts = "w"))
  # Unit-level slopes: the default forward-difference Jacobian of a
  # finite-difference slope is noisy at ~1e-5; Richardson extrapolation
  # removes it.
  r$slopes_head <- grab(slopes(m, variables = "x", numderiv = "richardson"))[1:5, , drop = FALSE]
  # For a GLM, predictions() defaults to type = "invlink(link)", which
  # reports no standard error; ask for the response scale explicitly.
  r$predictions_head <- grab(predictions(m, type = "response"), character(0))[1:5, , drop = FALSE]
  if (inherits(m, "glm")) r$predictions_link_head <- grab(predictions(m, type = "link"), character(0))[1:5, , drop = FALSE]
  r$comparisons_x_sd <- grab(avg_comparisons(m, variables = list(x = "sd")))
  r$comparisons_x_5 <- grab(avg_comparisons(m, variables = list(x = 5)))
  r$comparisons_x_2sd <- grab(avg_comparisons(m, variables = list(x = "2sd")))
  r$comparisons_x_iqr <- grab(avg_comparisons(m, variables = list(x = "iqr")))
  r$comparisons_x_minmax <- grab(avg_comparisons(m, variables = list(x = "minmax")))
  r$comparisons_x_pair <- grab(avg_comparisons(m, variables = list(x = c(-1, 2))))
  r$hypothesis_b2_b3 <- grab(hypotheses(m, hypothesis = "b2 - b3 = 0"), "term")
  out[[nm]] <- r
}
# Ratios on the response scale (meaningful for the GLMs).
for (nm in c("logit", "poisson")) {
  m <- models[[nm]]
  out[[nm]]$ratioavg_g <- grab(avg_comparisons(m, variables = "g", comparison = "ratioavg"))
  out[[nm]]$lnratioavg_g <- grab(avg_comparisons(m, variables = "g", comparison = "lnratioavg"))
  out[[nm]]$ratio_head <- grab(comparisons(m, variables = "x", comparison = "ratio"))[1:5, , drop = FALSE]
  out[[nm]]$lnratio_avg <- grab(avg_comparisons(m, variables = "x", comparison = "lnratio"))
}
# Binary 0/1 variable through comparisons and a hypothesis on a table.
ac <- avg_comparisons(m_logit, variables = "g")
out$logit$hypothesis_on_table <- grab(hypotheses(ac, hypothesis = "b1 - b2 = 0"), "term")
out$logit$comparisons_b <- grab(avg_comparisons(m_logit, variables = "b"))
out$logit$ratio_by_g <- grab(comparisons(m_logit, variables = "b", comparison = "ratio", by = "g"),
                             c("term", "contrast", "g"))

writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           file.path(here, "marginaleffects_R.json"))
