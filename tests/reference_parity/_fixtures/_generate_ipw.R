#!/usr/bin/env Rscript
# Base-R reference values for sp.ipw parity (no CRAN packages).
#
# sp.ipw fits an UNPENALIZED logistic propensity with intercept
# (statsmodels GLM Binomial; the sklearn fallback also sets
# penalty=None) — see src/statspai/inference/ipw.py:202-233.  That is
# exactly the MLE that base R's stats::glm(family = binomial) computes,
# so cross-language agreement holds to IRLS convergence tolerance
# (both R and statsmodels iterate deviance to ~1e-8).
#
# Hajek (normalized) weighting convention matched here is sp.ipw's
# default normalize=TRUE (ipw.py:263-269):
#   ATE: w1 = t/ps,  w0 = (1-t)/(1-ps), each normalized by its own sum.
#   ATT: w1 = t,     w0 = (1-t)*ps/(1-ps), each normalized by its own sum.
#
# JSON is written with sprintf/%.17g (full double precision) to keep
# this script free of jsonlite.

df <- read.csv("tests/reference_parity/_fixtures/ipw_data.csv")

fit <- glm(t ~ x1 + x2, family = binomial, data = df)
ps <- fitted(fit)

t <- df$t
y <- df$y

# --- Hajek ATE ---
w1 <- t / ps
w0 <- (1 - t) / (1 - ps)
hajek_ate <- sum(w1 * y) / sum(w1) - sum(w0 * y) / sum(w0)

# --- Hajek ATT ---
w1_att <- t
w0_att <- (1 - t) * ps / (1 - ps)
hajek_att <- sum(w1_att * y) / sum(w1_att) - sum(w0_att * y) / sum(w0_att)

co <- coef(fit)

num <- function(x) sprintf("%.17g", x)

json <- paste0(
  "{\n",
  '  "meta": {\n',
  '    "R_version": "', R.version.string, '",\n',
  '    "formula": "t ~ x1 + x2 (binomial logit, base stats::glm)",\n',
  '    "weighting": "Hajek (normalize=TRUE default of sp.ipw)",\n',
  '    "n": ', nrow(df), ",\n",
  '    "n_treated": ', sum(t), "\n",
  "  },\n",
  '  "glm_coef": {\n',
  '    "intercept": ', num(co[["(Intercept)"]]), ",\n",
  '    "x1": ', num(co[["x1"]]), ",\n",
  '    "x2": ', num(co[["x2"]]), "\n",
  "  },\n",
  '  "ps_min": ', num(min(ps)), ",\n",
  '  "ps_max": ', num(max(ps)), ",\n",
  '  "hajek_ate": ', num(hajek_ate), ",\n",
  '  "hajek_att": ', num(hajek_att), "\n",
  "}\n"
)

out <- "tests/reference_parity/_fixtures/ipw_R.json"
writeLines(json, out)
cat(sprintf("Hajek ATE = %.10f\n", hajek_ate))
cat(sprintf("Hajek ATT = %.10f\n", hajek_att))
cat(sprintf("ps range  = [%.6f, %.6f]\n", min(ps), max(ps)))
cat(sprintf("Wrote %s\n", out))
