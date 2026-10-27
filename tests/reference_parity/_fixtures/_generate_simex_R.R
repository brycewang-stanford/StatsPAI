# Reference values for tests/reference_parity/test_simex_r_screen.py
# R simex (Lederer and Kuechenhoff) on simex_data.csv: linear and logit naive
# models, measurement error SD 0.5 on w, lambda = (0.5, 1, 1.5, 2),
# B = 2000, quadratic and linear extrapolants, jackknife variance
# extrapolated with the quadratic. Stochastic: a screen, not parity.
# (fitting.method = "nonlinear" stops in stats::nls on these data: the path
# of z is flat up to simulation noise.)
# Run from this directory. Writes simex_R.txt ("key value" per line).
suppressPackageStartupMessages(library(simex))
df <- read.csv("simex_data.csv")
set.seed(20261028)
out <- character(0)
add <- function(key, value) out <<- c(out, sprintf("%s %.17g", key, value))
fits <- list(
  gaussian = lm(y ~ w + z, data = df, x = TRUE, y = TRUE),
  logit = glm(d ~ w + z, family = binomial, data = df, x = TRUE, y = TRUE)
)
for (fam in names(fits)) {
  for (ex in c("quadratic", "linear")) {
    s <- simex(fits[[fam]], SIMEXvariable = "w", measurement.error = 0.5,
               lambda = c(0.5, 1, 1.5, 2), B = 2000, fitting.method = ex,
               jackknife.estimation = "quadratic", asymptotic = FALSE)
    for (term in c("(Intercept)", "w", "z")) {
      key <- if (term == "(Intercept)") "Intercept" else term
      add(sprintf("%s_%s_b_%s", fam, ex, key), coef(s)[[term]])
      add(sprintf("%s_%s_se_%s", fam, ex, key),
          sqrt(s$variance.jackknife[term, term]))
    }
  }
}
add("simex_version", as.numeric(gsub("\\.", "", as.character(packageVersion("simex")))))
writeLines(out, "simex_R.txt")
cat(out, sep = "\n")
