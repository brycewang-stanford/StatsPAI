# Reference values for tests/reference_parity/test_simex_r_parity.py.
# R simex 1.8 (simex() and mcsimex()) with asymptotic = TRUE on
# simex_det_data.csv, fed the pseudo-errors of simex_det_draws.csv.gz and the
# re-misclassified vectors of simex_det_mc.csv.gz (both written by
# _generate_simex_det_data.py) instead of its own random draws, so that
# sp.simex can be compared deterministically. The draws are injected by
# replacing stats::rnorm in simex's imports environment and simex::misclass
# in its namespace; nothing else in simex() / mcsimex() consumes random
# numbers. B = 20, lambda = (0.5, 1, 1.5, 2), jackknife variance
# extrapolated with the quadratic.
# Run from this directory. Writes simex_det_R.txt ("key value" per line).
suppressPackageStartupMessages(library(simex))
df <- read.csv("simex_det_data.csv")
df$w3 <- df$w + 3  # the nonlinear case: a mean far from 0 bends the intercept path
draws <- as.matrix(read.csv(gzfile("simex_det_draws.csv.gz")))
mcv <- read.csv(gzfile("simex_det_mc.csv.gz"))
# glm's default deviance tolerance (1e-8) leaves the coefficients and the
# information-matrix weights about 1e-6 from the optimum; tighten it so the
# comparison measures the SIMEX layer, not the stopping rule (update() keeps
# the control argument for every simulated refit).
ctl <- glm.control(epsilon = 1e-16, maxit = 100)
# (some refits stop at maxit with "did not converge": the 1e-16 relative
# deviance change is below rounding noise; the coefficients are converged.)
counter <- 0L
fake_rnorm <- function(n, mean = 0, sd = 1) {
  counter <<- counter + 1L
  stopifnot(n == nrow(draws))
  draws[, counter]
}
imp <- parent.env(asNamespace("simex"))
unlockBinding("rnorm", imp)
assign("rnorm", fake_rnorm, envir = imp)
mc_prefix <- "x"
fake_misclass <- function(data.org, mc.matrix, k = 1) {
  counter <<- counter + 1L
  v <- mcv[[paste0(mc_prefix, counter - 1L)]]
  out <- data.org
  out[[1]] <- factor(v, levels = levels(data.org[[1]]))
  out
}
assignInNamespace("misclass", fake_misclass, "simex")

out <- character(0)
add <- function(key, value) out <<- c(out, sprintf("%s %.17g", key, value))
record <- function(case, s) {
  nm <- names(coef(s))
  for (i in seq_along(nm)) {
    key <- gsub("[()]", "", nm[i])
    add(sprintf("%s_b_%s", case, key), coef(s)[[i]])
    add(sprintf("%s_jse_%s", case, key), sqrt(s$variance.jackknife[i, i]))
    add(sprintf("%s_ase_%s", case, key), sqrt(s$variance.asymptotic[i, i]))
  }
}
lam <- c(0.5, 1, 1.5, 2)
B <- 20

fits <- list(
  gaussian = lm(y ~ w + z, data = df, x = TRUE, y = TRUE),
  gaussian1 = lm(y ~ w3, data = df, x = TRUE, y = TRUE),
  logit = glm(d ~ w + z, family = binomial, data = df, control = ctl, x = TRUE, y = TRUE),
  poisson = glm(cnt ~ w + z, family = poisson, data = df, control = ctl, x = TRUE, y = TRUE)
)
simex_cases <- list(
  c("gaussian", "quadratic"), c("gaussian", "linear"),
  c("gaussian1", "nonlinear"), c("logit", "quadratic"), c("logit", "linear"),
  c("poisson", "quadratic")
)
for (cs in simex_cases) {
  counter <- 0L
  s <- simex(fits[[cs[1]]], SIMEXvariable = if (cs[1] == "gaussian1") "w3" else "w", measurement.error = 0.5,
             lambda = lam, B = B, fitting.method = cs[2],
             jackknife.estimation = "quadratic", asymptotic = TRUE)
  stopifnot(counter == B * length(lam))
  record(sprintf("simex_%s_%s", cs[1], cs[2]), s)
}

dfm <- df
dfm$xs <- factor(dfm$xs, levels = c(0, 1))
dfm$dm <- factor(dfm$dm, levels = c(0, 1))
P_x <- matrix(c(0.85, 0.15, 0.10, 0.90), 2, dimnames = list(c("0", "1"), c("0", "1")))
P_y <- matrix(c(0.90, 0.10, 0.20, 0.80), 2, dimnames = list(c("0", "1"), c("0", "1")))
mc_cases <- list(
  list("gaussian", "quadratic", "x"), list("gaussian", "loglinear", "x"),
  list("logit", "quadratic", "x"), list("logit", "quadratic", "y"),
  list("logit", "loglinear", "y")
)
for (cs in mc_cases) {
  fam <- cs[[1]]; ex <- cs[[2]]; mc_prefix <- cs[[3]]
  counter <- 0L
  if (mc_prefix == "x") {
    f <- if (fam == "gaussian") yb ~ xs + z else db ~ xs + z
    m <- glm(f, family = if (fam == "gaussian") gaussian else binomial,
             data = dfm, control = ctl, x = TRUE, y = TRUE)
    s <- mcsimex(m, SIMEXvariable = "xs", mc.matrix = P_x, lambda = lam,
                 B = B, fitting.method = ex, jackknife.estimation = "quadratic",
                 asymptotic = TRUE)
  } else {
    m <- glm(dm ~ z, family = binomial, data = dfm, control = ctl, x = TRUE,
             y = TRUE)
    s <- mcsimex(m, SIMEXvariable = "dm", mc.matrix = P_y, lambda = lam,
                 B = B, fitting.method = ex, jackknife.estimation = "quadratic",
                 asymptotic = TRUE)
  }
  stopifnot(counter == B * length(lam))
  record(sprintf("mc%s_%s_%s", mc_prefix, fam, ex), s)
}
add("simex_version", as.numeric(gsub("\\.", "", as.character(packageVersion("simex")))))
writeLines(out, "simex_det_R.txt")
cat(out, sep = "\n")
