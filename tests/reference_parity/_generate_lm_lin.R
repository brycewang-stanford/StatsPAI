#!/usr/bin/env Rscript
# Reference values for sp.lm_lin from R estimatr::lm_lin (Lin 2013
# [@lin2013agnostic]).
#
# Writes, under tests/reference_parity/_fixtures/:
#   lm_lin.csv      simulated experiment: y, d, x1, x2, g (4 levels), cl
#   lm_lin_R.json   estimate, se, df and interval for every se_type
#
#   Rscript tests/reference_parity/_generate_lm_lin.R

suppressMessages({
  library(estimatr)
  library(jsonlite)
})
FIX <- "tests/reference_parity/_fixtures"
set.seed(20261004)
n <- 400
x1 <- rnorm(n)
x2 <- rbinom(n, 1, 0.4)
g <- sample(c("a", "b", "c", "d"), n, replace = TRUE)
cl <- (seq_len(n) - 1) %/% 8
dcl <- rbinom(50, 1, 0.4)
d <- rbinom(n, 1, 0.4)
y <- 1 + d * (1 + 0.8 * x1 - 0.5 * x2) + x1 + 0.3 * x2 + 0.5 * (g == "a") + rnorm(n) * (1 + 0.5 * d)
dat <- data.frame(y = y, d = d, dc = dcl[cl + 1], x1 = x1, x2 = x2, g = g, cl = cl)
write.csv(dat, file.path(FIX, "lm_lin.csv"), row.names = FALSE)
dat <- read.csv(file.path(FIX, "lm_lin.csv"))

row <- function(m, t) list(estimate = unname(coef(m)[t]), se = unname(m$std.error[t]),
                           df = unname(m$df[t]), lower = unname(m$conf.low[t]),
                           upper = unname(m$conf.high[t]), p = unname(m$p.value[t]))
out <- list(estimatr_version = as.character(packageVersion("estimatr")))
for (st in c("HC2", "HC0", "HC1", "HC3", "classical")) {
  out[[tolower(st)]] <- row(lm_lin(y ~ d, covariates = ~ x1 + x2 + g, data = dat, se_type = st), "d")
}
for (st in c("CR2", "stata")) {
  out[[tolower(st)]] <- row(lm_lin(y ~ dc, covariates = ~ x1 + x2 + g, data = dat,
                                   clusters = cl, se_type = st), "dc")
}
m <- lm_lin(y ~ d, covariates = ~ x1 + x2 + g, data = dat)
out$coefficients <- as.list(coef(m))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), file.path(FIX, "lm_lin_R.json"))
cat("wrote lm_lin fixtures\n")
