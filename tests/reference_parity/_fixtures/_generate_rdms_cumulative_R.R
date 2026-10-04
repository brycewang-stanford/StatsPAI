# Reference values for sp.rdms with one score and cumulative cutoffs,
# against R rdmulti 2.0: rdms(Y, X, C) and rdms(Y, X, C, rangemat = ).
library(rdmulti); library(jsonlite)
set.seed(7)
n <- 3000
x <- round(runif(n, 0, 100), 3)
y <- round(1 + 0.01 * x + 0.5 * (x >= 33) + 0.8 * (x >= 66) + rnorm(n, 0, 0.4), 6)
write.csv(data.frame(y = y, x = x), "rdms_cumulative_design.csv", row.names = FALSE)
d <- read.csv("rdms_cumulative_design.csv")
pick <- function(r, k = 2) list(
  coefs = unname(as.numeric(r$Coefs))[1:k], coefs_rb = unname(as.numeric(r$B))[1:k],
  var_rb = unname(as.numeric(r$V))[1:k], h = unname(as.numeric(r$H[1, 1:k])),
  Nh = unname(as.numeric(r$Nh[1, 1:k] + r$Nh[2, 1:k])),
  ci_lower = unname(r$CI[1, 1:k]), ci_upper = unname(r$CI[2, 1:k]))
sink("/dev/null")
full <- rdms(d$y, d$x, c(33, 66))
rng <- rdms(d$y, d$x, c(33, 66), rangemat = cbind(c(10, 40), c(50, 90)))
sink()
out <- list(cutoffs = c(33, 66), full = pick(full),
            ranges = list(c(10, 50), c(40, 90)), restricted = pick(rng),
            `_meta` = list(n = n, rdmulti_version = as.character(packageVersion("rdmulti"))))
write_json(out, "rdms_cumulative_R.json", auto_unbox = TRUE, digits = 15, pretty = TRUE)
cat("wrote rdms_cumulative_R.json\n")
