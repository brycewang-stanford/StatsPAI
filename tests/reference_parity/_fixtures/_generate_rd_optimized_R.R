# Reference numbers for sp.rd_optimized from optrdd (Imbens and Wager's own
# implementation), used as a black box.
#   remotes::install_github("swager/optrdd"); install.packages(c("quadprog", "jsonlite"))
suppressWarnings(suppressMessages(library(optrdd)))
d <- read.csv("rd_optimized_data.csv")
fit <- function(design, M, sigma.sq) {
  s <- d[d$design == design, ]
  r <- suppressWarnings(optrdd(X = s$x, Y = s$y, W = as.numeric(s$x >= 0),
    max.second.derivative = M, estimation.point = 0, sigma.sq = sigma.sq,
    optimizer = "quadprog", verbose = FALSE))
  list(M = M, sigma_sq = sigma.sq, tau_hat = r$tau.hat,
       tau_plusminus = r$tau.plusminus, max_bias = r$max.bias,
       sampling_se = r$sampling.se, gamma = as.numeric(r$gamma))
}
out <- list(
  version = as.character(packageVersion("optrdd")),
  continuous = fit("continuous", 4, 0.25),
  continuous_tight = fit("continuous", 1, 0.25),
  discrete = fit("discrete", 2, 0.25)
)
writeLines(jsonlite::toJSON(out, digits = 15, auto_unbox = TRUE, pretty = TRUE),
           "rd_optimized_optrdd.json")
