# Reference numbers for sp.residual_balance from balanceHD (Athey, Imbens
# and Wager's own implementation), used as a black box.
#   remotes::install_github("swager/balanceHD"); install.packages(c("quadprog", "jsonlite"))
suppressWarnings(suppressMessages(library(balanceHD)))
d <- read.csv("residual_balance_data.csv")
X <- as.matrix(d[, -(1:2)]); W <- d$W; Y <- d$Y
qp <- function(...) suppressWarnings(approx.balance(..., optimizer = "quadprog"))
est <- function(...) suppressWarnings(
  residualBalance.ate(X, Y, W, fit.method = "none", optimizer = "quadprog", ...))
out <- list(
  version = as.character(packageVersion("balanceHD")),
  gamma_treated_zeta05 = qp(X[W == 1, ], colMeans(X), zeta = 0.5),
  gamma_treated_zeta01 = qp(X[W == 1, ], colMeans(X), zeta = 0.1),
  gamma_treated_zeta09 = qp(X[W == 1, ], colMeans(X), zeta = 0.9),
  gamma_control_att = qp(X[W == 0, ], colMeans(X[W == 1, ]), zeta = 0.5),
  gamma_treated_negative = qp(X[W == 1, ], colMeans(X), zeta = 0.5,
                              allow.negative.weights = TRUE),
  ate_none = est(),
  ate_none_unscaled = est(scale.X = FALSE),
  att_none = est(target.pop = 1),
  atc_none = est(target.pop = 0),
  ate_none_zeta02 = est(zeta = 0.2)
)
writeLines(jsonlite::toJSON(out, digits = 15, auto_unbox = TRUE, pretty = TRUE),
           "residual_balance_balancehd.json")
