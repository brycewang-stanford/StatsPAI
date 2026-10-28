# StatsPAI bivariate probit parity (R side) -- Module 91.
#
# Reads data/91_biprobit.csv and fits the seemingly unrelated bivariate
# probit with GJRM::gjrm (Marra and Radice): two probit margins joined by
# a Gaussian copula, model = "B", which is the bivariate probit. GJRM's
# association parameter theta.star is atanh(rho) (Stata's athrho), and its
# standard errors come from the observed Hessian, the quantity Stata
# reports. VGAM::vglm(binom2.rho) uses expected information, so its
# standard errors are a different quantity; it is not the reference.
#
# GJRM evaluates the bivariate normal CDF with VGAM::pbinorm, accurate to
# about 1e-7, so its maximum sits about 1e-7 (relative) from Stata's and
# StatsPAI's, which evaluate the CDF to 1e-16 (module 91's Python
# docstring). Inside the 1e-6 budget.
#
# GJRM pulls in a large dependency tree (copula, VineCopula, scam,
# distrEx ...); a private library is fine: point STATSPAI_GJRM_LIB at it.
#
# Tolerance: rel < 1e-6.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

.gjrm_lib <- Sys.getenv("STATSPAI_GJRM_LIB")
if (nzchar(.gjrm_lib)) .libPaths(c(.gjrm_lib, .libPaths()))
suppressPackageStartupMessages(library(GJRM))

MODULE <- "91_biprobit"

df <- read_csv_strict(MODULE)

fit <- GJRM::gjrm(
  list(y1 ~ x1 + x3, y2 ~ x2 + x3),
  data = df, margins = c("probit", "probit"), model = "B", copula = "N"
)
b <- coef(fit)
se <- sqrt(diag(fit$Vb))
# coef() order: eq1 (Intercept, x1, x3), eq2 (Intercept, x2, x3), theta.star
stopifnot(length(b) == 7L, names(b)[7] == "theta.star")

rows <- list()
add <- function(stat, est, s = NA) {
  rows[[length(rows) + 1L]] <<- parity_row(
    module    = MODULE,
    statistic = stat,
    estimate  = unname(est),
    se        = if (is.na(s)) NULL else unname(s),
    n         = nrow(df)
  )
}
add("eq1_b_x1", b[2], se[2])
add("eq1_b_x3", b[3], se[3])
add("eq1_b_cons", b[1], se[1])
add("eq2_b_x2", b[5], se[5])
add("eq2_b_x3", b[6], se[6])
add("eq2_b_cons", b[4], se[4])
add("athrho", b[7], se[7])
add("rho", tanh(b[7]))
ll <- as.numeric(logLik(fit))
add("loglik", ll)

ll0 <- as.numeric(logLik(glm(y1 ~ x1 + x3, family = binomial("probit"), data = df,
                              control = glm.control(epsilon = 1e-14, maxit = 100)))) +
  as.numeric(logLik(glm(y2 ~ x2 + x3, family = binomial("probit"), data = df,
                        control = glm.control(epsilon = 1e-14, maxit = 100))))
add("lr_rho0", 2 * (ll - ll0))

write_results(
  MODULE, rows,
  extra = list(
    n_rows = length(rows),
    gjrm_call = "gjrm(list(y1 ~ x1 + x3, y2 ~ x2 + x3), margins = c('probit','probit'), model = 'B', copula = 'N')",
    gjrm_version = as.character(utils::packageVersion("GJRM"))
  )
)
