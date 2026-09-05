# StatsPAI fect parity (R side) -- Module 86.
#
# Reads data/86_fect.csv (the staggered two-factor panel written by
# 86_fect.py) and runs fect::fect three times on the same bytes:
#   fe  : method = "fe"
#   ife : method = "ife", r = 2
#   mc  : method = "mc",  lambda = 0.002
# all with force = "two-way", se = FALSE, CV = FALSE, tol = 1e-12,
# max.iteration = 20000. Rows mirror the Python side:
#   <m>_att_avg, <m>_att_avg_unit, <m>_beta_x1, <m>_beta_x2, <m>_mu,
#   <m>_rmse, <m>_att_on_<k> (n = fect's cell count at relative period k;
#   fect's coding: 0 = last untreated period, 1 = first treated period).
# Tolerance: rel_est 1e-6 (deterministic EM fixed point on both sides).

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(fect)
})

MODULE <- "86_fect"

# fect's initial two-way fit is a fixest::feols call with the package-wide
# demeaning tolerance (default 1e-6; the package floor is 2.2e-12). The Python port solves that initial
# least-squares problem exactly, and fect's one-pass "fe" model stops
# right after it, so the default tolerance would leave a ~1e-8 residual
# on the fe rows. Tightening fixest's own global option keeps the R side
# at the exact solution too.
fixest::setFixest_estimation(fixef.tol = 1e-11)

df <- read_csv_strict(MODULE)
n_obs <- nrow(df)

run_one <- function(method, ...) {
  fect::fect(
    Y ~ D + X1 + X2, data = df, index = c("id", "time"),
    method = method, force = "two-way", se = FALSE, CV = FALSE,
    tol = 1e-12, max.iteration = 20000, ...
  )
}

fits <- list(
  fe  = run_one("fe"),
  ife = run_one("ife", r = 2),
  mc  = run_one("mc", lambda = 0.002)
)

rows <- list()
extra <- list(
  force = "two-way", tol = 1e-12, max_iteration = 20000, r_ife = 2,
  lambda_mc = 0.002, fect_version = as.character(packageVersion("fect"))
)

for (m in names(fits)) {
  o <- fits[[m]]
  beta <- as.numeric(o$beta)
  add <- function(stat, est, n = n_obs) {
    rows[[length(rows) + 1L]] <<- parity_row(
      module = MODULE, statistic = stat, estimate = est, n = n
    )
  }
  add(paste0(m, "_att_avg"), o$att.avg)
  add(paste0(m, "_att_avg_unit"), o$att.avg.unit)
  add(paste0(m, "_beta_x1"), beta[1])
  add(paste0(m, "_beta_x2"), beta[2])
  add(paste0(m, "_mu"), o$mu)
  add(paste0(m, "_rmse"), o$rmse)
  for (j in seq_along(o$time)) {
    add(paste0(m, "_att_on_", o$time[j]), o$att[j], n = o$count[j])
  }
  extra[[paste0(m, "_niter")]] <- o$niter
  if (m == "mc") extra[["mc_lambda_norm"]] <- o$lambda.norm
}

write_results(MODULE, rows, extra = extra)
