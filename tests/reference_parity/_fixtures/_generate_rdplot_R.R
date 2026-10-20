# Black-box fixture for the clean-room rewrite of sp.rdplot's numbers
# (statspai.rd._rdplot_numbers). Runs R rdrobust::rdplot (rdrobust 4.0.0,
# GPL-3) only through its documented interface and records its return
# values; nothing here reads or reproduces its source.
#
#   Rscript tests/reference_parity/_fixtures/_generate_rdplot_R.R
#
# Writes, next to this file:
#   rdplot_cont.csv  continuous running variable, n = 900, outcome with
#                    curvature, two covariates, weights
#   rdplot_mass.csv  running variable on a 0.05 grid (>= 20% ties per side)
#   rdplot_qs.csv    201 / 191 observations left / right, so that quantile
#                    bin edges at j/J fall exactly on observations when
#                    J divides 200 or 190
#   rdplot_R.json    one entry per call: the arguments and R's J, J_IMSE,
#                    J_MV, coef, coef_covs, vars_bins (all columns),
#                    vars_poly, h, N, N_h, binselect, scale, rscale,
#                    bin_avg, bin_med
#
# Calls that R itself refuses (R 4.0.0 stops when a bin is empty, when
# `scale` is not an integer multiple or has two entries) are recorded with
# their error message instead of numbers.

suppressMessages({
  library(rdrobust)
  library(jsonlite)
})
stopifnot(as.character(packageVersion("rdrobust")) == "4.0.0")

here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) NULL)
if (is.null(here)) {
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  here <- dirname(normalizePath(f))
}

# ---- data -------------------------------------------------------------------
set.seed(20261020)
n <- 900
x <- c(runif(n / 2, -1, 0), 0.9 * rbeta(n / 2, 1.2, 1.6))
x <- sample(x)
z1 <- rnorm(n) + 0.5 * x
z2 <- rbinom(n, 1, 0.4)
w <- runif(n, 0.5, 1.5)
y <- 0.3 + 0.8 * x - 0.6 * x^2 + 0.9 * sin(3 * x) + 0.7 * (x >= 0) +
  0.4 * z1 - 0.3 * z2 + rnorm(n, 0, 0.25 + 0.2 * abs(x))
cont <- data.frame(y = y, x = x, z1 = z1, z2 = z2, w = w)
write.csv(cont, file.path(here, "rdplot_cont.csv"), row.names = FALSE)

nm <- 700
xm <- round(runif(nm, -1, 1) / 0.05) * 0.05
xm[xm == 0] <- 0.05
ym <- 1 + 0.5 * xm + 0.4 * xm^2 + 0.6 * (xm >= 0) + rnorm(nm, 0, 0.3)
mass <- data.frame(y = ym, x = xm)
write.csv(mass, file.path(here, "rdplot_mass.csv"), row.names = FALSE)

xq <- c(-runif(201), runif(191))
yq <- 0.5 + xq - 0.8 * xq^2 + 0.5 * (xq >= 0) + rnorm(392, 0, 0.3)
write.csv(data.frame(y = yq, x = xq), file.path(here, "rdplot_qs.csv"), row.names = FALSE)

# Re-read so that R works on exactly the bytes Python reads.
cont <- read.csv(file.path(here, "rdplot_cont.csv"))
mass <- read.csv(file.path(here, "rdplot_mass.csv"))
qsd <- read.csv(file.path(here, "rdplot_qs.csv"))

# ---- calls ------------------------------------------------------------------
num <- function(v) if (is.null(v)) NULL else unname(as.numeric(v))
POLY_IDX <- c(seq(1, 1000, by = 9), 500, 501, 1000)

record <- function(args, data) {
  call_args <- args
  call_args$y <- data$y
  call_args$x <- data$x
  if (!is.null(args$covs)) call_args$covs <- as.matrix(data[, args$covs, drop = FALSE])
  if (!is.null(args$weights)) call_args$weights <- data[[args$weights]]
  call_args$hide <- TRUE
  r <- tryCatch(suppressWarnings(suppressMessages(do.call(rdplot, call_args))),
                error = function(e) conditionMessage(e))
  if (is.character(r)) return(list(args = args, error = r))
  vb <- r$vars_bins
  list(
    args = args,
    J = num(r$J), J_IMSE = num(r$J_IMSE), J_MV = num(r$J_MV),
    coef = num(r$coef), coef_dim = dim(r$coef),
    coef_covs = num(r$coef_covs),
    h = num(r$h), N = num(r$N), N_h = num(r$N_h),
    binselect = r$binselect, kernel = r$kernel,
    scale = num(r$scale), rscale = num(r$rscale),
    bin_avg = num(r$bin_avg), bin_med = num(r$bin_med),
    vars_bins = lapply(as.list(vb), num),
    # every 9th point of the 2 x 500 grid, with R's 1-based index
    poly_idx = POLY_IDX,
    vars_poly = lapply(as.list(r$vars_poly[POLY_IDX, ]), num)
  )
}

cases <- list()
add <- function(key, data_name, ...) {
  args <- list(...)
  data <- switch(data_name, cont = cont, mass = mass, qs = qsd)
  out <- record(args, data)
  out$data <- data_name
  cases[[key]] <<- out
}

sel <- c("es", "espr", "esmv", "esmvpr", "qs", "qspr", "qsmv", "qsmvpr")

# every bin rule, three polynomial orders
for (b in sel) for (p in c(1, 2, 4)) add(sprintf("cont_%s_p%d", b, p), "cont", binselect = b, p = p)

# kernels, with and without a bandwidth (scalar and per side)
for (k in c("uni", "tri", "epa")) {
  add(sprintf("cont_%s_noh", k), "cont", kernel = k, p = 2)
  add(sprintf("cont_%s_h05", k), "cont", kernel = k, p = 2, h = 0.5)
  add(sprintf("cont_%s_h0406", k), "cont", kernel = k, p = 3, h = c(0.4, 0.6))
}

# manual bins (scalar and pair), scale
add("cont_nbins10", "cont", nbins = 10, p = 2)
add("cont_nbins_8_12", "cont", nbins = c(8, 12), kernel = "tri", h = 0.7, p = 1)
add("cont_nbins_with_scale", "cont", nbins = c(5, 7), scale = 3)
add("cont_scale2", "cont", scale = 2)
add("cont_scale3_qs", "cont", scale = 3, binselect = "qs")
add("cont_scale_pair", "cont", scale = c(2, 3))
add("cont_scale_1p5", "cont", scale = 1.5)

# covariates (covs_eval = "mean", the default), weights
add("cont_covs1", "cont", covs = "z1", p = 2)
add("cont_covs2", "cont", covs = c("z1", "z2"), p = 4)
add("cont_covs2_tri_h", "cont", covs = c("z1", "z2"), p = 1, kernel = "tri", h = c(0.6, 0.5))
add("cont_covs1_qsmv", "cont", covs = "z1", binselect = "qsmv")
add("cont_weights", "cont", weights = "w", p = 2)
add("cont_weights_tri_h", "cont", weights = "w", kernel = "tri", h = 0.6, p = 1)
add("cont_weights_espr", "cont", weights = "w", binselect = "espr")
add("cont_weights_covs", "cont", weights = "w", covs = "z1", p = 2)

# non-zero cutoff
add("cont_c02", "cont", c = 0.2)
add("cont_c02_qs", "cont", c = 0.2, binselect = "qs", p = 2)
add("cont_c02_covs_tri", "cont", c = 0.2, covs = "z1", kernel = "tri", h = 0.5, p = 2)
add("cont_cm03_nbins", "cont", c = -0.3, nbins = c(6, 9), p = 3)

# confidence level
add("cont_ci90", "cont", ci = 90, p = 2)

# mass points: the spacings rules switch to polynomial regression
for (b in sel) add(sprintf("mass_%s", b), "mass", binselect = b, p = 2)
add("mass_es_off", "mass", binselect = "es", masspoints = "off", p = 2)
add("mass_qsmv_off", "mass", binselect = "qsmv", masspoints = "off", p = 2)
add("mass_es_check", "mass", binselect = "es", masspoints = "check", p = 2)
add("mass_nbins", "mass", nbins = c(10, 10), p = 3)

# quantile edges landing on observations
add("qs_nbins_10_10", "qs", binselect = "qs", nbins = c(10, 10), p = 2)
add("qs_nbins_8_19", "qs", binselect = "qs", nbins = c(8, 19), p = 1)
add("qs_nbins_25_5", "qs", binselect = "qsmv", nbins = c(25, 5), p = 1)
add("qs_default_qs", "qs", binselect = "qs")
add("qs_default_qsmvpr", "qs", binselect = "qsmvpr")

meta <- list(
  rdrobust_version = as.character(packageVersion("rdrobust")),
  r_version = R.version.string,
  generated = "tests/reference_parity/_fixtures/_generate_rdplot_R.R"
)
writeLines(toJSON(list(meta = meta, cases = cases), digits = I(17), auto_unbox = TRUE,
                  null = "null", na = "null"),
           file.path(here, "rdplot_R.json"))
cat("wrote", length(cases), "cases\n")
