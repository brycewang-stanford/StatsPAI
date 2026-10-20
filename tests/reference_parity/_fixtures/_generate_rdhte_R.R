# Black-box fixture for the clean-room rewrite of sp.rdhte / sp.rdbwhte /
# sp.rdhte_lincom (statspai.rd.hte). Runs R rdhte 0.2.0 (Calonico, Cattaneo,
# Farrell, Palomba and Titiunik; GPL-3) only through its documented interface
# and records the returned values; nothing here reads or reproduces its source.
#
#   Rscript tests/reference_parity/_fixtures/_generate_rdhte_R.R
#
# Reads rd_iv_hte.csv (written by _generate_rd_iv_rd_R.R) next to this file;
# writes rdhte_extra.csv (a three-level moderator z3, seeded) and rdhte_R.json:
#   <cell>        rdhte(): args, coef names, coef (Estimate), coef_bc, se_rb,
#                 vcov (column-major), h and Nh (row-major, level x side), N
#   bw_<cell>     rdbwhte(): h, Nh
#   lincom_<cell> rdhte_lincom(): individual estimate / z / p / CI, joint Wald

suppressMessages({
  library(rdhte)
  library(jsonlite)
})
stopifnot(as.character(packageVersion("rdhte")) == "0.2.0")
stopifnot(as.character(packageVersion("rdrobust")) == "4.0.0")

here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) NULL)
if (is.null(here)) {
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  here <- dirname(normalizePath(f))
}

d <- read.csv(file.path(here, "rd_iv_hte.csv"))
set.seed(20261021)
z3 <- sample(0:2, nrow(d), replace = TRUE)
write.csv(data.frame(z3 = z3), file.path(here, "rdhte_extra.csv"), row.names = FALSE)
y <- d$y; x <- d$x; z <- d$z; zb <- d$zb; cl <- d$cl

out <- list()
models <- list()
quiet <- function(expr) {
  res <- NULL
  invisible(capture.output(res <- suppressWarnings(expr)))
  res
}
flat <- function(m) unname(as.numeric(t(m)))
run <- function(tag, moderator, args) {
  cov <- switch(moderator, z = z, zb = zb, z3 = z3, z3f = factor(z3))
  r <- quiet(do.call(rdhte, c(list(y = y, x = x, covs.hte = cov), args)))
  models[[tag]] <<- r
  out[[tag]] <<- list(
    moderator = moderator, args = args, names = names(r$Estimate),
    coef = unname(r$Estimate), coef_bc = unname(r$Estimate.bc),
    se_rb = unname(r$se.rb), vcov = unname(as.numeric(r$vcov)),
    h = flat(r$h), Nh = flat(r$Nh), N = unname(as.numeric(r$N)),
    vce = r$vce_select
  )
}
cl_arg <- function(...) c(list(cluster = cl), list(...))

# fixed bandwidths: variance types, windows, orders, kernels, cutoff
for (v in c("hc0", "hc1", "hc2", "hc3")) run(paste0("z_h04_", v), "z", list(h = 0.4, vce = v))
run("z_h04_cr1", "z", cl_arg(h = 0.4))
run("z_hl03_hr05", "z", list(h.l = 0.3, h.r = 0.5))
run("z_p2_h05", "z", list(p = 2, h = 0.5))
run("z_p1_q3_h05", "z", list(p = 1, q = 3, h = 0.5))
run("z_epa_h04", "z", list(kernel = "epa", h = 0.4))
run("z_uni_h04", "z", list(kernel = "uni", h = 0.4))
run("z_c02_h04", "z", list(c = 0.2, h = 0.4))
run("z3num_h04", "z3", list(h = 0.4))
for (v in c("hc0", "hc1", "hc2", "hc3")) run(paste0("zb_h04_", v), "zb", list(h = 0.4, vce = v))
run("zb_h04_cr1", "zb", cl_arg(h = 0.4))
run("zb_p2_h05", "zb", list(p = 2, h = 0.5))
run("z3f_h04", "z3f", list(h = 0.4))
# selected bandwidths
run("z_default", "z", list())
run("z_hc1", "z", list(vce = "hc1"))
run("z_cr1", "z", cl_arg())
run("z_msetwo", "z", list(bwselect = "msetwo"))
run("z_cerrd", "z", list(bwselect = "cerrd"))
run("z_uni_p2", "z", list(kernel = "uni", p = 2))
run("zb_default", "zb", list())
run("zb_cr1", "zb", cl_arg())
run("zb_msetwo", "zb", list(bwselect = "msetwo"))
run("z3f_default", "z3f", list())

bw <- function(tag, moderator, args) {
  cov <- switch(moderator, z = z, zb = zb, z3f = factor(z3))
  r <- quiet(do.call(rdbwhte, c(list(y = y, x = x, covs.hte = cov), args)))
  out[[paste0("bw_", tag)]] <<- list(moderator = moderator, args = args,
                                     h = flat(r$h), Nh = flat(r$Nh))
}
bw("z", "z", list()); bw("zb", "zb", list()); bw("z3f", "z3f", list())
bw("z_hc1", "z", list(vce = "hc1")); bw("z_cr1", "z", cl_arg())
bw("z_p2", "z", list(p = 2)); bw("z_uni", "z", list(kernel = "uni"))

lincom <- function(tag, model, linfct, L) {
  l <- quiet(rdhte_lincom(models[[model]], linfct = linfct, digits = 15))
  out[[paste0("lincom_", tag)]] <<- list(
    model = model, L = L,
    estimate = l$individual$estimate, z = l$individual$z_stat,
    p = l$individual$p_value,
    ci_low = l$individual$conf.low, ci_high = l$individual$conf.high,
    joint = l$joint$statistic
  )
}
lincom("z3f_two", "z3f_default",
       c("`covs.hte2` - `covs.hte0` = 0", "`covs.hte1` - `covs.hte0` = 0"),
       list(c(-1, 0, 1), c(-1, 1, 0)))
lincom("z_sum", "z_h04_hc3", c("T + `T:covs.hte` = 0"), list(c(1, 1)))

out[["_meta"]] <- list(
  n = nrow(d),
  rdhte_version = as.character(packageVersion("rdhte")),
  rdrobust_version = as.character(packageVersion("rdrobust")),
  R_version = R.version.string
)
write_json(out, file.path(here, "rdhte_R.json"), auto_unbox = TRUE,
           digits = NA, pretty = TRUE)
cat("wrote rdhte_R.json\n")
