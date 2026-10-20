# Reference values for tests/reference_parity/test_scpi_cleanroom_parity.py
#
# R scpi 4.0.1 (GPL-3) is used as a black box: scdata + scest + scpi through
# their documented interface, on the committed synthetic panels
# (_generate_scpi_cleanroom_data.py). The options exercised are the ones the
# clean-room rewrite of sp.scpi had to pin from R's outputs
# (docs/dev/cleanroom/scpi.md): rho rules, the simulation constraint radii,
# HC variants, residual designs, levels, and the constant out-of-sample design.
#
# As in _generate_did_synth_scpi_R.R, the in-sample draws are recovered from
# the RNG state at entry to scpi's simulation step (a wrapper that calls the
# original function unchanged), the same draws are re-solved with ECOS at
# 1e-12, and the out-of-sample bounds are re-run with Qtools::rrq(method =
# "br"), quantreg's exact simplex LP.
#
# scpi 4.0.1 needs CVXR > 1.9; put it in a private library first:
#   R_LIBS=<private lib> Rscript tests/reference_parity/_fixtures/_generate_scpi_cleanroom_R.R

suppressMessages({
  .libPaths(c(Sys.getenv("R_LIBS"), .libPaths()))
  library(scpi)
  library(jsonlite)
})
stopifnot(packageVersion("CVXR") > "1.9")
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))

ns <- asNamespace("scpi")
insample_orig <- get("insampleUncertaintyGetDiag", ns)
ns_ecos <- asNamespace("ECOSolveR")
ecos_orig <- get("ECOS_csolve", ns_ecos)
ecos_tight <- function(c = numeric(0), G = NULL, h = numeric(0),
                       dims = list(l = integer(0), q = NULL, e = integer(0)),
                       A = NULL, b = numeric(0), bool_vars = integer(0),
                       int_vars = integer(0), control = ECOSolveR::ecos.control()) {
  ecos_orig(c = c, G = G, h = h, dims = dims, A = A, b = b,
            control = ECOSolveR::ecos.control(maxit = 500L, feastol = 1e-12,
                                              reltol = 1e-12, abstol = 1e-12))
}
cap <- new.env()
insample_wrap <- function(...) {
  cap$seed <- get(".Random.seed", envir = globalenv())
  v <- insample_orig(...)
  cap$vsig <- v
  assign(".Random.seed", cap$seed, envir = globalenv())
  unlockBinding("ECOS_csolve", ns_ecos)
  assign("ECOS_csolve", ecos_tight, envir = ns_ecos)
  cap$vsig_tight <- tryCatch(insample_orig(...), finally = {
    assign("ECOS_csolve", ecos_orig, envir = ns_ecos)
    lockBinding("ECOS_csolve", ns_ecos)
  })
  v
}
unlockBinding("insampleUncertaintyGetDiag", ns)
assign("insampleUncertaintyGetDiag", insample_wrap, envir = ns)

ns_qt <- asNamespace("Qtools")
rrq_orig <- get("rrq", ns_qt)
rrq_br <- function(formula, tau, ...) rrq_orig(formula, tau, method = "br", ...)
with_rrq_br <- function(expr) {
  unlockBinding("rrq", ns_qt)
  assign("rrq", rrq_br, envir = ns_qt)
  on.exit({ assign("rrq", rrq_orig, envir = ns_qt); lockBinding("rrq", ns_qt) })
  expr
}

num <- function(x) if (is.null(x)) NULL else unname(as.numeric(x))
mat <- function(m) unname(as.matrix(m))
bnds <- function(ir) lapply(ir$bounds, mat)

panels <- list(long = list(file = "scpi_cleanroom_long.csv", t0 = 18),
               short = list(file = "scpi_cleanroom_short.csv", t0 = 13),
               tiny = list(file = "scpi_cleanroom_tiny.csv", t0 = 4))
data_of <- function(p) {
  d <- read.csv(file.path(here, p$file))
  tm <- sort(unique(d$time))
  scdata(df = d, id.var = "unit", time.var = "time", outcome.var = "y",
         period.pre = tm[tm < p$t0], period.post = tm[tm >= p$t0],
         unit.tr = "tr", unit.co = sort(setdiff(unique(d$unit), "tr")),
         verbose = FALSE)
}

cases <- list(
  simplex_e_order0 = list(panel = "long", w = list(name = "simplex"), opt = list(e.order = 0)),
  simplex_type1 = list(panel = "long", w = list(name = "simplex"), opt = list(rho = "type-1")),
  simplex_rho5 = list(panel = "long", w = list(name = "simplex"), opt = list(rho = 5)),
  simplex_HC2 = list(panel = "long", w = list(name = "simplex"), opt = list(u.sigma = "HC2")),
  simplex_HC4 = list(panel = "long", w = list(name = "simplex"), opt = list(u.sigma = "HC4")),
  simplex_nomissp = list(panel = "long", w = list(name = "simplex"), opt = list(u.missp = FALSE)),
  simplex_u_order0 = list(panel = "long", w = list(name = "simplex"), opt = list(u.order = 0)),
  simplex_alphas = list(panel = "long", w = list(name = "simplex"), opt = list(u.alpha = 0.1, e.alpha = 0.2)),
  lasso_Q13 = list(panel = "long", w = list(name = "lasso", Q = 1.3), opt = list()),
  lasso_Q15 = list(panel = "long", w = list(name = "lasso", Q = 1.5), opt = list()),
  lasso_Q05 = list(panel = "long", w = list(name = "lasso", Q = 0.5), opt = list()),
  ridge_Q03 = list(panel = "long", w = list(name = "ridge", Q = 0.3), opt = list()),
  L1L2_Q2_06 = list(panel = "long", w = list(name = "L1-L2", Q2 = 0.6), opt = list()),
  L1L2_Q2_066_rho03 = list(panel = "long", w = list(name = "L1-L2", Q2 = 0.66), opt = list(rho = 0.3)),
  ridge_short = list(panel = "short", w = list(name = "ridge"), opt = list()),
  L1L2_short = list(panel = "short", w = list(name = "L1-L2"), opt = list()),
  L1L2_tiny = list(panel = "tiny", w = list(name = "L1-L2"), opt = list())
)
sims <- 30L
out <- list()
for (nm in names(cases)) {
  cs <- cases[[nm]]
  sd <- data_of(panels[[cs$panel]])
  est <- scest(sd, w.constr = cs$w)
  wc <- est$est.results$w.constr
  args <- c(list(data = sd, w.constr = cs$w, sims = sims, e.method = "all",
                 cores = 1, verbose = FALSE), cs$opt)
  set.seed(8894L)
  pi <- do.call(scpi, args)
  ir <- pi$inference.results
  assign(".Random.seed", cap$seed, envir = globalenv())
  zraw <- matrix(rnorm(length(pi$est.results$b) * sims), ncol = sims)
  vsig <- mat(cap$vsig)
  vsig_tight <- mat(cap$vsig_tight)
  args$sims <- 10L
  set.seed(8894L)
  pi_br <- with_rrq_br(do.call(scpi, args))
  out[[nm]] <- list(
    panel = cs$panel, opt = cs$opt, w_constr = cs$w,
    donors = sub("^tr\\.", "", colnames(sd$B)),
    scest_w = num(est$est.results$w),
    Q = num(wc$Q), Q2 = num(wc$Q2), lambda = num(wc$lambda),
    w = num(pi$est.results$w),
    rho = num(ir$rho), Q_star = num(ir$Q.star),
    u_mean = num(ir$u.mean), u_var = unname(diag(as.matrix(ir$u.var))),
    Sigma = mat(ir$Sigma),
    u_T = ir$u.T, u_params = ir$u.params, u_order = ir$u.order,
    e_mean = num(ir$e.mean), e_var = num(ir$e.var),
    e_T = ir$e.T, e_params = ir$e.params, e_order = ir$e.order,
    bounds = bnds(ir), failed_sims = mat(ir$failed.sims),
    e_var_br = num(pi_br$inference.results$e.var),
    bounds_br = bnds(pi_br$inference.results),
    zraw = zraw, vsig = vsig, vsig_tight = vsig_tight
  )
}
out$versions <- list(R = R.version.string, scpi = as.character(packageVersion("scpi")),
                     CVXR = as.character(packageVersion("CVXR")),
                     ECOSolveR = as.character(packageVersion("ECOSolveR")),
                     Qtools = as.character(packageVersion("Qtools")),
                     quantreg = as.character(packageVersion("quantreg")))
writeLines(toJSON(out, digits = I(17), auto_unbox = TRUE, null = "null", na = "null",
                  matrix = "rowmajor"),
           file.path(here, "scpi_cleanroom_R.json"))
cat("wrote", file.path(here, "scpi_cleanroom_R.json"), "\n")
