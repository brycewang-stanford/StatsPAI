# Reference values for tests/reference_parity/test_did_synth_scpi_parity.py
#
#   sp.scdata / sp.scest / sp.scpi   vs   scpi::scdata / scest / scpi (R, 4.0.1)
#
# Data: scpi's own vignette panel ``scpi::scpi_germany`` (West Germany, GDP per
# capita in thousands of USD, pre 1960-1990, post 1991-2003, 16 OECD donors).
# The generator writes the three columns it uses to
# _fixtures/did_synth_scpi_germany.csv so both sides read identical bytes.
#
# scpi 4.0.1 imports CVXR (> 1.9). The shared site library carries CVXR 1.8.2
# (HonestDiD / DiSCos / synthdid fixtures were generated against it), so CVXR
# 1.9.2 + highs are installed into a PRIVATE library next to the fixtures
# (git-ignored) and put first on .libPaths() here only:
#
#   Rscript -e 'install.packages("CVXR", lib = "tests/reference_parity/_fixtures/_rlib_did_synth_scpi",
#                                repos = "https://cloud.r-project.org")'
#
# Regenerate (from the repository root):
#   Rscript tests/reference_parity/_generate_did_synth_scpi_R.R
#
# Writes _fixtures/did_synth_scpi_R.json.
#
# What is frozen for every w.constr in {simplex, lasso, ridge, ols, L1-L2}:
#   * scest: w, Q / Q2 / lambda of the constraint, Y.pre.fit, Y.post.fit
#   * scpi (sims = 200, e.method = "all", defaults otherwise): every
#     deterministic ingredient (rho, Q.star, lb, df, u.mean, Omega = u.var,
#     Sigma, e.mean, e.var) and the four PI tables + joint bounds.
#   * the raw N(0,1) matrix scpi's in-sample simulation drew and the per-draw
#     bounds ``vsig`` (sims x 2*T1) that ECOS returned.  CVXR consumes R's RNG
#     while solving the weights, so the draws are recovered from the RNG state
#     captured on entry to scpi's insampleUncertaintyGetDiag (a wrapper that
#     calls the original function unchanged).  Feeding those draws to
#     StatsPAI pins the simulated bounds draw by draw.
#   * vsig_tight: the same draws re-solved with ECOS at feastol = reltol =
#     abstol = 1e-12 (ECOSolveR defaults: 1e-8) -- a diagnostic showing that
#     the per-draw gap to StatsPAI's exact solutions is ECOS's stopping
#     tolerance, not a different problem.  Never what an R user gets.
#   * the average-effect interval of scdataMulti(effect = "unit") for the
#     simplex constraint (the construction behind sp.scpi(...).ci).
#
# Draw counts: sims = 200 (R default) for simplex, 50 for the other
# constraints (the algorithm does not depend on sims; this keeps the fixture
# small).

script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) "tests/reference_parity"
)
lib_private <- file.path(script_dir, "_fixtures", "_rlib_did_synth_scpi")
if (dir.exists(lib_private)) .libPaths(c(lib_private, .libPaths()))

suppressMessages({
  library(scpi)
  library(jsonlite)
})
stopifnot(packageVersion("CVXR") > "1.9")

fixtures <- file.path(script_dir, "_fixtures")

data(scpi_germany, package = "scpi")
germany <- scpi_germany[, c("country", "year", "gdp")]
germany <- germany[order(germany$country, germany$year), ]
csv_path <- file.path(fixtures, "did_synth_scpi_germany.csv")
write.csv(germany, csv_path, row.names = FALSE)
df <- read.csv(csv_path)   # read back: both sides use the CSV bytes

unit_tr <- "West Germany"
unit_co <- sort(setdiff(unique(df$country), unit_tr))
sd_obj <- scdata(df = df, id.var = "country", time.var = "year",
                 outcome.var = "gdp", period.pre = 1960:1990,
                 period.post = 1991:2003, unit.tr = unit_tr,
                 unit.co = unit_co, verbose = FALSE)

# ---- capture wrapper around scpi's in-sample simulation --------------------
ns <- asNamespace("scpi")
insample_orig <- get("insampleUncertaintyGetDiag", ns)
ns_ecos <- asNamespace("ECOSolveR")
ecos_orig <- get("ECOS_csolve", ns_ecos)
ecos_tight <- function(c = numeric(0), G = NULL, h = numeric(0), dims = list(l = integer(0), q = NULL, e = integer(0)),
                       A = NULL, b = numeric(0), bool_vars = integer(0), int_vars = integer(0),
                       control = ECOSolveR::ecos.control()) {
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
draws_used <- function(nb, sims) {
  assign(".Random.seed", cap$seed, envir = globalenv())
  matrix(rnorm(nb * sims), nrow = nb, ncol = sims)
}

# Qtools::rrq with quantreg's exact simplex LP (method = "br") instead of the
# default Frisch-Newton interior point ("fn"): a diagnostic for the
# out-of-sample moments (rrq enters e.var / the ls and qreg bounds).
ns_qt <- asNamespace("Qtools")
rrq_orig <- get("rrq", ns_qt)
rrq_br <- function(formula, tau, ...) rrq_orig(formula, tau, method = "br", ...)
with_rrq_br <- function(expr) {
  unlockBinding("rrq", ns_qt)
  assign("rrq", rrq_br, envir = ns_qt)
  on.exit({ assign("rrq", rrq_orig, envir = ns_qt); lockBinding("rrq", ns_qt) })
  expr
}
e_part <- function(ir) {
  b <- ir$bounds
  list(e_mean = num(ir$e.mean), e_var = num(ir$e.var),
       gaussian = list(lb = unname(b$subgaussian[, 1] - b$insample[, 1]),
                       ub = unname(b$subgaussian[, 2] - b$insample[, 2])),
       ls = list(lb = unname(b$ls[, 1] - b$insample[, 1]),
                 ub = unname(b$ls[, 2] - b$insample[, 2])),
       qreg = list(lb = unname(b$qreg[, 1] - b$insample[, 1]),
                   ub = unname(b$qreg[, 2] - b$insample[, 2])))
}

mat <- function(m) unname(as.matrix(m))
tbl <- function(ci) list(lb = unname(ci[, 1]), ub = unname(ci[, 2]))
num <- function(x) if (is.null(x)) NULL else unname(as.numeric(x))

out <- list()
out$meta <- list(
  R = R.version.string,
  scpi = as.character(packageVersion("scpi")),
  CVXR = as.character(packageVersion("CVXR")),
  ECOSolveR = as.character(packageVersion("ECOSolveR")),
  clarabel = as.character(packageVersion("clarabel")),
  osqp = as.character(packageVersion("osqp")),
  Qtools = as.character(packageVersion("Qtools")),
  quantreg = as.character(packageVersion("quantreg")),
  solver_scest = "CLARABEL (OSQP for lasso, forced by scest)",
  solver_insample = "ECOSolveR::ECOS_csolve (defaults feastol=reltol=abstol=1e-8)",
  seed = 8894L
)
out$scdata <- list(
  A = mat(sd_obj$A)[, 1], B = mat(sd_obj$B), P = mat(sd_obj$P),
  C_is_null = is.null(sd_obj$C),
  donors = sub("^West Germany\\.", "", colnames(sd_obj$B)),
  J = sd_obj$specs$J, KM = sd_obj$specs$KM,
  T0 = unname(sd_obj$specs$T0.features), T1 = sd_obj$specs$T1.outcome
)

constrs <- list(simplex = list(name = "simplex"),
                lasso = list(name = "lasso"),
                ridge = list(name = "ridge"),
                ols = list(name = "ols"),
                L1L2 = list(name = "L1-L2"))
n_sims <- c(simplex = 200L, lasso = 50L, ridge = 50L, ols = 50L, L1L2 = 50L)

for (nm in names(constrs)) {
  wc <- constrs[[nm]]
  sims <- n_sims[[nm]]
  est <- scest(sd_obj, w.constr = wc)
  wcs <- est$est.results$w.constr
  set.seed(8894L)
  pi <- scpi(sd_obj, w.constr = wc, sims = sims, e.method = "all",
             cores = 1, verbose = FALSE)
  ir <- pi$inference.results
  zraw <- draws_used(length(pi$est.results$b), sims)
  vsig <- mat(cap$vsig)
  vsig_tight <- mat(cap$vsig_tight)
  set.seed(8894L)
  pi_br <- with_rrq_br(scpi(sd_obj, w.constr = wc, sims = 10, e.method = "all",
                            cores = 1, verbose = FALSE))
  out[[nm]] <- list(
    scest = list(
      w = num(est$est.results$w),
      Q = num(wcs$Q), Q2 = num(wcs$Q2), lambda = num(wcs$lambda),
      Y_pre_fit = mat(est$est.results$Y.pre.fit)[, 1],
      Y_post_fit = mat(est$est.results$Y.post.fit)[, 1]
    ),
    scpi = list(
      sims = sims,
      w = num(pi$est.results$w),
      rho = num(ir$rho), Q_star = num(ir$Q.star),
      u_mean = num(ir$u.mean), u_var = unname(diag(as.matrix(ir$u.var))),
      Sigma = mat(ir$Sigma),
      u_T = ir$u.T, u_params = ir$u.params, u_order = ir$u.order,
      e_mean = num(ir$e.mean), e_var = num(ir$e.var),
      e_T = ir$e.T, e_params = ir$e.params, e_order = ir$e.order,
      insample = tbl(ir$bounds$insample),
      subgaussian = tbl(ir$bounds$subgaussian),
      ls = tbl(ir$bounds$ls),
      qreg = tbl(ir$bounds$qreg),
      joint = tbl(ir$bounds$joint),
      failed_sims = mat(ir$failed.sims),
      zraw = zraw,
      vsig = vsig,
      vsig_tight = vsig_tight,
      e_fn = e_part(ir),
      e_br = e_part(pi_br$inference.results)
    )
  )
}

# ---- donors outnumber pre-periods: California Prop 99 (J = 38, T0 = 19) ----
# Z'Z is singular, so the in-sample constraint set is a cylinder along its
# null space; the simplex bounds keep every problem bounded.
cal <- read.csv(file.path(fixtures, "did_synth_scpi_california.csv"))
cal_co <- sort(setdiff(unique(cal$state), "California"))
sd_cal <- scdata(df = cal, id.var = "state", time.var = "year",
                 outcome.var = "packspercapita", period.pre = 1970:1988,
                 period.post = 1989:2000, unit.tr = "California",
                 unit.co = cal_co, verbose = FALSE)
est_cal <- scest(sd_cal, w.constr = list(name = "simplex"))
set.seed(8894L)
pi_cal <- scpi(sd_cal, w.constr = list(name = "simplex"), sims = 50,
               e.method = "all", cores = 1, verbose = FALSE)
irc <- pi_cal$inference.results
out$california_simplex <- list(
  donors = sub("^California\\.", "", colnames(sd_cal$B)),
  scest_w = num(est_cal$est.results$w),
  w = num(pi_cal$est.results$w),
  rho = num(irc$rho), Sigma = mat(irc$Sigma),
  insample = tbl(irc$bounds$insample),
  subgaussian = tbl(irc$bounds$subgaussian),
  failed_sims = mat(irc$failed.sims),
  zraw = draws_used(length(pi_cal$est.results$b), 50L),
  vsig = mat(cap$vsig),
  vsig_tight = mat(cap$vsig_tight),
  e_fn = e_part(irc)
)

# ---- average-effect interval: scdataMulti(effect = "unit"), simplex ---------
dfm <- df
dfm$treat <- as.integer(dfm$country == unit_tr & dfm$year >= 1991)
sdm <- scdataMulti(df = dfm, id.var = "country", time.var = "year",
                   outcome.var = "gdp", treatment.var = "treat",
                   effect = "unit", verbose = FALSE)
set.seed(8894L)
pim <- scpi(sdm, w.constr = list(name = "simplex"), sims = 200,
            e.method = "all", cores = 1, verbose = FALSE)
irm <- pim$inference.results
out$unit_simplex <- list(
  donors = sub("^West Germany\\.", "", colnames(sdm$B)),
  P = mat(sdm$P)[1, ],
  w = num(pim$est.results$w),
  Y_post_fit = num(pim$est.results$Y.post.fit),
  Y_post = num(sdm$Y.post),
  insample = tbl(irm$bounds$insample),
  subgaussian = tbl(irm$bounds$subgaussian),
  ls = tbl(irm$bounds$ls),
  qreg = tbl(irm$bounds$qreg),
  zraw = draws_used(length(pim$est.results$b), 200L),
  vsig = mat(cap$vsig)
)

writeLines(toJSON(out, digits = I(17), auto_unbox = TRUE, null = "null",
                  matrix = "rowmajor", pretty = FALSE),
           file.path(fixtures, "did_synth_scpi_R.json"))
cat("wrote", file.path(fixtures, "did_synth_scpi_R.json"), "\n")
