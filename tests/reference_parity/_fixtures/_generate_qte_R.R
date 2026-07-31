#!/usr/bin/env Rscript
# =====================================================================
#  Reference values for the StatsPAI QTE / distributional family.
#
#  Produces:
#    qte_R.json              — reference point estimates (+ SEs where cheap)
#    qte_lalonde.csv         — cross-section (qte::lalonde.exp), row order locked
#    qte_lalonde_psid.csv    — cross-section (qte::lalonde.psid)
#    qte_lalonde_panel.csv   — 3-period panel (qte::lalonde.psid.panel)
#    qte_lalonde_exp_panel.csv — 3-period panel (qte::lalonde.exp.panel)
#
#  Environment (record any change in R_PACKAGE_VERSIONS.md):
#    R 4.5.2 / qte 1.3.1 / quantreg 6.1
#
#  ---------------------------------------------------------------
#  TWO FIXTURE TRAPS — both already stepped on, do not re-litigate:
#
#  1. `qte`'s SEs come from a bootstrap.  Point estimates are
#     deterministic, SEs are NOT.  Every se=TRUE block therefore sets
#     set.seed() immediately before the call AND records `iters`.
#     Parity tests must anchor POINT estimates tightly (1e-6) and SEs
#     loosely (a few %), never the reverse.
#
#  2. QDiD/CiC/MDiD take `t` (post) and `tmin1` (pre) as *values of the
#     tname column*, not indices.  Swapping them does not error — it
#     silently returns a sign-flipped estimate.  The panel here is
#     1974 / 1975 / 1978, so t=1978, tmin1=1975, tmin2=1974.
# =====================================================================

suppressPackageStartupMessages({
  library(qte)
  library(quantreg)
  library(jsonlite)
})

set.seed(20260731)

# Output next to this script regardless of the caller's cwd.
.args <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .args[grep("^--file=", .args)])
OUT_DIR <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

PROBS <- seq(0.05, 0.95, 0.05)
XF <- ~ age + education + black + hispanic + married + nodegree
BOOT_ITERS <- 100

data(lalonde)

results <- list()

msg <- function(...) cat("[qte-fixture]", ..., "\n")

# ---------------------------------------------------------------------
#  0. Freeze the datasets so Python reads byte-identical inputs
# ---------------------------------------------------------------------
write.csv(lalonde.exp, file.path(OUT_DIR, "qte_lalonde.csv"), row.names = FALSE)
write.csv(lalonde.psid, file.path(OUT_DIR, "qte_lalonde_psid.csv"), row.names = FALSE)
write.csv(lalonde.psid.panel, file.path(OUT_DIR, "qte_lalonde_panel.csv"),
          row.names = FALSE)
write.csv(lalonde.exp.panel, file.path(OUT_DIR, "qte_lalonde_exp_panel.csv"),
          row.names = FALSE)
msg("datasets written")

# ---------------------------------------------------------------------
#  1. Firpo (2007) unconditional QTE / QTT — cross-section
#     ci.qte  -> QTE  (weights D/p, (1-D)/(1-p))
#     ci.qtet -> QTT  (weights D,   (1-D)p/(1-p))
# ---------------------------------------------------------------------
grab <- function(obj) {
  list(
    probs = as.numeric(obj$probs),
    qte   = as.numeric(obj$qte),
    se    = if (!is.null(obj$qte.se)) as.numeric(obj$qte.se) else NULL,
    ate   = if (!is.null(obj$ate)) as.numeric(obj$ate) else NULL,
    ate_se = if (!is.null(obj$ate.se)) as.numeric(obj$ate.se) else NULL
  )
}

for (spec in list(
  list(key = "ci_qte_nocov",  fn = ci.qte,  xf = NULL, data = "exp"),
  list(key = "ci_qte_cov",    fn = ci.qte,  xf = XF,   data = "exp"),
  list(key = "ci_qtet_nocov", fn = ci.qtet, xf = NULL, data = "exp"),
  list(key = "ci_qtet_cov",   fn = ci.qtet, xf = XF,   data = "exp"),
  list(key = "ci_qte_psid_cov",  fn = ci.qte,  xf = XF, data = "psid"),
  list(key = "ci_qtet_psid_cov", fn = ci.qtet, xf = XF, data = "psid")
)) {
  dat <- if (spec$data == "exp") lalonde.exp else lalonde.psid
  set.seed(20260731)
  obj <- try(spec$fn(re78 ~ treat, xformla = spec$xf, data = dat,
                     probs = PROBS, se = TRUE, iters = BOOT_ITERS),
             silent = TRUE)
  if (inherits(obj, "try-error")) {
    msg("FAILED", spec$key, ":", as.character(obj))
    next
  }
  results[[spec$key]] <- c(
    grab(obj),
    list(dataset = spec$data, xformla = !is.null(spec$xf),
         iters = BOOT_ITERS, seed = 20260731)
  )
  msg("ok", spec$key)
}

# ---------------------------------------------------------------------
#  2. QDiD / CiC / MDiD — 2-period panel contrast (1975 -> 1978)
#     NOTE the trap: t = post = 1978, tmin1 = pre = 1975.
# ---------------------------------------------------------------------
for (spec in list(
  list(key = "qdid_nocov", fn = QDiD, xf = NULL),
  list(key = "qdid_cov",   fn = QDiD, xf = XF),
  list(key = "cic_nocov",  fn = CiC,  xf = NULL),
  list(key = "cic_cov",    fn = CiC,  xf = XF),
  list(key = "mdid_nocov", fn = MDiD, xf = NULL),
  list(key = "mdid_cov",   fn = MDiD, xf = XF)
)) {
  set.seed(20260731)
  obj <- try(spec$fn(re ~ treat, xformla = spec$xf,
                     t = 1978, tmin1 = 1975, tname = "year",
                     data = lalonde.psid.panel, panel = TRUE, idname = "id",
                     probs = PROBS, se = TRUE, iters = BOOT_ITERS),
             silent = TRUE)
  if (inherits(obj, "try-error")) {
    msg("FAILED", spec$key, ":", as.character(obj))
    next
  }
  results[[spec$key]] <- c(
    grab(obj),
    list(dataset = "psid.panel", t = 1978, tmin1 = 1975,
         xformla = !is.null(spec$xf), iters = BOOT_ITERS, seed = 20260731)
  )
  msg("ok", spec$key)
}

# ---------------------------------------------------------------------
#  3. Callaway & Li (2019) panel QTT — needs a third period (tmin2)
# ---------------------------------------------------------------------
for (m in c("qr", "pscore")) {
  key <- paste0("panel_qtet_", m)
  set.seed(20260731)
  obj <- try(panel.qtet(re ~ treat, xformla = XF,
                        t = 1978, tmin1 = 1975, tmin2 = 1974,
                        tname = "year", data = lalonde.psid.panel,
                        idname = "id", probs = PROBS,
                        method = m, se = TRUE, iters = BOOT_ITERS),
             silent = TRUE)
  if (inherits(obj, "try-error")) {
    msg("FAILED", key, ":", as.character(obj))
    next
  }
  results[[key]] <- c(
    grab(obj),
    list(dataset = "psid.panel", t = 1978, tmin1 = 1975, tmin2 = 1974,
         method = m, iters = BOOT_ITERS, seed = 20260731)
  )
  msg("ok", key)
}

# ---------------------------------------------------------------------
#  4. Conditional quantile regression (Koenker & Bassett 1978)
#     Anchors sp.qte(method="conditional_qr"), which is NOT Firpo.
#     rq default method "br" = Barrodale-Roberts exact LP.
# ---------------------------------------------------------------------
rq_taus <- c(0.1, 0.25, 0.5, 0.75, 0.9)
fit_nocov <- lapply(rq_taus, function(tt)
  coef(rq(re78 ~ treat, tau = tt, data = lalonde.exp)))
fit_cov <- lapply(rq_taus, function(tt)
  coef(rq(re78 ~ treat + age + education + black + hispanic + married +
            nodegree, tau = tt, data = lalonde.exp)))

results[["rq_nocov"]] <- list(
  taus = rq_taus,
  treat_coef = sapply(fit_nocov, function(b) unname(b["treat"])),
  intercept  = sapply(fit_nocov, function(b) unname(b["(Intercept)"])),
  dataset = "exp"
)
results[["rq_cov"]] <- list(
  taus = rq_taus,
  treat_coef = sapply(fit_cov, function(b) unname(b["treat"])),
  coef_names = names(fit_cov[[1]]),
  coefs = lapply(fit_cov, unname),
  dataset = "exp"
)
msg("ok rq_nocov / rq_cov")

# ---------------------------------------------------------------------
#  5. Plain weighted/unweighted empirical quantiles — cheap invariants
#     that let a Python test verify it loaded the SAME data before it
#     blames an estimator for a mismatch.
# ---------------------------------------------------------------------
results[["data_checksums"]] <- list(
  exp_n = nrow(lalonde.exp),
  exp_n_treat = sum(lalonde.exp$treat),
  exp_re78_mean = mean(lalonde.exp$re78),
  exp_re78_quantiles = unname(quantile(lalonde.exp$re78, PROBS, type = 7)),
  psid_n = nrow(lalonde.psid),
  psid_n_treat = sum(lalonde.psid$treat),
  panel_n = nrow(lalonde.psid.panel),
  panel_years = sort(unique(lalonde.psid.panel$year))
)

# ---------------------------------------------------------------------
results[["_meta"]] <- list(
  generated_by = "_generate_qte_R.R",
  r_version = R.version.string,
  qte_version = as.character(packageVersion("qte")),
  quantreg_version = as.character(packageVersion("quantreg")),
  probs = PROBS,
  boot_iters = BOOT_ITERS,
  seed = 20260731,
  note = paste("Point estimates are deterministic; SEs are bootstrap and",
               "only reproducible with the recorded seed AND iters.")
)

write(toJSON(results, digits = 12, auto_unbox = TRUE, null = "null"),
      file.path(OUT_DIR, "qte_R.json"))
msg("wrote qte_R.json with", length(results), "entries")
