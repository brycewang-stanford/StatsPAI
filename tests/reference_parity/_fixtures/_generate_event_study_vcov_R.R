# Reference for sp.event_study_vcov / sp.uniform_bands: the *full* joint
# covariance of four event-study estimators, and the sup-t critical value
# built on it.
#
# Track A modules 05 / 73 / 85 compare each event-time coefficient and its
# standard error, i.e. the diagonal only. A simultaneous band and the
# HonestDiD FLCI depend on the off-diagonal blocks, which no Track A row
# reaches; this fixture records them.
#
#   cs      did::aggte(type = "dynamic") on did::att_gt (dr, never-treated,
#           universal base period, analytic SEs), covariance
#           IF' IF / n^2 from the aggregated influence function. The
#           reference period e = -1 is dropped (R reports it as zero).
#   sunab   fixest::feols(... sunab(first_treat, year) ...), clustered by
#           county; event-time covariance A V A' with A fixest's documented
#           sunab aggregation (cohort shares fixed). A is checked to
#           reproduce fixest's own aggregated coefficients and SEs first.
#   twfe    fixest::feols(y ~ i(rel_f, treat, ref = -1) | unit + time),
#           clustered by unit, on Track A module 85's CSV.
#   did2s   did2s::did2s(second_stage = ~ i(rel_year, ref = Inf)), clustered
#           by county: every realised relative time of the treated units.
#
# supt: for each estimator, the 1 - alpha (alpha = 0.05) quantile of
# max_j |Z_j|, Z ~ N(0, R), R the correlation matrix of the covered
# coefficients, by mvtnorm::qmvnorm (Genz-Bretz quasi-Monte Carlo, abseps
# 1e-6), for the full vector ("all") and the e >= 0 block ("post").
#
# The mpdta estimators read Track A module 05's CSV bytes, which is
# sp.datasets.mpdta() written by tests/r_parity/05_sunab.py.
#
# Run from the repository root:
#   Rscript tests/reference_parity/_fixtures/_generate_event_study_vcov_R.R
suppressPackageStartupMessages({
  library(did)
  library(fixest)
  library(did2s)
  library(mvtnorm)
  library(jsonlite)
})

mp <- read.csv("tests/r_parity/data/05_sunab.csv")
mp$first_treat <- as.numeric(mp$first_treat)

block <- function(times, beta, V) {
  list(times = as.integer(times), beta = unname(as.numeric(beta)),
       vcov = unname(V))
}

# --- Callaway-Sant'Anna dynamic aggregation ------------------------------
a <- att_gt(yname = "lemp", tname = "year", idname = "countyreal",
            gname = "first_treat", data = mp, control_group = "nevertreated",
            est_method = "dr", base_period = "universal",
            bstrap = FALSE, cband = FALSE)
d <- aggte(a, type = "dynamic", bstrap = FALSE, cband = FALSE)
IF <- d$inf.function$dynamic.inf.func.e
V_cs <- t(IF) %*% IF / nrow(IF)^2
keep <- d$egt != -1
stopifnot(isTRUE(all.equal(sqrt(diag(V_cs))[keep], d$se.egt[keep],
                           tolerance = 1e-10)))
cs <- block(d$egt[keep], d$att.egt[keep], V_cs[keep, keep, drop = FALSE])

# --- Sun-Abraham via fixest ----------------------------------------------
fit <- feols(lemp ~ sunab(first_treat, year) | countyreal + year,
             data = mp, cluster = ~countyreal)
b_full <- coef(fit, agg = FALSE)
V_full <- vcov(fit)[names(b_full), names(b_full)]
cell_rel <- as.integer(sub("^year::(-?\\d+):cohort::.*$", "\\1", names(b_full)))
cell_coh <- as.numeric(sub("^.*:cohort::(\\d+)$", "\\1", names(b_full)))
n_cell <- mapply(function(g, e) sum(mp$first_treat == g & mp$year - g == e),
                 cell_coh, cell_rel)
b_agg <- coef(fit)
rel <- as.integer(sub("^year::(-?\\d+)$", "\\1", names(b_agg)))
A <- t(sapply(rel, function(e) {
  on <- cell_rel == e
  out <- rep(0, length(b_full))
  out[on] <- n_cell[on] / sum(n_cell[on])
  out
}))
V_sa <- A %*% V_full %*% t(A)
stopifnot(max(abs(A %*% b_full - b_agg)) < 1e-12)
stopifnot(max(abs(sqrt(diag(V_sa)) / se(fit) - 1)) < 1e-10)
sunab <- block(rel, b_agg, V_sa)

# --- Dynamic TWFE (Track A module 85's data and specification) ------------
tw <- read.csv("tests/r_parity/data/85_twfe_event_study.csv")
tw$rel <- ifelse(tw$g > 0, tw$time - tw$g, NA_integer_)
tw$treat <- as.integer(tw$g > 0)
tw$rel_f <- ifelse(is.na(tw$rel), -1L, tw$rel)
fit_tw <- feols(y ~ i(rel_f, treat, ref = -1) | unit + time,
                data = tw, cluster = ~unit)
b_tw <- coef(fit_tw)
twfe <- block(as.integer(sub("^.*::(-?[0-9]+).*$", "\\1", names(b_tw))),
              b_tw, vcov(fit_tw)[names(b_tw), names(b_tw)])

# --- Gardner two-stage event study ----------------------------------------
# did2s demeans its first stage with fixest's iterative solver, whose default
# tolerance (fixef.tol = 1e-6) leaves the point estimates ~1e-7 relative from
# the exact least-squares solution (the documented gap of Track A module 73).
# Tighten it to near fixest's floor so this fixture pins the exact solution.
setFixest_estimation(fixef.tol = 1e-11, fixef.iter = 100000)
mp$dpost <- as.integer(mp$first_treat > 0 & mp$year >= mp$first_treat)
mp$rel_year <- ifelse(mp$first_treat > 0, mp$year - mp$first_treat, Inf)
fit_2s <- suppressMessages(did2s(
  mp, yname = "lemp", first_stage = ~ 0 | countyreal + year,
  second_stage = ~ i(rel_year, ref = Inf), treatment = "dpost",
  cluster_var = "countyreal", verbose = FALSE
))
b_2s <- coef(fit_2s)
did2s_blk <- block(as.integer(sub("^rel_year::(-?[0-9]+)$", "\\1", names(b_2s))),
                   b_2s, vcov(fit_2s)[names(b_2s), names(b_2s)])

# --- sup-t critical values ------------------------------------------------
supt <- function(blk, alpha = 0.05) {
  one <- function(sel) {
    V <- blk$vcov[sel, sel, drop = FALSE]
    R <- cov2cor(V)
    set.seed(20260928)
    q <- qmvnorm(1 - alpha, tail = "both.tails", corr = R,
                 algorithm = GenzBretz(maxpts = 2e6, abseps = 1e-6))
    q$quantile
  }
  list(all = one(rep(TRUE, length(blk$times))), post = one(blk$times >= 0))
}

out <- list(
  cs = cs, sunab = sunab, twfe = twfe, did2s = did2s_blk,
  supt = list(cs = supt(cs), sunab = supt(sunab), twfe = supt(twfe),
              did2s = supt(did2s_blk)),
  alpha = 0.05,
  versions = list(
    r = R.version.string,
    did = as.character(packageVersion("did")),
    fixest = as.character(packageVersion("fixest")),
    did2s = as.character(packageVersion("did2s")),
    mvtnorm = as.character(packageVersion("mvtnorm"))
  )
)
write_json(out, "tests/reference_parity/_fixtures/event_study_vcov_R.json",
           digits = NA, auto_unbox = TRUE, pretty = TRUE)
cat("wrote event_study_vcov_R.json\n")
