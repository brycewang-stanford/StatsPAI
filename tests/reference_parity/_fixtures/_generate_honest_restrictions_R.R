# Reference values for the native Honest-DiD restriction families
# (statspai.did._honest_sets) vs R HonestDiD 0.2.8 (Rambachan & Roth 2023;
# Andrews, Roth & Pakes 2023).
#
#   cd tests/reference_parity/_fixtures
#   Rscript _generate_honest_restrictions_R.R <case> <delta> <method>
#   (one entry per run, merged into honest_restrictions_R.json; the list of
#   runs that make the fixture is RUNS in _generate_honest_restrictions_R.sh)
#
# Restrictions (HonestDiD's Delta names):
#   smoothness family   DeltaSD, DeltaSDPB / SDNB (biasDirection),
#                       DeltaSDI / SDD (monotonicityDirection)
#                       -> computeConditionalCS_DeltaSD / SDB / SDM, the
#                          functions createSensitivityResults calls
#   rel.-magnitudes     DeltaRMPB / RMNB, DeltaRMI / RMD, and the "deviation
#                       from linear trend" family DeltaSDRM, SDRMPB / SDRMNB,
#                       SDRMI / SDRMD
#                       -> createSensitivityResults_relativeMagnitudes
#
# Methods:
#   Conditional  no simulation: a correct port accepts the same grid points.
#   C-F          conditional-FLCI hybrid. Its first stage is the level
#                alpha/10 FLCI, whose folded-normal quantile HonestDiD
#                simulates (1e6 draws). Recorded twice: as shipped, and with
#                .qfoldednormal replaced by the exact quantile (the recipe of
#                _generate_honest_did_options_R.R); the grid is the FLCI
#                itself, so its ends are recorded too.
#   C-LF         least-favourable hybrid; its first-stage critical value is
#                simulated with R's RNG (a stochastic screen).
#
# Inputs: corr33 = did_synth_honest_es.csv (3 pre / 3 post, correlated);
# corr42 = 4 pre / 2 post AR(1)-type sigma with the post average as target;
# nonuis21 = 2 pre / 1 post (the no-nuisance test path); the same three
# inputs as _generate_honest_rm_R.R.
suppressPackageStartupMessages({ library(HonestDiD); library(jsonlite) })

es <- read.csv("did_synth_honest_es.csv")
S6 <- as.matrix(es[, paste0("s", 0:5)])
ar1 <- function(n, rho, sd) { m <- outer(1:n, 1:n, function(i, j) rho^abs(i - j)); m * outer(sd, sd) }

cases <- list(
  corr33 = list(b = es$betahat, S = S6, npre = 3, npost = 3, l = c(0, 1, 0),
                M = c(0, 0.01, 0.03)),
  corr42 = list(b = c(0.03, -0.01, 0.02, -0.015, 0.2, 0.25),
                S = ar1(6, 0.5, c(.02, .02, .02, .02, .05, .06)), npre = 4, npost = 2,
                l = c(0.5, 0.5), M = c(0, 0.01, 0.03)),
  nonuis21 = list(b = c(-0.02, 0.01, 0.15), S = ar1(3, 0.3, c(.03, .03, .06)),
                  npre = 2, npost = 1, l = 1, M = c(0, 0.01, 0.03))
)
MBAR <- c(0.5, 1, 2)

exact_q <- function(p, mu = 0, sd = 1, numSims = 10^6, seed = 0) {
  vapply(mu, function(m) {
    m <- abs(m)
    stats::uniroot(function(q) stats::pnorm((q - m) / sd) - stats::pnorm((-q - m) / sd) - p,
                   c(0, m + 20 * sd), tol = 1e-15)$root
  }, numeric(1))
}
shipped_q <- get(".qfoldednormal", asNamespace("HonestDiD"))

sd_call <- function(cs, delta, hybrid, M) {
  args <- list(betahat = cs$b, sigma = cs$S, numPrePeriods = cs$npre,
               numPostPeriods = cs$npost, l_vec = cs$l, alpha = 0.05, M = M,
               hybrid_flag = hybrid)
  if (delta %in% c("SDPB", "SDNB")) {
    args$biasDirection <- if (delta == "SDPB") "positive" else "negative"
    r <- do.call(computeConditionalCS_DeltaSDB, args)
  } else if (delta %in% c("SDI", "SDD")) {
    args$monotonicityDirection <- if (delta == "SDI") "increasing" else "decreasing"
    r <- do.call(computeConditionalCS_DeltaSDM, args)
  } else {
    r <- do.call(computeConditionalCS_DeltaSD, args)
  }
  acc <- r$grid[r$accept == 1]
  c(lb = if (length(acc)) min(acc) else NA, ub = if (length(acc)) max(acc) else NA,
    g0 = r$grid[1], g1 = r$grid[length(r$grid)],
    open = as.numeric(r$accept[1] == 1 || r$accept[length(r$accept)] == 1))
}

rm_call <- function(cs, delta, method) {
  args <- list(betahat = cs$b, sigma = cs$S, numPrePeriods = cs$npre,
               numPostPeriods = cs$npost, l_vec = cs$l, Mbarvec = MBAR,
               method = method, alpha = 0.05)
  if (startsWith(delta, "SDRM")) args$bound <- "deviation from linear trend"
  tail <- sub("^SDRM|^RM", "", delta)
  if (tail == "PB") args$biasDirection <- "positive"
  if (tail == "NB") args$biasDirection <- "negative"
  if (tail == "I") args$monotonicityDirection <- "increasing"
  if (tail == "D") args$monotonicityDirection <- "decreasing"
  r <- suppressWarnings(do.call(createSensitivityResults_relativeMagnitudes, args))
  list(lb = r$lb, ub = r$ub)
}

argv <- commandArgs(trailingOnly = TRUE)
nm <- argv[[1]]; delta <- argv[[2]]; meth <- argv[[3]]
cs <- cases[[nm]]
base <- list(betahat = cs$b, sigma = unname(cs$S), npre = cs$npre, npost = cs$npost,
             l_vec = cs$l)
path <- "honest_restrictions_R.json"
out <- if (file.exists(path)) fromJSON(path, simplifyVector = TRUE) else list()
if (delta %in% c("SD", "SDPB", "SDNB", "SDI", "SDD")) {
  runs <- list(Conditional = list(flag = "ARP", q = shipped_q),
               `C-F` = list(flag = "FLCI", q = exact_q),
               `C-F|shipped` = list(flag = "FLCI", q = shipped_q),
               `C-LF` = list(flag = "LF", q = shipped_q))
  assignInNamespace(".qfoldednormal", runs[[meth]]$q, "HonestDiD")
  res <- t(vapply(cs$M, function(M) suppressWarnings(sd_call(cs, delta, runs[[meth]]$flag, M)),
                  numeric(5)))
  entry <- c(base, list(family = "smoothness", delta = delta, method = meth, M = cs$M,
                        lb = res[, "lb"], ub = res[, "ub"], grid_lo = res[, "g0"],
                        grid_hi = res[, "g1"], open = res[, "open"]))
} else {
  r <- rm_call(cs, delta, meth)
  entry <- c(base, list(family = "relative_magnitude", delta = delta, method = meth,
                        Mbar = MBAR, lb = r$lb, ub = r$ub))
}
out[[paste(nm, delta, meth, sep = "|")]] <- entry
out$`_provenance` <- list(HonestDiD = as.character(packageVersion("HonestDiD")),
                          R = R.version.string,
                          note = paste("C-F rows use the exact folded-normal quantile;",
                                       "C-F|shipped rows use HonestDiD's simulated one."))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE, na = "null"), path)
