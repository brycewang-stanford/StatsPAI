#!/usr/bin/env Rscript
# R HonestDiD 0.2.8 references for sp.honest_did, one option value at a time:
# the target (a single post period other than the first, the post-period
# average, arbitrary weights), alpha = 0.10, a user-supplied relative-magnitudes
# grid, a restricted event-time window, and the event-study covariance taken
# from a Callaway-Sant'Anna fit (R did 2.5.1 att_gt + aggte, the pipeline of
# the HonestDiD README) as well as from raw arrays.
#
#   Rscript tests/reference_parity/_fixtures/_generate_honest_did_options_R.R
#
# Inputs
#   bc     HonestDiD's own BCdata_EventStudy (4 pre / 4 post, correlated sigma).
#   mpdta  tests/orig_parity/data/02_mpdta_original.csv through did::att_gt
#          (never-treated, universal base, dr, no bootstrap) and
#          aggte(type = "dynamic"); sigma = IF'IF / n^2 with the normalised
#          reference period e = -1 dropped. Event times -4..-2, 0..3.
#
# Relative magnitudes use method = "Conditional", which involves no
# simulation: both packages report the smallest and largest accepted point of
# the same grid, so a correct port returns the same grid points. Two cases
# also run HonestDiD's default "C-LF" hybrid, whose least-favourable critical
# value is simulated; those are a stochastic screen.
#
# The FLCI is computed from HonestDiD's own pieces with three like-for-like
# substitutions, each removing a numerical approximation of the shipped
# package (none changes the program being solved):
#   1. .qfoldednormal is the exact folded-normal quantile (a root of the CDF),
#      not the shipped 1e6-draw simulation (~3e-4 off) -- the recipe of Track A
#      module 10.
#   2. The ECOS cone solves of the worst-case-bias program run at
#      abstol = reltol = feastol = 1e-10 (CVXR defaults are 1e-8).
#   3. The optimal standard deviation h is found by optimize(tol = 1e-12)
#      instead of .findOptimalCIDerivativeBisection, whose step is
#      |h0| * eps^(1/3) (~6e-6 relative). The half-length is flat in h at its
#      minimum but the interval centre is not: that step alone moves the
#      bounds by ~8e-7 on bc (e = 0, M = 0.01) while the half-length moves
#      by 8e-11.
# Even so the bounds are determined only to ~1e-5 on a correlated covariance:
# the half-length is flat in h at its minimum (it moves by 1e-11 where the
# centre moves by 1e-6), so solver noise of 1e-11 in the half-length on either
# side leaves the centre uncertain at that level. The half-length itself, and
# the optimal h, are recorded so the test can compare the program's value and
# the affine estimator at a given h, which are well determined.
#
# At M = 0 the FLCI is the minimum-variance estimator unbiased under a linear
# trend through the reference period, plus or minus z sd: a closed form,
# computed below with base linear algebra and no HonestDiD call (the recipe of
# Track A module 10's analytic_ rows). The cone solve at h = hMin is
# near-degenerate, so the closed form, not the solver, is the M = 0 reference.
#
# Shipped HonestDiD's FLCI is pinned separately
# (test_honest_did_moments_R_parity.py, held to its simulation error).
suppressPackageStartupMessages({ library(HonestDiD); library(did); library(jsonlite) })
here <- "tests/reference_parity/_fixtures"

exact_q <- function(p, mu = 0, sd = 1, numSims = 10^6, seed = 0) {
  vapply(mu, function(m) {
    m <- abs(m)
    stats::uniroot(
      function(q) stats::pnorm((q - m) / sd) - stats::pnorm((-q - m) / sd) - p,
      c(0, m + 20 * sd), tol = 1e-14
    )$root
  }, numeric(1))
}
tight_psolve <- function(problem, ...) {
  value <- CVXR::psolve(problem, ..., abstol = 1e-10, reltol = 1e-10,
                        feastol = 1e-10, max_iters = 1000L)
  list(value = value, status = CVXR::status(problem),
       getValue = function(variable) CVXR::value(variable))
}
assignInNamespace(".qfoldednormal", exact_q, ns = "HonestDiD")
assignInNamespace(".psolve", tight_psolve, ns = "HonestDiD")

flci <- function(b, S, npre, npost, l, M, alpha) {
  l <- matrix(l, ncol = 1)
  h0 <- HonestDiD:::.findHForMinimumBias(S, npre, npost, l)
  hmin <- HonestDiD:::.findLowestH(S, npre, npost, l)
  f <- function(h) {
    bd <- HonestDiD:::.findWorstCaseBiasGivenH(h, S, npre, npost, l)
    exact_q(1 - alpha, M * bd$value / h) * h
  }
  hs <- stats::optimize(f, c(hmin, h0), tol = 1e-12)$minimum
  bd <- HonestDiD:::.findWorstCaseBiasGivenH(hs, S, npre, npost, l)
  centre <- sum(c(unlist(bd$optimal.l), l) * b)
  half <- exact_q(1 - alpha, M * bd$value / hs) * hs
  list(lb = centre - half, ub = centre + half, half = half, h = hs,
       status = bd$status)
}

# M = 0: min var(w'b_pre + l'b_post) s.t. w'r_pre + l'r_post = 0, where r is
# the event time relative to the omitted reference period.
flci_m0 <- function(b, S, t, l, alpha) {
  k <- sum(t < 0); r <- t + 1
  Spp <- S[1:k, 1:k, drop = FALSE]; g <- S[1:k, -(1:k), drop = FALSE] %*% l
  Si <- solve(Spp); rp <- r[1:k]
  lam <- drop((-sum(l * r[-(1:k)]) + t(rp) %*% Si %*% g) / (t(rp) %*% Si %*% rp))
  w <- drop(Si %*% (lam * rp - g))
  centre <- sum(w * b[1:k]) + sum(l * b[-(1:k)])
  sd <- sqrt(drop(t(w) %*% Spp %*% w + 2 * t(w) %*% g +
                  t(l) %*% S[-(1:k), -(1:k), drop = FALSE] %*% l))
  half <- stats::qnorm(1 - alpha / 2) * sd
  list(lb = centre - half, ub = centre + half, half = half, h = sd,
       status = "closed_form")
}

rm_cond <- function(b, S, npre, npost, l, Mbar, alpha, grid = NULL,
                    method = "Conditional") {
  args <- list(betahat = b, sigma = S, numPrePeriods = npre,
               numPostPeriods = npost, Mbarvec = Mbar,
               l_vec = matrix(l, ncol = 1), method = method,
               alpha = alpha, seed = 0)
  if (!is.null(grid)) {
    args$grid.lb <- grid$lb; args$grid.ub <- grid$ub
    args$gridPoints <- grid$points
  }
  r <- suppressWarnings(do.call(createSensitivityResults_relativeMagnitudes, args))
  list(lb = r$lb, ub = r$ub)
}

# ---- inputs ---------------------------------------------------------------
bc <- HonestDiD::BCdata_EventStudy
bc_b <- unname(bc$betahat); bc_S <- unname(bc$sigma)
bc_npre <- length(bc$prePeriodIndices); bc_npost <- length(bc$postPeriodIndices)

mp <- read.csv("tests/orig_parity/data/02_mpdta_original.csv")
mp$first_treat <- as.numeric(mp$first_treat)  # integer gname loses never-treated
att <- suppressWarnings(att_gt(yname = "lemp", tname = "year", idname = "countyreal",
                               gname = "first_treat", data = mp,
                               control_group = "nevertreated",
                               base_period = "universal", est_method = "dr",
                               bstrap = FALSE, cband = FALSE))
dyn <- aggte(att, type = "dynamic", bstrap = FALSE, cband = FALSE)
inf <- dyn$inf.function$dynamic.inf.func.e
V <- t(inf) %*% inf / nrow(inf)^2
keep <- dyn$egt != -1
mp_t <- dyn$egt[keep]; mp_b <- unname(dyn$att.egt[keep]); mp_S <- unname(V[keep, keep])

subset_window <- function(b, S, t, lo, hi) {
  k <- t >= lo & t <= hi
  list(b = b[k], S = S[k, k, drop = FALSE], t = t[k])
}

inputs <- list(
  bc = list(b = bc_b, S = bc_S, t = c(-(bc_npre + 1):-2, 0:(bc_npost - 1))),
  mpdta = list(b = mp_b, S = mp_S, t = mp_t)
)
w_mp <- subset_window(mp_b, mp_S, mp_t, -3, 2)
inputs$mpdta_window <- list(b = w_mp$b, S = w_mp$S, t = w_mp$t)

# ---- cases ----------------------------------------------------------------
# target: list(e = k) is the basis vector of post period k (0-based);
# "average" is 1 / npost on every post period; a numeric vector is l_vec.
cases <- list(
  list(id = "bc_flci_e2", input = "bc", method = "smoothness", target = list(e = 2), alpha = 0.05),
  list(id = "bc_flci_avg", input = "bc", method = "smoothness", target = "average", alpha = 0.05),
  list(id = "bc_flci_w", input = "bc", method = "smoothness", target = c(0.4, 0.3, 0.2, 0.1), alpha = 0.05),
  list(id = "bc_flci_e0_a10", input = "bc", method = "smoothness", target = list(e = 0), alpha = 0.10),
  list(id = "bc_rm_e2", input = "bc", method = "relative_magnitude", target = list(e = 2), alpha = 0.05),
  list(id = "bc_rm_w", input = "bc", method = "relative_magnitude", target = c(0.4, 0.3, 0.2, 0.1), alpha = 0.05),
  list(id = "bc_rm_avg_a10", input = "bc", method = "relative_magnitude", target = "average", alpha = 0.10),
  list(id = "bc_rm_e0_grid", input = "bc", method = "relative_magnitude", target = list(e = 0), alpha = 0.05,
       grid = list(lb = -0.5, ub = 1.0, points = 400)),
  list(id = "mp_flci_e0", input = "mpdta", method = "smoothness", target = list(e = 0), alpha = 0.05),
  list(id = "mp_flci_avg", input = "mpdta", method = "smoothness", target = "average", alpha = 0.05),
  list(id = "mp_flci_e1_a10", input = "mpdta", method = "smoothness", target = list(e = 1), alpha = 0.10),
  list(id = "mp_flci_e3", input = "mpdta", method = "smoothness", target = list(e = 3), alpha = 0.05),
  list(id = "mp_flci_w", input = "mpdta", method = "smoothness", target = c(0.4, 0.3, 0.2, 0.1), alpha = 0.05),
  list(id = "mp_rm_w", input = "mpdta", method = "relative_magnitude", target = c(0.4, 0.3, 0.2, 0.1), alpha = 0.05),
  list(id = "mp_rm_e0", input = "mpdta", method = "relative_magnitude", target = list(e = 0), alpha = 0.05),
  list(id = "mp_rm_avg", input = "mpdta", method = "relative_magnitude", target = "average", alpha = 0.05),
  list(id = "mp_rm_e1_a10", input = "mpdta", method = "relative_magnitude", target = list(e = 1), alpha = 0.10),
  # C-LF (HonestDiD's default) simulates its first-stage critical value with
  # R's RNG (seed = 0): a stochastic screen, compared to one grid step.
  list(id = "mp_clf_e0", input = "mpdta", method = "relative_magnitude", target = list(e = 0), alpha = 0.05, hm = "C-LF"),
  list(id = "mp_clf_avg", input = "mpdta", method = "relative_magnitude", target = "average", alpha = 0.05, hm = "C-LF"),
  list(id = "mpw_flci_avg", input = "mpdta_window", method = "smoothness", target = "average", alpha = 0.05),
  list(id = "mpw_rm_avg", input = "mpdta_window", method = "relative_magnitude", target = "average", alpha = 0.05)
)
M_GRID <- list(bc = c(0, 0.01, 0.02), mpdta = c(0, 0.01, 0.02), mpdta_window = c(0, 0.01, 0.02))
MBAR_GRID <- c(0, 0.5, 1)

out_cases <- list()
for (cs in cases) {
  inp <- inputs[[cs$input]]
  npre <- sum(inp$t < 0); npost <- sum(inp$t >= 0)
  l <- if (is.list(cs$target)) {
    v <- rep(0, npost); v[cs$target$e + 1] <- 1; v
  } else if (identical(cs$target, "average")) rep(1 / npost, npost) else cs$target
  if (cs$method == "smoothness") {
    grid <- M_GRID[[cs$input]]
    res <- lapply(grid, function(M) {
      if (M == 0) flci_m0(inp$b, inp$S, inp$t, l, cs$alpha)
      else flci(inp$b, inp$S, npre, npost, l, M, cs$alpha)
    })
    rec <- list(M = grid, lb = vapply(res, `[[`, 0, "lb"), ub = vapply(res, `[[`, 0, "ub"),
                half = vapply(res, `[[`, 0, "half"), h = vapply(res, `[[`, 0, "h"),
                status = vapply(res, `[[`, "", "status"))
  } else {
    hm <- if (is.null(cs$hm)) "Conditional" else cs$hm
    r <- rm_cond(inp$b, inp$S, npre, npost, l, MBAR_GRID, cs$alpha, cs$grid, hm)
    rec <- list(M = MBAR_GRID, lb = r$lb, ub = r$ub, honestdid_method = hm)
  }
  out_cases[[cs$id]] <- c(
    list(input = cs$input, method = cs$method,
         target = if (is.list(cs$target)) list(e = cs$target$e) else cs$target,
         l_vec = l, alpha = cs$alpha),
    if (!is.null(cs$grid)) list(grid = cs$grid) else list(),
    rec
  )
  cat(cs$id, "done\n")
}

write_json(
  list(
    meta = list(R = R.version.string,
                HonestDiD = as.character(packageVersion("HonestDiD")),
                did = as.character(packageVersion("did")),
                CVXR = as.character(packageVersion("CVXR"))),
    inputs = lapply(inputs, function(x) list(betahat = x$b, sigma = x$S, times = x$t)),
    cases = out_cases
  ),
  file.path(here, "honest_did_options_R.json"), digits = NA, auto_unbox = TRUE
)
cat("wrote honest_did_options_R.json\n")
