# Reference screen for tests/reference_parity/test_tvp_var_sv_parity.py
#
# bvarsv::bvar.sv.tvp (the sampler of Primiceri 2005 with the corrected
# ordering) on the committed synthetic file tvp_var_sv.csv, several seeds.
# Writes tvp_var_sv_R.json: per seed, the posterior medians of the standard
# deviations of the structural shocks, of the own-lag coefficient of x and of
# two impulse responses. This is a stochastic screen, not a parity: the two
# samplers share no random numbers, and bvarsv estimates on rows
# (tau + p + 1) .. n where StatsPAI estimates on (tau + 1) .. n.
# Requires: bvarsv 1.1, jsonlite. About 4 minutes.
#   Rscript tests/reference_parity/_fixtures/_generate_tvp_var_sv_R.R
suppressMessages({library(bvarsv); library(jsonlite)})
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
d = read.csv(file.path(here, "tvp_var_sv.csv"))
Y = as.matrix(d[, c("x", "z")])
seeds = c(11, 22, 33, 44)
one = function(seed) {
  set.seed(seed)
  fit = bvar.sv.tvp(Y, p = 1, tau = 40, nf = 1, nrep = 6000, nburn = 2000,
                    thinfac = 4, itprint = 1e9)
  M = 2; nT = dim(fit$H.postmean)[3]; nd = dim(fit$H.draws)[3]
  sd = array(NA, c(nd, nT, M))
  for (r in 1:nd) for (t in 1:nT) {
    H = fit$H.draws[, ((t - 1) * M + 1):(t * M), r]
    sd[r, t, ] = diag(t(chol(H)))
  }
  # Beta.draws stacks vec of [intercept, lag matrix] by column: rows 1..M are
  # the intercepts, rows M+1..2M the first column of the lag matrix
  a11 = t(fit$Beta.draws[M + 1, , ])
  irf = function(t, imp, resp)
    apply(impulse.responses(fit, impulse.variable = imp, response.variable = resp,
                            t = t, nhor = 6, scenario = 2, draw.plot = FALSE)$irf,
          2, median)
  list(seed = seed,
       sd_median = apply(sd, c(2, 3), median),
       a11_median = apply(a11, 2, median),
       a11_lower = apply(a11, 2, quantile, 0.025),
       a11_upper = apply(a11, 2, quantile, 0.975),
       irf_x_to_z_t40 = irf(40, 1, 2), irf_x_to_z_t180 = irf(180, 1, 2),
       irf_x_to_x_t40 = irf(40, 1, 1), irf_x_to_x_t180 = irf(180, 1, 1))
}
out = list(versions = list(R = R.version.string,
                           bvarsv = as.character(packageVersion("bvarsv"))),
           n_dates = nrow(Y) - 40 - 1, runs = lapply(seeds, one))
write_json(out, file.path(here, "tvp_var_sv_R.json"), digits = NA, auto_unbox = TRUE)
