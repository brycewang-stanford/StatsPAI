# Reference values for tests/reference_parity/test_bayes_mcmc_parity.py
#
# Runs on the committed synthetic files bayes_mcmc_chains.csv,
# bayes_mcmc_multichain.csv and bayes_bma.csv and writes bayes_mcmc_R.json
# next to them.
#
# Requires: coda, BMA, BMS, jsonlite.
#   Rscript tests/reference_parity/_fixtures/_generate_bayes_mcmc_R.R
suppressMessages({library(coda); library(BMA); library(BMS); library(jsonlite)})
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
out = list(versions = list(R = R.version.string, coda = as.character(packageVersion("coda")),
  BMA = as.character(packageVersion("BMA")), BMS = as.character(packageVersion("BMS"))))

## ---- coda: one chain -------------------------------------------------------
full = as.matrix(read.csv(file.path(here, "bayes_mcmc_chains.csv")))
num = function(m) { m = as.data.frame(m); rownames(m) = NULL; m }
for (nm in c("long", "short")) {
  x = if (nm == "long") mcmc(full) else mcmc(full[1:1237, 1:4])
  s = summary(x); h = heidel.diag(x); h2 = heidel.diag(x, eps = 0.02, pvalue = 0.2)
  res = list(names = colnames(x), stats = num(s$statistics), quant = num(s$quantiles),
    ess = unname(effectiveSize(x)), spec0 = unname(spectrum0.ar(x)$spec),
    geweke = unname(geweke.diag(x)$z), geweke_23 = unname(geweke.diag(x, frac1 = 0.2, frac2 = 0.3)$z),
    heidel = num(unclass(h)), heidel_strict = num(unclass(h2)),
    hpd95 = num(HPDinterval(x)), hpd80 = num(HPDinterval(x, prob = 0.8)),
    raftery_median = num(raftery.diag(x, q = 0.5, r = 0.05, s = 0.95)$resmatrix),
    raftery_decile = num(raftery.diag(x, q = 0.1, r = 0.02, s = 0.9, converge.eps = 0.01)$resmatrix))
  if (nm == "long") res$raftery_default = num(raftery.diag(x)$resmatrix)
  out[[nm]] = res
}

## ---- coda: several chains --------------------------------------------------
mc = read.csv(file.path(here, "bayes_mcmc_multichain.csv"))
ml = mcmc.list(lapply(1:4, function(i) mcmc(as.matrix(mc[mc$chain == i, c("a", "b", "c")]))))
g = gelman.diag(ml, autoburnin = FALSE, transform = FALSE)
g90 = gelman.diag(ml, autoburnin = FALSE, transform = FALSE, confidence = 0.9)
gb = gelman.diag(ml, autoburnin = TRUE, transform = FALSE)
out$gelman = list(psrf = num(g$psrf), mpsrf = g$mpsrf, psrf90 = num(g90$psrf),
  psrf_autoburnin = num(gb$psrf), mpsrf_autoburnin = gb$mpsrf)

## ---- BMA: BIC and Occam's window ------------------------------------------
d = read.csv(file.path(here, "bayes_bma.csv")); xn = paste0("x", 1:9)
pack = function(f) list(postprob = f$postprob, probne0 = f$probne0, postmean = unname(f$postmean),
  postsd = unname(f$postsd), condpostmean = unname(f$condpostmean), condpostsd = unname(f$condpostsd),
  bic = f$bic, r2 = f$r2, size = f$size, deviance = f$deviance,
  which = apply(f$which, 1, function(r) paste(as.integer(r), collapse = "")))
out$bicreg = pack(bicreg(d[, xn], d$y))
out$bicreg_or50 = pack(bicreg(d[, xn], d$y, OR = 50))
out$bicreg_strict = pack(bicreg(d[, xn], d$y, strict = TRUE, OR = 100))
rhs = paste(xn, collapse = "+")
out$bicglm_logit = pack(bic.glm(as.formula(paste("yb ~", rhs)), data = d, glm.family = binomial(), OR = 20))
out$bicglm_poisson = pack(bic.glm(as.formula(paste("yc ~", rhs)), data = d, glm.family = poisson(), OR = 20))
out$bicglm_gamma_log = pack(bic.glm(as.formula(paste("yg ~", rhs)), data = d, glm.family = Gamma(link = "log"), OR = 20))
out$bicglm_gamma_inverse = pack(bic.glm(as.formula(paste("yg ~", rhs)), data = d, glm.family = Gamma(), OR = 20))

## ---- BMS: g-prior, all 512 models -----------------------------------------
for (g in c("UIP", "BRIC", "RIC")) {
  b = bms(d[, c("y", xn)], mprior = "uniform", g = g, mcmc = "enumerate", nmodel = 512, user.int = FALSE)
  co = coef(b, order.by.pip = FALSE, exact = TRUE)
  tm = topmodels.bma(b)
  nvar = length(xn)
  out[[paste0("bms_", tolower(g))]] = list(names = rownames(co), pip = unname(co[, "PIP"]),
    postmean = unname(co[, "Post Mean"]), postsd = unname(co[, "Post SD"]), g = b$gprior.info$g,
    top_which = unname(apply(tm[1:nvar, 1:10], 2, function(r) paste(as.integer(r), collapse = ""))),
    top_pmp = unname(tm[nvar + 1, 1:10]))
}
write_json(out, file.path(here, "bayes_mcmc_R.json"), digits = 17, auto_unbox = TRUE, na = "null")
