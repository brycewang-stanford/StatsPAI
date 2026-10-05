# Reference posteriors for tests/external_parity/test_ramirez_hassan_bayes.py
#
# The examples of Ramirez-Hassan, "Introduction to Bayesian Econometrics"
# (2026), chapters 6 and 9, on the book's own data (the DataApp folder of
# https://github.com/besmarter/BSTApp, GPL-3, not redistributed here), with
# the packages the book uses. Long chains, so that the Monte Carlo error of
# the reference is small next to the posterior standard deviation.
#
#   export STATSPAI_RAMIREZ_HASSAN_DIR=<.../BSTApp/DataApp>
#   Rscript tests/external_parity/ramirez_hassan_bayes_reference.R
#
# The run takes about half an hour. Results are written after every section
# and a section already in the JSON is skipped, so an interrupted run resumes;
# delete the JSON to start over.
#
# Requires: MCMCpack, bayesm, coda, lme4, jsonlite.
suppressMessages({library(MCMCpack); library(bayesm); library(coda); library(jsonlite)})
dd = Sys.getenv("STATSPAI_RAMIREZ_HASSAN_DIR")
stopifnot(nzchar(dd), dir.exists(dd))
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
post = function(m, names = NULL) {
  m = as.matrix(m); if (!is.null(names)) colnames(m) = names
  list(names = colnames(m), mean = unname(colMeans(m)), sd = unname(apply(m, 2, sd)),
       mcse = unname(sqrt(spectrum0.ar(m)$spec / nrow(m))),
       q025 = unname(apply(m, 2, quantile, 0.025)), q975 = unname(apply(m, 2, quantile, 0.975)))
}
target = file.path(here, "data", "ramirez_hassan_bayes_R.json")
out = if (file.exists(target)) fromJSON(target, simplifyVector = TRUE) else list()
out$versions = list(R = R.version.string, MCMCpack = as.character(packageVersion("MCMCpack")),
  bayesm = as.character(packageVersion("bayesm")), coda = as.character(packageVersion("coda")))
save = function() write_json(out, target, digits = 10, auto_unbox = TRUE)
todo = function(key) is.null(out[[key]])
set.seed(10101)

## ---- 6.1 Gaussian linear model: market value of soccer players ------------
fp = read.csv(file.path(dd, "1ValueFootballPlayers.csv"))
fp$lv = log(fp$Value); fp$lvc = log(fp$ValueCens)
f1 = lv ~ Perf + Age + Age2 + NatTeam + Goals + Exp + Exp2
k = 8
if (todo("linear")) {
  m = MCMCregress(f1, data = fp, b0 = rep(0, k), B0 = diag(k) / 1000, c0 = 0.001, d0 = 0.001,
                  burnin = 5000, mcmc = 200000, seed = 1)
  out$linear = post(m); save()
}

## ---- 6.8 Tobit: the same regression, value censored below one million ------
out$tobit_n_censored = sum(fp$ValueCens <= 1e6)
if (todo("tobit")) {
  m = MCMCtobit(lvc ~ Perf + Age + Age2 + NatTeam + Goals + Exp + Exp2, data = fp, b0 = rep(0, k),
                B0 = diag(k) / 1000, c0 = 0.001, d0 = 0.001, below = log(1e6), above = Inf,
                burnin = 5000, mcmc = 200000, seed = 2)
  out$tobit = post(m); save()
}

## ---- 6.9 Quantile regression (asymmetric Laplace, scale fixed at 1) --------
for (q in c(0.5, 0.9)) if (todo(paste0("quantile_", q * 100))) {
  m = MCMCquantreg(f1, data = fp, tau = q, b0 = rep(0, k), B0 = diag(k) / 1000,
                   burnin = 5000, mcmc = 200000, seed = 3)
  out[[paste0("quantile_", q * 100)]] = post(m); save()
}

## ---- 6.3 Probit and 6.2 logit: hospitalisation ------------------------------
hm = read.csv(file.path(dd, "2HealthMed.csv"))
xb = c("SHI", "Female", "Age", "Age2", "Est2", "Est3", "Fair", "Good", "Excellent")
X = cbind(1, as.matrix(hm[, xb])); kb = ncol(X)
if (todo("probit")) {
  pr = rbprobitGibbs(Data = list(y = hm$Hosp, X = X), Prior = list(betabar = rep(0, kb), A = diag(kb)),
                     Mcmc = list(R = 60000, keep = 1, nprint = 0))
  out$probit = post(pr$betadraw[-(1:5000), ], c("Intercept", xb)); save()
}
if (todo("logit")) {
  fl = as.formula(paste("Hosp ~", paste(xb, collapse = "+")))
  m = MCMClogit(fl, data = hm, b0 = rep(0, kb), B0 = diag(kb) / 100, burnin = 5000, mcmc = 300000,
                thin = 10, tune = 0.8, seed = 4)
  out$logit = post(m); save()
}

## ---- 6.6 Ordered probit: preventive visits ----------------------------------
# The book passes X without a constant to rordprobitGibbs, whose first
# cutpoint is fixed at zero. Both versions are run. (The maximum likelihood
# benchmark is computed in the test with sp.oprobit.)
xo = c("SHI", "Female", "Age", "Age2", "Est2", "Est3", "Fair", "Good", "Excellent", "PriEd", "HighEd", "VocEd", "UnivEd")
yo = hm$MedVisPrevOr; L = length(table(yo)); ko = length(xo)
run_op = function(Xm) {
  kk = ncol(Xm)
  r = rordprobitGibbs(Data = list(y = yo, X = Xm, k = L),
        Prior = list(betabar = rep(0, kk), A = diag(kk) / 1000, dstarbar = rep(0, L - 2), Ad = diag(L - 2)),
        Mcmc = list(R = 60000, keep = 1, s = 1 / sqrt(L - 2), nprint = 0))
  list(beta = r$betadraw[-(1:10000), , drop = FALSE], cut = r$cutdraw[-(1:10000), , drop = FALSE])
}
if (todo("oprobit_constant")) {
  with_c = run_op(cbind(1, as.matrix(hm[, xo])))
  # cutdraw holds the interior cutpoints, the first of them fixed at 0
  cd = with_c$cut
  if (ncol(cd) == L + 1) cd = cd[, 2:L, drop = FALSE]
  stopifnot(ncol(cd) == L - 1, all(cd[, 1] == 0))
  out$oprobit_cuts = post(cd - with_c$beta[, 1], paste0("cut", 1:(L - 1)))
  out$oprobit_constant = post(with_c$beta, c("Intercept", xo)); save()
}
if (todo("oprobit_book")) {
  no_c = run_op(as.matrix(hm[, xo]))
  out$oprobit_book = post(no_c$beta, xo); save()
}

## ---- 9.1 Hierarchical normal model: public capital ---------------------------
gs = read.csv(file.path(dd, "8PublicCap.csv"))
gs$lgsp = log(gs$gsp); gs$lpcap = log(gs$pcap); gs$lpc = log(gs$pc); gs$lemp = log(gs$emp)
if (todo("hier")) {
m = MCMChregress(fixed = lgsp ~ lpcap + lpc + lemp + unemp, random = ~1, group = "id", data = gs,
                 burnin = 5000, mcmc = 100000, thin = 1, r = 5, R = diag(1), nu = 0.001, delta = 0.001,
                 mubeta = rep(0, 5), Vbeta = 1, seed = 5, verbose = 0)
dr = as.matrix(m$mcmc)
keep = c(grep("^beta\\.", colnames(dr)), grep("^VCV", colnames(dr)), grep("^sigma2", colnames(dr)))
out$hier = post(dr[, keep])
out$hier$names = colnames(dr)[keep]
save()
}
# Restricted maximum likelihood for the same model: an independent yardstick
# for the uncertainty of the fixed effects.
if (todo("hier_lmer")) {
  suppressMessages(library(lme4))
  lm4 = lmer(lgsp ~ lpcap + lpc + lemp + unemp + (1 | id), data = gs, REML = TRUE)
  vc = as.data.frame(VarCorr(lm4))
  out$hier_lmer = list(names = names(fixef(lm4)), est = unname(fixef(lm4)),
    se = unname(sqrt(diag(as.matrix(vcov(lm4))))), var_id = vc$vcov[1], sigma2 = vc$vcov[2],
    lme4 = as.character(packageVersion("lme4")))
  save()
}
