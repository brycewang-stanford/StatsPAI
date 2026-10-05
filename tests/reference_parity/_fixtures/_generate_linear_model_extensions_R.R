# Reference values for tests/reference_parity/test_linear_model_extensions_parity.py
#
# Runs on the committed synthetic file linear_model_extensions.csv and writes
# linear_model_extensions_R.json next to it.
#
# Requires: sandwich, MASS, leaps, gee, quantreg, survival, jsonlite.
#   Rscript tests/reference_parity/_fixtures/_generate_linear_model_extensions_R.R
suppressMessages({
  library(sandwich); library(MASS); library(leaps); library(gee)
  library(quantreg); library(survival); library(jsonlite)
})
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
d = read.csv(file.path(here, "linear_model_extensions.csv"))
d$ly = log(d$y)
tight = glm.control(epsilon = 1e-14, maxit = 200)
out = list(versions = list(R = R.version.string, sandwich = as.character(packageVersion("sandwich")),
  MASS = as.character(packageVersion("MASS")), leaps = as.character(packageVersion("leaps")),
  gee = as.character(packageVersion("gee")), quantreg = as.character(packageVersion("quantreg")),
  survival = as.character(packageVersion("survival"))))

## OLS: HC4, influence, regression through the origin
f = y ~ x1 + x2 + x3 + x4 + x5 + x6 + treat
m = lm(f, data = d); mw = lm(f, data = d, weights = w)
out$ols = list(names = names(coef(m)), est = unname(coef(m)),
  hc4 = unname(sqrt(diag(vcovHC(m, type = "HC4")))), hc4_w = unname(sqrt(diag(vcovHC(mw, type = "HC4")))),
  hat = unname(hatvalues(m)), rstudent = unname(rstudent(m)), dffits = unname(dffits(m)),
  cook = unname(cooks.distance(m)), press = sum((resid(m) / (1 - hatvalues(m)))^2))
m0 = lm(y ~ 0 + x1 + I(1 - x1), data = d)
out$origin = list(est = unname(coef(m0)), se = unname(summary(m0)$coef[, 2]))
mq = lm(y ~ x1 + I(x1^2), data = d)
out$square = list(est = unname(coef(mq)), se = unname(summary(mq)$coef[, 2]))

## cauchit link
gc = glm(d ~ x1 + x2 + x4 + treat, family = binomial(link = "cauchit"), data = d, control = tight)
out$cauchit = list(est = unname(coef(gc)), se = unname(summary(gc)$coef[, 2]), deviance = gc$deviance)

## generalized estimating equations (gee wants contiguous clusters in time order)
ds = d[order(d$id, d$t), ]
gs = function(g) { s = summary(g)$coef
  list(est = unname(s[, 1]), naive = unname(s[, 2]), robust = unname(s[, 4]), scale = g$scale,
       alpha = unname(g$working.correlation[1, 2])) }
quiet = function(expr) { sink(tempfile()); on.exit(sink()); suppressMessages(expr) }
out$gee = list()
for (fam in c("gaussian", "binomial", "poisson")) {
  fm = switch(fam, gaussian = ly ~ treat + x1 + x2 + x4, binomial = d ~ treat + x1 + x2 + x4,
              poisson = c ~ treat + x1 + x2 + x4)
  for (cs in c("independence", "exchangeable", "AR-M")) {
    key = paste0(fam, "_", sub("AR-M", "ar1", cs))
    out$gee[[key]] = gs(quiet(gee(fm, id = id, family = fam, corstr = cs, Mv = 1, data = ds,
                                  tol = 1e-11, maxiter = 500)))
  }
}

## ridge
lam = c(0, 1, 5, 25, 100)
rf = lm.ridge(f, data = d, lambda = lam)
out$ridge = list(lambda = lam, names = colnames(coef(rf)), coef = unname(coef(rf)), gcv = unname(rf$GCV),
                 kHKB = rf$kHKB, kLW = rf$kLW)

## Box-Cox profile
bc = boxcox(m, lambda = seq(-1, 1, 0.25), plotit = FALSE, interp = FALSE)
out$boxcox = list(lambda = bc$x, loglik = bc$y)

## best subset
sm = summary(regsubsets(f, data = d, nvmax = 7, method = "exhaustive"))
out$subsets = list(rss = sm$rss, adjr2 = sm$adjr2, cp = sm$cp,
  which = lapply(1:7, function(i) colnames(sm$which)[sm$which[i, ]][-1]))

## quantile regression with weights
fq = ly ~ x1 + x2 + x4 + treat
out$rq = list(taus = c(0.25, 0.5, 0.75))
for (tau in out$rq$taus) {
  r0 = rq(fq, tau = tau, data = d); rw = rq(fq, tau = tau, data = d, weights = w)
  out$rq[[paste0("tau", tau)]] = list(coef = unname(coef(r0)), coef_w = unname(coef(rw)),
    ker = unname(summary(r0, se = "ker")$coef[, 2]), nid = unname(summary(r0, se = "nid")$coef[, 2]),
    ker_w = unname(summary(rw, se = "ker")$coef[, 2]), nid_w = unname(summary(rw, se = "nid")$coef[, 2]))
}

## Cox with tied event times
cx = function(m) { s = summary(m)$coef
  list(est = unname(s[, "coef"]), se = unname(s[, "se(coef)"]), robust = unname(s[, "robust se"])) }
fs = Surv(time, event) ~ treat + x1 + x2 + x4
out$cox = list(
  efron = cx(coxph(fs, data = d, robust = TRUE)),
  breslow = cx(coxph(fs, data = d, robust = TRUE, ties = "breslow")),
  efron_cluster = cx(coxph(fs, data = d, cluster = id)),
  efron_strata = cx(coxph(Surv(time, event) ~ treat + x1 + x2 + x4 + strata(site), data = d, robust = TRUE)),
  n_tied_times = sum(table(d$time[d$event == 1]) > 1))

## Kaplan-Meier with survfit's default (log) interval
skm = summary(survfit(Surv(time, event) ~ 1, data = d), times = c(5, 15, 30, 60))
out$km = list(time = skm$time, surv = skm$surv, se = skm$std.err, lower = skm$lower, upper = skm$upper)

write(toJSON(out, digits = 15, auto_unbox = TRUE), file.path(here, "linear_model_extensions_R.json"))
cat("wrote linear_model_extensions_R.json\n")
