# Reference values for tests/reference_parity/test_probe_epi_parity.py
# (2026-11 epidemiology probe). Run from the repository root:
#
#   Rscript tests/reference_parity/_fixtures/_generate_probe_epi_R.R
#
# Needs survival, survRM2, WeightIt, metafor, MendelianRandomization, survey
# and jsonlite. survRM2 is not in the default library on the reference machine;
# install it into the gitignored private library first:
#
#   L=tests/reference_parity/_fixtures/_rlib_probe_epi
#   mkdir -p $L && Rscript -e 'install.packages("survRM2", lib="'$L'")'
#
# GPL packages are run as black boxes; nothing here is derived from their
# source.
args <- commandArgs(trailingOnly = FALSE)
fx <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
.libPaths(c(file.path(fx, "_rlib_probe_epi"), .libPaths()))
suppressMessages({
  library(survival); library(survRM2); library(WeightIt); library(metafor)
  library(MendelianRandomization); library(survey); library(jsonlite)
})
d <- read.csv(file.path(fx, "probe_epi_survival.csv"))
out <- list(versions = list(
  R = R.version.string,
  survival = as.character(packageVersion("survival")),
  survRM2 = as.character(packageVersion("survRM2")),
  WeightIt = as.character(packageVersion("WeightIt")),
  metafor = as.character(packageVersion("metafor")),
  MendelianRandomization = as.character(packageVersion("MendelianRandomization")),
  survey = as.character(packageVersion("survey"))
))

# ---- Kaplan-Meier quantiles with confidence limits --------------------
probs <- c(0.25, 0.5, 0.75)
kmq <- list()
for (ct in c("log", "log-log", "plain")) {
  f <- survfit(Surv(time, event) ~ arm, d, conf.type = ct)
  q <- quantile(f, probs)
  kmq[[ct]] <- list(quantile = unname(q$quantile), lower = unname(q$lower),
                    upper = unname(q$upper))
}
out$km_quantiles <- list(probs = probs, groups = c("0", "1"), by_conf_type = kmq)

# ---- G-rho and stratified log-rank tests ------------------------------
sd_out <- function(s) {
  obs <- if (is.matrix(s$obs)) rowSums(s$obs) else s$obs
  ex <- if (is.matrix(s$exp)) rowSums(s$exp) else s$exp
  list(chisq = s$chisq, obs = unname(obs), exp = unname(ex))
}
out$logrank <- list(
  rho0 = sd_out(survdiff(Surv(time, event) ~ arm, d)),
  rho1 = sd_out(survdiff(Surv(time, event) ~ arm, d, rho = 1)),
  rho05 = sd_out(survdiff(Surv(time, event) ~ arm, d, rho = 0.5)),
  strata_rho0 = sd_out(survdiff(Surv(time, event) ~ arm + strata(site), d)),
  strata_rho1 = sd_out(survdiff(Surv(time, event) ~ arm + strata(site), d, rho = 1)),
  ecog_rho1 = sd_out(survdiff(Surv(time, event) ~ ecog, d, rho = 1))
)

# ---- Cox: concordance of a stratified fit, baseline hazard ------------
bh <- function(fit) {
  b <- basehaz(fit, centered = FALSE)
  b <- b[!duplicated(b[, setdiff(names(b), "hazard")]), ]
  list(time = b$time, hazard = b$hazard,
       strata = if ("strata" %in% names(b)) as.character(b$strata) else NULL)
}
f_plain <- coxph(Surv(time, event) ~ age + ecog + arm, d, ties = "efron")
f_bres <- coxph(Surv(time, event) ~ age + ecog + arm, d, ties = "breslow")
f_str <- coxph(Surv(time, event) ~ age + ecog + arm + strata(site), d, ties = "efron")
f_w <- coxph(Surv(time, event) ~ age + ecog + arm, d, ties = "efron", weights = w)
f_cp <- coxph(Surv(entry, time, event) ~ age + ecog + arm, d, ties = "efron")
out$cox <- list(
  strata_concordance = unname(summary(f_str)$concordance[1]),
  plain_concordance = unname(summary(f_plain)$concordance[1]),
  coef_strata = unname(coef(f_str)),
  basehaz_efron = bh(f_plain),
  basehaz_breslow = bh(f_bres),
  basehaz_strata = bh(f_str),
  basehaz_weights = bh(f_w),
  basehaz_entry = bh(f_cp)
)

# ---- survRM2: default tau and a tau beyond the last time of arm 0 -----
rm_out <- function(r) list(
  tau = r$tau,
  rmst0 = unname(r$RMST.arm0$rmst[1:2]), rmst1 = unname(r$RMST.arm1$rmst[1:2]),
  contrasts = unname(as.matrix(r$unadjusted.result)))
out$rmst <- list(
  default = rm_out(rmst2(d$time, d$event, d$arm)),
  tau70 = rm_out(rmst2(d$time, d$event, d$arm, tau = 70)),
  max_time = c(max(d$time[d$arm == 0]), max(d$time[d$arm == 1]))
)

# ---- WeightIt: effective sample sizes of the tilted weights -----------
ess <- list()
for (est in c("ATE", "ATT", "ATC", "ATO", "ATM")) {
  W <- weightit(arm ~ age + ecog + site, d, method = "glm", estimand = est)
  e <- summary(W)$effective.sample.size
  ess[[est]] <- list(control = e["Weighted", "Control"], treated = e["Weighted", "Treated"])
}
out$weightit_ess <- ess

# ---- metafor: REML and Knapp-Hartung ----------------------------------
m <- read.csv(file.path(fx, "probe_epi_meta.csv"))
ct <- list(threshold = 1e-13, maxiter = 10000)
ma <- function(r) list(beta = as.numeric(r$beta), se = r$se, ci = c(r$ci.lb, r$ci.ub),
                       pval = r$pval, tau2 = r$tau2, QE = r$QE)
out$meta <- list(
  REML = ma(rma(yi, sei = sei, data = m, method = "REML", control = ct)),
  REML_knha = ma(rma(yi, sei = sei, data = m, method = "REML", test = "knha", control = ct)),
  DL = ma(rma(yi, sei = sei, data = m, method = "DL")),
  DL_knha = ma(rma(yi, sei = sei, data = m, method = "DL", test = "knha"))
)

# ---- mode-based MR estimate with the delta-method ratio SE ------------
g <- read.csv(file.path(fx, "mr_ldl.csv"))
o <- mr_input(bx = g$bx, bxse = g$bxse, by = g$by, byse = g$byse)
mbe <- list()
for (wt in c("weighted", "unweighted")) for (se in c("simple", "delta")) {
  set.seed(20261113)
  r <- mr_mbe(o, weighting = wt, stderror = se, phi = 1, iterations = 20000)
  mbe[[paste(wt, se, sep = "_")]] <- list(estimate = r@Estimate, se = r@StdError)
}
out$mr_mbe <- mbe

# ---- survey: design effects of regression coefficients ----------------
ds <- svydesign(ids = ~1, strata = ~site, weights = ~w, data = d)
dc <- svydesign(ids = ~age, weights = ~w, data = d)  # ages as clusters
tight <- glm.control(epsilon = 1e-14, maxit = 100)
sg <- function(f, des, fam) {
  g <- svyglm(f, des, family = fam, deff = TRUE, control = tight)
  list(coef = unname(coef(g)), se = unname(SE(g)), deff = unname(deff(g)))
}
out$svyglm_deff <- list(
  gaussian_strata = sg(time ~ age + ecog + arm, ds, gaussian()),
  gaussian_cluster = sg(time ~ ecog + arm, dc, gaussian()),
  binomial_strata = sg(event ~ age + ecog + arm, ds, quasibinomial()),
  poisson_strata = sg(time ~ age + ecog + arm, ds, quasipoisson())
)

writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE, null = "null"),
           file.path(fx, "probe_epi_R.json"))
