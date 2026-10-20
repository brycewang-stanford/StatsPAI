#!/usr/bin/env Rscript
# Frozen reference for replicate-weight designs (sp.svrepdesign /
# sp.as_svrepdesign and design= on the estimators): R survey 4.5
# svrepdesign + svymean / svytotal / svyglm / svycoxph / withReplicates.
# survey (GPL) is used as a black box: documentation and outputs only.
#
# Writes _fixtures/survey_replicate_data.csv (data + replicate weights
# exported from R, so both sides use the same replicate weights) and
# _fixtures/survey_replicate_R.json.
#
# Regenerate with:
#   Rscript tests/reference_parity/_generate_survey_replicate_R.R

suppressPackageStartupMessages({
  library(survey)
  library(survival)
  library(jsonlite)
})
here <- function(p) file.path("tests", "reference_parity", p)
set.seed(20261020)
H <- 10; per <- 20; n <- H * 2 * per
d <- data.frame(
  stratum = rep(1:H, each = 2 * per),
  psu = rep(1:(2 * H), each = per)
)
psu_eff <- rnorm(2 * H, sd = 0.6)[d$psu]
d$w <- round(runif(n, 1, 4), 3)
d$x1 <- round(rnorm(n) + 0.3 * psu_eff, 4)
d$x2 <- rbinom(n, 1, 0.45)
d$d <- rbinom(n, 1, plogis(-0.2 + 0.6 * d$x1 - 0.5 * d$x2))
d$y <- round(1 + 0.8 * d$x1 - 0.4 * d$x2 + 1.5 * d$d + psu_eff + rnorm(n), 4)
d$yb <- rbinom(n, 1, plogis(-0.3 + 0.7 * d$x1 + 0.4 * d$d))
t0 <- rexp(n, rate = exp(-1 + 0.5 * d$x1 - 0.4 * d$x2))
cens <- rexp(n, rate = 0.25)
d$time <- round(pmin(t0, cens), 3) + 0.001
d$status <- as.integer(t0 <= cens)
d$psu_only <- d$psu                          # unstratified view for JK1

lin <- svydesign(ids = ~psu, strata = ~stratum, weights = ~w, data = d)
lin1 <- svydesign(ids = ~psu, weights = ~w, data = d)
reps <- list(
  jkn = as.svrepdesign(lin, type = "JKn", compress = FALSE),
  brr = as.svrepdesign(lin, type = "BRR", compress = FALSE),
  fay = as.svrepdesign(lin, type = "Fay", fay.rho = 0.3, compress = FALSE),
  jk1 = as.svrepdesign(lin1, type = "JK1", compress = FALSE),
  boot = as.svrepdesign(lin, type = "subbootstrap", replicates = 30, compress = FALSE)
)
meta <- list()
for (nm in names(reps)) {
  rw <- unname(as.matrix(weights(reps[[nm]], "analysis")))
  for (j in seq_len(ncol(rw))) d[[paste0(nm, "_", j)]] <- rw[, j]
  meta[[nm]] <- list(R = ncol(rw), rscales = unname(reps[[nm]]$rscales),
                     scale = reps[[nm]]$scale)
}
write.csv(d, here("_fixtures/survey_replicate_data.csv"), row.names = FALSE)

# The references are built from the exported columns with svrepdesign, as a
# user with a public-use file would: type sets scale / rscales / mse.
mk <- function(nm, type, mse = FALSE, ...) {
  svrepdesign(data = d, repweights = paste0("^", nm, "_"), weights = ~w,
              type = type, combined.weights = TRUE, mse = mse, ...)
}
designs <- list(
  jkn = mk("jkn", "JKn", rscales = meta$jkn$rscales, scale = 1),
  jkn_mse = mk("jkn", "JKn", mse = TRUE, rscales = meta$jkn$rscales, scale = 1),
  brr = mk("brr", "BRR"),
  fay = mk("fay", "Fay", rho = 0.3),
  jk1 = mk("jk1", "JK1"),
  boot = mk("boot", "bootstrap"),
  boot_mse = mk("boot", "bootstrap", mse = TRUE)
)

# Hajek IPW ATE with a weighted logit score (sp.ipw's estimator with
# weights=). The weight argument must not be called `w`: glm() looks
# weights up in `data` first, which has a full-sample column `w`.
ipw_ate <- function(rw, data) {
  ps <- fitted(glm(d ~ x1 + x2, data = data, weights = rw, family = quasibinomial()))
  t <- data$d == 1
  m1 <- sum(rw[t] * data$y[t] / ps[t]) / sum(rw[t] / ps[t])
  m0 <- sum(rw[!t] * data$y[!t] / (1 - ps[!t])) / sum(rw[!t] / (1 - ps[!t]))
  m1 - m0
}

out <- list(meta = meta)
for (nm in names(designs)) {
  D <- designs[[nm]]
  m <- svymean(~y, D); tt <- svytotal(~y, D)
  g <- svyglm(y ~ x1 + x2, design = D)
  tight <- glm.control(epsilon = 1e-13, maxit = 100)
  b <- suppressWarnings(svyglm(yb ~ x1 + x2, design = D, family = quasibinomial(),
                               control = tight))
  cx <- svycoxph(Surv(time, status) ~ x1 + x2, design = D)
  cxb <- svycoxph(Surv(time, status) ~ x1 + x2, design = D, ties = "breslow")
  pr <- suppressWarnings(svyglm(yb ~ x1 + x2, design = D,
                                family = quasibinomial(link = "probit"),
                                control = tight))
  ip <- withReplicates(D, ipw_ate)
  out[[nm]] <- list(
    scale = D$scale, rscales = unname(D$rscales), mse = D$mse, degf = degf(D),
    mean = unname(coef(m)), mean_se = unname(SE(m)),
    total = unname(coef(tt)), total_se = unname(SE(tt)),
    glm_coef = unname(coef(g)), glm_se = unname(SE(g)),
    logit_coef = unname(coef(b)), logit_se = unname(SE(b)),
    cox_coef = unname(coef(cx)), cox_se = unname(SE(cx)),
    cox_breslow_coef = unname(coef(cxb)), cox_breslow_se = unname(SE(cxb)),
    probit_coef = unname(coef(pr)), probit_se = unname(SE(pr)),
    ipw = unname(coef(ip)), ipw_se = unname(SE(ip))
  )
}
out$versions <- list(R = R.version.string,
                     survey = as.character(packageVersion("survey")),
                     survival = as.character(packageVersion("survival")))
write_json(out, here("_fixtures/survey_replicate_R.json"), digits = NA,
           auto_unbox = TRUE)
