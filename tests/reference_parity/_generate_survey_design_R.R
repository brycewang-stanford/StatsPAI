#!/usr/bin/env Rscript
# Frozen reference for sp.svydesign (stratified, clustered, weighted, fpc
# designs; lonely-PSU handling; design df) and the estimators it exposes
# (mean / total / glm) vs R survey::svydesign + svymean / svytotal / svyglm.
#
# Regenerate (from the repository root):
#   Rscript tests/reference_parity/_generate_survey_design_R.R
#
# The data file _fixtures/survey_design_data.csv is written once by this
# script (fixed seed, values rounded to 6 decimals so the CSV bytes are the
# input both sides read) and never overwritten afterwards: delete it
# deliberately if the DGP must change.
#
# Design of the data
# ------------------
# * 6 strata with 2, 3, 4, 5, 3, 6 PSUs; PSU ids restart at 1 in every
#   stratum, so the ids are NOT unique across strata (R needs nest = TRUE;
#   Stata's svyset identifies PSUs within strata automatically).
# * 4-12 elements per PSU, unequal weights w.
# * fpc_N : population number of PSUs in the stratum (count form);
#   fpc_f : n_h / N_h, the same information as a sampling fraction;
#   fpc_el: population number of ELEMENTS in the stratum (for the
#           element-sampled design ids = ~1).
# * stratum_l: stratum 2's third PSU moved into its own stratum 7 -> one
#   single-PSU ("lonely") stratum, for survey.lonely.psu.
# * y continuous, x continuous, yb binary, yc count.
suppressPackageStartupMessages({library(survey); library(jsonlite)})

csv <- "tests/reference_parity/_fixtures/survey_design_data.csv"
if (!file.exists(csv)) {
  set.seed(20260918)
  n_psu <- c(2, 3, 4, 5, 3, 6)
  pop_mult <- c(5, 3, 10, 2, 8, 4)
  rows <- list()
  for (h in seq_along(n_psu)) {
    for (j in seq_len(n_psu[h])) {
      m <- sample(4:12, 1)
      u_psu <- rnorm(1, sd = 1.5)
      x <- round(rnorm(m, mean = 2 + 0.3 * h, sd = 1), 6)
      y <- round(5 + 0.8 * h + u_psu + 1.5 * x + rnorm(m, sd = 2), 6)
      eta <- -1 + 0.4 * x + 0.3 * u_psu
      yb <- rbinom(m, 1, 1 / (1 + exp(-eta)))
      yc <- rpois(m, exp(0.2 + 0.25 * x + 0.2 * u_psu))
      w <- round(pop_mult[h] * runif(m, 5, 15), 4)
      rows[[length(rows) + 1]] <- data.frame(
        stratum = h, psu = j, w = w, y = y, x = x, yb = yb, yc = yc,
        fpc_N = n_psu[h] * pop_mult[h],
        stratum_l = ifelse(h == 2 & j == 3, 7L, h))
    }
  }
  df <- do.call(rbind, rows)
  df$fpc_f <- round(ave(df$psu, df$stratum, FUN = function(p) length(unique(p))) /
                      df$fpc_N, 12)
  df$fpc_el <- ave(df$y, df$stratum, FUN = length) * 7
  write.csv(df, csv, row.names = FALSE)
}
df <- read.csv(csv)

dig <- function(x) unname(as.numeric(x))

one_mean <- function(des, form) {
  m <- svymean(form, des, deff = TRUE)
  dd <- degf(des)
  list(estimate = dig(coef(m)), se = dig(SE(m)), vcov = unname(vcov(m)),
       deff = dig(deff(m)),
       ci_t = unname(confint(m, df = dd)),
       ci_normal = unname(confint(m)))
}
one_total <- function(des, form) {
  t <- svytotal(form, des, deff = TRUE)
  list(estimate = dig(coef(t)), se = dig(SE(t)), deff = dig(deff(t)),
       ci_t = unname(confint(t, df = degf(des))))
}
# Non-gaussian fits: glm.fit returns the working weights of the iteration
# BEFORE its last update, and its deviance criterion stops while beta still
# moves ~1e-7 (deviance is flat at the optimum), so svyglm's bread is one
# step stale: ~2e-7 relative in the logit SEs and ~2e-8 in the Poisson SEs
# here. The reference fit is therefore refitted with start = the converged
# coefficients, so the weights glm.fit reports are evaluated at the MLE.
# The *_default_control block records the one-pass default output.
tight <- glm.control(epsilon = 1e-14, maxit = 100)
one_glm <- function(des, form, fam, control = tight, refit = TRUE) {
  g <- do.call(svyglm, list(formula = form, design = des, family = fam,
                            control = control))
  if (refit && fam$family != "gaussian")
    g <- do.call(svyglm, list(formula = form, design = des, family = fam,
                              control = control, start = unname(coef(g))))
  s <- summary(g)$coefficients
  list(coef = dig(coef(g)), se = dig(SE(g)), vcov = unname(vcov(g)),
       df_residual = g$df.residual,
       t = dig(s[, 3]), p = dig(s[, 4]),
       ci = unname(confint(g)))
}
design_block <- function(des, glms = TRUE) {
  out <- list(degf = degf(des),
              mean_y = one_mean(des, ~y),
              mean_yx = one_mean(des, ~y + x),
              total_y = one_total(des, ~y),
              glm_gaussian = one_glm(des, y ~ x, gaussian()))
  if (glms) {
    out$glm_binomial <- one_glm(des, yb ~ x, quasibinomial())
    out$glm_poisson <- one_glm(des, yc ~ x, quasipoisson())
    out$glm_binomial_default_control <- one_glm(des, yb ~ x, quasibinomial(),
                                                control = glm.control(),
                                                refit = FALSE)
  }
  out
}

res <- list()
res$full <- design_block(svydesign(ids = ~psu, strata = ~stratum, weights = ~w,
                                   fpc = ~fpc_N, nest = TRUE, data = df))
res$frac <- design_block(svydesign(ids = ~psu, strata = ~stratum, weights = ~w,
                                   fpc = ~fpc_f, nest = TRUE, data = df),
                         glms = FALSE)
res$nofpc <- design_block(svydesign(ids = ~psu, strata = ~stratum, weights = ~w,
                                    nest = TRUE, data = df))
res$element_fpc <- design_block(svydesign(ids = ~1, strata = ~stratum,
                                          weights = ~w, fpc = ~fpc_el,
                                          data = df), glms = FALSE)
# clusters without strata: psu ids are not unique, so build a global id
df$psu_global <- df$stratum * 100 + df$psu
res$cluster_only <- design_block(svydesign(ids = ~psu_global, weights = ~w,
                                           data = df), glms = FALSE)

# nest = FALSE on crossed ids is refused by R -- recorded, not skipped
res$crossed_nest_false_error <- tryCatch({
  svydesign(ids = ~psu, strata = ~stratum, weights = ~w, data = df)
  "no error"
}, error = function(e) conditionMessage(e))

# lonely PSU (stratum_l has one single-PSU stratum)
res$lonely_fail_error <- tryCatch({
  options(survey.lonely.psu = "fail")
  svymean(~y, svydesign(ids = ~psu, strata = ~stratum_l, weights = ~w,
                        nest = TRUE, data = df))
  "no error"
}, error = function(e) conditionMessage(e))
for (lp in c("remove", "certainty", "adjust", "average")) {
  options(survey.lonely.psu = lp)
  des <- svydesign(ids = ~psu, strata = ~stratum_l, weights = ~w, nest = TRUE,
                   data = df)
  res[[paste0("lonely_", lp)]] <- list(
    degf = degf(des),
    mean_y = list(estimate = dig(coef(svymean(~y, des))),
                  se = dig(SE(svymean(~y, des)))),
    mean_yx = list(vcov = unname(vcov(svymean(~y + x, des)))),
    total_y = list(se = dig(SE(svytotal(~y, des)))),
    glm_gaussian = list(se = dig(SE(svyglm(y ~ x, design = des)))))
}
options(survey.lonely.psu = "fail")

res$provenance <- list(
  r_version = R.version.string,
  survey_version = as.character(packageVersion("survey")),
  data = "tests/reference_parity/_fixtures/survey_design_data.csv",
  generated_by = "tests/reference_parity/_generate_survey_design_R.R")

writeLines(toJSON(res, auto_unbox = TRUE, digits = I(17), pretty = TRUE),
           "tests/reference_parity/_fixtures/survey_design_R.json")
cat("wrote survey_design_R.json\n")
