#!/usr/bin/env Rscript
# Frozen reference for SurveyDesign.calibrate(): svytotal and svyglm on a
# CALIBRATED design (calibration-adjusted linearisation), R survey.
#
# Regenerate (from the repository root):
#   Rscript tests/reference_parity/_generate_survey_calibrated_design_R.R
#
# Reads the committed _fixtures/survey_calib_data.csv (never rewrites it).
# Cases: calibrate(calfun = "raking") to sex x agegrp counts, and
# calibrate(~ 0 + income + age, calfun = "linear"). For each: svymean(~y),
# svytotal(~y), svytotal(~income) (a calibration variable for the linear
# case: SE = 0), svyglm(y ~ income + age) coefficients and SEs, and
# svyglm(yb ~ income, quasibinomial, epsilon = 1e-14 so the IRLS gap is not
# R's default early stop) with yb = 1[y > median(y)].
suppressPackageStartupMessages({library(survey); library(jsonlite)})

csv <- "tests/reference_parity/_fixtures/survey_calib_data.csv"
cal <- read.csv(csv)
cal$yb <- as.integer(cal$y > median(cal$y))
des0 <- svydesign(ids = ~psu, strata = ~stratum, weights = ~d, nest = TRUE,
                  data = cal)
pop_sex <- data.frame(sex = c("F", "M"), Freq = c(5100, 4900))
pop_age <- data.frame(agegrp = c("a", "b", "c"), Freq = c(3000, 4500, 2500))
T_income <- 10000 * 34.5
T_age <- 10000 * 44.0
dig <- function(x) unname(as.numeric(x))

summ <- function(cdes) {
  m <- svymean(~y, cdes)
  ty <- svytotal(~y, cdes)
  ti <- svytotal(~income, cdes)
  g <- svyglm(y ~ income + age, design = cdes)
  b <- suppressWarnings(svyglm(yb ~ income, design = cdes,
                               family = quasibinomial(),
                               control = glm.control(epsilon = 1e-14, maxit = 100)))
  list(mean_y = dig(coef(m)), se_mean_y = dig(SE(m)),
       total_y = dig(coef(ty)), se_total_y = dig(SE(ty)),
       total_income = dig(coef(ti)), se_total_income = dig(SE(ti)),
       glm_coef = dig(coef(g)), glm_se = dig(SE(g)),
       logit_coef = dig(coef(b)), logit_se = dig(SE(b)),
       weight_sum = sum(weights(cdes)))
}

res <- list()
res$raking <- summ(calibrate(des0, list(~sex, ~agegrp), list(pop_sex, pop_age),
                             calfun = "raking", epsilon = 1e-13, maxit = 1000))
res$linear <- summ(calibrate(des0, ~0 + income + age,
                             c(income = T_income, age = T_age),
                             calfun = "linear"))
res$provenance <- list(
  r_version = R.version.string,
  survey_version = as.character(packageVersion("survey")),
  data = csv,
  generated_by = "tests/reference_parity/_generate_survey_calibrated_design_R.R")
writeLines(toJSON(res, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "tests/reference_parity/_fixtures/survey_calibrated_design_R.json")
cat("wrote survey_calibrated_design_R.json\n")
