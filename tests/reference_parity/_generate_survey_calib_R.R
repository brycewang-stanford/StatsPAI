#!/usr/bin/env Rscript
# Frozen reference for sp.rake / sp.linear_calibration vs R survey::rake and
# survey::calibrate(calfun = "raking" / "linear").
#
# Regenerate (from the repository root):
#   Rscript tests/reference_parity/_generate_survey_calib_R.R
#
# _fixtures/survey_calib_data.csv is written once by this script (fixed seed)
# and never overwritten afterwards.
#
# What is frozen
# --------------
# * rake_tight: survey::rake to its fixed point (maxit 1000, epsilon 1e-13
#   relative to the weight total) from design weights d, margins sex x
#   agegrp given as population COUNTS (N = 10000).
# * rake_default: survey::rake with its default control (maxit = 10,
#   epsilon = 1 person) -- recorded to size its own stopping error.
# * calib_raking: calibrate(calfun = "raking") -- Newton on the raking
#   (multiplicative) distance; must equal the IPF fixed point.
# * rake_equal_start: rake from equal weights (sp.rake(weight=None)).
# * linear_noint: calibrate(~ 0 + income + age, calfun = "linear").
# * linear_int_factor: calibrate(~ sex + income, calfun = "linear") -- the
#   intercept and the dummy enter as columns one / male on the Python side.
# * For each: the calibrated weights, and svymean(~y) on (a) the calibrated
#   design (R's calibration-adjusted linearisation SE: residuals of y on the
#   calibration variables) and (b) a plain design that takes the calibrated
#   weights as fixed.
suppressPackageStartupMessages({library(survey); library(jsonlite)})

csv <- "tests/reference_parity/_fixtures/survey_calib_data.csv"
if (!file.exists(csv)) {
  set.seed(20260919)
  n <- 200
  stratum <- rep(1:4, each = 50)
  psu <- rep(rep(1:5, each = 10), 4)
  sex <- sample(c("F", "M"), n, replace = TRUE, prob = c(0.4, 0.6))
  agegrp <- sample(c("a", "b", "c"), n, replace = TRUE, prob = c(0.5, 0.3, 0.2))
  age <- round(ifelse(agegrp == "a", 25, ifelse(agegrp == "b", 45, 70)) +
                 rnorm(n, sd = 5), 4)
  income <- round(20 + 0.3 * age + 5 * (sex == "M") + rnorm(n, sd = 6), 4)
  y <- round(3 + 0.1 * income + 0.02 * age + rnorm(n), 6)
  d <- round(runif(n, 30, 70), 4)
  write.csv(data.frame(stratum, psu, sex, agegrp, age, income, y, d,
                       one = 1, male = as.integer(sex == "M")),
            csv, row.names = FALSE)
}
cal <- read.csv(csv)
des0 <- svydesign(ids = ~psu, strata = ~stratum, weights = ~d, nest = TRUE,
                  data = cal)
des_eq <- svydesign(ids = ~psu, strata = ~stratum, weights = ~one, nest = TRUE,
                    data = cal)

pop_sex <- data.frame(sex = c("F", "M"), Freq = c(5100, 4900))
pop_age <- data.frame(agegrp = c("a", "b", "c"), Freq = c(3000, 4500, 2500))
T_income <- 10000 * 34.5
T_age <- 10000 * 44.0

dig <- function(x) unname(as.numeric(x))
summ <- function(cdes) {
  w <- dig(weights(cdes))
  m_cal <- svymean(~y, cdes)
  fixed <- svydesign(ids = ~psu, strata = ~stratum, weights = ~wcal,
                     nest = TRUE, data = transform(cal, wcal = w))
  m_fix <- svymean(~y, fixed)
  list(weights = w, weight_sum = sum(w),
       mean_y = dig(coef(m_cal)), se_calibrated = dig(SE(m_cal)),
       se_fixed_weights = dig(SE(m_fix)))
}

res <- list()
tight <- list(maxit = 1000, epsilon = 1e-13)
res$rake_tight <- summ(rake(des0, list(~sex, ~agegrp), list(pop_sex, pop_age),
                            control = tight))
rk_def <- suppressWarnings(rake(des0, list(~sex, ~agegrp),
                                list(pop_sex, pop_age)))
res$rake_default <- list(weights = dig(weights(rk_def)))
res$calib_raking <- summ(calibrate(des0, list(~sex, ~agegrp),
                                   list(pop_sex, pop_age), calfun = "raking",
                                   epsilon = 1e-13, maxit = 1000))
res$rake_equal_start <- summ(rake(des_eq, list(~sex, ~agegrp),
                                  list(pop_sex, pop_age), control = tight))
res$linear_noint <- summ(calibrate(des0, ~ 0 + income + age,
                                   population = c(income = T_income,
                                                  age = T_age),
                                   calfun = "linear"))
res$linear_int_factor <- summ(calibrate(des0, ~ sex + income,
                                        population = c(`(Intercept)` = 10000,
                                                       sexM = 4900,
                                                       income = T_income),
                                        calfun = "linear"))
res$targets <- list(sex = list(F = 5100, M = 4900),
                    agegrp = list(a = 3000, b = 4500, c = 2500),
                    N = 10000, income = T_income, age = T_age)
res$provenance <- list(
  r_version = R.version.string,
  survey_version = as.character(packageVersion("survey")),
  data = "tests/reference_parity/_fixtures/survey_calib_data.csv",
  generated_by = "tests/reference_parity/_generate_survey_calib_R.R")

writeLines(toJSON(res, auto_unbox = TRUE, digits = I(17), pretty = TRUE),
           "tests/reference_parity/_fixtures/survey_calib_R.json")
cat("wrote survey_calib_R.json\n")
