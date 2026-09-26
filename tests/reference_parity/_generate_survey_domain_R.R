#!/usr/bin/env Rscript
# Frozen reference for survey domain estimation (subpop=) vs R survey.
# Regenerate from the repository root:
#   Rscript tests/reference_parity/_generate_survey_domain_R.R
# Reads the committed _fixtures/survey_calib_data.csv.
suppressPackageStartupMessages({library(survey); library(jsonlite)})
csv <- "tests/reference_parity/_fixtures/survey_calib_data.csv"
cal <- read.csv(csv)
cal$dom2 <- as.integer(cal$agegrp == "a" & cal$stratum <= 2)  # empty in strata 3, 4
des <- svydesign(ids = ~psu, strata = ~stratum, weights = ~d, nest = TRUE, data = cal)
dig <- function(x) unname(as.numeric(x))
one <- function(sub) {
  m <- svymean(~y, sub); t <- svytotal(~y, sub)
  g <- svyglm(y ~ income + age, design = sub)
  list(mean = dig(coef(m)), se_mean = dig(SE(m)), total = dig(coef(t)),
       se_total = dig(SE(t)), glm_coef = dig(coef(g)), glm_se = dig(SE(g)),
       degf = degf(sub))
}
pop_sex <- data.frame(sex = c("F", "M"), Freq = c(5100, 4900))
pop_age <- data.frame(agegrp = c("a", "b", "c"), Freq = c(3000, 4500, 2500))
calib <- calibrate(des, list(~sex, ~agegrp), list(pop_sex, pop_age),
                   calfun = "raking", epsilon = 1e-13, maxit = 1000)
res <- list(
  male = one(subset(des, sex == "M")),
  dom2 = one(subset(des, dom2 == 1)),
  calib_male = one(subset(calib, sex == "M")),
  provenance = list(r_version = R.version.string,
                    survey_version = as.character(packageVersion("survey"))))
writeLines(toJSON(res, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "tests/reference_parity/_fixtures/survey_domain_R.json")
cat("wrote survey_domain_R.json\n")
