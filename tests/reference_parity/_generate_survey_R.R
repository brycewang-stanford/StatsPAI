#!/usr/bin/env Rscript
# Frozen reference for sp.svymean / sp.svytotal vs R survey::svymean/svytotal
# (Horvitz-Thompson / Hajek estimators + Taylor-linearization SE, weights-only
# design ids=~1). Regenerate: Rscript tests/reference_parity/_generate_survey_R.R
suppressPackageStartupMessages({library(survey); library(jsonlite)})
df <- read.csv("tests/reference_parity/_fixtures/survey_data.csv")
des <- svydesign(ids = ~1, weights = ~w, data = df)
m <- svymean(~y, des); t <- svytotal(~y, des)
out <- list(
  svymean = list(estimate = unname(coef(m)), se = unname(SE(m))),
  svytotal = list(estimate = unname(coef(t)), se = unname(SE(t))),
  provenance = list(r_version = R.version.string,
                    survey_version = as.character(packageVersion("survey")),
                    generated_by = "tests/reference_parity/_generate_survey_R.R")
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = 16, pretty = TRUE),
           "tests/reference_parity/_fixtures/survey_R.json")
cat("wrote survey_R.json\n")
