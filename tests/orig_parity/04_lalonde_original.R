# Original-data parity for MatchIt::lalonde (Dehejia-Wahba 1999
# NSW + PSID-1 sample). Published values: naive OLS = -$8,498;
# PSM ATT = $1,794.

.script_dir <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m) > 0) dirname(normalizePath(sub("^--file=", "", m[1])))
  else getwd()
})()
DATA_DIR <- file.path(.script_dir, "data")
RESULTS_DIR <- file.path(.script_dir, "results")
dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(MatchIt)
  library(jsonlite)
})

data("lalonde", package = "MatchIt")

# Coerce factor `race` to dummies for sp consumption.
lalonde$black    <- as.integer(lalonde$race == "black")
lalonde$hispanic <- as.integer(lalonde$race == "hispan")
write.csv(lalonde, file.path(DATA_DIR, "04_lalonde_original.csv"),
          row.names = FALSE)

# Naive OLS on re78.
naive <- lm(re78 ~ treat, data = lalonde)
naive_se <- sqrt(diag(sandwich::vcovHC(naive, type = "HC1")))

# Covariate-adjusted OLS.
adj <- lm(re78 ~ treat + age + educ + black + hispanic +
            married + nodegree + re74 + re75, data = lalonde)
adj_se <- sqrt(diag(sandwich::vcovHC(adj, type = "HC1")))

# 1:1 NN PSM with replacement.
set.seed(42)
m <- MatchIt::matchit(
  treat ~ age + educ + black + hispanic + married + nodegree + re74 + re75,
  data = lalonde, method = "nearest", distance = "glm",
  link = "logit", replace = TRUE, ratio = 1L, estimand = "ATT"
)
md <- MatchIt::match.data(m, data = lalonde)
treated_y <- md$re78[md$treat == 1]
control_y <- md$re78[md$treat == 0]
control_w <- md$weights[md$treat == 0]
psm_att <- mean(treated_y) - sum(control_y * control_w) / sum(control_w)

build_row <- function(stat, est, se, n, published, citation) {
  list(module = jsonlite::unbox("04_lalonde_original"),
       side = jsonlite::unbox("R"),
       statistic = jsonlite::unbox(stat),
       estimate = jsonlite::unbox(est),
       se = jsonlite::unbox(se),
       n = jsonlite::unbox(n),
       published = jsonlite::unbox(published),
       citation = jsonlite::unbox(citation),
       extra = list())
}

rows <- list(
  build_row("naive_ols_att",
            unname(coef(naive)["treat"]),
            unname(naive_se["treat"]),
            nrow(lalonde),
            -8498,
            "Dehejia-Wahba (1999) Table 3, naive OLS"),
  build_row("adj_ols_att",
            unname(coef(adj)["treat"]),
            unname(adj_se["treat"]),
            nrow(lalonde),
            218,
            "Dehejia-Wahba (1999) Table 3, covariate-adjusted OLS"),
  build_row("psm_att",
            psm_att, NA_real_, nrow(lalonde),
            1794,
            "Dehejia-Wahba (1999) Table 4, PSM 1:1 NN")
)

payload <- list(
  module = jsonlite::unbox("04_lalonde_original"),
  side = jsonlite::unbox("R"),
  rows = rows,
  extra = list(data_source = jsonlite::unbox("MatchIt::lalonde"),
               n_obs = jsonlite::unbox(nrow(lalonde)))
)
writeLines(
  jsonlite::toJSON(payload, pretty = TRUE, na = "null", null = "null",
                    digits = NA),
  file.path(RESULTS_DIR, "04_lalonde_original_R.json")
)
message("OK -- wrote 04_lalonde_original_R.json")
