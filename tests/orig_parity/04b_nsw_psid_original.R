# Original-data parity for the *true* NSW+PSID-1 sample
# (causalsens::lalonde.psid, 2675 obs, 185 treated). Replaces the
# original module 04 which used MatchIt::lalonde (614 obs, not the
# Dehejia-Wahba PSID-1 sample). Published values:
#   naive OLS on re78           = -$15,205 (DW Table 3, NSW vs PSID-1)
#   covariate-adjusted OLS      = approximately $700 (DW Table 3)
#   PSM ATT                     = approximately $1,690 (DW Table 4)

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
  library(causalsens)
  library(MatchIt)
  library(jsonlite)
})

data("lalonde.psid", package = "causalsens")
df <- lalonde.psid

# Save canonical CSV.
write.csv(df, file.path(DATA_DIR, "04b_nsw_psid_original.csv"),
          row.names = FALSE)

# Naive OLS.
naive <- lm(re78 ~ treat, data = df)
naive_se <- sqrt(diag(sandwich::vcovHC(naive, type = "HC1")))

# Covariate-adjusted OLS.
adj <- lm(re78 ~ treat + age + education + black + hispanic +
            married + nodegree + re74 + re75, data = df)
adj_se <- sqrt(diag(sandwich::vcovHC(adj, type = "HC1")))

# 1:1 NN PSM with replacement.
set.seed(42)
m <- MatchIt::matchit(
  treat ~ age + education + black + hispanic + married + nodegree + re74 + re75,
  data = df, method = "nearest", distance = "glm",
  link = "logit", replace = TRUE, ratio = 1L, estimand = "ATT"
)
md <- MatchIt::match.data(m, data = df)
treated_y <- md$re78[md$treat == 1]
control_y <- md$re78[md$treat == 0]
control_w <- md$weights[md$treat == 0]
psm_att <- mean(treated_y) - sum(control_y * control_w) / sum(control_w)

build_row <- function(stat, est, se, n, published, citation) {
  list(module = jsonlite::unbox("04b_nsw_psid_original"),
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
  build_row("naive_ols_att", unname(coef(naive)["treat"]),
            unname(naive_se["treat"]), nrow(df), -15205,
            "Dehejia-Wahba (1999) naive OLS on NSW+PSID-1"),
  build_row("adj_ols_att", unname(coef(adj)["treat"]),
            unname(adj_se["treat"]), nrow(df), 700,
            "Dehejia-Wahba (1999) covariate-adjusted OLS on NSW+PSID-1"),
  build_row("psm_att", psm_att, NA_real_, nrow(df), 1690,
            "Dehejia-Wahba (1999) PSM 1:1 NN on NSW+PSID-1")
)
payload <- list(
  module = jsonlite::unbox("04b_nsw_psid_original"),
  side = jsonlite::unbox("R"), rows = rows,
  extra = list(data_source = jsonlite::unbox("causalsens::lalonde.psid"),
               n_obs = jsonlite::unbox(nrow(df)),
               n_treated = jsonlite::unbox(sum(df$treat == 1)),
               n_control = jsonlite::unbox(sum(df$treat == 0)))
)
writeLines(
  jsonlite::toJSON(payload, pretty = TRUE, na = "null", null = "null", digits = NA),
  file.path(RESULTS_DIR, "04b_nsw_psid_original_R.json")
)
message("OK -- wrote 04b_nsw_psid_original_R.json")
