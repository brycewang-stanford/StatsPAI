# StatsPAI PSM 1:1 NN parity (R side) -- Module 11.
#
# Reads data/11_psm.csv (NSW-DW replica) and runs MatchIt::matchit
# with logistic propensity score, 1:1 nearest-neighbour matching with
# replacement (matching the StatsPAI default). Tolerance: rel < 1e-2.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(MatchIt)
})

MODULE <- "11_psm"

df <- read_csv_strict(MODULE)

set.seed(PARITY_SEED)
m <- MatchIt::matchit(
  treat ~ age + education + black + hispanic + married + re74 + re75,
  data = df,
  method = "nearest",
  distance = "glm",        # logistic PS
  link = "logit",
  replace = TRUE,
  estimand = "ATT",
  ratio = 1L
)

# Extract matched data and compute the ATT on `re78` weighted by the
# matching weights. MatchIt's get_matches() / match.data() carry the
# weights for each treated and matched-control row.
md <- MatchIt::match.data(m, data = df)

# ATT = mean(y | treat=1) - mean weighted(y | treat=0)
treated_y <- md$re78[md$treat == 1]
control_y <- md$re78[md$treat == 0]
control_w <- md$weights[md$treat == 0]
att <- mean(treated_y) - sum(control_y * control_w) / sum(control_w)

# Abadie-Imbens analytical SE for the matching estimator.
# MatchIt::match.data does NOT compute SE; use a simple weighted
# regression to get a basic SE so the row is non-empty.
fit <- lm(re78 ~ treat, data = md, weights = md$weights)
se_basic <- summary(fit)$coefficients["treat", "Std. Error"]

rows <- list(
  parity_row(MODULE, "att_psm",   estimate = att,  se = se_basic, n = nrow(df)),
  parity_row(MODULE, "n_treated", estimate = sum(md$treat == 1L), n = nrow(df)),
  parity_row(MODULE, "n_control", estimate = sum(md$treat == 0L), n = nrow(df))
)

write_results(MODULE, rows,
              extra = list(distance = "glm-logit",
                           method = "nearest",
                           replace = TRUE,
                           ratio = 1L))
