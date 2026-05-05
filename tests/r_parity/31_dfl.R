# StatsPAI DFL reweighting parity (R side) -- Module 31.
#
# Reads data/31_dfl.csv and runs ddecompose::dfl_decompose.
# Tolerance: rel < 1e-2 on the gap, composition, and structure
# (propensity-score logit + reweighting; both packages share the
# DiNardo-Fortin-Lemieux 1996 algorithm).

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(ddecompose)
})

MODULE <- "31_dfl"

df <- read_csv_strict(MODULE)

# ddecompose's dfl_decompose: formula uses `outcome ~ covariates | group`
# syntax via Formula package. Group is passed as a bare symbol.
df$female <- as.factor(df$female)

fit <- ddecompose::dfl_decompose(
  formula = log_wage ~ educ + exper,
  data = df,
  group = female,
  reference_0 = TRUE,
  bootstrap = FALSE,
  statistics = c("mean")
)

# fit$decomposition_other_statistics is a data.frame with rows
# per statistic and columns: 'Observed difference', 'Composition
# effect', 'Structure effect'. ddecompose returns the (group=0) -
# (group=1) gap when reference_0=TRUE; sp returns the
# (group=0_mean) - (group=1_mean) gap with the same sign convention.
md <- fit$decomposition_other_statistics
gap_est   <- md[md$statistic == "mean", "Observed difference"][1]
comp_est  <- md[md$statistic == "mean", "Composition effect"][1]
struc_est <- md[md$statistic == "mean", "Structure effect"][1]

# Note: ddecompose's "Observed difference" is reference_minus_other,
# which equals -(sp's gap) when reference_0=TRUE because sp
# computes (group_b_mean - group_a_mean) by convention. Flip sign
# for direct comparison.
gap_est   <- -gap_est
comp_est  <- -comp_est
struc_est <- -struc_est

rows <- list(
  parity_row(MODULE, "gap",         estimate = gap_est,    n = nrow(df)),
  parity_row(MODULE, "composition", estimate = comp_est,   n = nrow(df)),
  parity_row(MODULE, "structure",   estimate = struc_est,  n = nrow(df))
)

write_results(MODULE, rows,
              extra = list(method = "ddecompose::dfl_decompose",
                           reference_0 = TRUE,
                           statistic = "mean",
                           sign_flipped = TRUE))
