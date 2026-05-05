# StatsPAI classical SCM parity (R side) -- Module 07.
#
# Reads data/07_scm.csv (the StatsPAI Basque replica) and runs
# Synth::synth with the pre-treatment outcome levels as the only
# predictors. Tolerance: rel < 1e-3 on the post-treatment average
# gap.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(Synth)
})

MODULE <- "07_scm"

df <- read_csv_strict(MODULE)

# Synth::dataprep needs numeric unit IDs.
df$unit_num <- as.integer(as.factor(df$region))
units <- unique(df[, c("region", "unit_num")])
units <- units[order(units$unit_num), ]
treated_id <- units$unit_num[units$region == "Basque Country"]
controls   <- setdiff(units$unit_num, treated_id)

# Use every pre-treatment year (1955..1969) as a "special predictor"
# of the outcome -- the standard ADH (2010) outcomes-only spec.
pre_years  <- 1955:1969
post_years <- 1970:1997
special_preds <- lapply(pre_years, function(yr) {
  list("gdppc", yr, "mean")
})

dp <- Synth::dataprep(
  foo = df,
  predictors = NULL,
  predictors.op = "mean",
  dependent = "gdppc",
  unit.variable = "unit_num",
  time.variable = "year",
  special.predictors = special_preds,
  treatment.identifier = treated_id,
  controls.identifier  = controls,
  time.predictors.prior = pre_years,
  time.optimize.ssr     = pre_years,
  time.plot             = c(pre_years, post_years),
  unit.names.variable   = "region"
)

set.seed(PARITY_SEED)
sy <- Synth::synth(data.prep.obj = dp, optimxmethod = "BFGS", verbose = FALSE)

w_unit <- as.numeric(sy$solution.w)
names(w_unit) <- units$region[match(rownames(dp$X1), units$unit_num)]

# Synth orders the donor matrix by ascending unit_num; the rownames
# of dp$Y0plot align with that ordering.
donor_names <- units$region[units$unit_num %in% controls]
names(w_unit) <- donor_names

# Synthetic counterfactual for the treated unit and the post-period gap.
Y0_synth <- dp$Y0plot %*% sy$solution.w
Y1_treat <- dp$Y1plot
gap <- Y1_treat - Y0_synth
post_idx <- which(rownames(Y1_treat) %in% as.character(post_years))
avg_post_gap <- mean(gap[post_idx])
pre_idx  <- which(rownames(Y1_treat) %in% as.character(pre_years))
pre_rmse <- sqrt(mean(gap[pre_idx]^2))

rows <- list(
  parity_row(
    module = MODULE, statistic = "avg_post_gap",
    estimate = avg_post_gap, n = nrow(df)
  ),
  parity_row(
    module = MODULE, statistic = "pre_treatment_rmse",
    estimate = pre_rmse, n = nrow(df)
  )
)

# Donor weights.
for (nm in sort(donor_names)) {
  rows[[length(rows) + 1L]] <- parity_row(
    module    = MODULE,
    statistic = paste0("weight_", nm),
    estimate  = unname(w_unit[nm]),
    n         = nrow(df)
  )
}

write_results(MODULE, rows,
              extra = list(method = "classic",
                           treatment_time = 1970,
                           treated_unit = "Basque Country"))
