# StatsPAI Blinder-Oaxaca parity (R side) -- Module 30.
#
# Reads data/30_oaxaca.csv and runs oaxaca::oaxaca with the
# threefold decomposition. Tolerance: rel < 1e-3 on the gap and
# explained/unexplained components.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(oaxaca)
})

MODULE <- "30_oaxaca"

df <- read_csv_strict(MODULE)

# oaxaca::oaxaca expects formula: y ~ x1 + x2 | group
# Ensure female is integer/logical so the package treats it as the
# group indicator.
df$female <- as.integer(df$female)

set.seed(PARITY_SEED)
fit <- oaxaca::oaxaca(
  formula = log_wage ~ educ + exper | female,
  data = df,
  R = 100  # bootstrap replications
)

# `fit$y` has overall y means: $y$y.A = group 0, $y$y.B = group 1.
# `fit$threefold$overall` has the gap decomposition.
y_a <- fit$y$y.A
y_b <- fit$y$y.B
gap <- y_a - y_b

three <- fit$threefold$overall
# Coefficients: fit$reg$reg.A$coefficients (group A = female = 1
# per oaxaca::oaxaca convention -- check $y to confirm).
# We follow the package's labelling A/B as documented.
reg_a <- fit$reg$reg.A$coefficients
reg_b <- fit$reg$reg.B$coefficients

# threefold$overall column names use 'coef(endowments)' etc.
explained <- unname(three["coef(endowments)"])
unexplained <- unname(three["coef(coefficients)"])
interaction <- unname(three["coef(interaction)"])
explained_se <- unname(three["se(endowments)"])
unexplained_se <- unname(three["se(coefficients)"])

rows <- list(
  parity_row(MODULE, "gap", estimate = gap, n = nrow(df)),
  parity_row(MODULE, "explained",
             estimate = explained, se = explained_se, n = nrow(df)),
  parity_row(MODULE, "unexplained",
             estimate = unexplained, se = unexplained_se, n = nrow(df)),
  parity_row(MODULE, "mean_y_a", estimate = y_a, n = nrow(df)),
  parity_row(MODULE, "mean_y_b", estimate = y_b, n = nrow(df)),
  # oaxaca::oaxaca has reg.A = group A (the "reference" group) which
  # sp's group_stats labels beta_a.
  parity_row(MODULE, "beta_a_educ",
             estimate = unname(reg_a["educ"]), n = nrow(df)),
  parity_row(MODULE, "beta_b_educ",
             estimate = unname(reg_b["educ"]), n = nrow(df)),
  parity_row(MODULE, "beta_a_exper",
             estimate = unname(reg_a["exper"]), n = nrow(df)),
  parity_row(MODULE, "beta_b_exper",
             estimate = unname(reg_b["exper"]), n = nrow(df))
)

# Detailed (variable-level) endowments contributions. The column
# names are 'coef(endowments)', 'se(endowments)' rather than the
# overall-block 'endowments', 'endowments SE'.
detail <- fit$threefold$variables
if (!is.null(detail)) {
  for (v in rownames(detail)) {
    if (v == "(Intercept)") next
    contribution <- detail[v, "coef(endowments)"]
    se <- detail[v, "se(endowments)"]
    rows[[length(rows) + 1L]] <- parity_row(
      MODULE, paste0("explained_", v),
      estimate = contribution, se = se, n = nrow(df))
  }
}

write_results(MODULE, rows,
              extra = list(method = "threefold",
                           reference = "fit$reg$reg.A"))
