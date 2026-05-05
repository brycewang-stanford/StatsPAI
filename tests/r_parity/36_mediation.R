# StatsPAI causal mediation parity (R side) -- Module 36.
#
# Reads data/36_mediation.csv and runs mediation::mediate (Imai,
# Keele & Tingley 2010). Tolerance: rel < 5e-2 on ACME/ADE/total
# (Monte Carlo from bootstrap is the dominant noise source).

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(mediation)
})

MODULE <- "36_mediation"

df <- read_csv_strict(MODULE)

set.seed(PARITY_SEED)

# Stage 1: mediator model.
fit_m <- lm(m ~ treat, data = df)
# Stage 2: outcome model.
fit_y <- lm(y ~ treat + m, data = df)

# Mediation analysis with 200 bootstrap reps (matches sp's default
# behaviour; conservative lower bound).
med <- mediation::mediate(
  fit_m, fit_y,
  treat = "treat",
  mediator = "m",
  boot = TRUE,
  sims = 200L
)

# mediate returns named scalars: d0/d1 = ACME, z0/z1 = ADE, tau =
# total, n0/n1 = proportion mediated. Average over treatment levels.
acme <- (med$d0 + med$d1) / 2
ade  <- (med$z0 + med$z1) / 2
total <- med$tau.coef
prop  <- (med$n0 + med$n1) / 2

# Bootstrap SEs.
se_acme <- (med$d0.ci[2] - med$d0.ci[1]) / (2 * 1.96)
se_ade  <- (med$z0.ci[2] - med$z0.ci[1]) / (2 * 1.96)

rows <- list(
  parity_row(MODULE, "acme",
             estimate = acme, se = se_acme, n = nrow(df)),
  parity_row(MODULE, "ade",
             estimate = ade, se = se_ade, n = nrow(df)),
  parity_row(MODULE, "total_effect",
             estimate = total, n = nrow(df)),
  parity_row(MODULE, "prop_mediated",
             estimate = prop, n = nrow(df))
)

write_results(MODULE, rows,
              extra = list(method = "mediation::mediate",
                           sims = 200L))
