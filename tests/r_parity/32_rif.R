# StatsPAI RIF/UQR decomposition parity (R side) -- Module 32.
#
# Reads data/32_rif.csv and runs the RIF approach manually using
# dineq::rif (compute the recentered influence function for the
# median) followed by an OLS-style decomposition.
# Tolerance: rel < 1e-2 on the total gap.
#
# Note: dineq's `rifr` returns the RIF regression coefficients only
# (no built-in decomposition wrapper). We compute the OLS-style
# decomposition by hand to match sp's output structure.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(dineq)
})

MODULE <- "32_rif"

df <- read_csv_strict(MODULE)

# Compute RIF for each group's outcome at the median.
group0 <- df[df$female == 0, ]
group1 <- df[df$female == 1, ]

rif0 <- dineq::rif(group0$log_wage, weights = NULL, method = "quantile",
                    quantile = 0.5)
rif1 <- dineq::rif(group1$log_wage, weights = NULL, method = "quantile",
                    quantile = 0.5)

# OLS regression of RIF on covariates within each group.
fit0 <- lm(rif0 ~ educ + exper, data = group0)
fit1 <- lm(rif1 ~ educ + exper, data = group1)

# Means of covariates per group.
X0 <- as.matrix(cbind(1, group0[, c("educ", "exper")]))
X1 <- as.matrix(cbind(1, group1[, c("educ", "exper")]))
colnames(X0)[1] <- colnames(X1)[1] <- "(Intercept)"

mean_X0 <- colMeans(X0)
mean_X1 <- colMeans(X1)

# Coefficients (Group 0 = reference).
beta0 <- coef(fit0)
beta1 <- coef(fit1)

# Total gap = sp's group-1 minus group-0 means of RIF.
total_diff <- mean(rif1) - mean(rif0)

# Decomposition: explained (composition) = (X1 - X0) * beta0;
# unexplained (structure) = X1 * (beta1 - beta0).
explained <- sum((mean_X1 - mean_X0) * beta0)
unexplained <- total_diff - explained

# Per-variable explained contributions
detailed_var <- (mean_X1 - mean_X0) * beta0

rows <- list(
  parity_row(MODULE, "total_diff",  estimate = total_diff,  n = nrow(df)),
  parity_row(MODULE, "explained",   estimate = explained,   n = nrow(df)),
  parity_row(MODULE, "unexplained", estimate = unexplained, n = nrow(df))
)

for (v in c("(Intercept)", "educ", "exper")) {
  vname <- if (v == "(Intercept)") "Intercept" else v
  rows[[length(rows) + 1L]] <- parity_row(
    MODULE, paste0("explained_", vname),
    estimate = unname(detailed_var[v]), n = nrow(df))
}

write_results(MODULE, rows,
              extra = list(statistic = "median",
                           method = "manual RIF + OLS decomposition"))
