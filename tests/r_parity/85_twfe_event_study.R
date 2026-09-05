# StatsPAI dynamic TWFE event study (R side) -- Module 85.
#
# Reference: fixest::feols with an i(rel, ref = -1) interaction and
# two-way fixed effects, clustered by unit -- the specification
# sp.event_study documents and implements.
#
# Never-treated units get no relative time and contribute only through
# the fixed effects, which is what "control" means in this design; fixest
# handles that by giving them no interaction dummy.
#
# Tolerance: rel < 1e-9 on estimate and SE.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages(library(fixest))

MODULE <- "85_twfe_event_study"
WINDOW <- c(-4, 4)

df <- read_csv_strict(MODULE)
df$rel <- ifelse(df$g > 0, df$time - df$g, NA_integer_)
df$treat <- as.integer(df$g > 0)
# Never-treated units must stay IN the sample as pure controls. Leaving
# their relative time NA does not do that: i(rel, ref=-1) drops every row
# with an NA on the right-hand side, so fixest silently estimates on the
# treated units alone -- 810 observations removed here, and coefficients
# that look like a large numerical disagreement rather than a different
# sample. Parking them at the reference level and interacting with the
# treated indicator keeps them in with no dummy of their own.
df$rel_f <- ifelse(is.na(df$rel), -1L, df$rel)

# fixest's default ssc (adj = TRUE, fixef.K = "nested", cluster.adj = TRUE)
# counts the absorbed time effects, which are not nested in the unit
# cluster, in K; reghdfe does the same. sp.event_study follows the same
# nested rule since 1.24.0, so all three sides agree at machine level
# with default settings (before, sp.event_study counted only the eight
# event-time coefficients and the R side had to pass fixef.K = "none").
fit <- fixest::feols(
  y ~ i(rel_f, treat, ref = -1) | unit + time,
  data = df, cluster = ~unit
)
ct <- summary(fit)$coeftable

rows <- list()
for (nm in rownames(ct)) {
  k <- suppressWarnings(as.integer(sub("^.*::(-?[0-9]+).*$", "\\1", nm)))
  if (is.na(k)) next
  label <- if (k >= 0) sprintf("+%d", k) else as.character(k)
  rows[[length(rows) + 1L]] <- parity_row(
    module    = MODULE,
    statistic = sprintf("es_%s", label),
    estimate  = unname(ct[nm, "Estimate"]),
    se        = unname(ct[nm, "Std. Error"])
  )
}

write_results(MODULE, rows,
              extra = list(reference = "fixest::feols i(rel, ref=-1) | unit + time",
                           cluster = "unit",
                           ref_period = -1,
                           window_covers_all_relative_times = TRUE,
                           ssc = "fixest default (adj=TRUE, fixef.K=nested, cluster.adj=TRUE); K = 8 + 9 time effects = 17",
                           fixest_version = as.character(packageVersion("fixest"))))
