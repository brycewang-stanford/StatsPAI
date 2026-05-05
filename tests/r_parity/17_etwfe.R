# StatsPAI Wooldridge ETWFE parity (R side) -- Module 17.
#
# Reads data/17_etwfe.csv (mpdta replica) and runs etwfe::etwfe.
# Tolerance: rel < 1e-3 on the pooled ATT.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(etwfe)
})

MODULE <- "17_etwfe"

df <- read_csv_strict(MODULE)
df$first_treat <- as.numeric(df$first_treat)
df$year <- as.integer(df$year)

# etwfe::etwfe API: yvar / tvar / gvar (cohort) / data
fit <- etwfe::etwfe(
  fml  = lemp ~ 0,
  tvar = year,
  gvar = first_treat,
  data = df,
  vcov = ~ countyreal
)

# Aggregate to a single pooled ATT.
agg <- etwfe::emfx(fit, type = "simple")

# emfx returns a marginaleffects-style data.frame.
rows <- list(
  parity_row(
    module    = MODULE,
    statistic = "att_etwfe",
    estimate  = agg$estimate[1],
    se        = agg$std.error[1],
    ci_lo     = agg$conf.low[1],
    ci_hi     = agg$conf.high[1],
    n         = nrow(df)
  )
)

write_results(MODULE, rows, extra = list(method = "etwfe::etwfe"))
