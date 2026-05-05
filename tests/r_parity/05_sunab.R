# StatsPAI Sun-Abraham parity (R side) -- Module 05.
#
# Reads data/05_sunab.csv (the StatsPAI mpdta replica) and runs
# fixest::feols(lemp ~ sunab(first_treat, year) | countyreal + year).
# Tolerance: rel < 1e-3 (iterative).

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(fixest)
})

MODULE <- "05_sunab"

df <- read_csv_strict(MODULE)
df$first_treat <- as.numeric(df$first_treat)

fit <- fixest::feols(
  lemp ~ sunab(first_treat, year) | countyreal + year,
  data = df,
  cluster = ~ countyreal
)

# Aggregated weighted-average ATT (post-treatment, all cohorts).
agg <- summary(fit, agg = "att")
agg_co <- coef(agg)
agg_se <- sqrt(diag(vcov(agg)))

# Locate the aggregated post-treatment row.
post_name <- grep("^ATT$|post|Post|^after$", names(agg_co), value = TRUE)
if (length(post_name) == 0) {
  # fixest names the aggregated row variously across versions; fall
  # back to the first coefficient.
  post_name <- names(agg_co)[1]
}

rows <- list(
  parity_row(
    module    = MODULE,
    statistic = "weighted_avg_ATT",
    estimate  = unname(agg_co[post_name]),
    se        = unname(agg_se[post_name]),
    n         = nrow(df)
  )
)

# Event-study coefficients: extract from the disaggregated summary.
es <- summary(fit)
co <- coef(es)
se <- sqrt(diag(vcov(es)))
# fixest's sunab() names the coefficients like "year::-4:cohort::2004"
# OR returns a single column per relative time (e.g. when use the
# "att" agg the names go away). To match per-rel-time, we re-run
# without the agg.
es2 <- summary(fit, agg = FALSE)
co2 <- coef(es2)
se2 <- sqrt(diag(vcov(es2)))
# Each name has form "year::<rel>:cohort::<g>" -- aggregate over
# cohorts using sunab's weighted aggregation by hand to mirror
# StatsPAI's per-rel-time output.
parse_rel <- function(nm) {
  m <- regmatches(nm, regexec("year::(-?\\d+):cohort::(\\d+)", nm))
  if (length(m[[1]]) != 3L) return(NA_integer_)
  as.integer(m[[1]][2])
}
rels <- sapply(names(co2), parse_rel)
keep <- !is.na(rels) & rels != -1L
co2 <- co2[keep]; se2 <- se2[keep]; rels <- rels[keep]

# Cohort-share weighted average per relative time.
# Get cohort counts from the data.
treated <- subset(df, first_treat > 0)
cohort_n <- table(treated$first_treat) / max(treated$first_treat - treated$first_treat + 1)
cohort_w <- cohort_n / sum(cohort_n)

agg_per_rel <- function(rt) {
  m <- regmatches(names(co2), regexec("year::(-?\\d+):cohort::(\\d+)", names(co2)))
  ms <- do.call(rbind, lapply(m, function(x) if (length(x) == 3L) c(x[2], x[3]) else c(NA, NA)))
  rel_match <- as.integer(ms[, 1]) == rt & !is.na(ms[, 1])
  if (!any(rel_match)) return(c(NA, NA))
  cohorts <- as.character(as.integer(ms[rel_match, 2]))
  est <- co2[rel_match]
  s   <- se2[rel_match]
  w   <- cohort_w[cohorts]
  if (any(is.na(w))) {
    return(c(weighted_average_unequal_weights = NA, NA))
  }
  w <- w / sum(w)
  c(sum(w * est), sqrt(sum(w^2 * s^2)))
}

for (rt in sort(unique(rels))) {
  out <- agg_per_rel(rt)
  rows[[length(rows) + 1L]] <- parity_row(
    module    = MODULE,
    statistic = paste0("att_rel_", rt),
    estimate  = out[1],
    se        = out[2],
    n         = nrow(df)
  )
}

write_results(MODULE, rows, extra = list(cluster = "countyreal"))
