# Critical values of the stacked DiD regression in R fixest, the reference
# for sp.stacked_did(crit_dist = "t"). Run as a black box (fixest is GPL-3:
# only its outputs are used, never its source). From the repo root:
#   Rscript tests/reference_parity/_fixtures/_generate_stacked_crit_R.R
#
# The stack is the hand-written one of Track A module 75
# (tests/r_parity/75_stacked.R) on that module's data: one sub-experiment
# per treated cohort with never-treated controls over the window [-3, 3],
# unit x cohort and year x cohort effects, k = -1 as reference, clustered
# on the unit. fixest refers the t statistics to t(G - 1), G = clusters.
suppressMessages({
  library(fixest)
  library(data.table)
})
df <- as.data.table(read.csv("tests/r_parity/data/75_stacked.csv"))
cohorts <- sort(unique(df[first_treat > 0, first_treat]))
never <- unique(df[first_treat == 0, id])
frames <- list()
for (g in cohorts) {
  coh <- unique(df[first_treat == g, id])
  sub <- df[id %in% union(coh, never) & year >= g - 3 & year <= g + 3]
  sub[, `:=`(cohort = g, rel = year - g, treated_unit = as.integer(id %in% coh))]
  frames[[length(frames) + 1]] <- sub
}
st <- rbindlist(frames)
st[, `:=`(uc = paste0(id, "_", cohort), tc = paste0(year, "_", cohort))]
st[, rel_f := factor(rel, levels = sort(unique(rel)))]
m <- feols(y ~ i(rel_f, treated_unit, ref = "-1") | uc + tc, data = st,
           cluster = ~id)
dump <- function(m, tag, level) {
  ct <- coeftable(m)
  ci <- confint(m, level = level)
  data.frame(
    tag = tag,
    rel = as.numeric(sub(".*rel_f::(-?[0-9]+):.*", "\\1", rownames(ct))),
    b = ct[, 1], se = ct[, 2], pvalue = ct[, 4], ll = ci[, 1], ul = ci[, 2],
    df = degrees_freedom(m, "t"), level = level
  )
}
res <- rbind(dump(m, "never_95", 0.95), dump(m, "never_90", 0.90))
res$fixest <- as.character(packageVersion("fixest"))
write.csv(format(res, digits = 17),
          "tests/reference_parity/_fixtures/stacked_crit_R.csv",
          row.names = FALSE, quote = FALSE)
