# Critical values of regression event studies in R fixest (parity='r' of
# sp.event_study / sp.sun_abraham / sp.did_2x2). Run as a black box (fixest
# is GPL-3: only its outputs are used, never its source). From the repo root:
#   Rscript tests/reference_parity/_fixtures/_generate_es_crit_R.R
suppressMessages(library(fixest))
out <- list()
dump <- function(m, tag, level = 0.95) {
  ct <- coeftable(m); ci <- confint(m, level = level)
  data.frame(tag = tag, term = sub("^year::", "", rownames(ct)),
             b = ct[, 1], se = ct[, 2], pvalue = ct[, 4], ll = ci[, 1], ul = ci[, 2],
             df = degrees_freedom(m, "t"))
}
d <- read.csv("src/statspai/datasets/data/castle_2013.csv")
d$region <- 1 * d$northeast + 2 * d$midwest + 3 * d$south + 4 * d$west
d$rel <- d$year - d$effyear
d$relb <- pmin(pmax(d$rel, -5), 4)
for (k in c(-5, -4, -3, -2, 0, 1, 2, 3, 4)) {
  nm <- if (k < 0) paste0("es_m", -k) else paste0("es_p", k)
  d[[nm]] <- as.numeric(!is.na(d$relb) & d$relb == k)
}
f <- l_homicide ~ es_m5 + es_m4 + es_m3 + es_m2 + es_p0 + es_p1 + es_p2 + es_p3 + es_p4 | sid + year
m <- feols(f, d, weights = ~popwt, cluster = ~sid)
out[[length(out) + 1]] <- dump(m, "es_castle_cl")
out[[length(out) + 1]] <- dump(m, "es_castle_cl90", 0.90)
m <- feols(f, d, weights = ~popwt, cluster = ~region)
out[[length(out) + 1]] <- dump(m, "es_castle_region")
d$gs <- ifelse(is.na(d$effyear), 10000, d$effyear)
m <- feols(l_homicide ~ sunab(gs, year) | sid + year, d, weights = ~popwt, cluster = ~sid)
out[[length(out) + 1]] <- dump(m, "sa_castle_cl")
m <- feols(l_homicide ~ sunab(gs, year) | gs + year, d, weights = ~popwt, vcov = "hetero")
out[[length(out) + 1]] <- dump(m, "sa_castle_rob")
d2 <- subset(d, effyear == 2006 | is.na(effyear))
d2$tr <- as.numeric(!is.na(d2$effyear)); d2$po <- as.numeric(d2$year >= 2006)
out[[length(out) + 1]] <- dump(feols(l_homicide ~ tr * po, d2, cluster = ~sid), "2x2_castle_cl")
out[[length(out) + 1]] <- dump(feols(l_homicide ~ tr * po, d2, vcov = "hetero"), "2x2_castle_rob")
out[[length(out) + 1]] <- dump(feols(l_homicide ~ tr * po, d2, weights = ~popwt, cluster = ~sid), "2x2_castle_clw")
b <- read.csv("tests/fixtures/divorce_no_fault_panel.csv")
fp <- tapply(ifelse(b$post == 1, b$year, NA), b$stfips, function(x) suppressWarnings(min(x, na.rm = TRUE)))
b$cohort <- fp[as.character(b$stfips)]; b$cohort[!is.finite(b$cohort)] <- 0
b <- subset(b, cohort != 1964)
b$rel <- ifelse(b$cohort > 0, b$year - b$cohort, NA); b$rel <- pmin(pmax(b$rel, -10), 15)
terms <- c()
for (k in 10:2) { nm <- paste0("lead", k); b[[nm]] <- as.numeric(!is.na(b$rel) & b$rel == -k); terms <- c(terms, nm) }
for (k in 0:15) { nm <- paste0("lag", k); b[[nm]] <- as.numeric(!is.na(b$rel) & b$rel == k); terms <- c(terms, nm) }
mb <- feols(as.formula(paste("asmrs ~", paste(terms, collapse = "+"), "| stfips + year")), b, cluster = ~stfips)
out[[length(out) + 1]] <- dump(mb, "es_divorce_cl")
res <- do.call(rbind, out)
res$fixest <- as.character(packageVersion("fixest"))
write.csv(format(res, digits = 17), "tests/reference_parity/_fixtures/es_crit_R.csv", row.names = FALSE, quote = FALSE)
