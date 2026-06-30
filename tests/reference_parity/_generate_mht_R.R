#!/usr/bin/env Rscript
# Frozen reference for sp.bonferroni / sp.holm / sp.benjamini_hochberg /
# sp.adjust_pvalues vs base R stats::p.adjust.
# Regenerate: Rscript tests/reference_parity/_generate_mht_R.R
suppressPackageStartupMessages(library(jsonlite))
p <- c(0.001, 0.008, 0.012, 0.012, 0.039, 0.041, 0.21, 0.5, 0.73, 0.99)
out <- list(
  pvalues = p,
  bonferroni = p.adjust(p, "bonferroni"),
  holm = p.adjust(p, "holm"),
  BH = p.adjust(p, "BH"),
  provenance = list(r_version = R.version.string,
                    generated_by = "tests/reference_parity/_generate_mht_R.R")
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = 16, pretty = TRUE),
           "tests/reference_parity/_fixtures/mht_R.json")
cat("wrote mht_R.json\n")
