# Reference values for test_rosenbaum_bounds_match_senwilcox in
# tests/reference_parity/test_the_effect_stata_parity.py.
# DOS2 0.5.2 (Rosenbaum's companion package to Design of Observational
# Studies, 2nd ed.). rb_diff.csv is written by
# _generate_the_effect_Stata.do: the 185 matched-pair differences of re78.
d <- read.csv("rb_diff.csv")$diff
for (g in c(1, 1.25, 1.5, 1.75, 2)) {
  two <- DOS2::senWilcox(d, gamma = g, conf.int = TRUE, alpha = 0.05,
                         alternative = "twosided")
  one <- DOS2::senWilcox(d, gamma = g)
  cat(sprintf("REF %.2f %.14g %.6f %.6f\n", g, one$pval, two$ci[1], two$ci[2]))
}
