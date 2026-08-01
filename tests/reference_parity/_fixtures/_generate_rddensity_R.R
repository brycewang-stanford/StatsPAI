# Reference values for sp.rddensity vs R rddensity 2.6 (Cattaneo, Jansson &
# Ma). Uses rdrobust's own senate data plus two synthetic designs: one with
# NO manipulation and one with a deliberate discontinuity in the density,
# so the fixture can tell a working test from one that always fails to
# reject.
library(rddensity); library(jsonlite)
set.seed(20260802)

d <- read.csv("rdsenate.csv")

n <- 4000
x_null <- rnorm(n, 0, 1)                       # smooth density at 0
# heaping: drop 45% of the mass just left of the cutoff
xm <- rnorm(n, 0, 1)
keep <- !(xm > -0.35 & xm < 0 & runif(n) < 0.45)
x_manip <- xm[keep]

grab <- function(r) list(
  T_jk = unname(r$test$t_jk), p_jk = unname(r$test$p_jk),
  T_q  = unname(r$test$t_q),  p_q  = unname(r$test$p_q),
  hl = unname(r$h$left), hr = unname(r$h$right),
  fl = unname(r$hat$left), fr = unname(r$hat$right),
  sel = unname(r$hat$diff), Nl = unname(r$N$eff_left), Nr = unname(r$N$eff_right)
)

out <- list()
for (p in c(2, 3)) {
  out[[sprintf("senate_p%d", p)]]  <- grab(rddensity(X = d$margin, c = 0, p = p))
  out[[sprintf("null_p%d", p)]]    <- grab(rddensity(X = x_null, c = 0, p = p))
  out[[sprintf("manip_p%d", p)]]   <- grab(rddensity(X = x_manip, c = 0, p = p))
}
out[["senate_unrestricted"]] <- grab(
  rddensity(X = d$margin, c = 0, p = 2, fitselect = "unrestricted"))

write.csv(data.frame(x = x_null), "rddensity_null.csv", row.names = FALSE)
write.csv(data.frame(x = x_manip), "rddensity_manip.csv", row.names = FALSE)
out[["_meta"]] <- list(n_null = length(x_null), n_manip = length(x_manip),
                       n_senate = nrow(d),
                       rddensity_version = as.character(packageVersion("rddensity")))
write_json(out, "rddensity_R.json", auto_unbox = TRUE, digits = 15, pretty = TRUE)
cat("wrote rddensity_R.json\n")
