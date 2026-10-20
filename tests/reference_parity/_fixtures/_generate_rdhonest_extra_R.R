# Reference values for sp.rd_honest / sp.rd_bias_aware_fuzzy branches that the
# older RDHonest fixtures do not exercise: selected bandwidths with the
# uniform and Epanechnikov kernels, the OCI criterion, the Taylor class, and
# the fuzzy default bandwidth with clusters. R RDHonest (GPL-3) is run as a
# black box on committed CSVs; nothing of its source is used.
# Run from this directory:  Rscript _generate_rdhonest_extra_R.R
suppressMessages({library(RDHonest); library(jsonlite)})
cur <- read.csv("rdhonest_curved.csv")
lee <- read.csv("rdhonest_lee08.csv")
frd <- read.csv("rd_iv_frd.csv")
grab <- function(r) {
  co <- r$coefficients
  keep <- intersect(c("estimate", "std.error", "maximum.bias", "conf.low", "conf.high",
                      "bandwidth", "M", "M.rf", "M.fs", "first.stage"), names(co))
  lapply(as.list(co[1, keep]), as.numeric)
}
out <- list()
for (k in c("uniform", "epanechnikov")) for (crit in c("MSE", "FLCI")) {
  out[[sprintf("curved_%s_%s", k, crit)]] <- grab(RDHonest(y ~ x, data = cur, M = 3, kern = k, opt.criterion = crit))
  out[[sprintf("lee08_%s_%s", k, crit)]] <- grab(RDHonest(y ~ x, data = lee, M = 0.02, kern = k, opt.criterion = crit))
}
out$curved_OCI <- grab(RDHonest(y ~ x, data = cur, M = 3, opt.criterion = "OCI"))
out$curved_taylor <- grab(RDHonest(y ~ x, data = cur, M = 3, sclass = "T"))
out$lee08_alpha10_FLCI <- grab(RDHonest(y ~ x, data = lee, M = 0.02, alpha = 0.1, opt.criterion = "FLCI"))
out$frd_cluster_default <- grab(suppressMessages(RDHonest(y | d ~ x, data = frd, clusterid = g, se.method = "EHW")))
out$frd_epanechnikov_default <- grab(suppressMessages(RDHonest(y | d ~ x, data = frd, kern = "epanechnikov")))
out$frd_uniform_default <- grab(suppressMessages(RDHonest(y | d ~ x, data = frd, kern = "uniform")))
out$`_version` <- as.character(packageVersion("RDHonest"))
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), "rdhonest_extra_R.json")
