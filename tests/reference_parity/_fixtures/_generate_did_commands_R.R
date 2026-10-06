# R `did` reference for the time-varying covariate of did_commands_data.csv
# (test_stata_did_commands_parity.py). did 2.5.1, DRDID 1.3.0 (did 2.3.0 / DRDID 1.2.3 give the same numbers).
# Run from this folder: Rscript _generate_did_commands_R.R
suppressMessages({library(did); library(jsonlite)})
d <- read.csv("did_commands_data.csv")
# did 2.3.0 writes Inf into gname for the never treated; an integer column
# turns that into NA and the never-treated units are dropped.
d$g <- as.numeric(d$g)
d$year <- as.numeric(d$year)
out <- list()
for (bp in c("varying", "universal")) for (cg in c("nevertreated", "notyettreated")) {
  r <- att_gt(yname = "y", tname = "year", idname = "id", gname = "g",
              xformla = ~xt, data = d, control_group = cg, base_period = bp,
              est_method = "dr", bstrap = FALSE, cband = FALSE)
  out[[paste(bp, cg, sep = "_")]] <- list(group = r$group, time = r$t,
                                           att = r$att, se = r$se)
}
out$versions <- list(did = as.character(packageVersion("did")),
                     DRDID = as.character(packageVersion("DRDID")))
write_json(out, "did_commands_R.json", digits = 17, auto_unbox = TRUE, na = "null")
