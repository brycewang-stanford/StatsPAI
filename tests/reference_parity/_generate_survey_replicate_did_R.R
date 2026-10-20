#!/usr/bin/env Rscript
# Frozen reference for replicate-weight designs on the DiD estimators:
# sp.callaway_santanna(design=svrepdesign) and sp.did_2x2(design=...).
# R survey 4.5 (as.svrepdesign, svrepdesign, withReplicates, svyglm) and
# did 2.5.1 (att_gt, aggte); both GPL, used as black boxes (documentation
# and outputs only).
#
# Data: _fixtures/survey_did_data.csv (the linearisation DiD fixture). The
# unit-level design (strata, PSUs nested in strata, weight w) is turned
# into JKn replicate weights by as.svrepdesign; they are exported per unit
# to _fixtures/survey_replicate_did_rw.csv so both sides use the same ones.
# CS: att_gt(weightsname =) recomputed with every replicate's unit weights
# (units of zero weight left out), estimator "dr", base_period "varying";
# the statistic is (simple ATT, dynamic ATTs, ATT(g,t)). 2x2: svyglm(y ~
# treat * post) on the long rows with the replicate weights of their unit.
#
# Regenerate with:
#   Rscript tests/reference_parity/_generate_survey_replicate_did_R.R

suppressPackageStartupMessages({
  library(survey)
  library(did)
  library(jsonlite)
})
here <- function(p) file.path("tests", "reference_parity", p)
df <- read.csv(here("_fixtures/survey_did_data.csv"))
units <- unique(df[, c("id", "stratum", "psu", "w")])
udes <- svydesign(ids = ~psu, strata = ~stratum, weights = ~w, nest = TRUE,
                  data = units)
rep <- as.svrepdesign(udes, type = "JKn", compress = FALSE)
RW <- unname(as.matrix(weights(rep, "analysis")))
R <- ncol(RW)
rw_out <- data.frame(id = units$id)
for (j in seq_len(R)) rw_out[[paste0("rw_", j)]] <- RW[, j]
write.csv(rw_out, here("_fixtures/survey_replicate_did_rw.csv"), row.names = FALSE)
rscales <- unname(rep$rscales)

ud <- svrepdesign(data = units, repweights = RW, weights = ~w, type = "JKn",
                  scale = 1, rscales = rscales, combined.weights = TRUE)

cs_stat <- function(rw, data) {
  keep <- data$id[rw > 0]
  dl <- df[df$id %in% keep, ]
  dl$rw <- rw[match(dl$id, data$id)]
  a <- suppressWarnings(att_gt(yname = "y", tname = "t", idname = "id",
              gname = "g", xformla = ~x, data = dl, weightsname = "rw",
              panel = TRUE, bstrap = FALSE, est_method = "dr",
              base_period = "varying", control_group = "nevertreated"))
  s <- aggte(a, type = "simple", bstrap = FALSE)
  dy <- aggte(a, type = "dynamic", bstrap = FALSE)
  c(s$overall.att, dy$att.egt, a$att)
}
cs <- withReplicates(ud, cs_stat)
a0 <- att_gt(yname = "y", tname = "t", idname = "id", gname = "g",
             xformla = ~x, data = df, weightsname = "w", panel = TRUE,
             bstrap = FALSE, est_method = "dr", base_period = "varying",
             control_group = "nevertreated")
dy0 <- aggte(a0, type = "dynamic", bstrap = FALSE)
n_dyn <- length(dy0$egt)

sub <- df[df$t %in% c(2, 3) & df$g %in% c(0, 3), ]
sub$treat <- as.integer(sub$g == 3)
sub$post <- as.integer(sub$t == 3)
# keep only the model columns: a data column named `t` masks base::t inside
# svyglm's replicate method ("$ operator is invalid for atomic vectors")
sub <- sub[, c("id", "y", "treat", "post", "w")]
SRW <- RW[match(sub$id, units$id), ]
sd2 <- svrepdesign(data = sub, repweights = SRW, weights = ~w, type = "JKn",
                   scale = 1, rscales = rscales, combined.weights = TRUE)
f <- svyglm(y ~ treat * post, design = sd2)

out <- list(
  R = R, rscales = rscales, degf = degf(ud), degf_2x2 = degf(sd2),
  cs = list(simple = unname(coef(cs))[1], se_simple = unname(SE(cs))[1],
            dynamic_e = dy0$egt,
            dynamic_att = unname(coef(cs))[1 + seq_len(n_dyn)],
            se_dynamic = unname(SE(cs))[1 + seq_len(n_dyn)],
            group = a0$group, time = a0$t,
            att = unname(coef(cs))[-(1:(1 + n_dyn))],
            se_cells = unname(SE(cs))[-(1:(1 + n_dyn))]),
  did2x2 = list(att = unname(coef(f)["treat:post"]),
                se = unname(SE(f)["treat:post"])),
  versions = list(R = R.version.string,
                  survey = as.character(packageVersion("survey")),
                  did = as.character(packageVersion("did")))
)
write_json(out, here("_fixtures/survey_replicate_did_R.json"), digits = NA,
           auto_unbox = TRUE)
