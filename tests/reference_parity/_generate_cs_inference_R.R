# Generate R `did` reference numbers for the Callaway–Sant'Anna
# multiplier-bootstrap inference options added in StatsPAI batch D:
#   - bstrap=TRUE per-(g,t) bootstrap SEs
#   - cband=TRUE uniform (sup-t) critical value
#   - clustervars beyond the unit id
#
# Uses the SAME panel as _fixtures/cs_data.csv (shared with
# test_callaway_santanna_parity.py) plus a deterministic cluster
# variable clust = id %% 15 (time-invariant by construction).
#
# Run from tests/reference_parity/:
#   Rscript _generate_cs_inference_R.R
#
# Output: _fixtures/cs_inference_R.json

suppressMessages({
  library(did)
  library(jsonlite)
})

set.seed(20260725)

df <- read.csv("_fixtures/cs_data.csv")
# did >= 2.3.0 recodes g==0 to Inf internally; on an integer column that
# truncates to NA and silently drops the never-treated group. Cast first.
df$first_treat <- as.numeric(df$first_treat)
df$clust <- df$id %% 15

BITERS <- 9999

fit <- function(clustervars = NULL) {
  att_gt(
    yname = "y",
    tname = "year",
    idname = "id",
    gname = "first_treat",
    data = df,
    control_group = "nevertreated",
    base_period = "universal",
    est_method = "dr",
    bstrap = TRUE,
    biters = BITERS,
    cband = TRUE,
    clustervars = clustervars
  )
}

extract <- function(m) {
  list(
    group = m$group,
    t = m$t,
    att = m$att,
    se = m$se,
    crit_val = as.numeric(m$c)
  )
}

# did >= 2.3.0 requires a scalar clustervars (idname is implied).
m_unit <- fit(clustervars = NULL)
m_clust <- fit(clustervars = "clust")

# Overall simple aggregation under bootstrap, for the headline ATT SE.
agg_simple <- aggte(m_unit, type = "simple", bstrap = TRUE, biters = BITERS)

out <- list(
  meta = list(
    generator = "_generate_cs_inference_R.R",
    did_version = as.character(packageVersion("did")),
    r_version = paste(R.version$major, R.version$minor, sep = "."),
    biters = BITERS,
    seed = 20260725,
    cluster_rule = "clust = id %% 15",
    spec = paste(
      "att_gt(control_group='nevertreated', base_period='universal',",
      "est_method='dr', bstrap=TRUE, cband=TRUE)"
    )
  ),
  unit = extract(m_unit),
  clustered = extract(m_clust),
  agg_simple = list(
    att = agg_simple$overall.att,
    se = agg_simple$overall.se
  )
)

write_json(out, "_fixtures/cs_inference_R.json",
           auto_unbox = TRUE, digits = 12)
cat("Wrote _fixtures/cs_inference_R.json\n")
cat("unit crit:", out$unit$crit_val, " clustered crit:", out$clustered$crit_val, "\n")
