# Reference for sp.icp(test = "normal"): InvariantCausalPrediction::ICP with
# its default test ("normal": each environment against the rest, t test and
# F test on the residuals of the pooled fit), every subset tested
# (selection = "all"), alpha = 0.05 and 0.01, on the 16 designs of
# icp_data.csv. Recorded: the accepted sets, whether the model is rejected,
# the per-variable p-values and the confidence intervals.
#
# InvariantCausalPrediction is GPL; black box. Install it into a private
# library and pass its path as the first argument if it is not on the
# default library path.
#
#   Rscript tests/reference_parity/_fixtures/_generate_icp_R.R [lib]
suppressMessages(library(jsonlite))
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
if (length(here) == 0) here <- "."
extra <- commandArgs(trailingOnly = TRUE)
if (length(extra)) .libPaths(c(extra[1], .libPaths()))
suppressMessages(library(InvariantCausalPrediction))
d <- read.csv(file.path(here, "icp_data.csv"))
cases <- list()
for (k in sort(unique(d$case))) {
  s <- d[d$case == k, ]
  xs <- setdiff(names(s), c("case", "env", "y"))
  xs <- xs[colSums(is.na(s[, xs])) == 0]
  X <- as.matrix(s[, xs])
  for (alpha in c(0.05, 0.01)) {
    r <- ICP(X, s$y, s$env, alpha = alpha, test = "normal", selection = "all",
             showAcceptedSets = FALSE, showCompletion = FALSE,
             stopIfEmpty = FALSE)
    cases[[length(cases) + 1]] <- list(
      case = k, alpha = alpha, columns = I(xs),
      n_env = length(unique(s$env)),
      accepted = unname(lapply(r$acceptedSets, function(a) I(xs[a]))),
      model_reject = isTRUE(r$modelReject),
      pvalues = I(as.numeric(r$pvalues)),
      ci_upper = I(as.numeric(r$ConfInt[1, ])),
      ci_lower = I(as.numeric(r$ConfInt[2, ]))
    )
  }
}
res <- list(
  versions = list(
    R = R.version.string,
    InvariantCausalPrediction =
      as.character(packageVersion("InvariantCausalPrediction"))
  ),
  cases = cases
)
writeLines(toJSON(res, auto_unbox = TRUE, digits = NA),
           file.path(here, "icp_R.json"))
cat(length(cases), "fits; accepted-set counts:",
    sapply(cases, function(c) length(c$accepted)), "\n")
