# Reference conditional inference trees for sp.ctree, from partykit::ctree.
# Run _generate_ctree_data.py first. partykit is used as a black box.
#   Rscript tests/reference_parity/_fixtures/_generate_ctree.R
suppressMessages({library(partykit); library(jsonlite)})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("--file=", "", args[grep("--file=", args)]))

describe <- function(fit, data) {
  ids <- nodeids(fit)
  term <- nodeids(fit, terminal = TRUE)
  nodes <- lapply(ids, function(i) {
    nd <- nodeapply(fit, i)[[1]]
    out <- list(id = i, terminal = i %in% term,
                n = length(fit[[i]]$fitted[[1]]))
    if (!(i %in% term)) {
      sp <- split_node(nd)
      v <- names(fit$data)[varid_split(sp)]
      out$variable <- v
      out$kids <- sapply(kids_node(nd), id_node)
      if (!is.null(breaks_split(sp))) out$threshold <- breaks_split(sp)
      if (!is.null(index_split(sp))) {
        idx <- index_split(sp); lv <- levels(fit$data[[v]])
        out$left <- lv[!is.na(idx) & idx == 1]
        out$right <- lv[!is.na(idx) & idx == 2]
      }
      crit <- info_node(nd)$criterion
      out$tested <- colnames(crit)
      out$statistic <- unname(crit["statistic", ])
      out$p_value <- unname(crit["p.value", ])
    }
    out
  })
  resp <- predict(fit, newdata = data)
  list(nodes = nodes,
       node_id = unname(predict(fit, newdata = data, type = "node")),
       fitted = if (is.factor(resp)) as.character(resp) else unname(resp))
}

reg <- read.csv(file.path(here, "ctree_reg.csv"), stringsAsFactors = TRUE)
multi <- read.csv(file.path(here, "ctree_multi.csv"), stringsAsFactors = TRUE)
multi$grade <- factor(multi$grade, levels = c("low", "mid", "high", "top"),
                      ordered = TRUE)
small <- read.csv(file.path(here, "ctree_small.csv"))
small$y <- factor(small$y)

cases <- list(
  reg_default = describe(ctree(y ~ ., data = reg), reg),
  reg_alpha30 = describe(ctree(y ~ ., data = reg,
                               control = ctree_control(alpha = 0.3)), reg),
  reg_univariate = describe(ctree(y ~ ., data = reg,
      control = ctree_control(testtype = "Univariate")), reg),
  reg_depth2 = describe(ctree(y ~ ., data = reg,
      control = ctree_control(maxdepth = 2)), reg),
  reg_bucket40 = describe(ctree(y ~ ., data = reg,
      control = ctree_control(minsplit = 90, minbucket = 40)), reg),
  multi_default = describe(ctree(y ~ ., data = multi), multi),
  multi_maximum = describe(ctree(y ~ ., data = multi,
      control = ctree_control(splitstat = "maximum")), multi),
  multi_alpha40 = describe(ctree(y ~ ., data = multi,
      control = ctree_control(alpha = 0.4)), multi),
  small_default = describe(ctree(y ~ ., data = small), small),
  small_alpha50 = describe(ctree(y ~ ., data = small,
      control = ctree_control(alpha = 0.5)), small),
  small_prob20 = describe(ctree(y ~ ., data = small,
      control = ctree_control(alpha = 0.5, minprob = 0.2)), small),
  small_bucket1 = describe(ctree(y ~ ., data = small,
      control = ctree_control(alpha = 0.5, minsplit = 4, minbucket = 1,
                              minprob = 0)), small)
)
out <- list(partykit = as.character(packageVersion("partykit")),
            R = R.version.string, cases = cases)
writeLines(toJSON(out, digits = 15, auto_unbox = TRUE, null = "null"),
           file.path(here, "ctree_R.json"))
for (k in names(cases)) cat(k, length(cases[[k]]$nodes), "nodes\n")
