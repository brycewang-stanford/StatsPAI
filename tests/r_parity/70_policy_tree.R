# StatsPAI policy-tree parity -- Module 70 (R side).
#
# Reads data/70_policy_tree.csv (written by 70_policy_tree.py at %.16g
# precision) and runs policytree::policy_tree on the *shared* AIPW score
# vector carried in the `gamma` column. Passing the two-action reward
# matrix cbind(0, gamma) makes policytree's objective
# mean(Gamma[i, pi(X_i)]) identical to StatsPAI's mean(gamma * pi), so
# the two engines solve the same optimisation problem over the same data.
#
# policytree returns 1-based actions (1 = control, 2 = treat); we map
# them to the {0, 1} policy StatsPAI reports.
#
# Registered tolerance: rel_est < 1e-6 (machine tier). See 70_policy_tree.py.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(policytree)
})

MODULE <- "70_policy_tree"
DEPTHS <- c(1L, 2L)
MIN_NODE_SIZE <- 1L

df <- read_csv_strict(MODULE)
covariates <- c("x1", "x2", "x3")
X <- as.matrix(df[, covariates])
gamma <- df$gamma
n <- nrow(df)

# Two-action reward matrix: action 1 = do not treat (reward 0),
# action 2 = treat (reward gamma).
Gamma <- cbind(0, gamma)

rows <- list()
policies <- list()

for (depth in DEPTHS) {
  fit <- policy_tree(X, Gamma, depth = depth,
                     min.node.size = MIN_NODE_SIZE, verbose = FALSE)
  action <- predict(fit, X)          # 1-based action index
  policy <- as.integer(action) - 1L  # -> {0, 1}

  value_policy <- mean(gamma * policy)
  fraction_treated <- mean(policy)

  # Root node of the fitted tree. policytree stores nodes in
  # fit$nodes; node 1 is the root, carrying split_variable (1-based)
  # and split_value when it is not a leaf.
  root <- fit$nodes[[1]]
  if (isTRUE(root$is_leaf)) {
    root_var <- NA
    root_val <- NA
  } else {
    root_var <- as.numeric(root$split_variable)
    root_val <- as.numeric(root$split_value)
  }

  policies[[paste0("depth", depth)]] <- policy
  rows <- c(rows, list(
    parity_row(MODULE, sprintf("value_policy_d%d", depth),
               estimate = value_policy, n = n),
    parity_row(MODULE, sprintf("fraction_treated_d%d", depth),
               estimate = fraction_treated, n = n),
    parity_row(MODULE, sprintf("root_split_variable_d%d", depth),
               estimate = root_var, n = n),
    parity_row(MODULE, sprintf("root_split_value_d%d", depth),
               estimate = root_val, n = n)
  ))
}

write_results(MODULE, rows,
              extra = list(
                depths = DEPTHS,
                min.node.size = MIN_NODE_SIZE,
                split.step = 1L,
                estimator = "policytree::policy_tree (exact tree search)",
                policy = policies
              ))
