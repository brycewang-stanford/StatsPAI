# ---------------------------------------------------------------------------
# Answer key for tests/external_parity/test_ness_causal_ai.py
#
# Ness, "Causal AI" (Manning, 2025). The book's code and data are at
# https://github.com/altdeep/causalML (MIT). This script reruns, with R
# packages that are independent of the Python ones the book uses, the
# graph-side steps of chapters 3, 4 and 7 on the book's own data and stores
# the numbers. The numbers the book prints (from pgmpy and DoWhy) are
# written into the test itself.
#
# Requires: dagitty (0.3.4), bnlearn (5.2.1), pcalg (2.7.12), jsonlite.
# Run:      STATSPAI_NESS_DIR=/path/to/causalML/datasets \
#               Rscript tests/external_parity/ness_causal_ai_reference.R
#
# It writes data/ness_causal_ai_R.json next to itself. The data are read
# from $STATSPAI_NESS_DIR and are not copied.
# ---------------------------------------------------------------------------
suppressMessages({library(dagitty); library(bnlearn); library(pcalg); library(jsonlite)})
args <- commandArgs(trailingOnly = FALSE)
self <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
D <- normalizePath(Sys.getenv("STATSPAI_NESS_DIR"), mustWork = TRUE)
out <- list()

# --- chapter 4: testable implications of the transportation DAG -------------
tr <- read.csv(file.path(D, "transportation_survey.csv"), stringsAsFactors = TRUE)
g <- dagitty("dag { A -> E; S -> E; E -> O; E -> R; O -> T; R -> T }")
ici <- impliedConditionalIndependencies(g)
lt <- localTests(g, tr, type = "cis.chisq")
out$transport_chisq <- unname(lapply(seq_along(ici), function(k) {
  z <- sort(unlist(ici[[k]]$Z))
  mi <- if (length(z)) ci.test(ici[[k]]$X, ici[[k]]$Y, z, data = tr, test = "mi-adf") else
    ci.test(ici[[k]]$X, ici[[k]]$Y, data = tr, test = "mi-adf")
  list(x = ici[[k]]$X, y = ici[[k]]$Y, z = as.list(z), x2 = lt$x2[k],
       df = lt$df[k], p = lt$p.value[k], g2 = unname(mi$statistic),
       g2_df = unname(mi$parameter), g2_p = mi$p.value)
}))

# --- chapter 3: kernels of the transportation model -------------------------
bn <- model2network("[A][S][E|A:S][O|E][R|E][T|O:R]")
mle <- bn.fit(bn, tr, method = "mle")
p <- as.data.frame(as.table(mle$T$prob))
out$transport_cpt_T <- lapply(seq_len(nrow(p)), function(k) as.list(p[k, ]))

# --- structure learning on the course's test data ---------------------------
x2 <- function(x, y, S, suffStat) {
  v <- suffStat$names
  r <- if (length(S)) ci.test(v[x], v[y], v[S], data = suffStat$d, test = "x2-adf") else
    ci.test(v[x], v[y], data = suffStat$d, test = "x2-adf")
  if (r$parameter == 0) 1 else r$p.value
}
edges <- function(d) {
  fit <- pcalg::pc(list(d = d, names = names(d)), indepTest = x2, alpha = 0.05,
                   labels = names(d), skel.method = "stable")
  a <- as(fit@graph, "matrix"); v <- names(d)
  directed <- list(); undirected <- list()
  for (i in seq_along(v)) for (j in seq_along(v)) {
    if (a[i, j] == 1 && a[j, i] == 0) directed[[length(directed) + 1]] <- c(v[i], v[j])
    if (i < j && a[i, j] == 1 && a[j, i] == 1)
      undirected[[length(undirected) + 1]] <- c(v[i], v[j])
  }
  list(directed = directed, undirected = undirected)
}
sl <- read.csv(file.path(D, "structure_learning_test.csv"), stringsAsFactors = TRUE)
out$pc_structure_learning <- edges(sl)
out$pc_transport <- edges(tr)

# --- chapter 7: the experiment the observational analysis should match ------
ex <- read.csv(file.path(D, "sidequests_and_purchases_exp.csv"))
tt <- t.test(In.game.Purchases ~ Side.quest.Engagement, data = ex)
out$experiment <- list(diff = unname(tt$estimate[1] - tt$estimate[2]),
                       se = unname(tt$stderr))
ob <- read.csv(file.path(D, "sidequests_and_purchases_full_obs.csv"))
ob$E <- as.numeric(ob$Side.quest.Engagement == "high")
ob$G <- as.numeric(ob$Guild.Membership == "member")
m <- lm(In.game.Purchases ~ E * G, data = ob)
nd1 <- transform(ob, E = 1); nd0 <- transform(ob, E = 0)
out$standardised <- mean(predict(m, nd1) - predict(m, nd0))

write_json(out, file.path(self, "data", "ness_causal_ai_R.json"), auto_unbox = TRUE,
           digits = 15, pretty = TRUE)
cat("wrote", file.path(self, "data", "ness_causal_ai_R.json"), "\n")
