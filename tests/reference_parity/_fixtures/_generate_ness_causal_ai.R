# ---------------------------------------------------------------------------
# Reference numbers for tests/reference_parity/test_ness_causal_ai_parity.py
#
# The graph-side workflow of Ness, "Causal AI" (Manning, 2025; code at
# https://github.com/altdeep/causalML): list the conditional independencies
# a DAG implies, test them on categorical data, fit the causal Markov
# kernels, compute an interventional distribution, and (in the course the
# book grew out of) learn the graph from data. The book does this
# in Python with pgmpy and y0; the references here are the R packages that
# implement the same things independently. Both sides read the CSV this
# script writes.
#
# Requires: dagitty (0.3.4), bnlearn (5.2.1), causaleffect (1.3.15),
#           pcalg (2.7.12), cfid (0.1.8), jsonlite.
# Run:      Rscript tests/reference_parity/_fixtures/_generate_ness_causal_ai.R
#           (from the repository root)
# ---------------------------------------------------------------------------
suppressMessages({
  library(dagitty); library(bnlearn); library(causaleffect); library(igraph)
  library(pcalg)
  library(jsonlite)
})
here <- "tests/reference_parity/_fixtures"
set.seed(20261006)
n <- 600
pick <- function(levels, p) {
  # one draw per row; p is a matrix of row-wise probabilities
  u <- runif(nrow(p)); cum <- t(apply(p, 1, cumsum))
  levels[1 + rowSums(u > cum[, -ncol(cum), drop = FALSE])]
}
A <- sample(c("young", "adult", "old"), n, TRUE, c(.3, .5, .2))
S <- sample(c("F", "M"), n, TRUE, c(.45, .55))
pE <- ifelse(A == "young", .35, ifelse(A == "adult", .25, .15)) + ifelse(S == "F", .1, 0)
E <- ifelse(runif(n) < pE, "uni", "high")
O <- ifelse(runif(n) < ifelse(E == "uni", .12, .04), "self", "emp")
R <- ifelse(runif(n) < ifelse(E == "uni", .85, .7), "big", "small")
pT <- cbind(car = ifelse(R == "big", .55, .45) + ifelse(O == "self", .1, 0),
            train = ifelse(R == "big", .3, .2))
pT <- cbind(pT, other = 1 - rowSums(pT))
Tr <- pick(c("car", "train", "other"), pT)
# One arrow the test graph leaves out, so that an implication fails.
Tr <- ifelse(A == "old" & runif(n) < .5, "car", Tr)
d <- data.frame(A, S, E, O, R, T = Tr, stringsAsFactors = TRUE)
write.csv(d, file.path(here, "ness_categorical.csv"), row.names = FALSE)

out <- list(n = n)
g <- dagitty("dag { A -> E; S -> E; E -> O; E -> R; O -> T; R -> T }")

# --- implied conditional independencies and their chi-square tests ----------
ici <- impliedConditionalIndependencies(g)
out$implied <- lapply(ici, function(i) list(x = i$X, y = i$Y, z = sort(unlist(i$Z))))
lt <- localTests(g, d, type = "cis.chisq")
out$dagitty_chisq <- lapply(seq_along(ici), function(k) {
  list(x = ici[[k]]$X, y = ici[[k]]$Y, z = sort(unlist(ici[[k]]$Z)),
       x2 = lt$x2[k], df = lt$df[k], p = lt$p.value[k])
})
out$bnlearn <- lapply(ici, function(i) {
  z <- sort(unlist(i$Z))
  f <- function(tst) {
    r <- if (length(z)) ci.test(i$X, i$Y, z, data = d, test = tst) else
      ci.test(i$X, i$Y, data = d, test = tst)
    list(stat = unname(r$statistic), df = unname(r$parameter), p = r$p.value)
  }
  list(x = i$X, y = i$Y, z = z, x2_adf = f("x2-adf"), mi_adf = f("mi-adf"))
})

# --- kernels: maximum likelihood and with a Dirichlet prior -----------------
bn <- model2network("[A][S][E|A:S][O|E][R|E][T|O:R]")
mle <- bn.fit(bn, d, method = "mle")
tab <- function(fit, node) {
  p <- as.data.frame(as.table(fit[[node]]$prob))
  lapply(seq_len(nrow(p)), function(k) as.list(p[k, ]))
}
out$cpt_mle <- list(E = tab(mle, "E"), T = tab(mle, "T"))
# A pseudo-count of one per cell, written out: bnlearn's own Bayesian fit
# spreads an imaginary sample size instead and is a different prior.
cnt <- table(d$O, d$R, d$T) + 1
pr <- prop.table(cnt, c(1, 2))
out$cpt_T_laplace <- lapply(seq_len(nrow(as.data.frame(pr))), function(k)
  as.list(setNames(as.data.frame(pr)[k, ], c("O", "R", "T", "Freq"))))

# --- an interventional distribution by the truncated factorisation ----------
# P(T | do(E = e)) = sum_{o, r} P(o | e) P(r | e) P(T | o, r), from the
# maximum-likelihood kernels above.
pO <- prop.table(table(d$E, d$O), 1); pR <- prop.table(table(d$E, d$R), 1)
pTor <- prop.table(table(d$O, d$R, d$T), c(1, 2))
do_e <- function(e) {
  acc <- setNames(numeric(nlevels(d$T)), levels(d$T))
  for (o in levels(d$O)) for (r in levels(d$R))
    acc <- acc + pO[e, o] * pR[e, r] * pTor[o, r, ]
  as.list(acc)
}
out$do_E <- list(high = do_e("high"), uni = do_e("uni"))
# and a conditional that needs the whole network: P(E | T = train)
pA <- prop.table(table(d$A)); pS <- prop.table(table(d$S))
pEas <- prop.table(table(d$A, d$S, d$E), c(1, 2))
num <- c(high = 0, uni = 0)
for (a in levels(d$A)) for (s in levels(d$S)) for (e in levels(d$E))
  for (o in levels(d$O)) for (r in levels(d$R))
    num[e] <- num[e] + pA[a] * pS[s] * pEas[a, s, e] * pO[e, o] * pR[e, r] *
      pTor[o, r, "train"]
out$E_given_T_train <- as.list(num / sum(num))

# --- adjustment sets on graphs the old search missed -------------------------
big <- paste0("dag { X -> Y; ",
              paste0("Z", 1:8, " -> X; Z", 1:8, " -> Y", collapse = "; "), " }")
out$adjust_eight <- lapply(adjustmentSets(dagitty(big), "X", "Y"), sort)
gaming <- dagitty('dag {
  PE [latent]
  PE -> Skill; PE -> Time; Time -> Skill
  Guild -> Engage; Guild -> Buy; Skill -> Engage; Skill -> Buy
  Time -> Engage; Time -> Buy; Assign -> Engage; Custom -> Engage
  Engage -> Won; Won -> Buy; Won -> Inventory; Buy -> Inventory }')
out$adjust_gaming <- lapply(adjustmentSets(gaming, "Engage", "Buy"), sort)
out$instruments_gaming <- sort(sapply(instrumentalVariables(gaming, "Engage", "Buy"),
                                      function(i) i$I))
out$implied_gaming_n <- length(impliedConditionalIndependencies(gaming))

# --- identification verdicts (Shpitser-Pearl ID, causaleffect) ---------------
ident <- function(edges, bidirected = character(), x = "X", y = "Y") {
  spec <- paste(c(edges, unlist(lapply(bidirected, function(b) {
    v <- strsplit(b, " <-> ")[[1]]; c(paste(v[1], "-+", v[2]), paste(v[2], "-+", v[1]))
  }))), collapse = ", ")
  gr <- eval(parse(text = paste0("graph_from_literal(", spec, ", simplify = FALSE)")))
  if (length(bidirected)) {
    nb <- 2 * length(bidirected); ne <- length(E(gr))
    gr <- set_edge_attr(gr, "description", (ne - nb + 1):ne, "U")
  }
  r <- tryCatch(causal.effect(y, x, G = gr, simp = TRUE), error = function(e) NA)
  !identical(r, NA)
}
out$identify <- list(
  backdoor   = ident(c("Z -+ X", "Z -+ Y", "X -+ Y")),
  bow        = ident(c("X -+ Y"), "X <-> Y"),
  frontdoor  = ident(c("X -+ M", "M -+ Y"), "X <-> Y"),
  iv         = ident(c("Z -+ X", "X -+ Y"), "X <-> Y"),
  napkin     = ident(c("W -+ Z", "Z -+ X", "X -+ Y"), c("W <-> X", "W <-> Y")),
  m_bias     = ident(c("X -+ Y"), c("X <-> Z", "Z <-> Y")),
  fd_broken  = ident(c("X -+ M", "M -+ Y"), c("X <-> Y", "M <-> Y"))
)

# --- structure learning: PC-stable (pcalg) -----------------------------------
graph_of <- function(fit, v) {
  a <- as(fit@graph, "matrix")
  directed <- list(); undirected <- list(); sep <- list()
  for (i in seq_along(v)) for (j in seq_along(v)) {
    if (a[i, j] == 1 && a[j, i] == 0) directed[[length(directed) + 1]] <- c(v[i], v[j])
    if (i < j && a[i, j] == 1 && a[j, i] == 1)
      undirected[[length(undirected) + 1]] <- c(v[i], v[j])
    if (i < j && a[i, j] == 0 && a[j, i] == 0) {
      s <- fit@sepset[[i]][[j]]; if (is.null(s)) s <- fit@sepset[[j]][[i]]
      sep[[length(sep) + 1]] <- list(x = v[i], y = v[j], s = as.list(v[s]))
    }
  }
  list(directed = directed, undirected = undirected, sepsets = sep)
}
# Gaussian: X1 -> X3 <- X2, X3 -> X4, X4 -> X6 <- X5, X2 -> X7, X6 -> X8
m <- 500
g1 <- rnorm(m); g2 <- rnorm(m); g5 <- rnorm(m)
g3 <- 0.8 * g1 + 0.7 * g2 + rnorm(m); g4 <- 0.9 * g3 + rnorm(m)
g6 <- 0.7 * g4 + 0.8 * g5 + rnorm(m); g7 <- 0.9 * g2 + rnorm(m)
g8 <- 0.8 * g6 + rnorm(m)
gd <- round(data.frame(X1 = g1, X2 = g2, X3 = g3, X4 = g4, X5 = g5, X6 = g6,
                       X7 = g7, X8 = g8), 6)
write.csv(gd, file.path(here, "ness_gaussian.csv"), row.names = FALSE)
gd <- read.csv(file.path(here, "ness_gaussian.csv"))
fit <- pcalg::pc(list(C = cor(gd), n = nrow(gd)), indepTest = gaussCItest,
                 alpha = 0.05, labels = names(gd), skel.method = "stable")
out$pc_gaussian <- graph_of(fit, names(gd))
# Categorical: pcalg's search with the stratified chi-square of bnlearn.
x2 <- function(x, y, S, suffStat) {
  v <- suffStat$names
  r <- if (length(S)) ci.test(v[x], v[y], v[S], data = suffStat$d, test = "x2-adf") else
    ci.test(v[x], v[y], data = suffStat$d, test = "x2-adf")
  if (r$parameter == 0) 1 else r$p.value
}
fit <- pcalg::pc(list(d = d, names = names(d)), indepTest = x2, alpha = 0.05,
                 labels = names(d), skel.method = "stable")
cat_graph <- graph_of(fit, names(d))
# Two colliders claim the edge R - T in this sample; which one keeps it is
# an implementation's choice, so only the skeleton and the separating sets
# are compared.
out$pc_categorical <- list(
  skeleton = lapply(c(cat_graph$directed, cat_graph$undirected), sort),
  sepsets = cat_graph$sepsets)
# --- FCI: the partial ancestral graph, edge marks included --------------------
pag_of <- function(dat, alpha) {
  ff <- pcalg::fci(list(C = cor(dat), n = nrow(dat)), indepTest = gaussCItest,
                   alpha = alpha, labels = names(dat), skel.method = "stable")
  a <- ff@amat; v <- names(dat); left <- c("", "o", "<", "-"); right <- c("", "o", ">", "-")
  e <- list()
  # amat[i, j] is the mark at the j end of the edge i - j
  for (i in seq_along(v)) for (j in seq_along(v)) if (i < j && a[i, j] != 0)
    e[[length(e) + 1]] <- c(v[i], paste0(left[a[j, i] + 1], "-", right[a[i, j] + 1]), v[j])
  e
}
out$fci_gaussian <- pag_of(gd, 0.05)
# Three common causes: X and Y are separated only by all three of them.
k <- 2000
ca <- rnorm(k); cb <- rnorm(k); cc <- rnorm(k)
fd <- round(data.frame(A = ca, B = cb, C = cc, X = ca + cb + cc + rnorm(k),
                       Y = ca + cb + cc + rnorm(k)), 6)
write.csv(fd, file.path(here, "ness_three_causes.csv"), row.names = FALSE)
fd <- read.csv(file.path(here, "ness_three_causes.csv"))
out$fci_three_causes <- pag_of(fd, 0.05)
# Two unobserved common causes (L1 of X and Y, L2 of Y and Z), dropped from
# the file: the graph has bidirected edges that PC cannot represent.
q <- 3000
L1 <- rnorm(q); L2 <- rnorm(q); la <- rnorm(q); lb <- rnorm(q)
lx <- 0.8 * la + 0.9 * L1 + rnorm(q)
ly <- 0.8 * lb + 0.9 * L1 + 0.7 * L2 + rnorm(q)
lw <- 0.8 * lx + rnorm(q)
lz <- 0.9 * L2 + 0.7 * lw + rnorm(q)
ld <- round(data.frame(A = la, B = lb, X = lx, Y = ly, W = lw, Z = lz), 6)
write.csv(ld, file.path(here, "ness_latent.csv"), row.names = FALSE)
ld <- read.csv(file.path(here, "ness_latent.csv"))
out$fci_latent <- pag_of(ld, 0.01)

# --- counterfactual identification verdicts (ID* / IDC*, cfid) ----------------
# cf(var, 0) is the event var = 0; the third argument is the intervention.
cfq <- function(spec, gamma, delta = NULL) {
  r <- if (is.null(delta)) cfid::identifiable(cfid::dag(spec), gamma) else
    cfid::identifiable(cfid::dag(spec), gamma, delta)
  r$id
}
y0 <- cfid::cf("Y", 1, c(X = 0)); x1 <- cfid::cf("X", 1); y1 <- cfid::cf("Y", 1)
out$cfid <- list(
  ett_backdoor   = cfq("Z -> X; Z -> Y; X -> Y", y0, x1),
  ett_bow        = cfq("X -> Y; X <-> Y", y0, x1),
  ett_frontdoor  = cfq("X -> W; W -> Y; X <-> Y", y0, x1),
  ett_iv         = cfq("Z -> X; X -> Y; X <-> Y", y0, x1),
  effect         = cfq("X -> Y; X <-> Z; Z -> Y", y0),
  necessity      = cfq("X -> Y", cfid::cf("Y", 0, c(X = 0)), cfid::conj(x1, y1)),
  two_worlds     = cfq("X -> Y", cfid::conj(cfid::cf("Y", 1, c(X = 0)),
                                             cfid::cf("Y", 1, c(X = 1)))),
  book_ett       = cfq("T -> W; W -> A; B -> V; V -> A; C -> T; C -> A; C -> B",
                       cfid::cf("A", 1, c(T = 0)), cfid::cf("T", 1)),
  paper_example  = cfq("X -> W; W -> Y; D -> Z; Z -> Y; X <-> Y",
                       cfid::cf("Y", 1, c(X = 0)),
                       cfid::conj(cfid::cf("X", 1), cfid::cf("Z", 1, c(D = 1)),
                                  cfid::cf("D", 1)))
)

write_json(out, file.path(here, "ness_causal_ai_R.json"), auto_unbox = TRUE,
           digits = 15, pretty = TRUE)
cat("wrote", file.path(here, "ness_causal_ai_R.json"), "\n")
