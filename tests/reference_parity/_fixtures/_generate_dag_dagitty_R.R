# Reference answers for tests/reference_parity/test_dag_dagitty_parity.py.
#
# Reads dag_dagitty_cases.json (written by _generate_dag_dagitty_cases.py)
# and answers every query with dagitty: minimal adjustment sets, implied
# conditional independencies, d-separation, instruments, the paths between
# exposure and outcome, and the d-separation statements behind the three
# rules of do-calculus on graphs mutilated here, edge by edge, from the
# definitions (Pearl 2009, Theorem 3.4.1).
#
# dagitty is GPL; it is run as a black box.
#
#   Rscript _generate_dag_dagitty_R.R

suppressMessages({
  library(dagitty)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
if (length(here) == 0) here <- "."
cases <- fromJSON(file.path(here, "dag_dagitty_cases.json"),
                  simplifyVector = FALSE)

as_chr <- function(x) as.character(unlist(x))

spec_of <- function(nodes, latent, directed, bidirected) {
  decl <- vapply(nodes, function(v) {
    if (v %in% latent) paste0(v, " [latent]") else v
  }, "")
  ed <- vapply(directed, function(e) paste(e[[1]], "->", e[[2]]), "")
  eb <- vapply(bidirected, function(e) paste(e[[1]], "<->", e[[2]]), "")
  paste0("dag { ", paste(c(decl, ed, eb), collapse = " ; "), " }")
}

# do-calculus surgery on an explicit edge list: every bidirected edge is
# written as a latent common parent first, so "edges into X" covers its
# arrowhead.
explicit_edges <- function(case) {
  ed <- lapply(case$directed, as_chr)
  lat <- as_chr(case$latent)
  k <- 0
  for (e in case$bidirected) {
    k <- k + 1
    l <- paste0("LAT", k)
    lat <- c(lat, l)
    ed <- c(ed, list(c(l, e[[1]])), list(c(l, e[[2]])))
  }
  list(nodes = c(as_chr(case$nodes), setdiff(lat, as_chr(case$nodes))),
       latent = lat, edges = ed)
}
drop_into <- function(edges, s) Filter(function(e) !(e[2] %in% s), edges)
drop_out_of <- function(edges, s) Filter(function(e) !(e[1] %in% s), edges)
graph_of <- function(ex, edges) {
  dagitty(spec_of(ex$nodes, ex$latent, edges, list()))
}
sep <- function(g, a, b, z) {
  isTRUE(dseparated(g, a, b, if (length(z)) z else list()))
}

out <- list()
for (case in cases) {
  nodes <- as_chr(case$nodes)
  latent <- as_chr(case$latent)
  g <- dagitty(spec_of(nodes, latent, case$directed, case$bidirected))
  x <- case$exposure
  y <- case$outcome

  adj <- adjustmentSets(g, x, y, type = "minimal", effect = "total")
  adj <- unname(lapply(adj, function(s) I(sort(as_chr(s)))))

  ici <- impliedConditionalIndependencies(g, type = "missing.edge",
                                          max.results = Inf)
  ici <- unname(lapply(ici, function(ci) {
    list(x = ci$X, y = ci$Y, z = I(sort(as_chr(ci$Z))))
  }))

  ds <- vapply(case$dsep, function(q) {
    sep(g, q$x, q$y, as_chr(q$z))
  }, TRUE)

  iv <- instrumentalVariables(g, x, y)
  iv <- unname(lapply(iv, function(i) {
    list(instrument = i$I, conditioning = I(sort(as_chr(i$Z))))
  }))

  pt <- paths(g, x, y, limit = 100000)
  paths_out <- list(paths = I(as_chr(pt$paths)), open = I(as.logical(pt$open)))

  ex <- explicit_edges(case)
  rules <- lapply(case$rules, function(q) {
    Y <- as_chr(q$Y); X <- as_chr(q$X); Z <- as_chr(q$Z); W <- as_chr(q$W)
    cond <- c(X, W)
    bar_x <- drop_into(ex$edges, X)
    g1 <- graph_of(ex, bar_x)
    g2 <- graph_of(ex, drop_out_of(bar_x, Z))
    an_w <- if (length(W)) unique(unlist(lapply(W, function(w) {
      ancestors(g1, w)
    }))) else character(0)
    zw <- setdiff(Z, an_w)
    g3 <- graph_of(ex, drop_into(bar_x, zw))
    list(rule1 = sep(g1, Y, Z, cond), rule2 = sep(g2, Y, Z, cond),
         rule3 = sep(g3, Y, Z, cond), z_w = I(sort(zw)))
  })

  out[[case$id]] <- list(
    case = case, spec = spec_of(nodes, latent, case$directed, case$bidirected),
    adjustment = adj, implied = ici, dsep = I(ds),
    instruments = iv, paths = paths_out, rules = rules
  )
}

res <- list(
  versions = list(R = R.version.string,
                  dagitty = as.character(packageVersion("dagitty"))),
  cases = out
)
writeLines(toJSON(res, auto_unbox = TRUE, digits = NA),
           file.path(here, "dag_dagitty_R.json"))
cat("wrote", length(out), "cases\n")
