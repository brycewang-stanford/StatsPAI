# Generate a renv.lock for the Track A R-parity reference environment.
#
# We do NOT depend on renv being installed: instead we read the actual
# install metadata from each package's DESCRIPTION (the same fields renv
# itself reads) and emit a schema-compatible renv.lock. Source/Remote
# fields are taken from the installed DESCRIPTION, never guessed, so a
# GitHub-installed package is recorded with its true RemoteUsername /
# RemoteRepo / RemoteSha and a CRAN package with Repository: CRAN.
#
# Scope: the recursive dependency closure of the canonical reference
# packages used by the parity modules, restricted to what is installed.
# This is the minimal-yet-complete set that renv::restore() needs to
# rebuild the reference environment.
#
#   Rscript tests/r_parity/_gen_renv_lock.R   ->  tests/r_parity/renv.lock

suppressWarnings(suppressMessages({
  library(jsonlite)
}))

HERE <- dirname(normalizePath(sub(
  "--file=", "",
  grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])))

REFERENCE_PKGS <- c(
  "AER", "augsynth", "bacondecomp", "did", "didimputation", "DoubleML",
  "etwfe", "fixest", "frontier", "grf", "gsynth", "HonestDiD", "lme4",
  "lpirfs", "MatchIt", "mediation", "oaxaca", "plm", "quantreg",
  "rddensity", "rdrobust", "sandwich", "survival", "Synth", "synthdid",
  "vars", "jsonlite",
  # Modules 23-53: sensitivity, SFA, decomposition, limited-dependent
  # and robust-SE references.
  "EValue", "sfaR", "ddecompose", "dineq", "censReg", "MASS",
  "sampleSelection", "nnet", "clubSandwich", "DRDID", "sensemakr",
  # Modules 57-64: GLM / IV / system / limited-dependent extensions.
  "ivmodel", "systemfit", "betareg", "truncreg", "pscl"
)

ip <- installed.packages()
inst <- rownames(ip)
ref <- intersect(REFERENCE_PKGS, inst)

# Recursive dependency closure (Depends, Imports, LinkingTo) of the
# reference set, restricted to installed packages. We resolve against the
# local installed-packages DB (db = ip) so no CRAN mirror / network is
# needed; base packages are included so the lock is explicit about the R
# version that ships them.
deps <- tools::package_dependencies(
  ref, db = ip, recursive = TRUE,
  which = c("Depends", "Imports", "LinkingTo")
)
closure <- sort(unique(c(ref, unlist(deps, use.names = FALSE))))
closure <- intersect(closure, inst)

pkg_entry <- function(p) {
  d <- as.list(packageDescription(p))
  entry <- list(
    Package = unbox(p),
    Version = unbox(as.character(packageVersion(p)))
  )
  remote_type <- d$RemoteType
  if (!is.null(remote_type) && !is.na(remote_type) &&
      tolower(remote_type) %in% c("github", "git")) {
    entry$Source <- unbox("GitHub")
    if (!is.null(d$RemoteHost))     entry$RemoteHost     <- unbox(d$RemoteHost)
    if (!is.null(d$RemoteUsername)) entry$RemoteUsername <- unbox(d$RemoteUsername)
    if (!is.null(d$RemoteRepo))     entry$RemoteRepo     <- unbox(d$RemoteRepo)
    if (!is.null(d$RemoteRef))      entry$RemoteRef      <- unbox(d$RemoteRef)
    if (!is.null(d$RemoteSha))      entry$RemoteSha      <- unbox(d$RemoteSha)
  } else {
    repo <- d$Repository
    priority <- d$Priority
    if (!is.null(priority) && priority %in% c("base", "recommended")) {
      entry$Source <- unbox("R")
    } else if (!is.null(repo) && nzchar(repo)) {
      entry$Source <- unbox("Repository")
      entry$Repository <- unbox(if (identical(repo, "CRAN")) "CRAN" else repo)
    } else {
      # No Repository field and no Remote* metadata: record honestly as
      # unknown rather than fabricating a CRAN provenance.
      entry$Source <- unbox("unknown")
    }
  }
  entry
}

packages <- list()
for (p in closure) packages[[p]] <- pkg_entry(p)

lockfile <- list(
  R = list(
    Version = unbox(paste(R.version$major, R.version$minor, sep = ".")),
    Repositories = list(list(
      Name = unbox("CRAN"),
      URL  = unbox("https://cloud.r-project.org")
    ))
  ),
  Packages = packages
)

out <- file.path(HERE, "renv.lock")
writeLines(toJSON(lockfile, pretty = TRUE, auto_unbox = FALSE), out)
cat(sprintf("Wrote %s (%d packages: %d reference + %d transitive deps)\n",
            out, length(packages), length(ref),
            length(packages) - length(ref)))
