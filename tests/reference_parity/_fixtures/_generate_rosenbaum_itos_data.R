# ---------------------------------------------------------------------------
# Input data for tests/reference_parity/test_rosenbaum_itos_parity.py
#
# Writes the CSV files that both the R reference script
# (_generate_rosenbaum_itos.R) and the Python tests read.
#
# The committed files are SIMULATED here, to the shape of the examples in
# Rosenbaum, "An Introduction to the Theory of Observational Studies"
# (Springer, 2025): blocks of four with integer outcomes and ties, blocks of
# three from two control groups, blocks with two treated individuals and
# many tied zeros, and an unmatched sample with a small treated group and
# two control reservoirs. The book's own data are not redistributed.
#
# To run the same comparison on the book's data, which ship with the CRAN
# packages iTOS, weightedRank and tightenBlock, write them to a directory of
# your own and point both the reference script and the tests at it:
#
#   Rscript _generate_rosenbaum_itos_data.R --book /some/dir
#   STATSPAI_ITOS_FIXTURES=/some/dir Rscript _generate_rosenbaum_itos.R
#   STATSPAI_ITOS_FIXTURES=/some/dir pytest test_rosenbaum_itos_parity.py
#
# Run:  Rscript tests/reference_parity/_fixtures/_generate_rosenbaum_itos_data.R
#       (from the repository root)
# ---------------------------------------------------------------------------
args <- commandArgs(TRUE)
book <- length(args) >= 2 && args[1] == "--book"
here <- if (book) args[2] else "tests/reference_parity/_fixtures"
w <- function(d, f) write.csv(d, file.path(here, f), row.names = FALSE)
binge_cols <- c("SEQN", "age", "ageC", "female", "education", "bmi", "waisthip", "vigor",
                "smokenow", "bpRX", "smokeQuit", "AlcGroup", "bpCombined")
block_cols <- c("SEQN", "z", "block", "age", "education", "bmi", "ibmi", "dentist1Y")

if (book) {
  get_data <- function(name, pkg) { e <- new.env(); data(list = name, package = pkg, envir = e); get(name, e) }
  aHDL <- get_data("aHDL", "iTOS")
  hdl <- as.data.frame(t(matrix(aHDL$hdl, 4, 406))); names(hdl) <- c("treated", "c1", "c2", "c3")
  aBP <- get_data("aBP", "weightedRank")
  yD <- t(matrix(aBP$bpDiastolic, 3, 207)); yS <- t(matrix(aBP$bpSystolic, 3, 207))
  vS <- c(yS[, 1] - yS[, 2], yS[, 1] - yS[, 3], yS[, 2] - yS[, 3])
  vD <- c(yD[, 1] - yD[, 2], yD[, 1] - yD[, 3], yD[, 2] - yD[, 3])
  y <- (yD / median(abs(vD))) + (yS / median(abs(vS)))
  bp <- data.frame(b = y[, 1], n = y[, 2], p = y[, 3])
  peri <- get_data("Peri24and15", "weightedRank")[, c("block", "z", "pd", "pair")]
  binge <- get_data("binge", "iTOS")
  binge$ageC <- as.integer(binge$ageC); binge$AlcGroup <- as.character(binge$AlcGroup)
  binge <- binge[, binge_cols]
  blocks <- get_data("aHDLt", "tightenBlock")[, block_cols]
} else {
  set.seed(20261006)
  # 406 blocks of four: one treated, three controls, integer outcomes ---------
  lvl <- rnorm(406, 52, 9)
  hdl <- data.frame(treated = round(lvl + rnorm(406, 13, 17)), c1 = round(lvl + rnorm(406, 0, 13)),
                    c2 = round(lvl + rnorm(406, 0, 13)), c3 = round(lvl + rnorm(406, 0, 13)))
  # 207 blocks of three: treated, first control, second control ---------------
  lvl <- rnorm(207, 0, 0.8)
  bp <- data.frame(b = round(lvl + 0.55 + rt(207, 5), 4), n = round(lvl + rt(207, 5), 4),
                   p = round(lvl - 0.1 + rt(207, 5), 4))
  # 606 blocks of four with two treated, 213 of five with one; many zeros -----
  mk <- function(nb, size, zrow, first, pair) {
    z <- rep(zrow, nb)
    pd <- round(pmax(0, rnorm(nb * size, -4 + 14 * z, 16)), 1)
    data.frame(block = rep(first:(first + nb - 1), each = size), z = z, pd = pd, pair = pair)
  }
  peri <- rbind(mk(606, 4, c(1, 0, 1, 0), 1, 1), mk(213, 5, c(1, 0, 0, 0, 0), 607, 0))
  # An unmatched sample: 206 treated (B), two control reservoirs (N, P) -------
  n <- 4600
  age <- round(pmin(80, pmax(20, rnorm(n, 52, 16))))
  female <- rbinom(n, 1, 0.56); education <- sample(1:5, n, TRUE, c(.06, .1, .23, .33, .28))
  bmi <- round(pmax(16, rnorm(n, 29.8, 6.8)), 1); waisthip <- round(rnorm(n, 0.94, 0.08), 4)
  vigor <- rbinom(n, 1, 0.38); smokenow <- sample(1:3, n, TRUE, c(.13, .04, .83))
  bpRX <- rbinom(n, 1, plogis(-3.6 + 0.055 * age)); smokeQuit <- rbinom(n, 1, 0.2) * (smokenow == 3)
  lin <- -2.2 - 0.03 * (age - 50) - 1.1 * female + 1.3 * (smokenow == 1) + 0.6 * vigor - 0.2 * (education - 3)
  u <- runif(n); pB <- plogis(lin) * 0.8; pP <- 0.11 + 0.05 * (smokenow < 3)
  AlcGroup <- ifelse(u < pB, "B", ifelse(u < pB + pP, "P", "N"))
  keepB <- which(AlcGroup == "B"); AlcGroup[keepB[-(1:min(206, length(keepB)))]] <- "N"
  bpCombined <- round(0.02 * (age - 50) + 0.03 * (bmi - 30) + 0.5 * (AlcGroup == "B") + rnorm(n), 6)
  binge <- data.frame(SEQN = 100000 + sample(1:20000, n), age, ageC = as.integer(cut(age, c(0, 29.5, 44.5, 59.5, 200))),
                      female, education, bmi, waisthip, vigor, smokenow, bpRX, smokeQuit, AlcGroup, bpCombined)
  stopifnot(sum(binge$AlcGroup == "B") == 206, !anyDuplicated(binge$SEQN))
  # A 1-to-3 block design to tighten ------------------------------------------
  nb <- 406; z <- rep(c(1, 0, 0, 0), nb); base_age <- rep(round(rnorm(nb, 50, 14)), each = 4)
  bmi <- round(pmax(17, rnorm(4 * nb, 28.5 - 1.5 * z, 6)), 1); miss <- rbinom(4 * nb, 1, 0.04)
  blocks <- data.frame(SEQN = 70000 + sample(1:20000, 4 * nb), z, block = rep(1:nb, each = 4),
                       age = base_age + sample(-2:2, 4 * nb, TRUE),
                       education = pmin(5, pmax(1, rep(sample(2:5, nb, TRUE), each = 4) + sample(-1:1, 4 * nb, TRUE))),
                       bmi = ifelse(miss == 1, 28.5, bmi), ibmi = miss,
                       dentist1Y = rbinom(4 * nb, 1, 0.55 + 0.12 * (1 - z)))
  stopifnot(!anyDuplicated(blocks$SEQN))
}
w(hdl, "rosenbaum_itos_hdl.csv")
w(bp, "rosenbaum_itos_bp.csv")
w(peri, "rosenbaum_itos_peri.csv")
w(binge, "rosenbaum_itos_binge.csv")
w(blocks, "rosenbaum_itos_blocks.csv")
# Matched-pair differences for the Noether test ----------------------------
set.seed(1)
w(data.frame(d = round(rnorm(1000) + 0.5, 6)), "rosenbaum_itos_pairs.csv")
