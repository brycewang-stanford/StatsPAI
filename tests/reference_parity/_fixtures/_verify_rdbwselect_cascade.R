# Hand-replicate rdbwselect's three-stage cascade by calling the internal
# rdrobust_bw, and assert it reproduces the package's h and b exactly.
# If this matches, the ASSEMBLY is understood and only rdrobust_bw itself
# remains to port.
suppressPackageStartupMessages({library(rdrobust)})
data(rdrobust_RDsenate); d <- rdrobust_RDsenate
d <- d[!is.na(d$margin) & !is.na(d$vote), ]

cascade <- function(y, x, c=0, p=1, kernel="triangular", variant="mserd",
                    deriv=0, scaleregul=1, nnmatch=3, vce="nn") {
  q <- p + 1
  # stdvars = FALSE is the DEFAULT: no standardization, and BWp uses the
  # unstandardized branch.  x_sd stays 1.
  x_sd <- 1
  x_iq <- unname(quantile(x, .75, type=2) - quantile(x, .25, type=2))
  BWp  <- min(sd(x), x_iq/1.349)
  C_c  <- if (kernel=="epanechnikov") 2.34 else if (kernel=="uniform") 1.843 else 2.576
  N <- length(y)
  # masspoints="adjust" is the default: N is replaced by the count of
  # UNIQUE running-variable values, and bwcheck=10 kicks in when >=20%
  # of either side is tied.
  ind_l0 <- x < c; ind_r0 <- x >= c
  M_l <- length(unique(x[ind_l0])); M_r <- length(unique(x[ind_r0])); M <- M_l + M_r
  c_bw <- C_c * BWp * M^(-1/5)
  bw_max <- max(abs(c - min(x)), abs(c - max(x)))
  c_bw <- min(c_bw, bw_max)
  mass_l <- 1 - M_l/sum(ind_l0); mass_r <- 1 - M_r/sum(ind_r0)
  if (mass_l >= 0.2 || mass_r >= 0.2) {
    Xu_l <- sort(unique(x[ind_l0]), decreasing=TRUE); Xu_r <- unique(x[ind_r0])
    bwc_l <- min(10, M_l); bwc_r <- min(10, M_r)
    c_bw <- max(c_bw, abs(Xu_l - c)[bwc_l] + 1e-08, abs(Xu_r - c)[bwc_r] + 1e-08)
  }

  ind_l <- x < c; ind_r <- x >= c
  Y_l <- y[ind_l]; X_l <- x[ind_l]; Y_r <- y[ind_r]; X_r <- x[ind_r]
  o1 <- order(X_l); Y_l <- Y_l[o1]; X_l <- X_l[o1]
  o2 <- order(X_r); Y_r <- Y_r[o2]; X_r <- X_r[o2]
  rl <- rle(as.vector(X_l)); dl <- rep(rl$lengths, rl$lengths); dil <- sequence(rl$lengths)
  rr <- rle(as.vector(X_r)); dr <- rep(rr$lengths, rr$lengths); dir_ <- sequence(rr$lengths)
  range_l <- abs(c - min(x)); range_r <- abs(c - max(x))

  # NOTE: stage 1 passes scale = 0; stages 2/3 pass scaleregul.  Passing
  # scaleregul to all three leaves h ~11% low.
  BW <- function(Y,X,o,nu,o_B,h_V,h_B,dups,dupsid,scale_)
    rdrobust:::rdrobust_bw(Y, X, T=NULL, Z=NULL, C=NULL, W=NULL, c=c, o=o, nu=nu,
      o_B=o_B, h_V=h_V, h_B=h_B, scale=scale_, vce=vce, nnmatch=nnmatch,
      kernel=kernel, dups=dups, dupsid=dupsid, covs_drop_coll=0, ginv.tol=1e-20)

  # stage 1: d_bw
  D_l <- BW(Y_l,X_l, q+1, q+1, q+2, c_bw, range_l, dl, dil, 0)
  D_r <- BW(Y_r,X_r, q+1, q+1, q+2, c_bw, range_r, dr, dir_, 0)
  d_l <- (D_l$V/(D_l$B^2 + scaleregul*D_l$R))^D_l$rate
  d_r <- (D_r$V/(D_r$B^2 + scaleregul*D_r$R))^D_r$rate
  d_d <- ((D_l$V+D_r$V)/((D_r$B-D_l$B)^2 + scaleregul*(D_r$R+D_l$R)))^D_l$rate

  # stage 2: b_bw   (mserd uses the difference form, fed d_d)
  B_l <- BW(Y_l,X_l, q, p+1, q+1, c_bw, d_d, dl, dil, scaleregul)
  B_r <- BW(Y_r,X_r, q, p+1, q+1, c_bw, d_d, dr, dir_, scaleregul)
  b_d <- ((B_l$V+B_r$V)/((B_r$B-B_l$B)^2 + scaleregul*(B_r$R+B_l$R)))^B_l$rate

  # stage 3: h_bw
  H_l <- BW(Y_l,X_l, p, deriv, q, c_bw, b_d, dl, dil, scaleregul)
  H_r <- BW(Y_r,X_r, p, deriv, q, c_bw, b_d, dr, dir_, scaleregul)
  h_d <- ((H_l$V+H_r$V)/((H_r$B-H_l$B)^2 + scaleregul*(H_r$R+H_l$R)))^H_l$rate

  c(h = x_sd*h_d, b = x_sd*b_d)
}

cat(sprintf("%-26s %10s %10s %10s %10s\n","spec","pkg h","mine h","pkg b","mine b"))
worst <- 0
for (k in c("triangular","uniform","epanechnikov")) for (p in c(1,2)) {
  ref <- rdrobust(y=d$vote, x=d$margin, c=0, p=p, kernel=k, bwselect="mserd")
  mine <- cascade(d$vote, d$margin, 0, p, k, "mserd")
  gh <- abs(mine["h"]-ref$bws[1,1])/ref$bws[1,1]; gb <- abs(mine["b"]-ref$bws[2,1])/ref$bws[2,1]
  worst <- max(worst, gh, gb)
  cat(sprintf("%-26s %10.5f %10.5f %10.5f %10.5f\n", paste0(k,"_p",p),
              ref$bws[1,1], mine["h"], ref$bws[2,1], mine["b"]))
}
cat(sprintf("\nworst relative deviation: %.3e\n", worst))
