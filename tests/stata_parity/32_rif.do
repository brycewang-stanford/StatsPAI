* tests/stata_parity/32_rif.do
*
* Module 32: RIF / UQR decomposition at the median.
*   StatsPAI:  sp.rif_decomposition(..., statistic="quantile", tau=0.5)
*   R:         dineq::rif + manual Oaxaca-style decomposition
*   Stata:     audited Stata/Mata implementation of the same RIF kernel
*
* There is a community rifhdreg package for RIF regressions, but it is not
* part of the portable test-machine baseline. This file therefore records an
* explicit Stata/Mata algorithm bridge: Hmisc type-7 quantile,
* R stats::density binned Gaussian density at the quantile, groupwise
* RIF-OLS, and the reference-group-0 Oaxaca split.

version 18
clear all

do _common.do
stata_parity_init, module(32_rif)
stata_parity_open, module(32_rif)

import delimited "${STATA_PARITY_DATA}/32_rif.csv", clear case(preserve)

mata:
real scalar _sp_quantile_type7(real colvector y, real scalar tau)
{
    real scalar n, h, lo, hi, frac
    real colvector ys

    ys = sort(y, 1)
    n = rows(ys)
    if (tau <= 0) return(ys[1])
    if (tau >= 1) return(ys[n])
    h = 1 + (n - 1) * tau
    lo = floor(h)
    hi = ceil(h)
    frac = h - lo
    return((1 - frac) * ys[lo] + frac * ys[hi])
}

real scalar _sp_bw_nrd0(real colvector y)
{
    real scalar n, hi, iqr, lo, mu
    real colvector centered

    n = rows(y)
    if (n < 2) return(1)
    mu = mean(y)
    centered = y :- mu
    hi = sqrt(quadcross(centered, centered) / (n - 1))
    iqr = _sp_quantile_type7(y, 0.75) - _sp_quantile_type7(y, 0.25)
    lo = min((hi, iqr / 1.34))
    if (lo == 0 | lo >= .) lo = hi
    if (lo == 0 | lo >= .) lo = abs(y[1])
    if (lo == 0 | lo >= .) lo = 1
    return(0.9 * lo * n^(-0.2))
}

real scalar _sp_density_dineq_unit(real colvector y, real scalar point)
{
    real scalar h, ngrid, nfft, lo, up, xdelta, xpos, fx, wi
    real scalar ix, i, r, j, m0, k0, left, right, frac, pos, s
    real colvector bins, kords, kernel

    h = max((_sp_bw_nrd0(y), 1e-12))
    ngrid = 512
    nfft = 2 * ngrid
    lo = point - 4 * h
    up = point + 4 * h
    xdelta = (up - lo) / (ngrid - 1)
    bins = J(nfft, 1, 0)
    wi = 1 / rows(y)

    for (i = 1; i <= rows(y); i++) {
        xpos = (y[i] - lo) / xdelta
        ix = floor(xpos)
        fx = xpos - ix
        if (0 <= ix & ix <= ngrid - 2) {
            bins[ix + 1] = bins[ix + 1] + (1 - fx) * wi
            bins[ix + 2] = bins[ix + 2] + fx * wi
        }
        else if (ix == -1) {
            bins[1] = bins[1] + fx * wi
        }
        else if (ix == ngrid - 1) {
            bins[ix + 1] = bins[ix + 1] + (1 - fx) * wi
        }
    }

    kords = J(nfft, 1, 0)
    for (r = 1; r <= nfft; r++) {
        kords[r] = (r - 1) * (((2 * ngrid - 1) / (ngrid - 1)) * (up - lo)) / (nfft - 1)
    }
    for (r = ngrid + 2; r <= nfft; r++) {
        kords[r] = -kords[2 * ngrid + 2 - r]
    }
    kernel = exp(-0.5 :* (kords :/ h):^2) :/ (h * sqrt(2 * pi()))

    pos = 1 + (point - lo) / xdelta
    left = floor(pos)
    right = ceil(pos)
    frac = pos - left
    for (j = left; j <= right; j++) {
        m0 = j - 1
        s = 0
        for (r = 1; r <= nfft; r++) {
            k0 = (r - 1) - m0
            if (k0 < 0) k0 = k0 + nfft
            if (k0 >= nfft) k0 = k0 - nfft
            s = s + bins[r] * kernel[k0 + 1]
        }
        if (j == left) left = max((s, 0))
        else right = max((s, 0))
    }
    return((1 - frac) * left + frac * right)
}

real colvector _sp_rif_quantile_unit(real colvector y, real scalar tau)
{
    real scalar q, fq

    q = _sp_quantile_type7(y, tau)
    fq = max((_sp_density_dineq_unit(y, q), 1e-12))
    return(q :+ (tau :- (y :< q)) :/ fq)
}

void _sp_rif_bridge()
{
    real matrix data, X0, X1
    real colvector y, g, y0, y1, rif0, rif1, beta0, beta1, xbar0, xbar1
    real rowvector delta, detail
    real scalar total, explained, unexplained
    real colvector i0, i1

    data = st_data(., ("log_wage", "educ", "exper", "female"))
    y = data[, 1]
    g = data[, 4]
    i0 = selectindex(g :== 0)
    i1 = selectindex(g :== 1)

    y0 = y[i0]
    y1 = y[i1]
    X0 = J(rows(i0), 1, 1), data[i0, (2, 3)]
    X1 = J(rows(i1), 1, 1), data[i1, (2, 3)]

    rif0 = _sp_rif_quantile_unit(y0, 0.5)
    rif1 = _sp_rif_quantile_unit(y1, 0.5)
    beta0 = qrsolve(X0, rif0)
    beta1 = qrsolve(X1, rif1)

    xbar0 = (colsum(X0) :/ rows(X0))'
    xbar1 = (colsum(X1) :/ rows(X1))'
    delta = (xbar1 :- xbar0)'
    detail = delta :* beta0'

    total = mean(rif1) - mean(rif0)
    explained = detail * J(cols(detail), 1, 1)
    unexplained = total - explained

    st_numscalar("SP_total_diff", total)
    st_numscalar("SP_explained", explained)
    st_numscalar("SP_unexplained", unexplained)
    st_numscalar("SP_explained_Intercept", detail[1])
    st_numscalar("SP_explained_educ", detail[2])
    st_numscalar("SP_explained_exper", detail[3])
}
_sp_rif_bridge()
end

local n = _N
stata_parity_row, statname("total_diff") estimate(`=SP_total_diff') nobs(`n')
stata_parity_row, statname("explained") estimate(`=SP_explained') nobs(`n')
stata_parity_row, statname("unexplained") estimate(`=SP_unexplained') nobs(`n')
stata_parity_row, statname("explained_Intercept") estimate(`=SP_explained_Intercept') nobs(`n')
stata_parity_row, statname("explained_educ") estimate(`=SP_explained_educ') nobs(`n')
stata_parity_row, statname("explained_exper") estimate(`=SP_explained_exper') nobs(`n')

stata_parity_extra, key(statistic) val("quantile")
stata_parity_extra_num, key(tau) val(0.5)
stata_parity_extra, key(quantile_convention) val("dineq")
stata_parity_extra, key(stata_bridge_status) val("audited Stata/Mata algorithm bridge")
stata_parity_extra, key(stata_reference_note) val("Portable bridge because rifhdreg is not part of the baseline Stata environment; implements the dineq-compatible RIF kernel and Oaxaca split used by StatsPAI/R parity.")
stata_parity_extra, key(stata_algorithm) val("Hmisc type-7 quantile + R stats::density binned Gaussian density + groupwise RIF-OLS decomposition")

stata_parity_close, module(32_rif)
