* Frozen-reference generator: Stata 18 stteffects (ra, wra, ipw, ipwra; ATE and
* ATET; weibull / exponential / lnormal outcome, probit treatment, lnormal
* censoring) for tests/reference_parity/test_survival_causal_parity.py.
* Run from the repository root:
*   stata-mp -b do tests/reference_parity/_generate_stteffects_stata.do
* Columns: b1/se1 = ATE (or ATET), b2/se2 = POmean of the control level,
* cov12 = their covariance.
version 18
set type double
local fx "tests/reference_parity/_fixtures"
import delimited using "`fx'/stteffects_data.csv", clear
stset time, failure(event)
tempname fh
file open `fh' using "`fx'/stteffects_Stata.csv", write replace
file write `fh' "tag,b1,se1,b2,se2,cov12,N" _n
capture program drop dump
program define dump
    args fh tag
    matrix b = e(b)
    matrix V = e(V)
    file write `fh' "`tag'," %21.0g (b[1,1]) "," %21.0g (sqrt(V[1,1])) "," %21.0g (b[1,2]) "," %21.0g (sqrt(V[2,2])) "," %21.0g (V[1,2]) "," (e(N)) _n
end
stteffects ra (x1 x2) (a)
dump `fh' ra_ate_weibull
stteffects ra (x1 x2) (a), atet
dump `fh' ra_atet_weibull
stteffects ra (x1 x2, exponential) (a)
dump `fh' ra_ate_exponential
stteffects ra (x1 x2, lnormal) (a)
dump `fh' ra_ate_lognormal
stteffects wra (x1 x2) (a) (x1)
dump `fh' wra_ate_weibull
stteffects ipw (a x1 x2) (x1)
dump `fh' ipw_ate_weibull
stteffects ipw (a x1 x2) (x1), atet
dump `fh' ipw_atet_weibull
stteffects ipw (a x1 x2, probit) (x1, lnormal)
dump `fh' ipw_ate_probit_lognormal
stteffects ipwra (x1 x2) (a x1 x2)
dump `fh' ipwra_lac_ate_weibull
stteffects ipwra (x1 x2) (a x1 x2), atet
dump `fh' ipwra_lac_atet_weibull
stteffects ipwra (x1 x2) (a x1 x2) (x1)
dump `fh' ipwra_wac_ate_weibull
stteffects ipwra (x1 x2) (a x1 x2) (x1), atet
dump `fh' ipwra_wac_atet_weibull
file close `fh'
