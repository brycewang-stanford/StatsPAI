* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_panel_stata_parity.py
*
* Requires: Stata 18 (xtgls, xtlogit, xtprobit are all official).
* Run:      stata -b do _generate_panel_stata.do   (from this directory)
*
* Conventions this fixture exists to pin
* --------------------------------------
* 1. `xtgls` is TWO-STEP by default. The variance parameters come from the
*    OLS residuals, one GLS solve follows, and that is the answer.
*    Re-estimating the variances from the GLS residuals and iterating is a
*    different estimator and is the `igls` option. StatsPAI iterated
*    unconditionally while claiming equivalence to the plain command.
*
* 2. `xtlogit, re` and `xtprobit, re` include a constant. StatsPAI's RE
*    fitters built their design from the regressor list alone -- correct for
*    conditional FE logit, where the constant is differenced out, and wrong
*    for RE, where it is a parameter. Both estimates and the log-likelihood
*    are recorded so the test can check convergence rather than a single
*    tolerance: StatsPAI integrates with non-adaptive Gauss-Hermite while
*    Stata's default is adaptive, so agreement improves with n_quadrature
*    and that improvement is the evidence the likelihood is right.
*
* Data comes from the committed CSV so both sides read the same bytes.
* ---------------------------------------------------------------------------
version 18
clear all
set type double
import delimited "panel_stata_data.csv", clear case(preserve)
xtset id t

tempname fh
file open `fh' using "panel_stata.json", write replace text
file write `fh' "{" _n

quietly xtgls y x1 x2, panels(hetero)
file write `fh' `"  "xtgls": {"' _n
file write `fh' `"    "x1": "' %21.16e (_b[x1]) `","' _n
file write `fh' `"    "x2": "' %21.16e (_b[x2]) `","' _n
file write `fh' `"    "cons": "' %21.16e (_b[_cons]) `","' _n
file write `fh' `"    "se_x1": "' %21.16e (_se[x1]) `","' _n
file write `fh' `"    "se_x2": "' %21.16e (_se[x2]) _n
file write `fh' `"  },"' _n

quietly xtgls y x1 x2, panels(hetero) igls
file write `fh' `"  "xtgls_igls": {"' _n
file write `fh' `"    "x1": "' %21.16e (_b[x1]) `","' _n
file write `fh' `"    "se_x1": "' %21.16e (_se[x1]) _n
file write `fh' `"  },"' _n

quietly xtlogit ybin x1 x2, re
file write `fh' `"  "xtlogit_re": {"' _n
file write `fh' `"    "x1": "' %21.16e (_b[x1]) `","' _n
file write `fh' `"    "x2": "' %21.16e (_b[x2]) `","' _n
file write `fh' `"    "cons": "' %21.16e (_b[_cons]) `","' _n
file write `fh' `"    "se_x1": "' %21.16e (_se[x1]) `","' _n
file write `fh' `"    "sigma_u": "' %21.16e (e(sigma_u)) `","' _n
file write `fh' `"    "rho": "' %21.16e (e(rho)) `","' _n
file write `fh' `"    "ll": "' %21.16e (e(ll)) _n
file write `fh' `"  },"' _n

quietly xtprobit ybin x1 x2, re
file write `fh' `"  "xtprobit_re": {"' _n
file write `fh' `"    "x1": "' %21.16e (_b[x1]) `","' _n
file write `fh' `"    "x2": "' %21.16e (_b[x2]) `","' _n
file write `fh' `"    "cons": "' %21.16e (_b[_cons]) `","' _n
file write `fh' `"    "se_x1": "' %21.16e (_se[x1]) `","' _n
file write `fh' `"    "sigma_u": "' %21.16e (e(sigma_u)) `","' _n
file write `fh' `"    "rho": "' %21.16e (e(rho)) `","' _n
file write `fh' `"    "ll": "' %21.16e (e(ll)) _n
file write `fh' `"  },"' _n

file write `fh' `"  "_meta": {"' _n
file write `fh' `"    "stata_version": "' `"""' "`c(stata_version)'" `"""' `","' _n
file write `fh' `"    "generated": "' `"""' "`c(current_date)'" `"""' _n
file write `fh' `"  }"' _n
file write `fh' "}" _n
file close `fh'
display "wrote panel_stata.json"
