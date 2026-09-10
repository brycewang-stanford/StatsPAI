* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_etregress_stata_parity.py
*
* Requires: Stata 18 (etregress is official; no ado to install).
* Run:      stata -b do _generate_etregress_stata.do   (from this directory)
*
* What this fixture pins, and why each piece is here
* --------------------------------------------------
* 1. `etregress` with NO options is the full-information MLE. StatsPAI's
*    `method='mle'` used to run the two-step and label it MLE, so the two
*    are recorded separately and the test asserts they DIFFER -- otherwise
*    a regression back to "two-step under an MLE label" would pass.
*
* 2. `twostep` is recorded with its standard errors, which carry Heckman's
*    correction for the estimated first stage. The uncorrected OLS standard
*    errors of the hazard-augmented regression are ~11% smaller on this
*    design, in the anti-conservative direction.
*
* 3. vce(robust) and vce(cluster) are recorded because `sp.etregress`
*    accepted `robust=` and `cluster=` as arguments and used neither.
*
* Data is generated and exported HERE rather than in Python so that both
* sides read the same bytes and the CSV carries Stata's own full-precision
* values.
* ---------------------------------------------------------------------------
version 18
clear all
set type double
set seed 20260911
set obs 2000

gen double experience   = rnormal(15, 7)
gen double education    = rnormal(13, 3)
gen byte   father_union = runiformint(0, 1)
gen byte   region       = runiformint(0, 3)
gen int    clust        = mod(_n, 50) + 1

* Two correlated errors: rho = 0.6, sigma = 2.
gen double u   = rnormal()
gen double eps = 0.6*2*u + sqrt(1 - 0.6^2)*2*rnormal()
gen byte   union = (-0.5 + 0.4*father_union + 0.15*region + u) > 0
gen double wage  = 8 + 0.3*experience + 0.5*education + 2*union + eps
drop u eps

format experience education wage %21.16e
export delimited wage experience education union father_union region clust ///
    using "etregress_data.csv", replace datafmt

tempname fh
file open `fh' using "etregress_stata.json", write replace text
file write `fh' "{" _n

* ---- 1. MLE (the default; no option) -------------------------------------
quietly etregress wage experience education, treat(union = father_union region)
file write `fh' `"  "mle": {"' _n
file write `fh' `"    "cons": "' %21.16e (_b[wage:_cons]) `","' _n
file write `fh' `"    "experience": "' %21.16e (_b[wage:experience]) `","' _n
file write `fh' `"    "education": "' %21.16e (_b[wage:education]) `","' _n
file write `fh' `"    "treat": "' %21.16e (_b[wage:1.union]) `","' _n
file write `fh' `"    "sel_cons": "' %21.16e (_b[union:_cons]) `","' _n
file write `fh' `"    "sel_father_union": "' %21.16e (_b[union:father_union]) `","' _n
file write `fh' `"    "sel_region": "' %21.16e (_b[union:region]) `","' _n
file write `fh' `"    "se_cons": "' %21.16e (_se[wage:_cons]) `","' _n
file write `fh' `"    "se_experience": "' %21.16e (_se[wage:experience]) `","' _n
file write `fh' `"    "se_education": "' %21.16e (_se[wage:education]) `","' _n
file write `fh' `"    "se_treat": "' %21.16e (_se[wage:1.union]) `","' _n
file write `fh' `"    "se_sel_father_union": "' %21.16e (_se[union:father_union]) `","' _n
file write `fh' `"    "athrho": "' %21.16e (_b[/athrho]) `","' _n
file write `fh' `"    "lnsigma": "' %21.16e (_b[/lnsigma]) `","' _n
file write `fh' `"    "rho": "' %21.16e (e(rho)) `","' _n
file write `fh' `"    "sigma": "' %21.16e (e(sigma)) `","' _n
file write `fh' `"    "lambda": "' %21.16e (e(lambda)) `","' _n
file write `fh' `"    "ll": "' %21.16e (e(ll)) _n
file write `fh' `"  },"' _n

* ---- 2. two-step ---------------------------------------------------------
quietly etregress wage experience education, treat(union = father_union region) twostep
file write `fh' `"  "twostep": {"' _n
file write `fh' `"    "cons": "' %21.16e (_b[wage:_cons]) `","' _n
file write `fh' `"    "experience": "' %21.16e (_b[wage:experience]) `","' _n
file write `fh' `"    "education": "' %21.16e (_b[wage:education]) `","' _n
file write `fh' `"    "treat": "' %21.16e (_b[wage:union]) `","' _n
file write `fh' `"    "hazard": "' %21.16e (_b[hazard:lambda]) `","' _n
file write `fh' `"    "se_cons": "' %21.16e (_se[wage:_cons]) `","' _n
file write `fh' `"    "se_experience": "' %21.16e (_se[wage:experience]) `","' _n
file write `fh' `"    "se_treat": "' %21.16e (_se[wage:union]) `","' _n
file write `fh' `"    "se_hazard": "' %21.16e (_se[hazard:lambda]) `","' _n
file write `fh' `"    "rho": "' %21.16e (e(rho)) `","' _n
file write `fh' `"    "sigma": "' %21.16e (e(sigma)) _n
file write `fh' `"  },"' _n

* ---- 3. MLE with vce(robust) --------------------------------------------
quietly etregress wage experience education, treat(union = father_union region) vce(robust)
file write `fh' `"  "mle_robust": {"' _n
file write `fh' `"    "treat": "' %21.16e (_b[wage:1.union]) `","' _n
file write `fh' `"    "se_cons": "' %21.16e (_se[wage:_cons]) `","' _n
file write `fh' `"    "se_experience": "' %21.16e (_se[wage:experience]) `","' _n
file write `fh' `"    "se_treat": "' %21.16e (_se[wage:1.union]) `","' _n
file write `fh' `"    "se_sel_father_union": "' %21.16e (_se[union:father_union]) _n
file write `fh' `"  },"' _n

* ---- 4. MLE with vce(cluster clust) -------------------------------------
quietly etregress wage experience education, treat(union = father_union region) vce(cluster clust)
file write `fh' `"  "mle_cluster": {"' _n
file write `fh' `"    "n_clusters": "' %21.16e (e(N_clust)) `","' _n
file write `fh' `"    "treat": "' %21.16e (_b[wage:1.union]) `","' _n
file write `fh' `"    "se_cons": "' %21.16e (_se[wage:_cons]) `","' _n
file write `fh' `"    "se_experience": "' %21.16e (_se[wage:experience]) `","' _n
file write `fh' `"    "se_treat": "' %21.16e (_se[wage:1.union]) `","' _n
file write `fh' `"    "se_sel_father_union": "' %21.16e (_se[union:father_union]) _n
file write `fh' `"  },"' _n

file write `fh' `"  "_meta": {"' _n
file write `fh' `"    "stata_version": "' `"""' "`c(stata_version)'" `"""' `","' _n
file write `fh' `"    "flavor": "' `"""' "`c(flavor)' `c(edition_real)'" `"""' `","' _n
file write `fh' `"    "generated": "' `"""' "`c(current_date)'" `"""' _n
file write `fh' `"  }"' _n
file write `fh' "}" _n
file close `fh'
display "wrote etregress_stata.json and etregress_data.csv"
