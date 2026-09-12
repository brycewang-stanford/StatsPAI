* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_decomp_R_parity.py
* (Stata-side block). Requires Stata 18 and the SSC packages b1x2
* (Gelbach's own command), ineqdeco (Jenkins) and descogini
* (Lopez-Feldman), installed into a private ado directory -- never PLUS:
*
*   net set ado "<dir>" ; adopath ++ "<dir>" ; ssc install b1x2 ...
*
* Run _generate_decomp_data.py first (writes decomp_stata.csv), then
*   do _generate_decomp_stata.do     (from this directory)
* Writes decomp_Stata.json (numbers as %23.16e: %g drops the leading zero,
* which JSON forbids).
*
* Conventions pinned here
* -----------------------
* * b1x2 ..., robust: each x2 variable is its own group; the covariance is
*   the stacked-system sandwich (_robust, default multiplier n/(n-1)) plus
*   the part due to the long-regression coefficients and both cross terms.
*   Plain b1x2 is the homoskedastic analogue. The __TC variance b1x2
*   reports equals 1'V1 over the per-variable contributions.
* * ineqdeco reports GE(-1..2) with between/within; its Gini is the plug-in
*   (population) Gini.
* * descogini uses the plug-in Gini with ranks _n/N (no ties in this data,
*   so the Gini correlation equals the midpoint-rank version).
* ---------------------------------------------------------------------------
version 18
set more off
import delimited using "decomp_stata.csv", clear asdouble

tempname fh
file open `fh' using "decomp_Stata.json", write replace text
file write `fh' "{" _n

* ---- Gelbach ----------------------------------------------------------------
quietly b1x2 log_wage, x1all(education) x2all(experience tenure union) ///
    x1only(education) x2delta(g1=experience : g2=tenure : g3=union) robust
matrix b = e(b)
matrix V = e(V)
file write `fh' `"  "gelbach": {"delta": ["' %23.16e (b[1,1]) ", " %23.16e (b[1,2]) ", " %23.16e (b[1,3]) "], "
file write `fh' `""total": "' %23.16e (b[1,4]) ", "
file write `fh' `""V": [["' %23.16e (V[1,1]) ", " %23.16e (V[1,2]) ", " %23.16e (V[1,3]) "], ["
file write `fh' %23.16e (V[2,1]) ", " %23.16e (V[2,2]) ", " %23.16e (V[2,3]) "], ["
file write `fh' %23.16e (V[3,1]) ", " %23.16e (V[3,2]) ", " %23.16e (V[3,3]) "]], "
file write `fh' `""V_total": "' %23.16e (V[4,4]) "}," _n

* ---- Gelbach, homoskedastic -------------------------------------------------
quietly b1x2 log_wage, x1all(education) x2all(experience tenure union) ///
    x1only(education) x2delta(g1=experience : g2=tenure : g3=union)
matrix V = e(V)
file write `fh' `"  "gelbach_homoskedastic": {"V": [["' %23.16e (V[1,1]) ", " %23.16e (V[1,2]) ", " %23.16e (V[1,3]) "], ["
file write `fh' %23.16e (V[2,1]) ", " %23.16e (V[2,2]) ", " %23.16e (V[2,3]) "], ["
file write `fh' %23.16e (V[3,1]) ", " %23.16e (V[3,2]) ", " %23.16e (V[3,3]) "]], "
file write `fh' `""V_total": "' %23.16e (V[4,4]) "}," _n

* ---- ineqdeco by female ------------------------------------------------------
quietly ineqdeco wage, bygroup(female)
file write `fh' `"  "ineqdeco": {"'
foreach s in ge0 ge1 ge2 within_ge0 within_ge1 within_ge2 between_ge0 between_ge1 between_ge2 gini {
    file write `fh' `""`s'": "' %23.16e (r(`s')) ", "
}
file write `fh' `""N": "' %9.0f (r(N)) "}," _n

* ---- descogini ----------------------------------------------------------------
gen double total = wage + capital + transfer
quietly descogini total wage capital transfer
file write `fh' `"  "descogini": {"gtotal": "' %23.16e (gtotal)
foreach v in wage capital transfer {
    file write `fh' `", "`v'": {"S": "' %23.16e (s`v') `", "G": "' %23.16e (g`v') ///
        `", "R": "' %23.16e (r`v') `", "share": "' %23.16e (sg`v') "}"
}
file write `fh' "}," _n
file write `fh' `"  "provenance": {"Stata": "`c(stata_version)'", "edition": "MP"}"' _n
file write `fh' "}" _n
file close `fh'
display "wrote decomp_Stata.json"
