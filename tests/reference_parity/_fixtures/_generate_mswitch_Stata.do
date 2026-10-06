* Stata 18 reference values for tests/reference_parity/test_mswitch_parity.py.
* Reads the committed mswitch.csv and writes mswitch_Stata_1.json .. _4.json; run from
* this directory. Every model is iterated to nrtolerance(1e-10): with numerical
* derivatives Stata does not always get below 1e-12 and then never stops.
clear all
import delimited "mswitch.csv", clear asdouble
tsset t

mata:
void jmat(real scalar fh, string scalar key, real matrix M, real scalar last)
{
    real scalar i, j
    fput(fh, sprintf(`"    "%s": ["', key))
    for (i = 1; i <= rows(M); i++) {
        fwrite(fh, "      [")
        for (j = 1; j <= cols(M); j++) {
            fwrite(fh, M[i, j] >= . ? "null" : strtrim(sprintf("%24.17e", M[i, j])))
            if (j < cols(M)) fwrite(fh, ", ")
        }
        fput(fh, i < rows(M) ? "]," : "]")
    }
    fput(fh, last ? "    ]" : "    ],")
}

void jcase(real scalar fh, string scalar key, string scalar cmd, real scalar last)
{
    string rowvector nm
    real scalar j
    nm = st_matrixcolstripe("e(b)")
    fput(fh, sprintf(`"  "%s": {"', key))
    fput(fh, sprintf(`"    "cmd": "mswitch %s","', cmd))
    fwrite(fh, `"    "names": ["')
    for (j = 1; j <= rows(nm); j++) {
        fwrite(fh, sprintf(`""%s:%s""', nm[j, 1], nm[j, 2]))
        if (j < rows(nm)) fwrite(fh, ", ")
    }
    fput(fh, "],")
    fput(fh, sprintf(`"    "ll": %24.17e,"', st_numscalar("e(ll)")))
    fput(fh, sprintf(`"    "n_obs": %g,"', st_numscalar("e(N)")))
    jmat(fh, "b", st_matrix("e(b)"), 0)
    jmat(fh, "V", st_matrix("e(V)"), 0)
    jmat(fh, "uncprob", st_matrix("e(uncprob)"), 0)
    jmat(fh, "prob", st_matrix("TP"), 0)
    jmat(fh, "prob_se", st_matrix("TS"), 0)
    jmat(fh, "prob_ci", st_matrix("TC"), 0)
    jmat(fh, "duration", st_matrix("DU"), 0)
    jmat(fh, "predicted", st_data(., "po*", "touse"), 0)
    jmat(fh, "filtered", st_data(., "pf*", "touse"), 0)
    jmat(fh, "smoothed", st_data(., "ps*", "touse"), 0)
    jmat(fh, "yhat_default", st_data(., "yd", "touse")', 0)
    jmat(fh, "yhat_filter", st_data(., "yh", "touse")', 1)
    fput(fh, last ? "  }" : "  },")
}
end

capture program drop onecase
program define onecase
    args key cmd last
    mswitch `cmd' nrtolerance(1e-10) iterate(100)
    capture drop po* pf* ps* yh yd touse
    gen byte touse = e(sample)
    predict double po*, pr
    predict double pf*, pr smethod(filter)
    predict double ps*, pr smethod(smooth)
    predict double yd, yhat
    predict double yh, yhat smethod(filter)
    estat transition
    matrix TP = r(prob)
    matrix TS = r(se)
    local k = e(states)
    matrix TC = J(`k' * `k', 2, .)
    forvalues s = 1/`=`k' * `k'' {
        matrix TC[`s', 1] = r(ci`s')
    }
    estat duration
    matrix DU = J(4, `k', .)
    forvalues s = 1/`k' {
        matrix DU[1, `s'] = r(d`s')
        matrix DU[2, `s'] = r(se`s')
        matrix DU[3, `s'] = r(ci`s')'
    }
    mata: jcase(fh, "`key'", `"`cmd'"', `last')
end

* Four files, so that each stays under the repository's 500 KB limit for
* added files; the test merges them.
capture program drop openjson
program define openjson
    args part
    capture erase "mswitch_Stata_`part'.json"
    mata: fh = fopen("mswitch_Stata_`part'.json", "w")
    mata: fput(fh, "{")
end
capture program drop closejson
program define closejson
    mata: fput(fh, "}")
    mata: fclose(fh)
end

openjson 1
onecase mean    "dr y_mean,"                                0
onecase var     "dr y_var, varswitch"                       0
onecase x       "dr y_x x,"                                 0
onecase z       "dr y_z, switch(z)"                         0
onecase k3      "dr y3, states(3)"                          1
closejson
openjson 2
onecase common  "dr y_z, switch(z, noconstant) constant"    0
onecase drlag   "dr y_ar L.y_ar,"                           0
onecase drlagsw "dr y_ar, switch(L.y_ar L2.y_ar)"           0
onecase varrob  "dr y_var, varswitch vce(robust)"           1
closejson
openjson 3
onecase ar1     "ar y_ar, ar(1)"                            0
onecase ar2     "ar y_ar, ar(1/2)"                          0
onecase ar4     "ar y_ar, ar(1/4)"                          0
onecase ar1x    "ar y_x x, ar(1)"                           0
onecase ar1z    "ar y_z, ar(1) switch(z)"                   1
closejson
openjson 4
onecase arsw    "ar y_ar, ar(1) arswitch"                   0
onecase arvar   "ar y_var, ar(1) varswitch"                 0
onecase ar2var  "ar y_var, ar(1/2) varswitch"               0
* three states with ar(1/2) does not converge in Stata on this series
onecase k3ar1   "ar y3_ar, states(3) ar(1)"                  1
closejson
