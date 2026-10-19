* Stata 18 gsbounds stopping boundaries for the sp.group_sequential parity test.
* Integration and information tolerances are tightened (intpointsscale(60),
* infotol(1e-12)); at the defaults Stata stops its search for the maximum
* information at infotol(1e-6), which moves futility bounds by up to 3e-5.
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_gsbounds_Stata.do
version 18
clear all
set linesize 255
capture program drop dumpgs
program define dumpgs
    args fh first
    tempname B
    matrix `B' = r(bounds)
    local cn : colfullnames `B'
    if !`first' file write `fh' "," _n
    file write `fh' `"  {"cmd": "`r(cmdline)'", "info_ratio": "' %25.16e (r(info_ratio)) `", "cols": "`cn'", "rows": ["'
    forvalues i = 1/`=rowsof(`B')' {
        if `i' > 1 file write `fh' ", "
        file write `fh' "["
        forvalues j = 1/`=colsof(`B')' {
            if `j' > 1 file write `fh' ", "
            if missing(`B'[`i',`j']) file write `fh' "null"
            else file write `fh' %25.16e (`B'[`i',`j'])
        }
        file write `fh' "]"
    }
    file write `fh' "]}"
end
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/gsbounds_Stata.json", write replace
file write `fh' "[" _n
quietly gsbounds, efficacy(obfleming) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 1
quietly gsbounds, efficacy(obfleming) nlooks(5) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) information(1 3 4 10) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(pocock) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(pocock) nlooks(5) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(pocock) information(1 3 4 10) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(wtsiatis(0.25)) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(wtsiatis(0.25)) nlooks(5) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(wtsiatis(0.25)) information(1 3 4 10) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(wtsiatis(-0.3)) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(wtsiatis(-0.3)) nlooks(5) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(wtsiatis(-0.3)) information(1 3 4 10) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) nlooks(5) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) information(1 3 4 10) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errpocock) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errpocock) nlooks(5) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errpocock) information(1 3 4 10) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(kdemets(2)) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(kdemets(2)) nlooks(5) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(kdemets(2)) information(1 3 4 10) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(kdemets(0.75)) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(kdemets(0.75)) nlooks(5) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(kdemets(0.75)) information(1 3 4 10) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(-4)) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(-4)) nlooks(5) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(-4)) information(1 3 4 10) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(1)) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(1)) nlooks(5) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(1)) information(1 3 4 10) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) nlooks(5) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) information(1 3 4 10) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(pocock) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(pocock) nlooks(5) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(pocock) information(1 3 4 10) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(wtsiatis(0.25)) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(wtsiatis(0.25)) nlooks(5) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(wtsiatis(0.25)) information(1 3 4 10) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(wtsiatis(-0.3)) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(wtsiatis(-0.3)) nlooks(5) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(wtsiatis(-0.3)) information(1 3 4 10) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) nlooks(5) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) information(1 3 4 10) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errpocock) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errpocock) nlooks(5) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errpocock) information(1 3 4 10) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(kdemets(2)) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(kdemets(2)) nlooks(5) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(kdemets(2)) information(1 3 4 10) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(kdemets(0.75)) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(kdemets(0.75)) nlooks(5) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(kdemets(0.75)) information(1 3 4 10) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(-4)) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(-4)) nlooks(5) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(-4)) information(1 3 4 10) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(1)) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(1)) nlooks(5) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(1)) information(1 3 4 10) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) information(0.2 0.5 1) lower intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) information(0.2 0.5 1) lower intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(-4)) information(0.2 0.5 1) lower intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) nlooks(4) alpha(0.025) power(0.9) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) nlooks(4) upper alpha(0.025) power(0.9) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) nlooks(4) alpha(0.025) power(0.9) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) nlooks(4) upper alpha(0.025) power(0.9) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(-2)) nlooks(4) alpha(0.025) power(0.9) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(-2)) nlooks(4) upper alpha(0.025) power(0.9) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) nlooks(4) alpha(0.1) power(0.85) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) nlooks(4) upper alpha(0.1) power(0.85) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) nlooks(4) alpha(0.1) power(0.85) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) nlooks(4) upper alpha(0.1) power(0.85) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(-2)) nlooks(4) alpha(0.1) power(0.85) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(hsdecani(-2)) nlooks(4) upper alpha(0.1) power(0.85) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) futility(errpocock) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) futility(errpocock, binding) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) futility(errpocock) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) futility(errpocock, binding) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) futility(errobfleming) information(0.3 0.5 0.8 1) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) futility(errobfleming, binding) information(0.3 0.5 0.8 1) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) futility(errobfleming) information(0.3 0.5 0.8 1) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) futility(errobfleming, binding) information(0.3 0.5 0.8 1) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(kdemets(3)) futility(hsdecani(-2)) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(kdemets(3)) futility(hsdecani(-2), binding) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(kdemets(3)) futility(hsdecani(-2)) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(kdemets(3)) futility(hsdecani(-2), binding) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errpocock) futility(kdemets(2)) information(0.3 0.5 0.8 1) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errpocock) futility(kdemets(2), binding) information(0.3 0.5 0.8 1) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errpocock) futility(kdemets(2)) information(0.3 0.5 0.8 1) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errpocock) futility(kdemets(2), binding) information(0.3 0.5 0.8 1) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) futility(pocock) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) futility(pocock, binding) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) futility(pocock) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) futility(pocock, binding) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) futility(obfleming) information(0.3 0.5 0.8 1) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) futility(obfleming, binding) information(0.3 0.5 0.8 1) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) futility(obfleming) information(0.3 0.5 0.8 1) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(obfleming) futility(obfleming, binding) information(0.3 0.5 0.8 1) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(pocock) futility(wtsiatis(0.1)) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(pocock) futility(wtsiatis(0.1), binding) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(pocock) futility(wtsiatis(0.1)) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(pocock) futility(wtsiatis(0.1), binding) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(wtsiatis(0.25)) futility(obfleming) information(0.3 0.5 0.8 1) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(wtsiatis(0.25)) futility(obfleming, binding) information(0.3 0.5 0.8 1) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(wtsiatis(0.25)) futility(obfleming) information(0.3 0.5 0.8 1) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(wtsiatis(0.25)) futility(obfleming, binding) information(0.3 0.5 0.8 1) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, futility(errpocock) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, futility(errpocock, binding) nlooks(3) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, futility(errpocock) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, futility(errpocock, binding) nlooks(3) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, futility(pocock) information(0.3 0.5 0.8 1) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, futility(pocock, binding) information(0.3 0.5 0.8 1) upper intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, futility(pocock) information(0.3 0.5 0.8 1) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, futility(pocock, binding) information(0.3 0.5 0.8 1) intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) futility(errpocock) nlooks(3) lower intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
quietly gsbounds, efficacy(errobfleming) futility(errobfleming) nlooks(3) lower intpointsscale(60) infotol(1e-12)
dumpgs `fh' 0
file write `fh' _n "]" _n
file close `fh'
