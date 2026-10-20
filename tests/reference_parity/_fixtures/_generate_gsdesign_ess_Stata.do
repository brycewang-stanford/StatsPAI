* Stata 18 gsdesign expected sample sizes (r(ESS0), r(ESS1)) for the
* sp.group_sequential parity test (test_group_sequential_stata_parity.py).
* Sample sizes are fractional and the tolerances tightened as for gsbounds.
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_gsdesign_ess_Stata.do
version 18
clear all
set linesize 255
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/gsdesign_ess_Stata.json", write replace
file write `fh' "[" _n
local o "nfractional intpointsscale(60) infotol(1e-12)"
local first 1
foreach spec in ///
    "10 15, sd(12) onesided power(0.9) efficacy(errobfleming)" ///
    "10 15, sd(12) onesided power(0.9) efficacy(obfleming) nlooks(4)" ///
    "10 15, sd(12) power(0.9) efficacy(pocock) nlooks(3)" ///
    "10 15, sd(12) onesided power(0.9) efficacy(errobfleming) futility(errpocock)" ///
    "10 15, sd(12) onesided power(0.9) efficacy(errobfleming) futility(errpocock, binding)" ///
    "10 15, sd(12) onesided power(0.9) efficacy(obfleming) futility(pocock)" ///
    "10 15, sd(12) onesided power(0.9) efficacy(obfleming) futility(pocock, binding)" ///
    "10 15, sd(12) power(0.9) efficacy(errobfleming) futility(errpocock)" ///
    "10 15, sd(12) power(0.9) efficacy(errobfleming) futility(errpocock, binding)" ///
    "10 15, sd(12) power(0.9) efficacy(obfleming) futility(obfleming)" ///
    "10 15, sd(12) onesided power(0.9) futility(errpocock)" ///
    "10 15, sd(12) onesided power(0.9) futility(errpocock, binding)" ///
    "10 15, sd(12) onesided power(0.9) efficacy(errobfleming) futility(errpocock) nlooks(4)" ///
    "10 15, sd(12) onesided power(0.8) efficacy(kdemets(2)) futility(kdemets(3)) information(1 3 4 10)" ///
    "10 15, sd(12) onesided power(0.8) efficacy(hsdecani(-2)) futility(hsdecani(1)) nlooks(5)" ///
    "10 5, sd(12) onesided power(0.9) efficacy(wtsiatis(0.25)) futility(wtsiatis(0.1)) nlooks(3)" {
    quietly gsdesign onemean `spec' `o'
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `"  {"cmd": "gsdesign onemean `spec'", "direction": "`r(direction)'", "binding": "' %1.0f (r(binding)) `", "N_fixedfrac": "' %25.16e (r(N_fixedfrac)) `", "N_max": "' %25.16e (r(N_max)) `", "info_ratio": "' %25.16e (r(info_ratio)) `", "ESS0": "' %25.16e (r(ESS0)) `", "ESS1": "' %25.16e (r(ESS1)) "}"
}
file write `fh' _n "]" _n
file close `fh'
