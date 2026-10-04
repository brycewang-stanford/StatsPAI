* Run every chapter with a text log. Start Stata in this directory.
* egranger and kpss are SSC commands; they are installed into ./_ado.
set more off
set linesize 255
capture mkdir _ado
net set ado "`c(pwd)'/_ado"
adopath ++ "`c(pwd)'/_ado"
foreach pkg in egranger kpss {
    capture which `pkg'
    if _rc ssc install `pkg'
}
do _data.do
set graphics off
capture log close _all
foreach c in 01 02 03 04 06 07 08 09 10 11 12 {
    clear all
    set seed 12345
    log using ch`c'.log, text replace
    do ch`c'.do, nostop
    log close
}
