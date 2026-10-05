* Run every chapter with a text log. Start Stata in a folder that holds
* these do-files and the files of the book (net from
* http://www.stata-press.com/data/kkh4/, package daus4).
set more off
set linesize 255
set graphics off
adopath ++ "`c(pwd)'"
capture log close _all
foreach c in 01 03 05 07 08 09 10 11 12 {
    clear all
    set seed 12345
    set varabbrev on
    log using ch`c'.log, text replace
    do ch`c'.do, nostop
    log close
}
