* Fetch the datasets the chapter do-files use and save local copies.
* They ship with Stata (sysuse) or come from the Stata Press site (webuse)
* and are not redistributed here. senate.dta is written by README step 1.
foreach d in auto nlsw88 {
    sysuse `d', clear
    save `d', replace
}
foreach d in womenwk union3 nlswork cattaneo2 lutkepohl2 wpi1 grunfeld mroz klein {
    webuse `d', clear
    save `d', replace
}
