use lutkepohl2, clear
tsset qtr
regress dln_consump L(0/4).dln_inc
test L1.dln_inc L2.dln_inc L3.dln_inc L4.dln_inc
lincom dln_inc + L1.dln_inc + L2.dln_inc + L3.dln_inc + L4.dln_inc
regress dln_consump dln_inc L.dln_consump
nlcom _b[dln_inc]/(1-_b[L.dln_consump])
estat durbinalt
generate z0 = dln_inc + L1.dln_inc + L2.dln_inc + L3.dln_inc + L4.dln_inc
generate z1 = L1.dln_inc + 2*L2.dln_inc + 3*L3.dln_inc + 4*L4.dln_inc
generate z2 = L1.dln_inc + 4*L2.dln_inc + 9*L3.dln_inc + 16*L4.dln_inc
regress dln_consump z0 z1 z2
var dln_inv dln_inc dln_consump if qtr<=tq(1978q4), lags(1/2)
varstable
varlmar
varwle
vargranger
varnorm
irf create order1, set(irf_xu, replace) step(8)
irf table irf oirf fevd, impulse(dln_inc) response(dln_consump)
fcast compute f_, step(4)
matrix A = (1,0,0 \ .,1,0 \ .,.,1)
matrix B = (.,0,0 \ 0,.,0 \ 0,0,.)
svar dln_inv dln_inc dln_consump if qtr<=tq(1978q4), lags(1/2) aeq(A) beq(B)
irf create svar1, set(irf_xu2, replace) step(8)
irf table sirf, impulse(dln_inc) response(dln_consump)
matrix C = (.,0,0 \ .,.,0 \ .,.,.)
svar dln_inv dln_inc dln_consump if qtr<=tq(1978q4), lags(1/2) lreq(C)
lpirf dln_inv dln_inc dln_consump if qtr<=tq(1978q4), lags(1/2) step(8)
