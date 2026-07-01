#!/usr/bin/env Rscript
# Frozen reference for sp.odds_ratio / relative_risk / risk_difference /
# mantel_haenszel vs canonical epidemiological closed-form formulas computed
# in base R (the same Woolf / Katz-log / Wald / Robins-Breslow-Greenland
# estimators epiR::epi.2by2 and Stata epitab implement). No CRAN packages —
# the formulas are unambiguous arithmetic on 2x2 cell counts.
# Regenerate: Rscript tests/reference_parity/_generate_epi_R.R
suppressPackageStartupMessages(library(jsonlite))
z <- qnorm(0.975)
a<-30; b<-70; c<-20; d<-80                       # single 2x2 table

# Odds ratio (Woolf logit)
or <- a*d/(b*c); se_or <- sqrt(1/a+1/b+1/c+1/d)
OR <- list(estimate=or, se_log=se_or,
           ci_lo=exp(log(or)-z*se_or), ci_hi=exp(log(or)+z*se_or),
           p=2*(1-pnorm(abs(log(or)/se_or))))
# Relative risk (Katz-log)
r1<-a/(a+b); r0<-c/(c+d); rr<-r1/r0
se_rr<-sqrt(1/a-1/(a+b)+1/c-1/(c+d))
RR<-list(estimate=rr, se_log=se_rr,
         ci_lo=exp(log(rr)-z*se_rr), ci_hi=exp(log(rr)+z*se_rr),
         p=2*(1-pnorm(abs(log(rr)/se_rr))))
# Risk difference (Wald)
rd<-r1-r0; se_rd<-sqrt(r1*(1-r1)/(a+b)+r0*(1-r0)/(c+d))
RD<-list(estimate=rd, se=se_rd, ci_lo=rd-z*se_rd, ci_hi=rd+z*se_rd,
         p=2*(1-pnorm(abs(rd/se_rd))))
# Mantel-Haenszel OR over two strata (RBG se)
tabs<-list(c(a=30,b=70,c=20,d=80), c(a=40,b=60,c=25,d=75))
num<-0;den<-0;Pse<-0;Qse<-0;Rse<-0;Sse<-0
for(t in tabs){n<-sum(t)
  num<-num+t["a"]*t["d"]/n; den<-den+t["b"]*t["c"]/n
  P<-(t["a"]+t["d"])/n; Q<-(t["b"]+t["c"])/n
  R<-t["a"]*t["d"]/n; S<-t["b"]*t["c"]/n
  Pse<-Pse+P*R; Qse<-Qse+P*S+Q*R; Rse<-Rse+R; Sse<-Sse+S; Qse2<-0
  Qse2<-Qse2+Q*S}
mhor<-num/den
# Robins-Breslow-Greenland variance of log(OR_MH)
Pr<-0;PSQR<-0;QS<-0;Rsum<-0;Ssum<-0
for(t in tabs){n<-sum(t)
  Pi<-(t["a"]+t["d"])/n; Qi<-(t["b"]+t["c"])/n
  Ri<-t["a"]*t["d"]/n; Si<-t["b"]*t["c"]/n
  Pr<-Pr+Pi*Ri; PSQR<-PSQR+Pi*Si+Qi*Ri; QS<-QS+Qi*Si; Rsum<-Rsum+Ri; Ssum<-Ssum+Si}
var_log<-Pr/(2*Rsum^2)+PSQR/(2*Rsum*Ssum)+QS/(2*Ssum^2)
se_mh<-sqrt(var_log)
MH<-list(estimate=unname(mhor), se_log=unname(se_mh),
         ci_lo=unname(exp(log(mhor)-z*se_mh)), ci_hi=unname(exp(log(mhor)+z*se_mh)))
# Prevalence ratio (Katz-log, same estimator family as RR)
PR<-list(estimate=rr, se_log=se_rr,
         ci_lo=exp(log(rr)-z*se_rr), ci_hi=exp(log(rr)+z*se_rr))
# Number needed to treat = 1/RD (estimate only; CI convention differs when the
# risk-difference CI crosses zero, so we pin only the point estimate)
NNT<-list(estimate=1/rd)
# Incidence rate ratio + conditional-binomial exact CI
e1<-40; pt1<-100; e2<-20; pt2<-100
irr<-(e1/pt1)/(e2/pt2)
bt<-binom.test(e1, e1+e2)$conf.int
IRR<-list(estimate=irr, rate_exposed=e1/pt1, rate_unexposed=e2/pt2,
          ci_lo=(bt[1]/(1-bt[1]))*(pt2/pt1), ci_hi=(bt[2]/(1-bt[2]))*(pt2/pt1))
writeLines(toJSON(list(table=list(a=a,b=b,c=c,d=d), OR=OR, RR=RR, RD=RD,
           MH=MH, PR=PR, NNT=NNT, IRR=IRR, strata=tabs,
           provenance=list(r_version=R.version.string,
           generated_by="tests/reference_parity/_generate_epi_R.R")),
           auto_unbox=TRUE, digits=16, pretty=TRUE),
           "tests/reference_parity/_fixtures/epi_R.json")
cat("wrote epi_R.json: OR",or,"RR",rr,"RD",rd,"MH",unname(mhor),
    "PR",rr,"NNT",1/rd,"IRR",irr,"\n")
