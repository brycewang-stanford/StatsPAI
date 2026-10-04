# ---------------------------------------------------------------------------
# Answer key for tests/external_parity/test_ding_first_course.py
#
# Peng Ding, "A First Course in Causal Inference" (2024), replication files
# at https://doi.org/10.7910/DVN/ZX3VEV. This script reruns the deterministic
# part of every chapter's R program (the simulations and the Monte Carlo
# randomization tests are left out) and stores the numbers. The functions
# are the book's own, shortened. Neither the programs nor the data are
# redistributed here.
#
# Requires: R with Matching, car, sandwich, foreign, rdrobust, sensitivitymw,
#           mlbench and jsonlite.
# Run:      STATSPAI_DING_DIR=/path/to/dataverse_ZX3VEV \
#               Rscript tests/external_parity/ding_first_course_reference.R
#
# It writes data/ding_first_course_R.json next to itself and, into
# $STATSPAI_DING_DIR/_statspai/, the few tables the Python side cannot build
# from the dataverse files alone: Matching::lalonde, the matched pairs of
# chapter 19 and the model matrices of the JOBS II analyses.
# ---------------------------------------------------------------------------
suppressMessages({library(Matching); library(car); library(sandwich); library(foreign); library(jsonlite); library(rdrobust); library(sensitivitymw)})
D = paste0(normalizePath(Sys.getenv("STATSPAI_DING_DIR")), "/")
Sdir = paste0(D, "_statspai"); dir.create(Sdir, showWarnings = FALSE)
out = list()
## ch1
dat <- read.table(paste0(D,"cps1re74.csv"), header = TRUE)
dat$u74 <- as.numeric(dat$re74==0); dat$u75 <- as.numeric(dat$re75==0)
out$ch1_lm_all = summary(lm(re78 ~ ., data = dat))$coef[2, ]
out$ch1_lm_treat = summary(lm(re78 ~ treat, data = dat))$coef[2, ]
resume = read.csv(paste0(D,"resume.csv")); tb = table(resume$race, resume$call)
ft = fisher.test(tb); out$ch1_fisher = c(p=ft$p.value, or=unname(ft$estimate), lo=ft$conf.int[1], hi=ft$conf.int[2]); out$ch1_tab = as.vector(tb)
## ch3
data(lalonde); z = lalonde$treat; y = lalonde$re78
out$ch3 = c(t_eq=unname(t.test(y[z==1],y[z==0],var.equal=TRUE)$statistic), t_uneq=unname(t.test(y[z==1],y[z==0])$statistic),
  W=unname(wilcox.test(y[z==1],y[z==0])$statistic), D=unname(ks.test(y[z==1],y[z==0])$statistic),
  p_eq=t.test(y[z==1],y[z==0],var.equal=TRUE)$p.value, p_uneq=t.test(y[z==1],y[z==0])$p.value,
  p_W=wilcox.test(y[z==1],y[z==0])$p.value, p_D=ks.test(y[z==1],y[z==0])$p.value)
write.csv(lalonde, paste0(Sdir,"/lalonde_matching.csv"), row.names=FALSE)
## ch4
n1=sum(z); n0=length(z)-n1
ols = lm(y~z)
out$ch4 = c(tau=mean(y[z==1])-mean(y[z==0]), se=sqrt(var(y[z==1])/n1+var(y[z==0])/n0), ols_se=summary(ols)$coef[2,2],
  hc3=sqrt(hccm(ols)[2,2]), hc0=sqrt(hccm(ols,type="hc0")[2,2]), hc2=sqrt(hccm(ols,type="hc2")[2,2]))
## ch5
Neyman_SRE = function(z, y, x){ xl=unique(x); K=length(xl); P=T_=V=rep(0,K)
 for(k in 1:K){zk=z[x==xl[k]]; yk=y[x==xl[k]]; P[k]=length(zk)/length(z); T_[k]=mean(yk[zk==1])-mean(yk[zk==0]); V[k]=var(yk[zk==1])/sum(zk)+var(yk[zk==0])/sum(1-zk)}
 c(sum(P*T_), sqrt(sum(P^2*V)))}
penn = read.table(paste0(D,"Penn46_ascii.txt")); out$ch5_penn = Neyman_SRE(penn$treatment, log(penn$duration), penn$quarter)
stat_SRE = function(z,y,x){xl=unique(x);K=length(xl);P=T_=W=rep(0,K); for(k in 1:K){zk=z[x==xl[k]];yk=y[x==xl[k]];P[k]=length(zk)/length(z);T_[k]=mean(yk[zk==1])-mean(yk[zk==0]);W[k]=wilcox.test(yk[zk==1],yk[zk==0])$statistic}; c(sum(P*T_),sum(W/P))}
out$ch5_penn_stat = stat_SRE(penn$treatment, log(penn$duration), penn$quarter)
chong = read.dta(paste0(D,"chong.dta"))
dp = subset(chong, treatment != "Soccer Player", select=c("treatment","gradesq34","class_level","anemic_base_re")); dp$z = (dp$treatment=="Physician")
out$ch5_chong_S = with(dp, Neyman_SRE(z, gradesq34, class_level))
out$ch5_chong_SPS = with(dp, Neyman_SRE(z, gradesq34, interaction(class_level, anemic_base_re)))
## ch6
star = read.dta(paste0(D,"star.dta")); a2 = subset(star, control==1|sfsp==1)
y6 = a2$GPA_year1; y6 = ifelse(is.na(y6), mean(y6,na.rm=TRUE), y6); z6=a2$sfsp; x6 = scale(a2[,c("female","gpa0")])
f0=lm(y6~z6); f1=lm(y6~z6*x6)
out$ch6 = c(unadj=unname(coef(f0)[2]), se_unadj=sqrt(hccm(f0,type="hc2")[2,2]), adj=unname(coef(f1)[2]), se_adj=sqrt(hccm(f1,type="hc2")[2,2]))
## ch7
diffd = c(6.125,-8.375,1,2,0.75,2.875,3.5,5.125,1.75,3.625,7,3,9.375,7.5,-6)
MP = function(i, n.pairs){a=2^((n.pairs-1):0); b=2*a; 2*sapply(i-1,function(x) as.integer((x%%b)>=a))-1}
t.ran = sapply(1:2^15, function(x) sum(MP(x,15)*abs(diffd)))/15
out$ch7_darwin_p = mean(t.ran >= mean(diffd))
dx = c(12.9,12.0,54.6,60.6,15.1,12.3,56.5,55.5,16.8,17.2,75.2,84.8,15.8,18.9,75.6,101.9,13.9,15.3,55.3,70.6,14.5,16.6,59.3,78.4,17.0,16.0,87.0,84.2,15.8,20.1,73.7,108.6)
dx = matrix(dx,8,4,byrow=TRUE); diffx=dx[,2]-dx[,1]; diffy=dx[,4]-dx[,3]
unadj=summary(lm(diffy~1))$coef; adj=summary(lm(diffy~diffx))$coef
tr = sapply(1:2^8,function(x){zm=MP(x,8); dy=diffy*zm; dxx=diffx*zm; c(summary(lm(dy~1))$coef[1,3], summary(lm(dy~dxx))$coef[1,3])})
out$ch7_tv = c(unadj=unadj[1,1:3], adj=adj[1,1:3], p_unadj=mean(abs(tr[1,])>=abs(unadj[1,3])), p_adj=mean(abs(tr[2,])>=abs(adj[1,3])))
## ch8 deterministic stats
cre_stat = function(z,y,x){f=lm(y~z); tn=f$coef[2]; sn=sqrt(vcovHC(f,type="HC2")[2,2]); x=scale(x); fl=lm(y~z+x+z*x); tl=fl$coef[2]; sl=sqrt(vcovHC(fl,type="HC2")[2,2]); c(tn,sn,tn/sn,tl,sl,tl/sl,length(z))}
sre_stat = function(z,y,block,x){x=as.matrix(x); s=sapply(unique(block),function(k) cre_stat(z[block==k],y[block==k],x[block==k,])); nn=length(z)
 c(sum(s[1,]*s[7,])/nn, sqrt(sum(s[2,]^2*s[7,]^2)/nn^2), sum(s[4,]*s[7,])/nn, sqrt(sum(s[5,]^2*s[7,]^2)/nn^2))}
ds = subset(chong, treatment!="Physician", select=c("treatment","gradesq34","class_level","anemic_base_re")); ds$z=(ds$treatment=="Soccer Player"); ds$x=(ds$anemic_base_re=="Yes")
dp$x = (dp$anemic_base_re=="Yes")
out$ch8_soccer = with(ds, sre_stat(z, gradesq34, class_level, x)); out$ch8_phys = with(dp, sre_stat(z, gradesq34, class_level, x))
out$ch8_soccer_k = sapply(1:5,function(k) with(subset(ds,class_level==k), cre_stat(z,gradesq34,x)))
write.csv(data.frame(treatment=as.character(chong$treatment), gradesq34=chong$gradesq34, class_level=chong$class_level, anemic=as.character(chong$anemic_base_re)), paste0(Sdir,"/chong.csv"), row.names=FALSE)
## ch9
linestimator = function(Z,Y,X){X=scale(X);n=dim(X)[1];p=dim(X)[2];lr=lm(Y~Z*X);est=coef(lr)[2];vehw=hccm(lr)[2,2];inter=coef(lr)[(p+3):(2*p+2)];vs=vehw+sum(inter*(cov(X)%*%inter))/n;c(est,sqrt(vehw),sqrt(vs))}
x9 = as.matrix(lalonde[,c("age","educ","black","hisp","married","nodegr","re74","re75")])
out$ch9_lalonde = unname(linestimator(z, y, x9))
## ch11/12/13
nh = read.csv(paste0(D,"nhanes_bmi.csv"))[,-1]; zn=nh$School_meal; yn=nh$BMI; xn=scale(as.matrix(nh[,-c(1,2)]))
DiM=lm(yn~zn); Fi=lm(yn~zn+xn); Li=lm(yn~zn+xn+zn*xn)
out$ch11_reg = c(coef(DiM)[2],hccm(DiM)[2,2]^.5,coef(Fi)[2],hccm(Fi)[2,2]^.5,coef(Li)[2],hccm(Li)[2,2]^.5)
ps = glm(zn~xn,family=binomial)$fitted.values
out$ch11_strat = sapply(c(5,10,20,50,80),function(nn){q=quantile(ps,(1:(nn-1))/nn); st=cut(ps,breaks=c(0,q,1),labels=1:nn); Neyman_SRE(zn,yn,st)})
ipw.est=function(z,y,x,tr=c(0,1)){p=glm(z~x,family=binomial)$fitted.values;p=pmax(tr[1],pmin(tr[2],p)); c(mean(z*y/p-(1-z)*y/(1-p)), mean(z*y/p)/mean(z/p)-mean((1-z)*y/(1-p))/mean((1-z)/(1-p)))}
out$ch11_ipw = sapply(list(c(0,1),c(.01,.99),c(.05,.95),c(.1,.9)),function(t) ipw.est(zn,yn,xn,t))
OS_est=function(z,y,x,tr=c(0,1)){p=glm(z~x,family=binomial)$fitted.values;p=pmax(tr[1],pmin(tr[2],p));o1=glm(y~x,weights=z)$fitted.values;o0=glm(y~x,weights=1-z)$fitted.values
 reg=mean(o1-o0); yt=mean(z*y/p);yc=mean((1-z)*y/(1-p));ot=mean(z/p);oc=mean((1-z)/(1-p)); dr=reg+mean(z*(y-o1)/p)-mean((1-z)*(y-o0)/(1-p)); c(reg,yt-yc,yt/ot-yc/oc,dr)}
out$ch12 = OS_est(zn,yn,xn); out$ch12_trunc = OS_est(zn,yn,xn,c(.1,.9))
ATT.est=function(z,y,x,U=1){nn=length(z);nn1=sum(z);p=pmin(U,glm(z~x,family=binomial)$fitted.values);od=p/(1-p);o0=glm(y~x,weights=1-z)$fitted.values
 r0=lm(y~z+x)$coef[2]; r=mean(y[z==1])-mean(o0[z==1]); i0=mean(y[z==1])-mean(od*(1-z)*y)*nn/nn1; i1=mean(y[z==1])-mean(od*(1-z)*y)/mean(od*(1-z)); dr=r-mean(od*(1-z)*(y-o0))*nn/nn1; unname(c(r0,r,i0,i1,dr))}
out$ch13 = ATT.est(zn,yn,xn); out$ch13_trunc = ATT.est(zn,yn,xn,.9)
## ch15
m1 = Match(Y=y,Tr=z,X=x9,BiasAdjust=TRUE); out$ch15_exp = c(m1$est, m1$se, m1$se.standard)
yo=dat$re78; zo=dat$treat; xo=as.matrix(dat[,c("age","educ","black","hispan","married","nodegree","re74","re75","u74","u75")])
m2 = Match(Y=yo,Tr=zo,X=xo,BiasAdjust=TRUE); m3 = Match(Y=yo,Tr=zo,X=xo)
out$ch15_obs_adj = c(m2$est,m2$se,m2$se.standard, length(m2$index.treated)); out$ch19_obs = c(m3$est,m3$se,m3$se.standard)
dd = yo[m2$index.treated]-yo[m2$index.control]; out$ch15_pairs = summary(lm(dd~1))$coef[1,]
dxm = xo[m2$index.treated,]-xo[m2$index.control,]; out$ch15_pairs_adj = summary(lm(dd~dxm))$coef[1,]
## ch17
NC = read.table(paste0(D,"NCHS2003.txt"),header=TRUE,sep="\t")
yl = glm(PTbirth~ageabove35+mar+smoking+drinking+somecollege+hispanic+black+nativeamerican+asian,data=NC,family=binomial); lo=summary(yl)$coef[2,1:2]
out$ch17 = c(est=exp(lo[1]), lower=exp(lo[1]-1.96*lo[2]))
## ch18
OS_est_sa=function(z,y,x,e1=1,e0=1){p=glm(z~x,family=binomial)$fitted.values;o1=glm(y~x,weights=z)$fitted.values;o0=glm(y~x,weights=1-z)$fitted.values
 reg=mean(z*y)+mean((1-z)*o1/e1)-mean(z*o0*e0)-mean((1-z)*y); w1=p+(1-p)/e1;w0=p*e0+(1-p); i0=mean(z*y*w1/p)-mean((1-z)*y*w0/(1-p)); i1=mean(z*y*w1/p)/mean(z/p)-mean((1-z)*y*w0/(1-p))/mean((1-z)/(1-p)); aug=o1/p/e1+o0*e0/(1-p); c(reg,i0,i1,i0+mean((z-p)*aug))}
E=c(1/2,1/1.7,1/1.5,1/1.3,1,1.3,1.5,1.7,2); out$ch18 = outer(1:9,1:9,Vectorize(function(i,j) OS_est_sa(zn,yn,xn,E[i],E[j])[4]))
out$ch18_all = OS_est_sa(zn,yn,xn,1.5,1/1.3)
## ch19
dm = cbind(yo[m3$index.treated], yo[m3$index.control]); out$ch19_npairs = nrow(dm)
G = seq(1,1.4,0.001); P = sapply(G,function(g) senmw(dm,gamma=g,method="t")$pval); out$ch19_gammastar = G[which(P>=0.05)[1]]
out$ch19_p = sapply(c(1,1.1,1.2,1.3),function(g) senmw(dm,gamma=g,method="t")$pval)
write.csv(dm, paste0(Sdir,"/lalonde_pairs.csv"), row.names=FALSE)
data(erpcp); out$ch19_erpcp = sapply(c(1,2,3,4),function(g) senmw(erpcp,gamma=g,method="t")$pval); write.csv(erpcp, paste0(Sdir,"/erpcp.csv"), row.names=FALSE)
## ch20
house = read.csv(paste0(D,"house.csv"))[,-1]; r = rdrobust(house$y, house$x); out$ch20_rd = cbind(r$coef, r$se, r$ci); out$ch20_bw = r$bws
house$z = (house$x>=0); out$ch20_local = sapply(c(0.05,0.25,0.5,1),function(h){g=lm(y~z+x+z*x,data=house,subset=(abs(x)<=h)); c(coef(g)[2],confint(g,'zTRUE'))})
## ch21
jobs = read.csv(paste0(D,"jobsdata.csv")); Z=jobs$treat; Dd=jobs$comply; Y=jobs$job_seek
X = model.matrix(lm(treat~sex+age+marital+nonwhite+educ+income,data=jobs))[,-1]
IV_Wald=function(Z,D,Y){tD=mean(D[Z==1])-mean(D[Z==0]);tY=mean(Y[Z==1])-mean(Y[Z==0]);c(tD,tY,tY/tD)}
e=IV_Wald(Z,Dd,Y); A=Y-Dd*e[3]; out$ch21_wald = c(e, sqrt(var(A[Z==1])/sum(Z)+var(A[Z==0])/sum(1-Z))/abs(e[1]))
Xs=scale(X); out$ch21_lin = unname(c(lm(Dd~Z+Xs+Z*Xs)$coef[2], lm(Y~Z+Xs+Z*Xs)$coef[2]))
FARci=function(Z,D,Y,L,U,g){r=seq(L,U,g);p=sapply(r,function(t){Yt=Y-t*D;ta=mean(Yt[Z==1])-mean(Yt[Z==0]);v=var(Yt[Z==1])/sum(Z)+var(Yt[Z==0])/sum(1-Z);(1-pnorm(abs(ta/sqrt(v))))*2});range(r[p>=0.05])}
out$ch21_far = FARci(Z,Dd,Y,-0.2,0.4,0.001)
FARciX=function(Z,D,Y,X,L,U,g){r=seq(L,U,g);X=scale(X);p=sapply(r,function(t){l=linestimator(Z,Y-t*D,X);(1-pnorm(abs(l[1]/l[3])))*2});range(r[p>=0.05])}
out$ch21_farx = FARciX(Z,Dd,Y,X,-0.2,0.4,0.001)
write.csv(cbind(jobs[,c("treat","comply","job_seek","depress2")], X), paste0(Sdir,"/jobs_X.csv"), row.names=FALSE)
## ch23
card = read.csv(paste0(D,"card1995.csv")); Yc=card$lwage; Dc=card$educ; Zc=card$nearc4
Xc=as.matrix(card[,c("exper","expersq","black","south","smsa","reg661","reg662","reg663","reg664","reg665","reg666","reg667","reg668","smsa66")])
Dh=lm(Dc~Zc+Xc)$fitted.values; ts=lm(Yc~Dh+Xc); te=coef(ts)[2]; ts$residuals=as.vector(Yc-cbind(1,Dc,Xc)%*%coef(ts)); out$ch23_tsls=c(te, sqrt(hccm(ts,type="hc0")[2,2]))
B=seq(-0.1,0.4,0.001); PA=sapply(B,function(b){ar=lm(I(Yc-b*Dc)~Zc+Xc); (1-pnorm(abs(coef(ar)[2]/sqrt(hccm(ar)[2,2]))))*2}); out$ch23_far=c(B[which.max(PA)], range(B[PA>=0.05]))
## ch24
road=read.csv(paste0(D,"indianroad.csv")); road$runv=road$left+road$right
fr=function(h){s=subset(road,abs(runv)<=h); s$hat=lm(r2012~t+left+right,data=s)$fitted.values; tr=lm(occupation_index_andrsn~hat+left+right,data=s); tr$residuals=as.vector(s$occupation_index_andrsn-cbind(1,s$r2012,s$left,s$right)%*%coef(tr)); c(coef(tr)[2],sqrt(hccm(tr,type="hc2")[2,2]),nrow(s))}
out$ch24_road_h = sapply(c(10,40,80),fr)
rr = with(road, rdrobust(y=occupation_index_andrsn,x=runv,c=0,fuzzy=r2012)); out$ch24_road_rd = cbind(rr$coef,rr$se); out$ch24_road_bw = rr$bws
italy=read.csv(paste0(D,"italy.csv")); ri = with(italy, rdrobust(y=outcome,x=rv0,c=0,fuzzy=D)); out$ch24_italy_rd=cbind(ri$coef,ri$se); out$ch24_italy_bw = ri$bws
italy$left=pmin(italy$rv0,0); italy$right=pmax(italy$rv0,0)
fi=function(h){s=subset(italy,abs(rv0)<=h); s$hat=lm(D~Z+left+right,data=s)$fitted.values; tr=lm(outcome~hat+left+right,data=s); tr$residuals=as.vector(s$outcome-cbind(1,s$D,s$left,s$right)%*%coef(tr)); c(coef(tr)[2],sqrt(hccm(tr,type="hc2")[2,2]),nrow(s))}
out$ch24_italy_h = sapply(c(0.1,0.5,1),fi)
## ch25
bs = read.csv(paste0(D,"mr_bmisbp.csv")); bs$iv=bs$beta.outcome/bs$beta.exposure; bs$se.iv=bs$se.outcome/bs$beta.exposure; bs$se.iv1=sqrt(bs$se.outcome^2+bs$iv^2*bs$se.exposure^2)/bs$beta.exposure
fw=function(est,se){c(sum(est/se^2)/sum(1/se^2), sqrt(1/sum(1/se^2)))}
out$ch25_fw = c(fw(bs$iv,bs$se.iv), fw(bs$iv,bs$se.iv1))
out$ch25_egger0 = summary(lm(beta.outcome~0+beta.exposure,data=bs,weights=1/se.outcome^2))$coef
out$ch25_egger = summary(lm(beta.outcome~beta.exposure,data=bs,weights=1/se.outcome^2))$coef
## ch26
psw=function(Z,M,Y,X){p10=mean(M[Z==1]);p00=1-p10;ps10=glm(M~X,family=binomial,weights=Z)$fitted.values;ps00=1-ps10; c(mean(Y[Z==1&M==1])-mean(Y[Z==0]*ps10[Z==0])/p10, mean(Y[Z==1&M==0])-mean(Y[Z==0]*ps00[Z==0])/p00)}
out$ch26_psw = psw(Z,Dd,Y,X)
## ch27
BK=function(Z,M,Y,X){mr=lm(M~Z+X);a=mr$coef[2];sa=sqrt(hccm(mr)[2,2]);or=lm(Y~Z+M+X);d=or$coef[2];sd_=sqrt(hccm(or)[2,2]);b=or$coef[3];sb=sqrt(hccm(or)[3,3]);unname(c(d,b*a,sd_,sqrt(sb^2*a^2+b^2*sa^2)))}
X2 = model.matrix(lm(treat~econ_hard+depress1+sex+age+occp+marital+nonwhite+educ+income,data=jobs))[,-1]
out$ch27 = BK(jobs$treat, jobs$job_seek, jobs$depress2, X2)
write.csv(cbind(jobs[,c("treat","job_seek","depress2")], X2), paste0(Sdir,"/jobs_X2.csv"), row.names=FALSE)
## A2
suppressMessages(library(mlbench)); data(BostonHousing); of=lm(medv~.,data=BostonHousing)
out$A2 = cbind(coef(of), summary(of)$coef[,2], sqrt(diag(hccm(of,type="hc0"))), sqrt(diag(hccm(of,type="hc1"))), sqrt(diag(hccm(of,type="hc2"))), sqrt(diag(hccm(of,type="hc3"))))
BH = BostonHousing; BH$chas = as.numeric(as.character(BH$chas)); write.csv(BH, paste0(Sdir,"/boston.csv"), row.names=FALSE)
lg = glm(I(re78>0)~., family=binomial, data=lalonde); out$A2_logit = summary(lg)$coef[,1:2]
write_json(out, "tests/external_parity/data/ding_first_course_R.json", digits=NA, auto_unbox=TRUE, pretty=TRUE)
cat("done\n")
