use cattaneo2, clear
ttest bweight, by(mbsmoke)
regress bweight mbsmoke
regress bweight mbsmoke mage medu mmarried fbaby prenatal1
teffects ra (bweight mage medu mmarried fbaby prenatal1) (mbsmoke)
teffects ra (bweight mage medu mmarried fbaby prenatal1) (mbsmoke), atet
teffects nnmatch (bweight mage medu) (mbsmoke)
teffects nnmatch (bweight mage medu) (mbsmoke), atet
teffects nnmatch (bweight mage medu) (mbsmoke), nneighbor(4) biasadj(mage medu)
teffects nnmatch (bweight mage medu) (mbsmoke), ematch(mmarried fbaby)
teffects nnmatch (bweight mage medu) (mbsmoke), metric(euclidean)
teffects psmatch (bweight) (mbsmoke mage medu mmarried fbaby prenatal1)
teffects psmatch (bweight) (mbsmoke mage medu mmarried fbaby prenatal1), atet
teffects psmatch (bweight) (mbsmoke mage medu mmarried fbaby prenatal1, probit)
teffects psmatch (bweight) (mbsmoke mage medu mmarried fbaby prenatal1), nneighbor(3)
teffects ipw (bweight) (mbsmoke mage medu mmarried fbaby prenatal1)
teffects ipw (bweight) (mbsmoke mage medu mmarried fbaby prenatal1), atet
teffects ipwra (bweight mage medu mmarried fbaby prenatal1) (mbsmoke mage medu mmarried fbaby prenatal1)
teffects aipw (bweight mage medu mmarried fbaby prenatal1) (mbsmoke mage medu mmarried fbaby prenatal1)
tebalance summarize
teffects overlap
logit mbsmoke mage medu mmarried fbaby prenatal1
