* Stata 18 egen on egen_data.csv, for sp.egen (tests/test_egen_public.py).
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_egen_Stata.do
* Every result is stored as double; rows are written in id order.
version 18
clear all
set type double
import delimited using "tests/reference_parity/_fixtures/egen_data.csv", clear asdouble
egen double mean_g = mean(x), by(g)
egen double count_g = count(x), by(g)
egen double sd_g = sd(x), by(g)
egen double median_g = median(x), by(g)
egen double p25_g = pctile(x), p(25) by(g)
egen double iqr_g = iqr(x), by(g)
egen double skew_g = skew(x), by(g)
egen double kurt_g = kurt(x), by(g)
egen double mad_g = mad(x), by(g)
egen double mdev_g = mdev(x), by(g)
egen double min_gh = min(x), by(g h)
egen double max_gh = max(x), by(g h)
egen double total_gh = total(x), by(g h)
egen double totalm_gh = total(x), by(g h) missing
egen double mode_a = mode(a), by(g)
egen double minmode_a = mode(a), by(g) minmode
egen double maxmode_a = mode(a), by(g) maxmode
egen double mean_if = mean(x) if w == 1, by(g)
egen double rank_g = rank(x), by(g)
egen double field_g = rank(x), by(g) field
egen double track_g = rank(x), by(g) track
egen double std_x = std(x)
egen double std_if = std(x) if w == 1
egen double seq_x = seq(), from(2) to(4) block(2)
egen double group_gh = group(g h)
egen double groupm_gh = group(g h), missing
egen double tag_gh = tag(g h)
egen double rowtotal_abc = rowtotal(a b c)
egen double rowtotalm_abc = rowtotal(a b c), missing
egen double rowmean_abc = rowmean(a b c)
egen double rowmin_abc = rowmin(a b c)
egen double rowmax_abc = rowmax(a b c)
egen double rowsd_abc = rowsd(a b c)
egen double rownonmiss_abc = rownonmiss(a b c)
egen double rowmiss_abc = rowmiss(a b c)
egen double rowfirst_abc = rowfirst(a b c)
egen double rowlast_abc = rowlast(a b c)
egen double anycount_ac = anycount(a c), values(0 3/4)
egen double anymatch_ac = anymatch(a c), values(5)
egen double cut_at = cut(x), at(0, 6, 9, 12, 30)
egen double cut_icodes = cut(x), at(0, 6, 9, 12, 30) icodes
egen double cut_group = cut(x), group(4)
sort id
format _all %24.17g
export delimited using "tests/reference_parity/_fixtures/egen_Stata.csv", replace datafmt
