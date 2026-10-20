# Draft issue for R PanelMatch on three behaviours

Status: draft, not sent. Bryce posts it himself on PanelMatch's GitHub
issue tracker, as one issue or as three.

The evidence in this repository is
`tests/reference_parity/test_panel_match_R_parity.py`
(`test_forbid_reversal_drops_the_set_r_keeps`,
`test_placebo_r_truncation_at_panel_end`), fixture
`tests/reference_parity/_fixtures/panel_match_data.csv`, R output
`_fixtures/panel_match_R.json` from `_fixtures/_generate_panel_match_R.R`,
and the note on `sp.panel_match` in `docs/dev/modules/did.md` (item 3 of
its departures, the `ps.match` tie, is stated there; no committed fixture
case triggers it).

PanelMatch is GPL-3. It was used here as a black box. This draft and the
snippets rest on its documentation and outputs only. Nobody preparing it
read the package source. Keep it that way when following up.

The snippets were run on 2026-10-20 with PanelMatch 3.1.5 on R 4.5.2. The
third one is new for this draft and was checked once.

---

Title: Three behaviours in PanelMatch 3.1.5 (forbid.treatment.reversal, placebo_test near the panel end, ps.match ties)

Thank you for PanelMatch. We compared an independent implementation with
it on a synthetic panel. Matched sets, refinement weights, estimates,
analytic standard errors and balance tables agree to 1e-10 in 15 settings.
Three behaviours remain that we could not match to the method as we
understand it. Each may be intended. We would be grateful to know.

All three use this public CSV. It is a balanced panel of 60 units over 16
periods with no missing values, and treatment switches on and off.

```r
library(PanelMatch)
url <- paste0("https://raw.githubusercontent.com/brycewang-stanford/StatsPAI/",
              "6fdc22bef90d03defabfc45b5b26efd9cde60e8f/",
              "tests/reference_parity/_fixtures/panel_match_data.csv")
full <- read.csv(url)
pd <- PanelData(full, unit.id = "id", time.id = "t", treatment = "d", outcome = "y")
D <- with(full, tapply(d, list(id, t), function(v) v[1]))
Y <- with(full, tapply(y, list(id, t), function(v) v[1]))
```

### 1. forbid.treatment.reversal keeps a set whose controls all violate it

```r
pm <- PanelMatch(pd, lag = 4, refinement.method = "mahalanobis", qoi = "att",
                 lead = 0:3, size.match = 5, covs.formula = ~ x1 + x2,
                 forbid.treatment.reversal = TRUE)
viol <- sapply(names(pm$att), function(key) {
  ctrl <- as.character(pm$att[[key]]); if (length(ctrl) == 0) return(c(0, 0))
  t <- as.integer(strsplit(key, ".", fixed = TRUE)[[1]][2])
  lead_t <- as.character(t:(t + 3)); lead_t <- lead_t[lead_t %in% colnames(D)]
  c(sum(apply(D[ctrl, lead_t, drop = FALSE], 1, max) == 1), length(ctrl))
})
names(pm$att)[viol[1, ] > 0]                 # sets with a control treated in t..t+3
s <- pm$att[["15.7"]]; as.numeric(s); attr(s, "weights")
D[as.character(s), as.character(6:10)]
```

Only one set contains controls that are treated during the lead window,
and in that set all of them are. Set `15.7` keeps controls 8, 59 and 60
with weight 1/3 each. Each is untreated at t = 6 and treated at some point
in 7 to 10. No other set contains such a control. We expected this set to
be dropped, since none of its controls satisfies the restriction. All other
sets and weights agree with ours to 1e-10.

### 2. placebo_test drops Y[t-1] for treated observations near the panel end

```r
pl <- PanelMatch(pd, lag = 4, refinement.method = "mahalanobis", qoi = "att",
                 lead = 0:1, size.match = 5, covs.formula = ~ x1 + x2,
                 placebo.test = TRUE)
pt <- placebo_test(pl, pd, se.method = "conditional")
last <- max(full$t)
full_did <- trunc_did <- 0; n <- 0
for (key in names(pl$att)) {
  ctrl <- pl$att[[key]]; if (length(ctrl) == 0) next
  i <- strsplit(key, ".", fixed = TRUE)[[1]][1]
  t <- as.integer(strsplit(key, ".", fixed = TRUE)[[1]][2])
  w <- attr(ctrl, "weights"); cc <- as.character(ctrl)
  keep <- as.numeric(t + 2 <= last)
  yi <- function(tt) Y[i, as.character(tt)]; yc <- function(tt) Y[cc, as.character(tt)]
  full_did  <- full_did  + (yi(t-1) - yi(t-4)) - sum(w * (yc(t-1) - yc(t-4)))
  trunc_did <- trunc_did + (keep*yi(t-1) - yi(t-4)) - sum(w * (keep*yc(t-1) - yc(t-4)))
  n <- n + 1
}
c(placebo_t_minus_4 = pt$estimates[1], did = full_did / n, did_truncated = trunc_did / n)
```

| Quantity | Value |
| --- | --- |
| `placebo_test` estimate at t-4 | 0.561639071038252 |
| DiD from t-4 to t-1 over the matched sets | 0.427064316939891 |
| same, with Y[t-1] set to 0 when t + 2 > 16 | 0.561639071038251 |

Eight of the 61 sets have t + 2 past the last period. For those, the
estimate at t-4 behaves as if Y[t-1] were zero for the treated unit and its
controls. The panel is balanced, so Y[t-1] exists for every one of them. The estimates at t-3 and t-2,
which need no rows past the panel end, agree with ours to 1e-10, and so do
their analytic standard errors.

### 3. ps.match gives zero weight to a control with exactly the treated unit's score

We add a twin of treated unit 1 at t = 9. It has the same treatment history
before t = 9, is untreated from then on, and has the same y, x1 and x2 in
every period. Its propensity score equals the treated unit's.

```r
run <- function(df, method) {
  pd <- PanelData(df, unit.id = "id", time.id = "t", treatment = "d", outcome = "y")
  PanelMatch(pd, lag = 4, refinement.method = method, qoi = "att", lead = 0:2,
             size.match = 5, covs.formula = ~ I(lag(y, 1:2)) + x1 + x2)
}
key <- "1.9"
twin <- full[full$id == 1, ]; twin$id <- 999L; twin$d[twin$t >= 9] <- 0L
dat <- rbind(full, twin)
for (m in c("ps.match", "mahalanobis")) {
  s <- run(dat, m)$att[[key]]; w <- attr(s, "weights")
  cat(m, " twin weight:", w[as.numeric(s) == 999], "  positive weights:", sum(w > 0), "\n")
}
```

Output on our machine.

```
ps.match  twin weight: 0   positive weights: 5
mahalanobis  twin weight: 0.2   positive weights: 5
```

The twin is in the matched set in both cases. Mahalanobis refinement ranks
it first (distance 0). `ps.match` leaves it out of the five and gives the
weight to five other controls. With continuous covariates exact ties are
rare, but with discrete covariates they are common, and the closest
control would be the one excluded.
