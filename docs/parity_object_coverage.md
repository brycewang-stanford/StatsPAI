# Reference-claim audit: reported objects vs. pinned evidence

StatsPAI 1.22.0. 9 estimators probed on `sp.datasets.mpdta()`.

- reported objects: **65**
- of those, carrying a reference value to compare against: **38**  ("pinned" = the archive has a number for this object; whether it *agrees* is the tolerance registry's question, not this table's)
- unpinned: **27**, of which **22** belong to a function whose documentation names a runnable reference somewhere

The second count is the softer one and is reported as such: naming a reference in a docstring is not the same as promising parity on the particular object in the row. The first two counts need no such judgement, which is why the headline rests on them.

| Function | Object | Pinned sides | Verdict | Docs name a reference |
| --- | --- | --- | --- | --- |
| `callaway_santanna` | headline ATT | R, Stata | pinned | Stata |
| `callaway_santanna` | headline SE | R, Stata | pinned | Stata |
| `callaway_santanna` | event-study coefficients (post) | R, Stata | pinned | Stata |
| `callaway_santanna` | event-study SEs (post) | R, Stata | pinned | Stata |
| `callaway_santanna` | pre-treatment leads | R, Stata | pinned | Stata |
| `callaway_santanna` | pre-treatment lead SEs | R, Stata | pinned | Stata |
| `callaway_santanna` | joint pre-trend test | -- | unpinned | Stata |
| `callaway_santanna` | cohort (group) ATT vector | R, Stata | pinned | Stata |
| `callaway_santanna` | cohort (group) ATT SEs | R, Stata | pinned | Stata |
| `callaway_santanna` | calendar-time ATT vector | R, Stata | pinned | Stata |
| `callaway_santanna` | calendar-time ATT SEs | R, Stata | pinned | Stata |
| `sun_abraham` | headline ATT | R, Stata | pinned | Stata |
| `sun_abraham` | headline SE | -- | unpinned | Stata |
| `sun_abraham` | event-study coefficients (post) | R, Stata | pinned | Stata |
| `sun_abraham` | event-study SEs (post) | R, Stata | pinned | Stata |
| `sun_abraham` | pre-treatment leads | R, Stata | pinned | Stata |
| `sun_abraham` | pre-treatment lead SEs | R, Stata | pinned | Stata |
| `sun_abraham` | joint pre-trend test | -- | unpinned | Stata |
| `did_imputation` | headline ATT | R, Stata | pinned | Stata |
| `did_imputation` | headline SE | R, Stata | pinned | Stata |
| `did_imputation` | event-study coefficients (post) | R, Stata | pinned | Stata |
| `did_imputation` | event-study SEs (post) | R, Stata | pinned | Stata |
| `did_imputation` | pre-treatment leads | Stata | pinned | Stata |
| `did_imputation` | pre-treatment lead SEs | Stata | pinned | Stata |
| `did_imputation` | joint pre-trend test | -- | unpinned | Stata |
| `gardner_did` | headline ATT | R, Stata | pinned | did2s |
| `gardner_did` | headline SE | R, Stata | pinned | did2s |
| `gardner_did` | event-study coefficients (post) | -- | unpinned | did2s |
| `gardner_did` | event-study SEs (post) | -- | unpinned | did2s |
| `gardner_did` | pre-treatment leads | -- | unpinned | did2s |
| `gardner_did` | pre-treatment lead SEs | -- | unpinned | did2s |
| `gardner_did` | joint pre-trend test | -- | unpinned | did2s |
| `wooldridge_did` | headline ATT | -- | unpinned | etwfe |
| `wooldridge_did` | headline SE | -- | unpinned | etwfe |
| `wooldridge_did` | event-study coefficients (post) | -- | unpinned | etwfe |
| `wooldridge_did` | event-study SEs (post) | -- | unpinned | etwfe |
| `wooldridge_did` | pre-treatment leads | -- | unpinned | etwfe |
| `wooldridge_did` | pre-treatment lead SEs | -- | unpinned | etwfe |
| `wooldridge_did` | joint pre-trend test | -- | unpinned | etwfe |
| `etwfe` | headline ATT | R, Stata | pinned | Stata |
| `etwfe` | headline SE | R, Stata | pinned | Stata |
| `etwfe` | event-study coefficients (post) | -- | unpinned | Stata |
| `etwfe` | event-study SEs (post) | -- | unpinned | Stata |
| `etwfe` | joint pre-trend test | -- | unpinned | Stata |
| `stacked_did` | headline ATT | R, Stata | pinned | -- |
| `stacked_did` | headline SE | -- | unpinned | -- |
| `stacked_did` | event-study coefficients (post) | R, Stata | pinned | -- |
| `stacked_did` | event-study SEs (post) | R, Stata | pinned | -- |
| `stacked_did` | pre-treatment leads | R, Stata | pinned | -- |
| `stacked_did` | pre-treatment lead SEs | R, Stata | pinned | -- |
| `stacked_did` | joint pre-trend test | -- | unpinned | -- |
| `lp_did` | headline ATT | -- | unpinned | -- |
| `lp_did` | headline SE | -- | unpinned | -- |
| `lp_did` | event-study coefficients (post) | R, Stata | pinned | -- |
| `lp_did` | event-study SEs (post) | R, Stata | pinned | -- |
| `lp_did` | pre-treatment leads | R, Stata | pinned | -- |
| `lp_did` | pre-treatment lead SEs | R, Stata | pinned | -- |
| `lp_did` | joint pre-trend test | -- | unpinned | -- |
| `event_study` | headline ATT | -- | unpinned | Stata |
| `event_study` | headline SE | -- | unpinned | Stata |
| `event_study` | event-study coefficients (post) | R, Stata | pinned | Stata |
| `event_study` | event-study SEs (post) | R, Stata | pinned | Stata |
| `event_study` | pre-treatment leads | R, Stata | pinned | Stata |
| `event_study` | pre-treatment lead SEs | R, Stata | pinned | Stata |
| `event_study` | joint pre-trend test | -- | unpinned | Stata |
