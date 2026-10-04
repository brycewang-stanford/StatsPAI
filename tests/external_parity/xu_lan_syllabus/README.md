# A syllabus audit of `sp.stata`

徐小君、蓝嘉俊《因果推断计量经济学》(清华大学出版社, 2025) has no public
data or code. These do-files follow its table of contents instead: one file
per chapter, each running the commands that chapter's methods are taught
with, on datasets that ship with Stata. Chapter 5 (causal graphs) has no
Stata counterpart and no file.

The logs are an answer key. `scripts/stata_log_replay.py` runs every command
through `sp.stata` and compares every number Stata printed.

1. Write the one dataset Stata does not ship:

   ```bash
   python -c "import statspai as sp; sp.datasets.lee_2008_senate().to_stata('senate.dta', write_index=False)"
   ```

2. In Stata 18, from this directory: `do _master.do`. It needs `rdrobust`
   and `rddensity` (SSC) and installs `egranger` and `kpss` into `./_ado`.
3. Replay:

   ```bash
   python scripts/stata_log_replay.py tests/external_parity/xu_lan_syllabus/*.log \
       --data tests/external_parity/xu_lan_syllabus
   ```

The `.dta` and `.log` files are gitignored: the datasets are Stata's.
Findings are in `docs/dev/2026-10-05-xu-lan-causal-econometrics-review.md`.
The numbers that were added or corrected are pinned on committed synthetic
data in `tests/reference_parity/test_textbook_syllabus_stata_parity.py`.
