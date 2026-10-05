# A syllabus audit of `sp.stata` on an introductory Stata text

Kohler, Kreuter and Haensch, *Data Analysis Using Stata* (4th ed., Stata
Press 2026) ships its datasets and a few do-files (`net from
http://www.stata-press.com/data/kkh4/`, package `daus4`), but the commands
of the book are typed into the running text. These do-files follow its
table of contents instead: one file per chapter, each running the commands
the chapter teaches, on the book's own datasets. Chapters 2, 4 and 6 (how
to organise do-files, Python inside Stata, graphs) have no file.

The logs are the answer key. `scripts/stata_log_replay.py` runs every
command through one `sp.stata` session and compares every number Stata
printed.

1. Put the files of `daus4` and these do-files in one folder.
2. In Stata 18, from that folder: `do _master.do`.
3. Replay:

   ```bash
   python scripts/stata_log_replay.py <folder>/ch*.log --data <folder>
   STATSPAI_KK4E_DIR=<folder> pytest tests/external_parity/test_kohler_kreuter_logs.py
   ```

Neither the datasets nor the logs are in the repository: the data are a
teaching sample of the German Socio-Economic Panel that the authors
distribute. Findings are in
`docs/dev/2026-10-05-kohler-kreuter-4e-review.md`. The numbers that were
added or corrected are pinned on committed synthetic data, with Stata 18
reference values, in
`tests/reference_parity/test_kohler_kreuter_stata_parity.py`.
