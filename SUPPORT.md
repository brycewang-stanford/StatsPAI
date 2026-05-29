# Getting Support for StatsPAI

Thanks for using StatsPAI. Pick the channel that fits what you need.

## Asking a question or discussing usage

For "**how do I**", "**which estimator should I use**", "**why does my result differ from Stata / R**", or general usage discussion:

- **GitHub Discussions** — <https://github.com/brycewang-stanford/StatsPAI/discussions>
  - Best place for open-ended questions, methodological discussion, and Q&A that may help future users.
  - Please search existing threads first — many common questions may already be answered.

When asking, please include:

1. StatsPAI version: `python -c "import statspai as sp; print(sp.__version__)"`
2. Python version + OS
3. Minimal reproducible example (data + code, ideally using one of the bundled `sp.datasets`)
4. What you expected vs. what you got

## Reporting a bug

If something is **wrong** — a crash, an incorrect number, a documentation error, or behavior that contradicts the docstring / paper:

- **GitHub Issues** — <https://github.com/brycewang-stanford/StatsPAI/issues>
- Use the **Bug report** template.
- Numerical-correctness issues (estimator returns a different value than Stata / R / the source paper) are top priority. Please cite the reference implementation and version when relevant.

## Requesting a feature or new estimator

- **GitHub Issues** — use the **Feature request / new estimator** template.
- For new estimators, please link to the source paper (with DOI) and, if possible, a reference implementation in Stata, R, or another package.

## Security issues

If you believe you have found a security vulnerability (e.g., a code path that allows arbitrary code execution from data input), please **do not** open a public issue. Email the maintainer directly:

- **Email:** brycew6m@stanford.edu

## Commercial support and collaboration

StatsPAI is developed by the team behind [CoPaper.AI](https://copaper.ai) (Stanford REAP). For applied-econometrics consulting, custom-estimator collaborations, or institutional licensing questions, email brycew6m@stanford.edu.

## Response time expectations

StatsPAI is maintained primarily by a small team. We aim to triage issues within 1–2 weeks. For urgent numerical-correctness bugs or breakage in core estimators (`did`, `iv`, `rd`, `synth`, `dml`, `panel`), we will respond faster.
