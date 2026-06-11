# Security Policy

## Supported Versions

| Version  | Supported          |
| -------- | ------------------ |
| 1.17.x   | :white_check_mark: |
| 1.16.x   | :white_check_mark: |
| < 1.16   | :x:                |

## Reporting a Vulnerability

Please report suspected security vulnerabilities **privately** to
**brycew6m@stanford.edu** with the subject line `[StatsPAI SECURITY]`.
Include:

- The affected version(s) (`python -c "import statspai; print(statspai.__version__)"`)
- A minimal proof-of-concept or reproduction steps
- The impact you believe the issue has (e.g., arbitrary code execution via
  untrusted model files, path traversal in export helpers)
- A suggested patch, if you have one (optional)

**Do not open a public GitHub issue for security reports.** We will
acknowledge receipt within 72 hours and aim to ship a patch release within
14 days for confirmed issues. Credit is given in the CHANGELOG unless you
ask to remain anonymous.

## Scope notes

StatsPAI is a statistical computing library. The following are *in scope*:

- Code execution triggered by loading untrusted data files through
  `sp.datasets` / IO helpers
- Path traversal or arbitrary file overwrite in export methods
  (`.to_latex()`, `.to_word()`, `.to_excel()`, report writers)
- Injection through formula parsing or LLM-integration entry points
  (`sp.llm_dag_propose`, `sp.causal_text`)
- Vulnerabilities in the bundled Rust extension (`statspai_hdfe`)

Out of scope: numerical disagreements with reference implementations
(report those as regular bugs — we take them just as seriously, but they
are tracked publicly), and vulnerabilities in optional third-party
dependencies themselves (report upstream; we will bump pins).
