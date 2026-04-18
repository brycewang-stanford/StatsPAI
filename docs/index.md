# StatsPAI

**Python causal inference toolkit** — reimplemented from the original
papers, zero upstream DID-package runtime dependencies.

```python
import statspai as sp

rpt = sp.cs_report(
    data, y='y', g='g', t='t', i='id',
    n_boot=500, random_state=0,
    save_to='~/study/cs_v1',   # writes .txt / .md / .tex / .xlsx / .png
)
```

## What's inside

StatsPAI brings the R `did` / `HonestDiD` + Python `csdid` / `differences`
stack into a single consistent Python API, with every core algorithm
reimplemented from the original paper (no wrappers).

### Staggered Difference-in-Differences

- **Callaway & Sant'Anna (2021)** — `sp.callaway_santanna()` with
  DR / IPW / REG estimators, never-treated or not-yet-treated control,
  `anticipation=δ`, and `panel=False` for repeated cross-sections
  (with optional covariate regression adjustment).
- **Four aggregations + Mammen uniform bands** — `sp.aggte()` with
  `type='simple' | 'dynamic' | 'group' | 'calendar'` and simultaneous
  sup-t confidence bands (Callaway–Sant'Anna 2021 §4.2).
- **Sensitivity analysis** — `sp.honest_did()` and `sp.breakdown_m()`
  (Rambachan–Roth 2023).  Both accept either a `callaway_santanna()`
  result or an `aggte(type='dynamic')` result.
- **One-call report** — `sp.cs_report()` runs the full pipeline under a
  single seed and produces a structured `CSReport` with Markdown /
  LaTeX / Excel export and a 2×2 summary figure (`.plot()`).
- **Sun & Abraham (2021) IW** with Liang–Zeger cluster-robust SEs.
- **Borusyak–Jaravel–Spiess (2024)** imputation estimator +
  `sp.bjs_pretrend_joint()` cluster-bootstrap joint Wald pre-trend test.
- **de Chaisemartin–D'Haultfoeuille** with joint placebo Wald and
  average cumulative effect (dCDH 2024).

### Broader toolkit

OLS / GLM / quantile / survival / panel / spatial / RD / IV / matching /
synthetic control / DML / DeepIV / neural causal / causal forests /
policy learning / conformal causal / TMLE / Mendelian randomization /
sensitivity analysis / reporting via `outreg2` + `modelsummary`.

See the left-hand navigation for guides.

## Installation

```bash
pip install statspai
```

Optional extras for exports:

```bash
pip install openpyxl matplotlib          # Excel + figures
```

## Citation

If you use StatsPAI in research, please cite the underlying papers
implemented by each estimator (every `CausalResult` carries a
`.cite()` method that returns the correct BibTeX entry) and this
package:

```bibtex
@software{statspai,
  author  = {Wang, Biaoyue},
  title   = {StatsPAI: Python Causal Inference Toolkit},
  year    = {2026},
  url     = {https://github.com/brycewang-stanford/StatsPAI}
}
```
