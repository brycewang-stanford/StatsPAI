# Causal Text — Family Guide (Experimental, v1.6 MVP)

> Causal estimation when the treatment is text — or when the treatment
> indicator was produced by an LLM.

## When to use

- Your treatment, confounder, or outcome lives in **unstructured text**
  (reviews, complaints, statements, articles, …).
- You used an **LLM** (or any imperfect classifier) to derive a binary
  label and you want to debias the downstream coefficient.
- You want a **defensible MVP** today, not a deep neural pipeline you'll
  spend a week debugging.

## Status

**Experimental.** Two methods ship in v1.6 — the rest of the family
(text-as-confounder via topic models, text-as-outcome, full Veitch
CEVAE) is on the v1.7 roadmap. Both estimators flag themselves with
`result.diagnostics["status"] == "experimental"`.

## Functions

| Function | Treatment type | Method |
|----------|---------------|--------|
| `sp.text_treatment_effect` | Text-derived (binary or continuous) | OLS with embedding projection as adjustment (Veitch-Wang-Blei 2020 MVP) |
| `sp.llm_annotator_correct` | Binary, LLM-labelled | Hausman-style measurement-error correction (Egami et al. 2024) |

## Decision tree

```
Treatment derived from text?
├── No, but the LABEL came from an LLM?
│   └── sp.llm_annotator_correct  ← bias correction with validation set
└── Yes, treatment is the text itself?
    └── sp.text_treatment_effect  ← embedding-projected OLS
```

## Quick start: text-as-treatment

```python
import statspai as sp
import pandas as pd

df = pd.DataFrame({
    "review_text": ["great product, will buy again", ...],
    "purchased":    [1, 0, 1, ...],   # downstream outcome
    "positive":     [1, 0, 1, ...],   # text-derived treatment
})

result = sp.text_treatment_effect(
    df,
    text_col="review_text",
    outcome="purchased",
    treatment="positive",
    n_components=20,        # hash embedder dimensionality
    embedder="hash",        # 'hash' (default), 'sbert', or callable
)

print(result.summary())
# TextTreatmentResult (experimental)
# ============================================================
#   Method            : text_treatment_effect
#   Estimand          : ATE
#   Estimate          : 0.4521
#   SE                : 0.0382
#   95% CI            : [0.3772, 0.5270]
#   Embedding dim     : 20
#   Embedder          : hash
#   Status            : experimental
```

### Embedder options

| Value | Behaviour | Dependency |
|-------|-----------|------------|
| `'hash'` (default) | Deterministic hashing-vectoriser; reproducible across runs | None |
| `'sbert'` | `sentence-transformers/all-MiniLM-L6-v2` | `pip install sentence-transformers` |
| `callable` | Your own `f(texts) -> np.ndarray` | Your responsibility |

### Caveats (text-as-treatment)

1. **Strong assumption**: all text-based confounding is captured by the
   embedding projection. The MVP uses a single shared low-dimensional
   space — the full Veitch et al. (2020) recipe uses a CEVAE with
   separate treatment- and outcome-relevant subspaces.
2. **Hash embedder is coarse**: works for prototyping; switch to
   `embedder='sbert'` when you have the budget.
3. **No nonlinear outcome model**: linear OLS only. For nonlinear
   outcomes, post-process the embedding into your own DML pipeline.

## Quick start: LLM-annotator measurement-error correction

```python
import statspai as sp
import pandas as pd
import numpy as np

# 10000 LLM-labelled rows; 200 hand-labelled validation rows
T_llm = pd.Series(...)        # binary
y     = pd.Series(...)
human = pd.Series(...)        # NaN where unavailable; ~200 valid

result = sp.llm_annotator_correct(
    annotations_llm=T_llm,
    annotations_human=human,   # required
    outcome=y,
    method="hausman",
)

print(result.summary())
# LLMAnnotatorResult (experimental)
# ============================================================
#   Method            : llm_annotator_correct
#   Estimand          : ATE
#   Naive estimate    : 0.6445 (SE = 0.0561)
#   Correction factor : 0.6218
#   Corrected estimate: 1.0366 (SE = 0.0902)
#   Validation N      : 200
#   Agreement rate    : 0.8100
#   P(T_obs=1|T=0)    : 0.1818
#   P(T_obs=0|T=1)    : 0.1964
#   SE correction     : first_order
#   Status            : experimental
```

### How the Hausman correction works

For OLS of `y` on a binary `T_obs`, the coefficient is attenuated by
the misclassification rate:

```
β_obs = (1 - p_01 - p_10) · β_true
```

Estimate `p_01` and `p_10` on a hand-validated subset where both LLM
and human labels exist, then divide:

```
β_corrected = β_obs / (1 - p_01 - p_10)
```

### Caveats (LLM-annotator)

1. **Validation set size matters**: at least 30 rows with both labels;
   ideally a few hundred for stable correction.
2. **First-order SE correction**: we inflate the SE by the same factor
   but ignore the additional uncertainty from estimating `p_01` /
   `p_10` from the validation set. Inflate manually if you need
   conservative inference.
3. **Binary treatment only** in v1.6. Continuous-treatment correction
   (regression-calibration) is on the roadmap.
4. **If `1 - p_01 - p_10 ≤ 0`** the LLM has no information about the
   true label and the function raises `IdentificationFailure`.

## For Agents

- **Pre-conditions**:
  - `text_treatment_effect`: text/outcome/treatment columns must exist; `n_obs >= max(20, n_components+4)`.
  - `llm_annotator_correct`: ≥30 validation rows spanning both `T_human` classes.
- **Recovery on `DataInsufficient`**:
  - text version → reduce `n_components` or supply more rows.
  - annotator version → hand-label more rows.
- **Recovery on `IdentificationFailure`** (annotator): the LLM is
  effectively random; either re-prompt or hand-label the full sample.
- **Determinism**: with `embedder='hash'` and pinned `seed`, output is
  bit-for-bit reproducible.
- **No external network calls**: both methods are self-contained.

## References

- Veitch, V., Wang, Y., & Blei, D. M. (2020). "Adapting text embeddings for causal inference." *UAI*. [arXiv:1905.12741](https://arxiv.org/abs/1905.12741).
- Egami, N., Hinck, M., Stewart, B., & Wei, H. (2024). "Using imperfect surrogates for downstream inference: Design-based supervised learning for social science applications of large language models." *NeurIPS*. [arXiv:2306.04746](https://arxiv.org/abs/2306.04746).
- Hausman, J., Abrevaya, J., & Scott-Morton, F. (1998). "Misclassification of the dependent variable in a discrete-response setting." *Journal of Econometrics*, 87, 239–269.
- Aigner, D. J. (1973). "Regression with a binary independent variable subject to errors of observation." *Journal of Econometrics*, 1(1), 49–59.
- Roberts, M. E., Stewart, B. M., & Nielsen, R. A. (2020). "Adjusting for confounding with text matching." *American Journal of Political Science*. (Not yet implemented — v1.7 roadmap.)
