# Proximal Causal Inference

Tchetgen Tchetgen et al. (2020). Identifies the ATE in the presence
of an **unmeasured confounder** using two proxies of that
confounder:

* **`Z`** — "treatment-inducing confounding proxy":
  `Z ⊥ Y | (D, U, X)`.
* **`W`** — "outcome-inducing confounding proxy":
  `W ⊥ D | (U, X)`.

plus measured covariates `X`.

```text
       ┌── Z ──► D ──► Y ──┐
       │                   │
       U ───────────────── U     (U unobserved)
       │                   │
       └────── W ──────────┘
```

## `sp.proximal(...)` — linear 2SLS on the outcome bridge

Assumes a linear outcome bridge :math:`h(W, D, X) = \gamma_0 +
\gamma_D D + \gamma_W^\top W + \gamma_X^\top X`. Under that
restriction the bridge equation reduces to a standard 2SLS problem
where `W` is endogenous and `Z` is its instrument.

```python
# Unobserved U → biased OLS; (Z, W) unblocks identification.
r = sp.proximal(
    df, y='lung_cancer', treat='smoker',
    proxy_z=['occupation'],             # treatment-side proxy (IV for W)
    proxy_w=['shs_exposure'],           # outcome-side proxy (endogenous)
    covariates=['age', 'sex'],          # exogenous controls
    bridge='linear',                    # currently the only option
)

r.estimate                              # γ_D — ATE under the linear bridge
r.model_info['first_stage_F']           # weak-instrument diagnostic (k_w==1 only)
r.model_info['bridge']                  # 'linear' — recorded for audit
```

## Inference options

* **Closed-form SE** (default): 2SLS sandwich, homoskedastic.
  `r.model_info['se_method'] == '2sls_sandwich'`.
* **Bootstrap SE**: pass `n_boot=500` to switch. Failures are
  tracked and warned rather than silently falling back.

```python
r = sp.proximal(df, y='y', treat='d',
                proxy_z=['z'], proxy_w=['w'],
                n_boot=500, seed=0)
r.model_info['se_method']             # 'bootstrap'
r.model_info['n_boot_failed']         # 0 for clean runs
```

## Weak-instrument diagnostics

For a single endogenous proxy (`k_w == 1`), StatsPAI reports the
first-stage F statistic of regressing `W` on the excluded `Z`
(conditioning on `X`). A warning fires when F < 10.

```python
if r.model_info['first_stage_F'] is not None:
    print(f"First-stage F: {r.model_info['first_stage_F']:.2f}")
```

For multiple endogenous proxies (`k_w > 1`), the correct diagnostic
is the **Cragg-Donald / Kleibergen-Paap minimum-eigenvalue**
statistic. That is not yet implemented — `first_stage_F` is `None`
and a `RuntimeWarning` explains why. See `docs/ROADMAP.md` §2.

## Class API

```python
from statspai import ProximalCausalInference

obj = ProximalCausalInference(
    y='y', treat='d', proxy_z=['z'], proxy_w=['w'],
    covariates=['x'],
).fit(df)
r = obj.result_            # CausalResult
```

## Non-linear bridges

Kernel (Mastouri et al. 2021) and sieve / RKHS (Deaner 2018)
bridges are on the roadmap but not yet shipped.
`bridge='kernel'` and `bridge='sieve'` currently raise
`NotImplementedError` with a pointer to `docs/ROADMAP.md` §1.

The scaffold lives in `statspai.proximal.p2sls` — contributors
interested in shipping a non-linear bridge should start there.

## References

- Tchetgen Tchetgen, E.J., Ying, A., Cui, Y., Shi, X. and Miao, W.
  (2020). An Introduction to Proximal Causal Learning.
  *arXiv:2009.10982*.
- Miao, W., Geng, Z. and Tchetgen Tchetgen, E.J. (2018). Identifying
  causal effects with proxy variables of an unmeasured confounder.
  *Biometrika* 105(4).
- Cui, Y., Pu, H., Shi, X., Miao, W. and Tchetgen Tchetgen, E.J.
  (2024). Semiparametric proximal causal inference. *JASA*.
- Mastouri, A. et al. (2021). Proximal Causal Learning with Kernels.
  *ICML*. (kernel bridge — future work)
- Deaner, B. (2018). Proxy Controls and Panel Data.
  *arXiv:1810.00283*. (sieve bridge — future work)
