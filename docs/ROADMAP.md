# StatsPAI Roadmap

Tracks **deliberately deferred** features so future contributors (and
future Claude sessions) don't re-discover the same gap. Each entry
documents:

* What is missing.
* **Why** it was deferred in the release it could have landed in.
* **Trigger** — the concrete signal that would move it back onto the
  active sprint.
* **Rough design** — enough notes to hit the ground running.

When an item ships, move the entry to the relevant CHANGELOG.md
release and delete it here.

---

## 1 · Proximal Causal Inference — non-linear bridges

**Shipped in 0.9.6**: ``sp.proximal(..., bridge='linear')`` — Cui et al.
(2024) linear 2SLS on the outcome-confounding bridge.

**Deferred**: kernel (Mastouri et al. 2021, *Proximal Causal Learning
with Kernels*) and sieve / RKHS (Deaner 2018) bridge estimators.
``bridge='kernel'`` and ``bridge='sieve'`` currently raise
``NotImplementedError``.

**Why deferred**: each estimator is a ~500-line implementation on its
own (PMMR kernel mean matching, tuning λ, kernel design choices) and
requires hyper-parameter selection harness. Doing one carefully is
better than shipping both badly.

**Trigger**: user issues requesting non-linear bridge, or a benchmark
showing the linear bridge is the bias bottleneck on a realistic DGP.

**Rough design**:

* `proximal/kernel_bridge.py` — implement Mastouri PMMR with Gaussian
  kernel, median-heuristic length-scale, λ by cross-validation on
  moment imbalance.
* `proximal/sieve_bridge.py` — polynomial / B-spline sieve with
  ℓ₂-penalty; cross-validated regularisation.
* Reuse `_linear_iv_fit`'s variance-by-bootstrap path; closed-form
  sandwich is not generally available for non-linear bridges.

---

## 2 · PLIV / IIVM — native multi-instrument support

**Current state (0.9.7)**: StatsPAI's PLIV and IIVM cross-fit
reduce-form nuisance ``r(X) = E[Z|X]`` as a **scalar** learner output.
Passing multiple instruments raises a clear error that points the user
to ``sp.scalar_iv_projection()``, which builds an OLS first-stage
scalar index.

**Deferred**: vector-valued ``r(X)`` and a matching Neyman-orthogonal
score that pools across instruments, plus a Cragg-Donald /
Kleibergen-Paap minimum-eigenvalue first-stage statistic for
weak-instrument diagnostics.

**Why deferred**: the orthogonal score for vector Z is not a trivial
generalisation — different score constructions give different
efficiencies, and the weak-instrument statistic matters more than the
point estimate when ``k_z > 1``. Shipping a toy implementation would
mislead users.

**Trigger**: request for a multi-instrument PLIV/IIVM on a real data
set, or a benchmark showing the scalar-projection escape hatch is
materially worse than proper pooled 2SLS on the same DGP.

**Rough design**:

* Extend `dml/_base.py` with `_REQUIRES_SCALAR_INSTRUMENT = False`
  override in PLIV/IIVM (the scaffold already exists).
* Implement a PLIV variant whose cross-fit learner returns an
  `n × k_z` matrix; the score becomes a stacked moment.
* Add `first_stage_min_eig` (Cragg-Donald) to `model_info`.

---

## 3 · MSM per-period trimming — upgrade to user-defined quantile

**Shipped in 0.9.7**: ``sp.msm(..., trim_per_period=False)`` and the
helper ``sp.stabilized_weights(..., trim_per_period=0.0)`` allow the
user to switch between Robins (1998) post-cumulative trimming and
Cole & Hernán (2008 §3) per-period trimming. When enabled, the
per-period trim quantile reuses the ``trim`` value.

**Deferred**: allow a *different* quantile for per-period trimming
than for post-cumulative trimming (e.g. heavier per-period trim +
lighter post-cumulative trim), and support asymmetric trims.

**Why deferred**: the single-quantile default covers the common
convention. Two-quantile support multiplies the API surface for a
rarely-needed ergonomic gain.

**Trigger**: a user reports that they want to trim per-period at 0.05
while keeping the post-cumulative trim at 0.01.

**Rough design**:

* Replace ``trim_per_period: bool`` with ``trim_per_period: float``
  in one release.
* Leave ``trim_per_period=True`` as an alias for ``trim_per_period=trim``
  during the deprecation period.

---

## 4 · Front-door with continuous treatment

**Current state (0.9.7)**: ``sp.front_door`` requires binary D. When
D is continuous the error message points to
``sp.g_computation(..., estimand='dose_response')`` which handles the
continuous-D case under the standard unconfoundedness assumption.

**Deferred**: a proper front-door dose-response estimator — i.e. the
continuous-D front-door formula::

    E[Y | do(D=d)] = Σ_m P(M=m|D=d) · Σ_{d'} P(D=d') · E[Y|D=d', M=m]

marginalised over a d-grid.

**Why deferred**: in practice the front-door criterion is almost
always invoked with a binary treatment (program take-up, policy
adoption). A continuous version requires a conditional-density
estimate of ``P(M|D)`` that generalises the Gaussian-linear model we
ship for binary-D continuous-M. Until a user case shows up, the
escape hatch (``g_computation`` if unconfoundedness holds) is a
good-enough stop-gap.

**Trigger**: an applied paper / user request using continuous-D
front-door.

**Rough design**:

* Accept ``sp.front_door(..., treat_values=[d_1, ..., d_K])``.
* For each d_k, evaluate the closed-form / MC integration of the
  continuous-D front-door formula.
* Report a curve (like ``g_computation`` dose-response) in
  ``result.detail``.

---

## 5 · `sp.mediate()` — align p-value convention across the mediation family

**Current state (0.9.7)**: ``sp.mediate_interventional()`` exposes a
``pvalue_method='bootstrap_sign' | 'wald'`` kwarg (default
``'bootstrap_sign'`` for consistency with ``sp.mediate()``). The
classical ``sp.mediate()`` only supports the sign-based convention.

**Deferred**: add the same ``pvalue_method`` kwarg to ``sp.mediate()``,
and consider flipping the default to ``'wald'`` across the family
(the convention used by every other causal-inference module in
StatsPAI).

**Why deferred**: changing the default would alter existing users'
summaries. A coordinated rollout should (a) add the kwarg with the
old default, (b) deprecate the old default in one release, (c)
flip the default in a minor-version bump with clear CHANGELOG.

**Trigger**: user complaint about the inconsistency, or before any
release that bumps to 1.0 (alignment is a 1.0 story).

**Rough design**:

* Mirror the ``pvalue_method`` kwarg addition across ``sp.mediate()``
  (same branch in ``_boot_pvalue`` logic).
* Add a deprecation warning when ``pvalue_method`` is omitted for
  one release.
* Flip the default in the release after that.

---

## 6 · `sp.compare_estimators` — hint-aware Sprint-B support

**Shipped in 0.9.x**: ``sp.compare_estimators`` accepts ``methods=``
from ``{'ols', 'matching', 'ipw', 'aipw', 'dml', 'g_computation',
'causal_forest', 'did', 'panel_fe'}``. Each branch shares the same
``(data, y, treatment, covariates, id, time, instrument)`` signature.

**Deferred**: the four Sprint-B estimators that need extra arguments
the shared signature does not expose — ``'proximal'`` (needs
``proxy_z`` + ``proxy_w``), ``'msm'`` (needs ``time_varying``),
``'principal_strat'`` (needs a post-treatment ``strata`` column), and
the mediation family (needs ``mediator`` and optionally
``tv_confounders``). Calling them requires bespoke args per method.

**Why deferred**: bloating the shared signature with a dozen
optional kwargs would regress the ergonomics for the common
methods, while a hint-forwarding map (``method_hints={'proximal':
{'proxy_z': [...], 'proxy_w': [...]}}``) needs design work to avoid
confusing multi-method semantics.

**Trigger**: a user request to compare proximal against DML / TMLE
on the same data set.

**Rough design**:

* Add ``method_hints: Dict[str, Dict[str, Any]]`` parameter that
  maps a method name to its per-method kwargs.
* Dispatch each Sprint-B branch to the corresponding estimator using
  both the shared args and the per-method hints.
* Keep backward compatibility: callers who don't pass
  ``method_hints`` get the existing behaviour.
* **Collision rule**: per-method hints take precedence over the
  shared kwargs for the method they name. If
  ``covariates=['age']`` is the top-level shared arg and
  ``method_hints={'proximal': {'covariates': ['age', 'educ']}}`` is
  supplied, proximal uses ``['age', 'educ']`` and every other method
  uses ``['age']``. Emit a ``UserWarning`` on conflict so the
  override is visible in the log.

---
