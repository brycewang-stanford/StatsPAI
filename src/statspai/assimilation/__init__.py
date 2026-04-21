"""
Assimilative Causal Inference (``sp.assimilation``).

Bridges Bayesian data assimilation — the workhorse of numerical
weather prediction, climate reanalysis, and oceanography — with
causal inference.  Proposed in

    *Assimilative Causal Inference*,
    Nature Communications, 2026.

The core idea is to treat the *causal effect* as a latent time-varying
state, and update its posterior belief as new randomised or
observational data batches arrive.  Each update fuses:

1. A **forecast** step propagating the prior through a user-supplied
   dynamics model (default: random-walk).
2. An **analysis** step that incorporates the fresh observational or
   experimental batch via a Kalman-style innovation.

The result is a running posterior over the causal effect, with an
effective-sample-size diagnostic that flags when new evidence should
trigger a re-design of the experiment.

Why this module exists in StatsPAI
----------------------------------
* Streaming A/B tests: the treatment effect may drift over seasons;
  assimilation lets you pool evidence without pretending the effect is
  static.
* Adaptive monitoring: public-health or policy evaluation that wants to
  update the target estimate monthly instead of waiting for a single
  large study.
* Multi-source evidence synthesis: combine an RCT prior with streaming
  RWE (real-world-evidence) updates under a transport-compatible
  framework (pairs naturally with :mod:`sp.transport`).
"""

from .kalman import (
    assimilative_causal,
    causal_kalman,
    AssimilationResult,
)
from .particle import (
    particle_filter,
    assimilative_causal_particle,
)

__all__ = [
    "assimilative_causal",
    "causal_kalman",
    "AssimilationResult",
    "particle_filter",
    "assimilative_causal_particle",
]
