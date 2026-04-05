"""
Spatial econometrics module — StatsPAI's answer to R's ``spatialreg``
and Stata's ``spregress``.

Provides maximum-likelihood estimation for the three core spatial models:

- **SAR** (Spatial Autoregressive / Spatial Lag):  Y = ρWY + Xβ + ε
- **SEM** (Spatial Error Model):  Y = Xβ + u,  u = λWu + ε
- **SDM** (Spatial Durbin Model):  Y = ρWY + Xβ + WXθ + ε

>>> import statspai as sp
>>> result = sp.sar(W, data=df, formula='y ~ x1 + x2')
>>> result = sp.sem(W, data=df, formula='y ~ x1 + x2')
>>> result = sp.sdm(W, data=df, formula='y ~ x1 + x2')
"""

from .models import sar, sem, sdm, SpatialModel

__all__ = ["sar", "sem", "sdm", "SpatialModel"]
