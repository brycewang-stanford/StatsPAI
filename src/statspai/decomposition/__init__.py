"""
Decomposition Analysis module for StatsPAI.

Implements wage/outcome decomposition methods widely used in labor and
applied economics:

- **Oaxaca-Blinder (1973)** — decompose mean outcome gaps between groups
  into "explained" (endowment) and "unexplained" (coefficient) components.
- **Gelbach (2016)** — decompose the change in a coefficient when
  additional controls are added, attributing omitted variable bias to
  each added variable.

References
----------
Blinder, A.S. (1973). "Wage Discrimination: Reduced Form and Structural
Estimates." *Journal of Human Resources*, 8(4), 436-455.

Oaxaca, R. (1973). "Male-Female Wage Differentials in Urban Labor Markets."
*International Economic Review*, 14(3), 693-709.

Neumark, D. (1988). "Employers' Discriminatory Behavior and the Estimation
of Wage Discrimination." *Journal of Human Resources*, 23(3), 279-295.

Gelbach, J.B. (2016). "When Do Covariates Matter? And Which Ones, and How
Much?" *Journal of Labor Economics*, 34(2), 509-543.
"""

from .oaxaca import oaxaca, gelbach, OaxacaResult, GelbachResult

__all__ = ['oaxaca', 'gelbach', 'OaxacaResult', 'GelbachResult']
