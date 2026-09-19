"""Write the simulated inputs of tests/reference_parity/test_panel_glmm_parity.py.

Run before ``_generate_panel_glmm_R.R`` and ``_generate_panel_glmm_stata.do``.
Deterministic (fixed seed); the CSVs are committed so the Python, R and Stata
sides read identical bytes (``%.17g``).

``panel_glmm_data.csv`` -- one clustered cross-section, 60 groups of unequal
size (4 to 14), a random intercept u_g ~ N(0, 0.6^2) and one outcome per
GLMM family:

* ``y_pois``   Poisson, log link, offset ``log(expo)``
* ``y_nb``     NB-2 (alpha = 0.5), log link
* ``y_gam``    Gamma (shape 3, i.e. phi = 1/3), log link
* ``y_bin``    Bernoulli, logit link
* ``y_succ`` / ``n_trials``  binomial counts, logit link
* ``y_ord``    four ordered categories (latent logistic, cuts -1, 0.3, 1.5)
* ``y_gau``    Gaussian, identity link, residual sd 0.8
* ``y_rs``     Gaussian with an additional random slope on x1 (sd 0.4)

``panel_count_data.csv`` -- a balanced count panel (40 units x 6 periods)
with a unit-level gamma heterogeneity term for the conditional /
random-effects negative-binomial models (``xtnbreg``).

``panel_ife_data.csv`` -- a balanced 40 x 15 panel with two common factors
whose loadings also drive the regressors (``interactive_fe``); no additive
unit or time effects and no intercept.

``gmm_nl_data.csv`` -- n = 500 cross-section for a nonlinear (exponential
mean, multiplicative-error) IV moment problem (``sp.gmm``).

``absorb_ols_data.csv`` -- n = 1200, firm and year fixed effects, six
singleton firms, analytic weights, for ``sp.absorb_ols`` against
``reghdfe`` (weights, singleton dropping, two-way clustering).

``mixlogit_data.csv`` -- 150 individuals x 4 choice situations x 3
alternatives; price fixed, quality and comfort normal random coefficients.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent

rng = np.random.default_rng(20260918)
G = 60
sizes = rng.integers(4, 15, G)
gid = np.repeat(np.arange(1, G + 1), sizes)
n = gid.size
u = rng.normal(0.0, 0.6, G)[gid - 1]
x1 = rng.normal(size=n)
x2 = rng.binomial(1, 0.4, n).astype(float)
expo = rng.uniform(0.5, 2.0, n)

eta = 0.3 + 0.5 * x1 - 0.4 * x2 + u
y_pois = rng.poisson(np.exp(eta + np.log(expo)))

alpha_nb = 0.5
mu_nb = np.exp(0.4 + 0.5 * x1 - 0.4 * x2 + u)
y_nb = rng.poisson(rng.gamma(1.0 / alpha_nb, alpha_nb * mu_nb))

mu_gam = np.exp(0.5 + 0.3 * x1 - 0.2 * x2 + u)
y_gam = rng.gamma(3.0, mu_gam / 3.0)

p_bin = 1.0 / (1.0 + np.exp(-(-0.2 + 0.8 * x1 - 0.5 * x2 + u)))
y_bin = (rng.uniform(size=n) < p_bin).astype(int)
n_trials = rng.integers(1, 9, n)
y_succ = rng.binomial(n_trials, p_bin)

latent = 0.7 * x1 - 0.5 * x2 + u + rng.logistic(size=n)
y_ord = 1 + np.searchsorted(np.array([-1.0, 0.3, 1.5]), latent)

y_gau = 1.0 + 0.5 * x1 - 0.4 * x2 + u + rng.normal(0.0, 0.8, n)

# Random slope on x1 (sd 0.4) for the variance-component LR tests.
v = rng.normal(0.0, 0.4, G)[gid - 1]
y_rs = 1.0 + 0.5 * x1 - 0.4 * x2 + u + v * x1 + rng.normal(0.0, 0.8, n)

pd.DataFrame(
    {
        "gid": gid,
        "x1": x1,
        "x2": x2,
        "expo": expo,
        "lexpo": np.log(expo),
        "y_pois": y_pois,
        "y_nb": y_nb,
        "y_gam": y_gam,
        "y_bin": y_bin,
        "y_succ": y_succ,
        "n_trials": n_trials,
        "y_ord": y_ord,
        "y_gau": y_gau,
        "y_rs": y_rs,
    }
).to_csv(OUT / "panel_glmm_data.csv", index=False, float_format="%.17g")

# ---- balanced count panel for xtnbreg ------------------------------------
rng = np.random.default_rng(918)
N, T = 40, 6
pid = np.repeat(np.arange(1, N + 1), T)
year = np.tile(np.arange(1, T + 1), N)
a_i = rng.normal(0.0, 0.5, N)[pid - 1]
z1 = rng.normal(size=N * T) + 0.5 * a_i
z2 = rng.uniform(size=N * T)
lam = np.exp(0.5 + 0.4 * z1 - 0.6 * z2 + a_i)
y = rng.poisson(rng.gamma(2.0, lam / 2.0))
pd.DataFrame({"pid": pid, "year": year, "z1": z1, "z2": z2, "y": y}).to_csv(
    OUT / "panel_count_data.csv", index=False, float_format="%.17g"
)

# ---- balanced interactive-fixed-effects panel (Bai 2009 design) -----------
rng = np.random.default_rng(2009)
N, T, r = 40, 15, 2
F = rng.normal(size=(T, r))
L = rng.normal(size=(N, r))
common = L @ F.T
xa = 0.5 * common + rng.normal(size=(N, T))
xb = rng.normal(size=(N, T)) + 0.3 * (L[:, [0]] * F[:, 1][None, :])
yf = 1.0 * xa - 0.5 * xb + common + rng.normal(0.0, 0.7, size=(N, T))
pd.DataFrame(
    {
        "unit": np.repeat(np.arange(1, N + 1), T),
        "period": np.tile(np.arange(1, T + 1), N),
        "y": yf.ravel(),
        "xa": xa.ravel(),
        "xb": xb.ravel(),
    }
).to_csv(OUT / "panel_ife_data.csv", index=False, float_format="%.17g")

# ---- exponential-mean IV moment problem for sp.gmm -------------------------
# Mullahy's multiplicative moments E[z (y exp(-x'b) - 1)] = 0 with one
# endogenous regressor and three excluded instruments (over-identified).
rng = np.random.default_rng(1982)
n = 500
z = rng.normal(size=(n, 3))
vv = rng.normal(size=n)
xg = 0.6 * z[:, 0] + 0.4 * z[:, 1] - 0.3 * z[:, 2] + vv
eta = np.exp(0.5 * vv + rng.normal(0.0, 0.5, n))
eta = eta / np.exp(0.5 * (0.25 + 0.25))  # E[eta] = 1
yg = rng.poisson(np.exp(0.2 + 0.4 * xg) * eta)
pd.DataFrame({"y": yg, "x1": xg, "z1": z[:, 0], "z2": z[:, 1], "z3": z[:, 2]}).to_csv(
    OUT / "gmm_nl_data.csv", index=False, float_format="%.17g"
)

# ---- absorbed-FE OLS with singletons, weights and two cluster dimensions ---
rng = np.random.default_rng(1998)
n = 1200
firm = rng.integers(1, 151, n)
year = rng.integers(1, 13, n)
# a handful of singleton firms (ids above 150), dropped by reghdfe
firm[:6] = 150 + np.arange(1, 7)
x1 = rng.normal(size=n) + 0.3 * (firm % 7)
x2 = rng.normal(size=n)
w = rng.uniform(0.5, 2.0, n)
y = (
    1.5 * x1
    - 0.7 * x2
    + rng.normal(size=160)[firm - 1]
    + rng.normal(size=13)[year]
    + rng.normal(0, 1.0, n)
)
pd.DataFrame({"firm": firm, "year": year, "x1": x1, "x2": x2, "w": w, "y": y}).to_csv(
    OUT / "absorb_ols_data.csv", index=False, float_format="%.17g"
)

# ---- mixed logit panel (sp.mixlogit vs Stata mixlogit) --------------------
rng = np.random.default_rng(2007)
N, T, J = 150, 4, 3
bq = rng.normal(1.0, 0.8, N)
bc = rng.normal(0.5, 0.5, N)
rows = []
for i in range(N):
    for t in range(T):
        price = rng.uniform(0.5, 2.0, J)
        quality = rng.uniform(0.0, 3.0, J)
        comfort = rng.normal(size=J)
        u = -1.2 * price + bq[i] * quality + bc[i] * comfort + rng.gumbel(size=J)
        ch = int(np.argmax(u))
        for j in range(J):
            rows.append(
                {
                    "pid": i + 1,
                    "cs": i * T + t + 1,
                    "alt": j + 1,
                    "chosen": int(j == ch),
                    "price": price[j],
                    "quality": quality[j],
                    "comfort": comfort[j],
                }
            )
pd.DataFrame(rows).to_csv(OUT / "mixlogit_data.csv", index=False, float_format="%.17g")
