"""
Synthetic datasets for decomposition analysis tutorials and tests.

All datasets are generated programmatically with fixed seeds so results
are reproducible. Sample sizes and coefficient values are chosen to
resemble canonical labour-economics applications (gender wage gap,
racial education gap, rural/urban income gap).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def cps_wage(n: int = 3000, seed: Optional[int] = 42) -> pd.DataFrame:
    """
    CPS-style wage data with a gender gap.

    Columns:
      - female : int {0, 1}
      - education : years
      - experience : years
      - tenure : years
      - union : int {0, 1}
      - married : int {0, 1}
      - log_wage : float
    """
    rng = np.random.default_rng(seed)
    female = rng.binomial(1, 0.48, n)

    # Different X distributions by group
    educ = np.where(female == 1,
                    rng.normal(13.0, 2.5, n),
                    rng.normal(12.5, 2.8, n)).clip(8, 22)
    exp = np.where(female == 1,
                   rng.gamma(shape=3, scale=5, size=n),
                   rng.gamma(shape=3.3, scale=5.2, size=n)).clip(0, 45)
    tenure = np.clip(exp - rng.normal(3.5, 2.0, n), 0, None)
    union = rng.binomial(1, np.where(female == 1, 0.14, 0.22), n)
    married = rng.binomial(1, np.where(female == 1, 0.55, 0.60), n)

    # Different return structures → nontrivial unexplained component
    beta_m = np.array([1.7, 0.10, 0.04, 0.012, 0.15, 0.08])  # constant, educ, exp, tenure, union, married
    beta_f = np.array([1.5, 0.09, 0.035, 0.010, 0.10, 0.05])

    X = np.column_stack([np.ones(n), educ, exp, tenure, union, married])
    log_wage = np.where(
        female == 1,
        X @ beta_f,
        X @ beta_m,
    ) + rng.normal(0, 0.35, n)

    return pd.DataFrame({
        "female": female.astype(int),
        "education": educ,
        "experience": exp,
        "tenure": tenure,
        "union": union.astype(int),
        "married": married.astype(int),
        "log_wage": log_wage,
    })


def chilean_households(n: int = 2500, seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Chilean-style household income with urban/rural gap.

    Columns:
      - rural : int {0, 1}
      - head_education : years
      - head_age : years
      - household_size : int
      - log_income : float
    """
    rng = np.random.default_rng(seed)
    rural = rng.binomial(1, 0.38, n)

    educ = np.where(rural == 1,
                    rng.normal(8.0, 2.5, n),
                    rng.normal(11.2, 3.2, n)).clip(0, 20)
    age = rng.normal(43, 10, n).clip(18, 80)
    hh_size = rng.poisson(np.where(rural == 1, 4.5, 3.3), n).clip(1, 12)

    beta_u = np.array([6.5, 0.12, 0.012, -0.05])
    beta_r = np.array([6.2, 0.08, 0.010, -0.03])

    X = np.column_stack([np.ones(n), educ, age, hh_size])
    log_inc = np.where(
        rural == 1, X @ beta_r, X @ beta_u,
    ) + rng.normal(0, 0.42, n)

    return pd.DataFrame({
        "rural": rural.astype(int),
        "head_education": educ,
        "head_age": age,
        "household_size": hh_size,
        "log_income": log_inc,
    })


def mincer_wage_panel(n: int = 5000, seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Two-period Mincer wage distribution with a structural shift.

    Useful for DFL / FFL examples where Group 0 is early period and
    Group 1 is late period.

    Columns:
      - period : int {0, 1}
      - education : years
      - experience : years
      - union : int
      - occupation_high_skill : int
      - log_wage : float
    """
    rng = np.random.default_rng(seed)
    period = rng.binomial(1, 0.5, n)
    educ = np.where(period == 1,
                    rng.normal(13.5, 2.5, n),
                    rng.normal(12.5, 2.8, n)).clip(8, 22)
    exp = rng.gamma(3, 5, n).clip(0, 45)
    union = rng.binomial(1, np.where(period == 1, 0.12, 0.22), n)
    hs = rng.binomial(1, np.where(period == 1, 0.42, 0.30), n)

    beta_early = np.array([1.5, 0.08, 0.03, 0.20, 0.15])
    beta_late = np.array([1.4, 0.11, 0.035, 0.10, 0.25])

    X = np.column_stack([np.ones(n), educ, exp, union, hs])
    log_wage = np.where(
        period == 1, X @ beta_late, X @ beta_early,
    ) + rng.normal(0, 0.35, n)

    return pd.DataFrame({
        "period": period.astype(int),
        "education": educ,
        "experience": exp,
        "union": union.astype(int),
        "occupation_high_skill": hs.astype(int),
        "log_wage": log_wage,
    })


def disparity_panel(n: int = 3000, seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Synthetic disparity panel with treatment, mediator, outcome.

    Columns:
      - group : int {0, 1}  (disadvantaged = 1)
      - education : years   (mediator)
      - parent_income : float (confounder)
      - age : years (confounder)
      - income : float (outcome)
    """
    rng = np.random.default_rng(seed)
    group = rng.binomial(1, 0.4, n)
    parent_inc = rng.normal(np.where(group == 1, 9.5, 10.2), 0.6, n)
    age = rng.normal(36, 9, n).clip(18, 65)
    # Education (mediator) is affected by group and parents
    educ = (12 - 2.5 * group + 0.4 * (parent_inc - parent_inc.mean())
            + 0.02 * (age - age.mean()) + rng.normal(0, 1.2, n)).clip(6, 20)
    # Outcome income depends on group, education, parent_inc, age
    income = (8.5 - 0.4 * group + 0.12 * educ
              + 0.25 * (parent_inc - parent_inc.mean())
              + 0.01 * (age - age.mean())
              + rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "group": group.astype(int),
        "education": educ,
        "parent_income": parent_inc,
        "age": age,
        "income": income,
    })
