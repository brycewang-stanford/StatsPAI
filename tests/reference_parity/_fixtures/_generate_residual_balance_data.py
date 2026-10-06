"""Write the data set behind the balanceHD reference numbers.

Run this first, then ``_generate_residual_balance_R.R`` (needs the R
package balanceHD from github.com/swager/balanceHD and quadprog).
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261007)
n, p = 120, 12
X = rng.normal(size=(n, p))
e = 1 / (1 + np.exp(-(0.7 * X[:, 0] - 0.5 * X[:, 1])))
W = rng.binomial(1, e)
Y = X[:, 0] + 2 * X[:, 2] + W * (1 + X[:, 0]) + rng.normal(size=n)
df = pd.DataFrame(X, columns=[f"x{j + 1}" for j in range(p)]).round(6)
df.insert(0, "W", W)
df.insert(0, "Y", np.round(Y, 6))
df.to_csv(Path(__file__).with_name("residual_balance_data.csv"), index=False)
