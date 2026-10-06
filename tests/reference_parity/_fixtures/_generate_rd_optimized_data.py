"""Write the two data sets behind the optrdd reference numbers.

Run this first, then ``_generate_rd_optimized_R.R`` (needs the R package
optrdd from github.com/swager/optrdd and quadprog).
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261008)
n = 400
x = rng.uniform(-1, 1, n)
y = 1.0 * (x >= 0) + np.sin(2 * x) + rng.normal(scale=0.5, size=n)
cont = pd.DataFrame({"y": y, "x": x, "design": "continuous"})

n = 500
xd = (np.floor(rng.uniform(-1, 1, n) * 8) + 0.5) / 8  # sixteen support points
yd = 0.5 * (xd >= 0) + xd**2 + rng.normal(scale=0.5, size=n)
disc = pd.DataFrame({"y": yd, "x": xd, "design": "discrete"})

out = pd.concat([cont, disc], ignore_index=True)
out[["y", "x"]] = out[["y", "x"]].round(6)
out.to_csv(Path(__file__).with_name("rd_optimized_data.csv"), index=False)
