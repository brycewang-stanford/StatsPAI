# Preparing data the Stata way: xtset, lags with gaps, reshape, collapse, egen

Most of the work before an estimate is data preparation, and most of the
silent mistakes happen there. A lag taken by shifting rows is wrong on a
panel that skips a year. A `groupby().mean()` treats missing by-values
differently from Stata's `collapse`. This guide shows the StatsPAI
functions that do these steps with Stata's rules, and how they connect to
formulas and to `sp.stata`.

Every number printed below comes from running the code (the test
`tests/test_data_layer_guide.py` executes each block and compares its
output with what is shown here). The same steps were run in Stata 18; the
references are in `tests/test_data_layer.py`, `tests/test_stata_formula_syntax.py`,
`tests/test_egen_public.py` and `tests/test_stata_xtset_attrs.py`.

| Stata | StatsPAI |
| --- | --- |
| `xtset id t` / `tsset t` | `sp.xtset(df, id="id", time="t")` / `sp.xtset(df, time="t")` |
| `L.x`, `F2.x`, `D.x` | `sp.lag(df, "x")`, `sp.lead(df, "x", 2)`, `sp.diff(df, "x")`, or `L.x` in a formula |
| `reshape wide inc, i(hid) j(year)` | `sp.reshape(df, "wide", stubs="inc", id="hid", j="year")` |
| `collapse (mean) mx=x, by(g)` | `sp.collapse(df, "(mean) mx=x", by="g")` |
| `egen m = mean(x), by(g)` | `sp.egen(df, "mean", "x", by="g")` |
| `regress y i.g##c.x` | `sp.regress("y ~ i.g##c.x", data=df)` |

## Declaring a panel

`sp.xtset` records the panel and time variables on the DataFrame
(`df.attrs`) and describes the panel as Stata does. The rows are not
reordered.

```python
import numpy as np
import pandas as pd
import statspai as sp

panel = pd.DataFrame({
    "id": [2, 1, 1, 3, 2, 1, 2, 3, 3, 1],
    "t":  [3, 4, 1, 2, 1, 2, 2, 5, 4, 6],
    "x":  [7.5, 9.25, 1.5, 6.125, 2.75, 4.5, np.nan, 11, 8, 13.5],
})
decl = sp.xtset(panel, id="id", time="t")
print(decl.balanced, decl.gaps, decl.n_units, decl.t_min, decl.t_max)
print(panel.attrs["statspai_xtset"])
```

```text
unbalanced True 3 1.0 6.0
{'id': 'id', 'time': 't', 'delta': 1.0}
```

Stata refuses a time variable that repeats a period within a panel or has
non-integer values, and so does `sp.xtset`.

## Lags, leads and differences when periods are missing

A lag is the value one *period* earlier in the same panel, not the row
above. Panel 1 is observed at t = 1, 2, 4, 6, so the lag at t = 4 is
missing: there is no t = 3.

```python
print(sp.lag(panel, "x").tolist())
print(sp.lead(panel, "x").tolist())
print(sp.diff(panel, "x").tolist())
print(sp.lag(panel, "x", 2).tolist())
```

```text
[nan, nan, nan, nan, nan, 1.5, 2.75, 8.0, nan, nan]
[nan, nan, 4.5, nan, nan, nan, 7.5, nan, 11.0, nan]
[nan, nan, nan, nan, nan, 3.0, nan, 3.0, nan, nan]
[2.75, 4.5, nan, nan, nan, nan, nan, nan, 6.125, 9.25]
```

These are the values Stata gives for `L.x`, `F.x`, `D.x` and `L2.x` after
`xtset id t`. The common pandas idiom shifts rows instead, and puts a value
where Stata has none (t = 4 gets the value at t = 2, t = 6 the value at
t = 4):

```python
shifted = panel.sort_values(["id", "t"]).groupby("id")["x"].shift()
print(shifted.reindex(panel.index).tolist())
```

```text
[nan, 4.5, nan, nan, nan, 1.5, 2.75, 8.0, 6.125, 9.25]
```

With `delta=` one period is several time units, as Stata's `delta()`:

```python
ts = pd.DataFrame({"t": [1, 3, 5, 9], "x": [1.0, 4, 9, 16]})
sp.xtset(ts, time="t", delta=2)
print(sp.lag(ts, "x").tolist())
```

```text
[nan, 1.0, 4.0, nan]
```

## Time-series operators in formulas and in `sp.stata`

After `sp.xtset`, a formula may use `L.x`, `F.x`, `D.x` and `L(1/3).x`;
each term becomes a column named `x_L1`, `x_D1`, ... in a copy of the data
(the caller's frame is left alone). Without a declaration the formula is
refused with a hint to call `sp.xtset`.

```python
r = np.random.default_rng(5)
fv = pd.DataFrame({"id": np.repeat(np.arange(1, 31), 6), "t": np.tile(np.arange(1, 7), 30)})
fv = fv.drop(index=r.choice(len(fv), 25, replace=False)).reset_index(drop=True)
n = len(fv)
fv["g"] = r.integers(1, 4, n)
fv["h"] = r.integers(0, 2, n)
fv["x"] = np.round(r.normal(size=n), 6)
fv["z"] = np.round(r.normal(size=n), 6)
fv["y"] = np.round(
    1 + 0.5 * fv.x - 0.3 * fv.g + 0.2 * fv.g * fv.x + 0.4 * fv.h + r.normal(size=n), 6
)
fv = fv.sample(frac=1, random_state=2).reset_index(drop=True)

sp.xtset(fv, id="id", time="t")
fit = sp.regress("y ~ L.x + D.z + F.x", data=fv)
print(fit.params.round(6).to_dict(), int(fit.data_info["nobs"]))
```

```text
{'Intercept': 0.422123, 'x_L1': -0.004098, 'z_D1': -0.130575, 'x_F1': -0.084621} 71
```

Stata 18's `regress y L.x D.z F.x` after `xtset id t` gives the same four
coefficients and N = 71. `sp.stata` reads the declaration too, as if the
snippet began with `xtset id t` (including the sort by panel and time that
`xtset` does, so `_n` is Stata's row number):

```python
same = sp.stata("regress y L.x D.z F.x", data=fv)
print(bool(np.allclose(same.params.to_numpy(), fit.params.to_numpy(), rtol=1e-12)))
```

```text
True
```

## Factor variables in formulas

Stata's factor-variable terms work in every formula that goes through the
common design-matrix builder: `i.g` (first level is the base), `ib2.g`,
`c.x`, `g#h`, `i.g##c.x`, `c.x#c.x` (a square). They are translated by the
same code as `sp.from_stata`, so the base level and the cells kept in an
interaction are Stata's.

```python
fit = sp.regress("y ~ i.g##c.x", data=fv)
print(fit.params.round(6).to_dict())
```

```text
{'Intercept': 0.848664, 'C(g)[T.2]': -0.207153, 'C(g)[T.3]': -0.6183, 'x': 0.707342, 'C(g)[T.2]:x': 0.113235, 'C(g)[T.3]:x': 0.349368}
```

These are Stata's `_cons`, `2.g`, `3.g`, `x`, `2.g#c.x` and `3.g#c.x`.
Only whole terms are read: `i.g*x` mixes Stata and patsy notation and is
refused with the correct spelling. In `sp.feols` / `sp.fepois` /
`sp.feglm`, the fixed-effects part after `|` is left as it is, because
`i.f#c.x` there is reghdfe's absorbed slope, not a regressor.

## reshape

`sp.reshape` follows Stata's `reshape`: the same column order, the same
rows, and the same refusals (`j` not unique within `i`, a variable that is
not a stub but varies within `i`).

```python
long_inc = pd.DataFrame({
    "hid": [1, 1, 2, 2, 3],
    "year": [80, 81, 80, 82, 81],
    "inc": [5.5, 6, 7, 8.25, 9],
    "sex": [0, 0, 1, 1, 0],
})
wide = sp.reshape(long_inc, "wide", stubs="inc", id="hid", j="year")
print(wide.to_string(index=False))
back = sp.reshape(wide, "long", stubs="inc", id="hid", j="year")
print(back.shape, back.columns.tolist())
```

```text
 hid  inc80  inc81  inc82  sex
   1    5.5    6.0    NaN    0
   2    7.0    NaN   8.25    1
   3    NaN    9.0    NaN    0
(9, 4) ['hid', 'year', 'inc', 'sex']
```

Back to long, every household gets every year (nine rows), as in Stata.

## collapse

`sp.collapse` takes Stata's statistic list or a dict. Missing values are
skipped; a group with a missing by-value is a group of its own, listed
last; `(first)` takes the first row whether or not it is missing
(`(firstnm)` skips missing values); `cw=True` drops a row missing in any of
the source variables first.

```python
grouped = pd.DataFrame({
    "g": [1, 1, 1, 2, 2, 2, np.nan, np.nan],
    "x": [1.5, 2.5, 4, np.nan, 7, 9.5, 3, 6],
    "y": [2, np.nan, 5, 1, 8, 4, 3, np.nan],
})
out = sp.collapse(grouped, "(mean) mx=x (count) nx=x (sum) tx=x (first) f0=x", by="g")
print(out.to_string(index=False))
cw = sp.collapse(grouped, {"mx": ("mean", "x"), "my": ("mean", "y")}, by="g", cw=True)
print(cw.to_string(index=False))
```

```text
  g       mx  nx   tx  f0
1.0 2.666667   3  8.0 1.5
2.0 8.250000   2 16.5 NaN
NaN 4.500000   2  9.0 3.0
  g   mx  my
1.0 2.75 3.5
2.0 8.25 6.0
NaN 3.00 3.0
```

Weights (`weights=`, `weight_type='aweight' | 'fweight' | 'pweight' |
'iweight'`) are read as Stata defines them. The public function keeps
double precision; inside `sp.stata` a new variable is single precision
unless declared `double`, as in Stata.

## egen

`sp.egen` returns a new variable as a Series aligned with the data. The
missing-value rules are those of `[D] egen`: statistics skip missing
values; `count` is 0 and `total` 0 for a group with no value (`total`
missing with `missing=True`); row functions skip missing cells; rows not
selected by `where=` (Stata's `if`) get missing.

```python
df = pd.DataFrame({
    "g": [1, 1, 1, 2, 2],
    "x": [3.0, 1.0, 3.0, 5.0, np.nan],
    "a": [1.0, np.nan, 2.0, np.nan, np.nan],
})
print(sp.egen(df, "mean", "x", by="g").tolist())
print(sp.egen(df, "count", "x", by="g").tolist())
print(sp.egen(df, "rank", "x", by="g", rank_ties="field").tolist())
print(sp.egen(df, "rowtotal", ["x", "a"]).tolist())
print(sp.egen(df, "rowmean", ["x", "a"]).tolist())
print(sp.egen(df, "mean", "x", where=df["a"].notna()).tolist())
```

```text
[2.3333333333333335, 2.3333333333333335, 2.3333333333333335, 5.0, 5.0]
[3.0, 3.0, 3.0, 1.0, 1.0]
[1.0, 3.0, 1.0, 1.0, nan]
[4.0, 1.0, 5.0, 5.0, 0.0]
[2.0, 1.0, 2.5, 5.0, nan]
[3.0, nan, 3.0, nan, nan]
```

`rank_ties="field"` ranks the largest value 1 and gives ties the same
rank; `"track"` ranks the smallest 1. `egen` in `sp.stata` runs the same
code:

```python
run = sp.stata("egen double m = mean(x), by(g)", data=df, return_data=True)
print(run.data["m"].tolist())
```

```text
[2.3333333333333335, 2.3333333333333335, 2.3333333333333335, 5.0, 5.0]
```

## See also

- [Migrating Stata / R commands automatically](translator.md): `sp.from_stata`
  and `sp.stata` for whole do-file snippets.
- [Reproducing Stata defaults](stata_defaults.md): standard errors and
  small-sample corrections.
- [Variable labels and value labels](stata_labels.md).
