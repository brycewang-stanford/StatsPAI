# Variable labels and value labels

A Stata `.dta` file carries more than numbers: a label for each variable, a
label for each code of a categorical variable, a label for the dataset, and
labels on the extended missing values `.a` to `.z`. This guide shows what
StatsPAI keeps, where it keeps it, and how the labels reach your tables.

## Reading a .dta file

```python
import statspai as sp

df = sp.read_data("survey.dta")
sp.describe(df)
```

```text
  variable     type    n  n_missing          label              value_labels
0       id    int64  500          0  Respondent id
1   region  float64  491          9  Census region  1=North, 2=South, 3=East, ...
2   female    int64  500          0                               0=No, 1=Yes
```

Labelled variables stay numeric, as Stata stores them, so a regression on
`female` or `i.region` uses the codes. The labels sit beside the data:

| What | Where | Stata command |
| --- | --- | --- |
| Variable labels | `df.attrs['_labels']` | `label variable` |
| Value labels | `df.attrs['_value_labels']` | `label define` + `label values` |
| Labels of `.a` to `.z` | `df.attrs['_missing_labels']` | `label define x .a "Refused"` |
| Dataset label | `df.attrs['_data_label']` | `label data` |
| Display formats that say something | `df.attrs['_formats']` | `format` |
| Name of each variable's label set | `df.attrs['_value_label_names']` | `label values` |
| Notes | `df.attrs['_notes']` | `notes` |
| Other characteristics | `df.attrs['_characteristics']` | `char`, `xtset`, `tsset` |

Notes and characteristics are keyed by variable, with those of the dataset
itself under `'_dta'`:

```python
df.attrs["_notes"]
# {'_dta': ['Wave 3, released 2021'], 'wage': ['top-coded at 40']}
df.attrs["_characteristics"]["_dta"]["_TSpanel"]   # 'id', from xtset
```

The result is the same with or without the optional `pyreadstat`.

An SPSS `.sav` file is read the same way (this one needs `pyreadstat`):
variable labels, and value labels for numeric variables with whole-number
codes.

## Attaching labels yourself

```python
sp.label_var(df, "wage", "Hourly wage (USD)")
sp.label_values(df, ["female", "union"], {0: "No", 1: "Yes"})
sp.label_values(df, "region", {1: "North", 2: "South", ".a": "Refused"})
```

`sp.label_values` takes integer codes, and `'.a'` to `'.z'` for the extended
missing values. It replaces the labels a variable already has; `None`
removes them.

## Using the labels

**Tabulations** print labels in the order of the codes, as Stata does:

```python
sp.tab(df, "region")                 # North, South, East, West
sp.tab(df, "region", labels=False)   # 1, 2, 3, 4  (Stata's nolabel)
```

**Regression tables** take the estimation data through `labels=`:

```python
m = sp.regress("wage ~ tenure + C(region) + female:tenure", data=df)
sp.regtable(m, labels=df)
```

```text
Census region: South        0.104
Tenure                      0.040
Female × Tenure            -0.030
```

`coef_labels=` still overrides any single row.

**Readable columns** for plots and exports come from `sp.decode`, the
counterpart of Stata's `decode`:

```python
readable = sp.decode(df)             # every labelled column
readable = sp.decode(df, "region")   # one column
```

Each decoded column is an ordered categorical whose categories follow the
codes, so a plot or a set of dummies keeps the order of the original
variable. A code without a label is shown as the number.

## Extended missing values

Surveys use `.a`, `.b` and so on for "refused", "don't know" and similar.
pandas has one missing value, so by default every Stata missing value
becomes `NaN` and StatsPAI warns when the file labelled some of them, since
the rows can no longer be told apart. To keep the codes:

```python
df = sp.read_data("survey.dta", extended_missing="column")
df[["region", "region__miss"]]
```

```text
   region region__miss
0     2.0         None
1     NaN           .a
2     NaN           .b
3     NaN         None      # a plain '.'
```

`region` is still numeric with `NaN`, so estimation is unchanged. The
`region__miss` column says which missing value each row held, and
`sp.decode(df, "region", missing=True)` shows "Refused" on those rows.

## Writing labels back

```python
sp.write_data(df, "out.dta")       # Stata sees every label again
sp.write_data(df, "out.parquet")   # Parquet keeps them too
sp.write_data(df, "out.csv")       # warns: CSV cannot store labels
```

A .dta file that is read and written back keeps what Stata stored in it:

- **Label sets keep their names.** Twenty yes/no items that shared the set
  `yesno` in Stata share one set `yesno` in the file again. If you change
  the labels of one of them, that variable gets a set named after itself
  and the others keep `yesno`.
- **Extended missing values go back as they came.** A `region__miss`
  column is not written as a variable; the rows it marks are written as
  `.a`, `.b` and so on in `region`. Pass `extended_missing="nan"` to write
  the column as text and every missing value as `.`.
- **Notes and characteristics** are written for the columns still in the
  frame, so `xtset` and `tsset` declarations survive.
- **Display formats** are written when Stata would accept them for the
  column, and left out with a warning otherwise.

The file is written beside the target and moved into place when it is
complete, so a failed write leaves an existing file untouched.

Four things are not kept: a label set that is defined but attached to no
variable, a label name attached to a variable without being defined, the
sort order (pandas may have reordered the rows since), and default
formats such as a left-aligned `%-9s`.

## Files from Stata 18 with alias variables

Stata 18 saves a dataset that has alias variables (`fralias add`) in format
120 or 121, which neither pandas nor pyreadstat opens. An alias is a view
of a variable in another frame, and the file stores none of its values.
`sp.read_data` reads everything else, warns, and lists the aliases it left
out in `df.attrs['_alias_variables']`.

## Labels and pandas operations

Labels live in `DataFrame.attrs`, which pandas carries through row
filters, column selections, `assign`, `sort_values` and `groupby`, and
drops in `merge`, `pd.get_dummies`, `pivot`, and in a `pd.concat` of
frames whose labels differ. `rename` keeps the labels under the old
names. Copy them back from the frame that had them:

```python
merged = df.merge(firms, on="firm_id")           # labels gone
sp.label_vars(merged, df)                        # and back

short = df.rename(columns={"wage": "w"})
sp.label_vars(short, df, rename={"wage": "w"})
```

## In a do-file run through `sp.stata`

`label variable`, `label define` (with `add`, `modify`, `replace`),
`label values`, `label data` and `label drop` change the labels of the
session's data; `decode` and `encode` work as in Stata; `rename` takes the
labels along; `tabulate` prints labels unless `nolabel` is given. The
session knows the label sets a .dta file came with, so
`label define yesno 2 "Maybe", add` changes every variable that uses
`yesno`. The frame you passed in is never modified.
