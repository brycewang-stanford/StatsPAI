# polars, Arrow and other DataFrame libraries

Every `sp.*` function takes a polars `DataFrame` or `LazyFrame` wherever it
takes a pandas DataFrame, and so does any argument that holds data
(`data=`, `newdata=`, a positional frame). The same goes for a pyarrow
`Table` or `RecordBatch`, for any object that speaks the Arrow PyCapsule
stream protocol (`__arrow_c_stream__`), and for any object that speaks
the DataFrame interchange protocol (`__dataframe__`).

```python
import polars as pl
import statspai as sp

df = pl.read_parquet("panel.parquet")
fit = sp.regress("y ~ x + C(region)", data=df, vce="cluster firm")
cs = sp.callaway_santanna(df.lazy(), y="y", g="first_treat", t="year", i="firm")
```

## What happens

The frame is converted to pandas when the function is called; a
`LazyFrame` is collected first. Results are the usual StatsPAI objects,
and tables inside them are pandas. Nothing is converted back.

The numbers are the pandas numbers: a test fits the same models on the
same data as pandas, polars, a polars `LazyFrame` and an Arrow table and
requires identical coefficients and standard errors
(`tests/test_frame_input.py`).

polars needs pyarrow to hand a frame to pandas, so install both:
`pip install "statspai[polars]"`. Neither is imported unless such an
object is passed.

## What it does not do

The conversion copies the data once, so a frame that only fits in memory
as Arrow will not fit as pandas. For very large HDFE problems,
`sp.fast.demean_polars` and `sp.fast.fepois_polars` extract only the
columns they need.

Because a converted frame exists only during the call, `sp.vcov(result,
vce=...)` cannot refit from it afterwards; pass the data again with
`data=`.
