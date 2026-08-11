# Spatial Econometrics

`statspai.spatial` -- weights construction, exploratory spatial data
analysis (ESDA), spatial regression (ML and GMM), geographically
weighted regression (GWR / MGWR), spatial panel models, and spatial
difference-in-differences with local spillovers.

## Weights

```python
W = sp.queen_weights(gdf)                              # contiguity (queen)
W = sp.rook_weights(gdf)                               # contiguity (rook)
W = sp.knn_weights(coords, k=8)                        # k-nearest neighbors
W = sp.kernel_weights(coords, bandwidth=50_000)        # distance kernel
W.row_standardise()
W.transform                                            # 'r' | 'b' | 'v'
```

## ESDA — Exploratory Spatial Data Analysis

```python
sp.moran(y, W, permutations=999)          # global Moran's I
sp.geary(y, W, permutations=999)          # global Geary's C
sp.getis_ord_g(y, W)                      # Getis-Ord G
sp.moran_local(y, W)                      # LISA — local Moran
sp.join_counts(y_bin, W)                  # categorical autocorrelation
```

## Spatial regression (cross-section)

```python
# Maximum likelihood
sp.sar(W, df, 'price ~ rooms + age')       # spatial autoregressive
sp.sem(W, df, 'price ~ rooms + age')       # spatial error
sp.sdm(W, df, 'price ~ rooms + age')       # Durbin (lag + lagged X)
sp.slx(W, df, 'price ~ rooms + age')       # Spatially-lagged X only
sp.sarar_gmm(W, df, 'price ~ rooms + age') # combo SAR+SEM (GMM)

# GMM / IV
sp.sar_gmm(W, df, 'price ~ rooms + age')
sp.sem_gmm(W, df, 'price ~ rooms + age')
```

## GWR / MGWR — locally varying coefficients

```python
# Array API: projected coords (UTM), y vector, X design matrix.
sp.gwr(coords, y, X, bw=40, kernel='bisquare')   # bw = n neighbours
sp.mgwr(coords, y, X, kernel='bisquare')         # per-covariate bandwidths
```

## Spatial panel

```python
sp.spatial_panel(df, 'gdp ~ k + l', entity='country', time='year', W=W,
                 model='sar',            # 'sar' | 'sem' | 'sdm'
                 effects='twoways')      # 'fe' | 'twoways'
```

## Spatial DiD

`sp.spatial_did` estimates a two-way fixed-effect DiD specification
with an own-treatment effect and a spatially lagged treatment exposure:

```python
r = sp.spatial_did(
    df,
    y='outcome',
    treat='treated',
    unit='county',
    time='year',
    W=W,
    covariates=['income', 'population'],
    cluster='county',
)

r.direct_effect
r.spillover_effect
r.total_effect
r.summary()
```

The result follows StatsPAI's tidy/export contract:

```python
r.tidy()
r.glance()
r.to_csv('spatial_did_results.csv')
r.to_excel('spatial_did_results.xlsx')
r.to_markdown('spatial_did_results.md')
r.to_latex(caption='Spatial DiD estimates')
```

Plots are built in:

```python
r.plot(kind='coef')       # direct, spillover, and total effects
r.plot(kind='exposure')   # treatment vs. spatial exposure distribution
```

For spatially correlated residuals, use Conley-style spatial HAC
standard errors with either a unit-level distance matrix or
latitude/longitude columns:

```python
r = sp.spatial_did(
    df,
    y='outcome',
    treat='treated',
    unit='county',
    time='year',
    W=W,
    lat='lat',
    lon='lon',
    se_type='conley',
    conley_cutoff=100,      # kilometers when lat/lon are supplied
)
```

Spatial event-study paths decompose dynamic effects into own-treatment
and neighbor-exposure components:

```python
r = sp.spatial_did(
    df,
    y='outcome',
    treat='treated',
    unit='county',
    time='year',
    W=W,
    event_study=True,
    event_window=(-4, 4),
)

r.plot(kind='event_study')
r.detail['pretrend_test']
```

Methodologically, this is the local-spillover DiD design associated
with Delgado--Florax and related spatial DiD applications. It is a
direct/spillover workflow, not a claim that spatial interference is
fully solved: recent work on staggered adoption with spillovers,
spatial synthetic DiD, and network-interference DiD remains an active
frontier.

## Result objects

Spatial regression results expose:

```python
r.summary(); r.to_latex(); r.cite()
r.direct_effects; r.indirect_effects; r.total_effects   # LeSage-Pace 2009
r.plot(kind='residuals')
r.plot(kind='moran')         # Moran scatter of residuals
```

`SpatialDiDResult` uses singular effect names for the causal estimand
and adds tidy/export helpers:

```python
r.summary()
r.direct_effect; r.spillover_effect; r.total_effect
r.tidy(); r.glance()
r.to_csv(); r.to_latex(); r.to_excel(...)
r.plot(kind='coef')
```
