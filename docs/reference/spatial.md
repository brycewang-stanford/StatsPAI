# Spatial Econometrics

`statspai.spatial` — weights construction, exploratory spatial data
analysis (ESDA), spatial regression (ML and GMM), geographically-
weighted regression (GWR / MGWR), and spatial panel models. The full
stack in one module (v0.8.0, 38 API symbols).

## Weights

```python
W = sp.spatial_weights(gdf, method='queen')            # contiguity
W = sp.spatial_weights(gdf, method='knn', k=8)         # k-nearest neighbors
W = sp.spatial_weights(gdf, method='distance', band=50_000)
W.row_standardise()
W.transform                                            # 'r' | 'b' | 'v'
```

## ESDA — Exploratory Spatial Data Analysis

```python
sp.moran_i(y, W, permutations=999)        # global Moran's I
sp.geary_c(y, W, permutations=999)        # global Geary's C
sp.getis_ord_g(y, W)                      # Getis-Ord G
sp.local_moran(y, W)                      # LISA — local Moran
sp.local_geary(y, W)
sp.join_count(y_bin, W)                   # categorical autocorrelation
```

## Spatial regression (cross-section)

```python
# Maximum likelihood
sp.sar(df, y='price', x=[...], W=W)                  # spatial autoregressive
sp.sem(df, y='price', x=[...], W=W)                  # spatial error
sp.sdm(df, y='price', x=[...], W=W)                  # Durbin (lag + lagged X)
sp.slx(df, y='price', x=[...], W=W)                  # Spatially-lagged X only
sp.sarar(df, y='price', x=[...], W1=W1, W2=W2)       # combo SAR+SEM
sp.sdem(df, y='price', x=[...], W=W)                 # Durbin error

# GMM / IV
sp.sar_gmm(df, ..., W=W)
sp.sem_gmm(df, ..., W=W)
```

## GWR / MGWR — locally varying coefficients

```python
sp.gwr(df, y='price', x=['income','crime'], coords=('lon','lat'),
       kernel='bisquare', bandwidth='aic')
sp.mgwr(df, y='price', x=['income','crime'], coords=('lon','lat'))
```

## Spatial panel

```python
sp.spatial_panel(df, y='gdp', x=['k','l'], W=W,
                 i='country', t='year',
                 model='within_sar',     # 'pooled_sar' | 'within_sem' | ...
                 fe='two-way')
```

## Result objects

All spatial regression results expose:

```python
r.summary(); r.to_latex(); r.cite()
r.direct_effects; r.indirect_effects; r.total_effects   # LeSage-Pace 2009
r.plot(kind='residuals')
r.plot(kind='moran')         # Moran scatter of residuals
```
