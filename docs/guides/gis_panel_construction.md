# Building a historical-GIS panel: from shapefiles to an estimator

This guide walks the full path from raw historical shapefiles (CHGIS, the China
historical county shapefiles, or any comparable archive) to an analysis-ready
panel DataFrame, and then hands that panel to a StatsPAI estimator.

The motivating design is the one used throughout historical-GIS economics: a
county-year panel where treatment intensity is a *geographic* quantity —
kilometres of canal inside each county, the share of market towns within 10 km
of the canal, the distance from the county seat to the canal — interacted with
the timing of some historical shock.

StatsPAI does not reimplement GIS. It ships three thin wrappers around
`geopandas` so you stop hand-rolling the same spatial joins, and so the CRS
mistakes that silently corrupt these variables become loud errors.

## Installing the optional GIS stack

`geopandas` is an **optional extra**. The core install does not pull it in, and
the helpers below soft-import it and raise an actionable `ImportError` if it is
missing.

```bash
pip install statspai[spatial]
```

## Step 0 — the CRS decision (read this before anything else)

This is the step that quietly ruins historical-GIS panels.

Shapefiles almost always arrive in **EPSG:4326** (WGS 84), a *geographic*
coordinate system whose units are **degrees of latitude and longitude**. If you
compute a length, a buffer, or a distance in that CRS, geopandas happily
returns a number — it is just a number of degrees, measured on a plane where
one degree of longitude is ~111 km at the equator and ~85 km at 40°N. Canal
length "per county" computed this way is systematically wrong, and wrong in a
way that correlates with latitude, which is exactly the kind of measurement
error that survives a fixed-effects regression and shows up as a coefficient.

So: **reproject to a projected, metre-based CRS before measuring anything**, and
choose the projection to match the quantity you care about.

| What you are computing | Projection family to prefer |
| --- | --- |
| Lengths and distances | equidistant / conformal at your latitudes |
| Areas, area shares | equal-area (Albers, Lambert azimuthal) |
| Buffers | equidistant, centred on the study region |

For a China-wide historical panel, a national metre-based CRS such as
`EPSG:4479` (China Geodetic Coordinate System 2000) or a study-region Albers
equal-area definition is appropriate. `EPSG:3395` (World Mercator) is metric but
badly distorts distance away from the equator — acceptable for a small
low-latitude study area, not for a continental one. Whatever you pick, state it
in the paper: it is a measurement decision, not a technical detail.

StatsPAI enforces this rather than guessing. Every helper below:

- **raises** if a frame has no CRS at all;
- **raises** if a frame is in a geographic CRS, naming the CRS it found
  (e.g. `'WGS 84'`, `EPSG:4326`) and suggesting a projected alternative;
- **raises** if a frame is projected but its linear unit is not metres
  (US survey feet, for instance);
- **raises** if the two frames are in *different* CRS;
- **never reprojects silently.**

If you genuinely know what you are doing — say your data is already in metres
under a CRS that pyproj reports oddly — pass `allow_geographic=True` to opt out
explicitly. There is no way to opt out by accident.

## Step 1 — load and reproject the layers

```python
import geopandas as gpd
import statspai as sp

CRS = "EPSG:4479"  # projected, metres — see Step 0

counties = gpd.read_file("chgis/counties_1820.shp").to_crs(CRS)
canal = gpd.read_file("gis/grand_canal.shp").to_crs(CRS)
towns = gpd.read_file("gis/market_towns.shp").to_crs(CRS)
seats = gpd.read_file("gis/county_seats.shp").to_crs(CRS)
```

Repair geometries *now*, not later. Historical shapefiles are full of
self-intersecting rings and unclosed polygons; the helpers refuse to run on
invalid, empty, or missing geometries rather than dropping the offending rows
behind your back.

```python
counties["geometry"] = counties.geometry.buffer(0)     # fix self-intersections
counties = counties[~counties.geometry.is_empty].dropna(subset=["geometry"])
```

## Step 2 — build the cross-sectional geographic variables

### Canal kilometres per county

```python
canal_km = sp.line_length_in_polygon(
    canal, counties, polygon_id="county_id", unit="km"
)
#   county_id  line_length_km
# 0     11001            0.00
# 1     11002           37.42
# 2     11003           12.08
```

Counties the canal never touches get `0.0`, not `NaN` — a genuine zero, which
is what you want on the right-hand side of a regression. (If you want them
missing instead, recode explicitly; the helper will not guess.)

### Share of market towns within 10 km of the canal

```python
near_canal = sp.share_within_buffer(
    towns, canal, buffer_km=10.0, group_col="county_id"
)
#   county_id  n_points  n_within  share
# 0     11001        14         0   0.00
# 1     11002        22        19   0.86
```

Drop `group_col` to get a single overall share as a float. The buffer radius is
in kilometres and is applied in the projected CRS — which is why Step 0 matters:
a "10 km" buffer built in degrees is not 10 km anywhere.

### Distance from the county seat to the canal

```python
seats["dist_canal_km"] = sp.distance_to_feature(seats, canal, unit="km")
```

The distance is to the *nearest* feature in the target layer. The target may be
any geometry type; points falling inside a target polygon get `0`.

### Assemble the cross-section

```python
geo = (
    counties[["county_id"]]
    .merge(canal_km, on="county_id", how="left")
    .merge(near_canal[["county_id", "share"]], on="county_id", how="left")
    .merge(seats[["county_id", "dist_canal_km"]], on="county_id", how="left")
    .rename(columns={"share": "share_towns_10km"})
)
geo["canal_km"] = geo["line_length_km"].fillna(0.0)
geo["treated"] = (geo["canal_km"] > 0).astype(int)
```

## Step 3 — cross the geography with time

The geography is time-invariant (or changes only at redistricting years); the
outcome is a panel. Cross the two:

```python
import pandas as pd

outcomes = pd.read_csv("data/county_year_outcomes.csv")  # county_id, year, y
panel = outcomes.merge(geo, on="county_id", how="left", validate="many_to_one")
```

Two things worth checking before you estimate:

- **Boundary changes.** Historical county boundaries move. If your outcome
  series is coded on a different vintage than your shapefile, harmonise to a
  single consistent set of units first — an unmatched merge here is a much
  bigger problem than any CRS issue.
- **`validate="many_to_one"`.** Duplicated `county_id` in the geographic frame
  silently multiplies your panel rows. Let pandas catch it.

## Step 4 — hand the panel to an estimator

The panel is now ordinary tabular data, so any StatsPAI estimator applies.

Binary treatment with staggered adoption:

```python
res = sp.callaway_santanna(
    data=panel, y="y", g="canal_year", t="year", i="county_id"
)
res.summary()
```

Continuous treatment intensity (canal kilometres, or inverse distance):

```python
res = sp.continuous_did(
    data=panel, y="y", dose="canal_km", time="year", id="county_id", post="post"
)
res.summary()
```

Two-way fixed effects with the geographic intensity interacted with time:

```python
res = sp.feols("y ~ canal_km:post | county_id + year", data=panel)
res.summary()
```

Because treatment is spatial, plain clustering by county is usually not enough —
nearby counties share unobserved shocks. Use a spatial HAC (Conley) covariance.
Note that `sp.conley` takes the *fitted* result plus the lat/lon columns, and
`dist_cutoff` is in kilometres:

```python
fit = sp.feols("y ~ canal_km:post | county_id + year", data=panel)
res = sp.conley(fit, data=panel, lat="lat", lon="lon", dist_cutoff=100)
res.summary()
```

`sp.feols` can also do it inline via `conley_lat` / `conley_lon` /
`conley_cutoff`. Either way, check for residual spatial autocorrelation:

```python
w = sp.queen_weights(counties)
sp.moran(res.resid, w)
```

## Common failure modes

| Symptom | Cause |
| --- | --- |
| Canal lengths ~0.02 "km" | Measured in degrees; you skipped `.to_crs()` |
| `ValueError: ... geographic CRS ('WGS 84', EPSG:4326)` | Working as intended — reproject |
| `ValueError: CRS mismatch` | One layer reprojected, the other not |
| `ValueError: ... invalid geometries` | Unrepaired historical polygons; `buffer(0)` |
| All shares are 0 or 1 | Buffer radius wrong by 1000× (metres vs km) |
| Panel row count jumped after merge | Duplicate ids; use `validate="many_to_one"` |

## API reference

| Function | Returns |
| --- | --- |
| `sp.line_length_in_polygon(lines_gdf, polygons_gdf, polygon_id=None, unit='km', allow_geographic=False)` | DataFrame: one row per polygon, `line_length_<unit>` |
| `sp.share_within_buffer(points_gdf, lines_gdf, buffer_km=10.0, group_col=None, allow_geographic=False)` | float, or DataFrame with `n_points`, `n_within`, `share` |
| `sp.distance_to_feature(points_gdf, target_gdf, unit='km', allow_geographic=False)` | Series `distance_<unit>`, indexed like `points_gdf` |
