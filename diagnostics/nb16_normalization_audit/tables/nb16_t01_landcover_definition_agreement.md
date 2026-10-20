How far the two land-cover definitions agree on burned sampled pixels. Notebook 13 used `lc_dominant` (modal class 2018-2025); the polygon workflow uses the fixed 2018 baseline raster. If agreement is high, Notebook 13's conclusions carry over to the 2018 definition.

**Sources.**

- `lc_dominant` — data/processed/X_scaled_meta.csv
- `2018 class` — data/raw/landcover_2018_30m.tif sampled at each pixel's coordinate

| comparison | n | pct_same_class | largest_disagreement |
|---|---|---|---|
| burned sampled pixels on the grid | 295006 | 85.0610 | nan |
| lc_dominant = bare | 5784 | 64.4364 | grassland (21.0%) |
| lc_dominant = built-up | 821 | 72.8380 | shrubland (10.0%) |
| lc_dominant = covered agriculture | 108 | 80.5556 | fallow (13.0%) |
| lc_dominant = cropland | 48360 | 92.9715 | grassland (2.7%) |
| lc_dominant = fallow | 5778 | 51.0038 | grassland (34.1%) |
| lc_dominant = forest | 38283 | 85.7482 | shrubland (10.6%) |
| lc_dominant = grassland | 123241 | 89.2236 | shrubland (4.9%) |
| lc_dominant = orchards | 4677 | 73.8935 | cropland (17.7%) |
| lc_dominant = shrubland | 66474 | 76.7458 | grassland (13.1%) |
| lc_dominant = water | 1480 | 91.6216 | shrubland (5.2%) |
