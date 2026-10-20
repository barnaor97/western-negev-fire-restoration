Baseline (2017) NBR by the fixed 2018 land-cover class, over burned sampled pixels. This is the DENOMINATOR of the recovery-gap target. A class whose pre-fire NBR is low divides by a small number, so the same absolute spectral loss produces a larger normalized value.

**Sources.**

- `baseline NBR` — nbr_raw.npy[:, 0:12] mean, exactly the window compute_targets uses
- `land cover` — data/raw/landcover_2018_30m.tif (2018 baseline)

| land_cover | n | baseline_NBR_2017_mean | baseline_NBR_2017_median | p25 | p75 | sd | pct_baseline_below_0_05 |
|---|---|---|---|---|---|---|---|
| forest | 37383 | 0.2147 | 0.2093 | 0.1397 | 0.2852 | 0.1064 | 4.9113 |
| orchards | 4069 | 0.1366 | 0.1047 | 0.0455 | 0.1905 | 0.1279 | 27.4269 |
| cropland | 50573 | 0.1084 | 0.1076 | 0.0661 | 0.1521 | 0.0673 | 18.7570 |
| shrubland | 62858 | 0.1033 | 0.1000 | 0.0381 | 0.1669 | 0.0939 | 29.2819 |
| covered agriculture | 296 | 0.0573 | 0.0576 | -0.0414 | 0.1608 | 0.1322 | 47.6351 |
| built-up | 1125 | 0.0535 | 0.0498 | 0.0046 | 0.0982 | 0.0787 | 50.1333 |
| fallow | 8063 | 0.0508 | 0.0444 | 0.0069 | 0.0900 | 0.0625 | 53.4044 |
| grassland | 124015 | 0.0458 | 0.0402 | -0.0013 | 0.0883 | 0.0709 | 55.9505 |
| bare | 5099 | 0.0353 | 0.0377 | -0.0027 | 0.0743 | 0.0614 | 58.7174 |
| water | 1525 | 0.0337 | 0.0613 | -0.0494 | 0.1028 | 0.1070 | 44.7869 |
