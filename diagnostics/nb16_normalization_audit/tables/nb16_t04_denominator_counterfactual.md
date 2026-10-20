Isolating the denominator. Every class is given the SAME absolute NBR loss (the pooled median, 0.0421) and only the class baseline is allowed to vary. The resulting spread is produced entirely by the normalization, with no difference in observed spectral change whatsoever.

**Sources.**

- `counterfactual` — clip(pooled median absolute loss / class mean baseline NBR, 0, 1)
- `all inputs` — nbr_raw.npy and data/raw/landcover_2018_30m.tif

| land_cover | n | denominator_baseline_NBR | actual_absolute_loss | actual_recovery_gap | counterfactual_absolute_loss | recovery_gap_if_all_classes_lost_the_same |
|---|---|---|---|---|---|---|
| bare | 5099 | 0.0353 | 0.0182 | 0.4153 | 0.0421 | 1.0000 |
| water | 1525 | 0.0337 | -0.0706 | 0.0353 | 0.0421 | 1.0000 |
| grassland | 124015 | 0.0458 | 0.0432 | 0.6078 | 0.0421 | 0.9202 |
| fallow | 8063 | 0.0508 | 0.0401 | 0.5805 | 0.0421 | 0.8286 |
| built-up | 1125 | 0.0535 | 0.0187 | 0.3637 | 0.0421 | 0.7868 |
| covered agriculture | 296 | 0.0573 | -0.0339 | 0.1907 | 0.0421 | 0.7346 |
| shrubland | 62858 | 0.1033 | 0.0534 | 0.4969 | 0.0421 | 0.4076 |
| cropland | 50573 | 0.1084 | 0.0454 | 0.4807 | 0.0421 | 0.3886 |
| orchards | 4069 | 0.1366 | -0.0553 | 0.1629 | 0.0421 | 0.3082 |
| forest | 37383 | 0.2147 | 0.0394 | 0.2494 | 0.0421 | 0.1961 |
| SPEARMAN actual vs counterfactual (all classes) | 10 | nan | nan | nan | nan | 0.1581 |
| SPEARMAN actual vs counterfactual (6 main vegetated) | 6 | nan | nan | nan | nan | 0.9429 |
