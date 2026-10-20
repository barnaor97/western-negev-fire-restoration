# Final Figures 10 and 11 — provenance

Generated 2026-09-27T14:57:09Z by `notebooks/21_final_figures_10_11.py`.

No generative or AI-synthesized imagery was used; every panel is a deterministic matplotlib rendering of the research data.

## Files and SHA-256

| file | sha256 |
|---|---|
| `figure10_final.png` | `eb2d1b3b20572dbcbdd5d9946176c6a032bb53d713cfac985cb8a3c8d5b4fa6d` |
| `figure10_final.pdf` | `f6058df1c004a5539c92d652ceaf7857ccf76809c960d466e57a942e947961e5` |
| `figure11_final.png` | `5a3590a8c5adc81aeb8f55a243b76cf6b526aedc570199a296681ed0810238be` |
| `figure11_final.pdf` | `db932bf79173658dbea77a0b31b7b1c54a75975bf72b5832e77117dbe320b9c7` |

## Display limits (Figure 10)

| panel attribute | mode | vmin | vmax | observed range | units clipped |
|---|---|---|---|---|---|
| `mean_severity_model` | full | 0.0150 | 0.9560 | 0.0150 – 0.9560 | 0 |
| `mean_persistence_model` | full | 0.0720 | 0.9411 | 0.0720 – 0.9411 | 0 |
| `mean_recovery_gap_predicted` | full | 0.0009 | 0.9986 | 0.0009 – 0.9986 | 0 |
| `mean_recurrence_model` | robust | 0.0000 | 0.3777 | 0.0000 – 0.7429 | 22 |

## QA

| check | result |
|---|---|
| all rasters on the validated 30 m grid | PASS |
| reference: unit_id set identical to the final table | PASS |
| reference: no duplicated unit_id | PASS |
| reference: join is one-to-one by unit_id | PASS |
| reference: pixel counts match the final table | PASS |
| model: unit_id set identical to the final table | PASS |
| model: no duplicated unit_id | PASS |
| model: join is one-to-one by unit_id | PASS |
| model: pixel counts match the final table | PASS |
| reference units n = 1,064 | PASS |
| reference unit area = 2,819.70 ha | PASS |
| model units n = 2,209 | PASS |
| model unit area = 5,397.48 ha | PASS |
| plotted unit count equals expected (reference) | PASS |
| plotted unit count equals expected (model) | PASS |
| raw reference footprint = 2,984 ha | PASS |
| raw model footprint = 6,794 ha | PASS |
| reference only = 1,317 ha | PASS |
| overlap = 1,667 ha | PASS |
| model only = 5,127 ha | PASS |
| footprint Jaccard = 0.2055 | PASS |
| reference area also detected = 55.9% | PASS |
| model area outside inventory = 75.5% | PASS |
| within-LC hidden units (six main vegetated classes) = 37 | PASS |
| highlighted set equals the published 37 | PASS |
| cropland hidden = 25 of 27 | PASS |
| forest hidden = 8 of 16 | PASS |
| orchards hidden = 4 of 4 | PASS |
| 10a attribute 'mean_severity_model' present and finite | PASS |
| 10b attribute 'mean_persistence_model' present and finite | PASS |
| 10c attribute 'mean_recovery_gap_predicted' present and finite | PASS |
| 10d attribute 'mean_recurrence_model' present and finite | PASS |
| 11d attribute 'within_landcover_RIS_percentile' present and finite | PASS |
| outlined units = 37 | PASS |
| outline geometry repair preserves area | PASS |
| outlined geometries valid | PASS |
