# Geometry materialisation and replacement maps

Generated 2026-09-27T14:56:19Z by `notebooks/20_unit_geometries_and_replacement_maps.py`.

- masks: reference 33157 px (2984 ha) | model 75492 px (6794 ha)
- reference rebuilt 1064 units, identity asserted against the final table; pixel counts match
- model     rebuilt 2209 units, identity asserted against the final table; pixel counts match
- reference geometry: 1064 features | area from geometry 2819.70 ha vs pixel count 2819.70 ha | exact match: True
- model     geometry: 2209 features | area from geometry 5397.48 ha vs pixel count 5397.48 ha | exact match: True
- agreement: Jaccard 0.2055 | reference also detected 55.9% | model outside inventory 75.5%
- 
Figure 10 attribute ranges over the 1064 reference-defined units:
-    mean_severity_model          min 0.0150  median 0.4941  max 0.9560
-    mean_persistence_model       min 0.0720  median 0.5671  max 0.9411
-    mean_recovery_gap_predicted  min 0.0009  median 0.4970  max 0.9986
-    mean_recurrence_model        min 0.0000  median 0.0667  max 0.7429
- wrote figure10_replacement_unit_indicators.{png,pdf}
- 
within-LC top decile but below the global cut: 37 units across the six main vegetated classes
-    forest     8 of 16
-    orchards   4 of 4
-    shrubland  0 of 25
-    grassland  0 of 29
-    cropland   25 of 27
-    fallow     0 of 5
- wrote figure11_replacement_unit_prioritization.{png,pdf}
