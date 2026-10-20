The recovery-gap target decomposed by the fixed 2018 class: the absolute NBR loss it is built from (numerator), the pre-fire level it is divided by (denominator), the unclipped ratio, and how much of each class is destroyed by the clip to [0, 1].

**Sources.**

- `numerator` — mean(nbr[:, 0:12]) - mean(nbr[:, 96:108])
- `denominator` — abs(mean(nbr[:, 0:12])) + 1e-6
- `recovery_gap_target` — targets_aux.npz['recovery_gap']

| land_cover | n | numerator_absolute_NBR_loss | denominator_baseline_NBR | unclipped_ratio_mean | recovery_gap_target_mean | recovery_gap_target_median | pct_clipped_at_0 | pct_clipped_at_1 |
|---|---|---|---|---|---|---|---|---|
| grassland | 124015 | 0.0432 | 0.0458 | 1.4086 | 0.6078 | 0.7565 | 22.3820 | 38.8816 |
| fallow | 8063 | 0.0401 | 0.0508 | 0.2810 | 0.5805 | 0.6784 | 22.8203 | 32.8662 |
| shrubland | 62858 | 0.0534 | 0.1033 | 0.4807 | 0.4969 | 0.5087 | 22.7195 | 20.4970 |
| cropland | 50573 | 0.0454 | 0.1084 | -0.1504 | 0.4807 | 0.5166 | 22.3874 | 9.7819 |
| bare | 5099 | 0.0182 | 0.0353 | 0.2383 | 0.4153 | 0.3547 | 32.3789 | 18.5330 |
| built-up | 1125 | 0.0187 | 0.0535 | -0.6689 | 0.3637 | 0.2004 | 39.4667 | 17.5111 |
| forest | 37383 | 0.0394 | 0.2147 | -0.1787 | 0.2494 | 0.1619 | 32.0814 | 2.4423 |
| covered agriculture | 296 | -0.0339 | 0.0573 | -2.1543 | 0.1907 | 0.0000 | 63.1757 | 5.7432 |
| orchards | 4069 | -0.0553 | 0.1366 | -4.7951 | 0.1629 | 0.0000 | 62.1283 | 1.4500 |
| water | 1525 | -0.0706 | 0.0337 | -1.7859 | 0.0353 | 0.0000 | 87.2787 | 0.5902 |
