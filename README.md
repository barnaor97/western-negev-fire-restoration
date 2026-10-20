# Prioritizing Post-Fire Ecological Restoration in the Western Negev (2017-2025) using Sentinel-2 Time Series and Deep Learning

Research code for the MSc thesis of the same title (Bar Naor, Department of Industrial
Engineering and Management, Ben-Gurion University of the Negev), supervised by
Prof. Tarin Paz-Kagan and Prof. Yael Edan.

This repository holds the code, configuration and non-sensitive result tables behind
the thesis. Trained model weights are not included in this release; see section 11. It does **not** hold the two restricted third-party
datasets the study depends on; see [Datasets](#6-datasets).

---

## 1. Research purpose

The study builds a decision-support framework for prioritizing post-fire ecological
restoration across the Western Negev (Gaza Envelope) over 2017-2025, a region whose
fire regime changed substantially during the 2023-2025 conflict period. It asks where
restoration effort should be directed when field survey is not possible and the only
continuous record is satellite imagery.

Four indicators describe each burned location: **severity**, **spectral persistence**,
**recovery gap** and **recurrence**. The first three are analytical indicators computed
from the Sentinel-2 record and are also estimated by a temporal deep-learning model;
recurrence is derived after inference from the model's monthly fire classifications.
The four are combined into a Restoration Indicator Score (RIS) under explicit weighting
profiles, and burned areas are ranked both across the region and within their own
land-cover class.

## 2. Methodology in brief

- **Inputs.** Monthly Sentinel-2 L2A composites, 2017-2025, cloud-masked with Cloud
  Score+ (`cs_cdf >= 0.60`), over a 1,938 km2 region of interest, plus WorldClim 2.1
  precipitation and the SRTM GL1 elevation model.
- **Features.** 52 per-month inputs: 5 raw bands, 7 spectral indices, 14 anomaly
  features, 4 month-to-month deltas, 2 harmonic seasonality terms, 20 land-cover
  one-hot indicators.
- **Sample.** 608,922 pixels - every burned pixel of each recurrence stratum plus a
  1:1 matched set of pixels with no documented fire - split 70/15/15, stratified.
- **Model.** A Lightweight Temporal Attention Encoder (L-TAE) adapted to pixel-level
  multi-task regression, with one per-month fire-classification head and three
  regression heads, trained under a loss that gives the regression heads gradient only
  from pixels that actually burned. 66,436 trainable parameters.
- **Evaluation.** Held-out test metrics, five-fold cross-validation, spatial, temporal
  and combined holdouts, four comparison models, four architectural ablations, a
  deployment-realism reweighting, and indicator-definition sensitivity analyses.
- **Application.** Region-wide inference at 10 m, post-hoc recurrence derivation,
  aggregation over burned spatial units intersected with fixed 2018 land cover, and
  RIS ranking globally and within land-cover class.

The architecture is **not** a contribution of this thesis. It is the published L-TAE of
Sainte Fare Garnot and Landrieu (2020), adapted here; see [NOTICE.md](NOTICE.md).

## 3. Repository structure

```
config.py  model_defs.py  metrics.py  gee_project.py   core modules; imported as `import config, model_defs, metrics`
configs/            feature and split definitions, the fitted scaler, the GEE project example
pipeline/      8    A. the main research workflow, in execution order
inference_10m/ 15   the 10 m region-wide inference and prioritization - the REPORTED results
supplementary_30m/ 13   B. the earlier 30 m workflow and the analyses built on it
experiments/   3    C. model experiments that extend the core
analysis/      2    C. pixel-level analyses
figures/       2    scientific figure generators
verification/  4    independent verification of quantitative claims
gee/           5    standalone Earth Engine export and extraction utilities
diagnostics/   52   nb13 land-cover timing and nb16 normalisation audit, with their tables
models/        64   per-experiment results and histories, and the 16 trained checkpoints
outputs/       225  aggregate result tables, figure notes, provenance records
data/          7    the ROI polygon and the restricted-data documentation
docs/          2    sanitisation log and table screening record
```

Notebook numbers are research provenance and have deliberately **not** been renumbered.
Some script names carry pre-renumbering figure numbers; the mapping to the final thesis
numbers is in `outputs/README.md`.

## 4. Execution order

### A. Main research workflow

Data preparation -> sampling -> feature selection -> analytical indicators -> model
training and evaluation -> **10 m region-wide inference** -> restoration prioritization.

| # | File | Purpose | Inputs | Outputs | Thesis |
|---|---|---|---|---|---|
| 1 | `pipeline/01_data_to_gee.ipynb` | Sentinel-2 monthly composites, monthly fire labels, seasonal baselines, as Earth Engine assets | Sentinel-2 L2A, Cloud Score+, fire-reference inventory, land-cover product | GEE image collections | 3.2, 4.1 |
| 2 | `pipeline/02_sampling_computepixels.ipynb` | Stratified pixel sample through the high-volume `computePixels` endpoint | the GEE assets | `data/raw/full_pixels_10m.csv` | 4.2.4 |
| 3 | `pipeline/03_eda.ipynb` | Exploratory analysis, feature screening, variance-inflation check | `full_pixels_10m.csv` | `configs/selected_features.json`; Figures 5, 7, A.1 | 4.2.3, A.1 |
| 4 | `pipeline/04_data_prep.ipynb` | Model tensors, the three analytical indicators, the splits | the above | `X_scaled.npy`, `y_burned.npy`, `targets_aux.npz`, `configs/splits.json`, `configs/scaler.pkl` | 4.2, 4.3 |
| 5 | `pipeline/05_main_model.ipynb` | Train the L-TAE; evaluate on the held-out test set | the tensors | `models/main/model_main.pt`, `results_main.json`; Figures 9, 10 | 4.4, 5.2-5.3 |
| 6 | `pipeline/06_experiments.ipynb` | Experiments 1-8 | the tensors | `models/experiments/exp1`-`exp8` | Tables 11, 14, 15, 16 |
| 7 | **`inference_10m/run_pipeline.sh`** | **The reported region-wide inference and prioritization at 10 m** | 10 m feature sources, `model_main.pt` | indicator surfaces, spatial units, RIS, rankings | 4.5, 4.6, 5.11-5.13 |

Stage 7 is the source of every region-wide result in the thesis. Its stages, in order:

```
build_10m_sources.py -> run_inference_10m.py -> mosaic_10m.py -> qa_10m.py
  -> geometries_10m.py -> polygon_10m.py -> apply_exceptions.py
  -> tables_412_10m.py, table20_10m.py, migration_table.py
  -> figures_10m.py, figure15_10m.py
```

### B. Supplementary 30 m inference workflow

`pipeline/07a_export_to_drive.ipynb` -> `pipeline/07b_inference_full_roi.ipynb`.

**This is an earlier workflow and is not the source of the final region-wide results.**
It is retained because several diagnostics and appendix figures were computed on the
30 m surfaces it produced, and because it documents how the method was developed. Where
a thesis figure was rendered by this chain rather than by the 10 m pipeline,
`outputs/README.md` says so in its **Implementation** column.

Built on those 30 m surfaces: `supplementary_30m/07c_upload_to_gee.ipynb` (visual
inspection only, no thesis result), `compute_model_recurrence.py`, `11_`, `12_`,
`13_landcover_timing_diagnostic.ipynb`, `14_landuse_presentation.ipynb`,
`15_polygon_workflow.py`, `17_`, `18_`, `19_`, `20_`, `21_`, and
`compose_fig_4_2_and_4_3.ipynb`.

### C. Independent experiments and supporting analyses

These were conducted independently and have **no single execution order**. Each needs
only the artefacts named in its own header.

| File | What it does | Thesis |
|---|---|---|
| `experiments/06b_experiments_extras.ipynb` | Experiments 9-12: recovery-gap window sensitivity, composite weighting sensitivity, BiGRU and TempCNN comparison models | Tables 14, 17, 18 |
| `experiments/24_exp5b_burned_only_baselines.py` | Experiment 5b: the burned-pixel-only regression comparison that makes Table 14 compare like with like | Table 14, 5.6 |
| `experiments/feature_importance_analysis_v4.ipynb` | Permutation importance over all 52 features | Figure 11, Tables 19, 24 |
| `analysis/08_target_diagnostics.ipynb` | The indicator-definition audit: baselines, normalisation, saturation, boundaries, window and threshold sensitivity | 4.3, 6.4 |
| `analysis/regenerate_table_4_5_burned_only.ipynb` | Per-land-cover indicators recomputed on burned test pixels only | Table 13 |
| `supplementary_30m/13_landcover_timing_diagnostic.ipynb` | Is the recovery-gap ordering in the indicator, in fire timing, or in the model? | Figure 14, 6.4 |
| `diagnostics/nb16_normalization_audit/nb16_normalization_audit.py` | The normalisation audit behind the within-class comparison constraint | 6.4 |
| `supplementary_30m/17_`, `18_` | Where the between-class ordering appears along the processing chain | 6.4 |
| `figures/22_study_area_environment.py` | Precipitation and elevation panels, with a stats table | Figure 3, 3.2.1 |
| `figures/23_figure7_seasonal_control.py` | Event-aligned NBR with a land-cover- and calendar-aligned no-fire comparison | Figure 8 |
| `verification/09_`, `10_`, `verify_fig06_population.py`, `thesis_verification_notebook.ipynb` | Independent recomputation of quantities quoted in the thesis | various |

### Which code implements which component

| Component | Implementation |
|---|---|
| **Region-wide inference** | `inference_10m/run_inference_10m.py` + `mosaic_10m.py` (reported); `pipeline/07b_inference_full_roi.ipynb` (supplementary 30 m) |
| **Monthly fire detection** | the classification head in `model_defs.py` (`FireRecoveryModel.head_burned`), thresholded at 0.502 in `inference_10m/qa_10m.py` and `polygon_10m.py` |
| **Model-derived recurrence** | computed **after** inference by normalising a count of detected fire months: `supplementary_30m/compute_model_recurrence.py` (30 m) and `inference_10m/polygon_10m.py` (10 m). It is **not** a fourth regression head, and it is distinct from the reference recurrence used to stratify the sample in step 2 |
| **Spatial-unit construction** | `inference_10m/geometries_10m.py` and `polygon_10m.py` (burned footprint intersected with fixed 2018 land cover); `supplementary_30m/15_polygon_workflow.py` at 30 m |
| **RIS calculation** | `inference_10m/polygon_10m.py` and `tables_412_10m.py`; `supplementary_30m/11_`, `15_`, `19_` at 30 m |
| **Ranking** (global and within land-cover class) | `inference_10m/tables_412_10m.py`, `table20_10m.py`, `figure15_10m.py` |
| **Sensitivity analyses** | indicator definitions: `analysis/08_target_diagnostics.ipynb`; recovery-gap window and RIS weight profiles: `experiments/06b_experiments_extras.ipynb` (Experiments 9, 10); architecture: `pipeline/06_experiments.ipynb` (Experiment 7); normalisation: `diagnostics/nb16_normalization_audit/` |
| **Analytical indicator definitions** | `pipeline/04_data_prep.ipynb`, audited by `analysis/08_target_diagnostics.ipynb` |
| **Model architecture and losses** | `model_defs.py` |
| **Training and evaluation helpers** | `metrics.py` |

### Not published

Two scripts that generate Figures 1 and 6 are deliberately excluded: both are
deterministic box-and-arrow diagrams of the thesis text with no data access at all.
Four superseded notebooks are also excluded; each was replaced by a notebook that is
published here - `02_sampling.ipynb` and `02_sampling_fast.ipynb` by
`pipeline/02_sampling_computepixels.ipynb`, `feature_importance_analysis.ipynb` by
`experiments/feature_importance_analysis_v4.ipynb`, and `sensitivity_analyses.ipynb` by
`experiments/06b_experiments_extras.ipynb`. None produced a thesis result.

## 5. What can actually be run

| Tier | What | Which code |
|---|---|---|
| **FULLY EXECUTABLE** | Inspect the model definition and its parameter counts, read every published result table and provenance record, re-derive the figure-to-code mapping | `model_defs.py`, `models/**/*.json`, `outputs/**` |
| **REQUIRES EXTERNAL DATA** | Training, evaluation, the indicator diagnostics, the polygon workflow, the prioritization | `pipeline/04`-`06`, `experiments/**`, `analysis/**`, `supplementary_30m/**`, `inference_10m/**` - all need `data/processed/*` rebuilt from the restricted inputs |
| **REQUIRES AUTHENTICATION** | Everything that touches Earth Engine | `gee/**`, `pipeline/01`, `02`, `07a`, `supplementary_30m/07c`, `inference_10m/build_10m_sources.py`, `run_inference_10m.py` |
| **NOT FULLY REPRODUCIBLE** | Figures 2 and 4 (no surviving generator). Figures 1 and 6 are a separate case: their generators exist but are document-formatting diagrams, deliberately not published | No generating code survives in the project; see `REPRODUCIBILITY_GAPS.md`. They are not reconstructed here, and no replacement script is presented as their source |

Running the pipeline end to end requires authorized access to the two restricted
datasets. Section 8 separates the three things that are often conflated: reproducing
the thesis results, applying the method to other data, and running the published
checkpoints.

## 6. How the four indicators and the RIS are derived

**Three indicators are model outputs.** The regression heads estimate burn severity,
persistence and recovery gap for every eligible pixel, against analytical indicators
computed from the Sentinel-2 record in `pipeline/04_data_prep.ipynb`. The analytical
indicators are constructed from spectral data, not from independently observed
ecological variables, so agreement between a head and its indicator measures how well
the model reproduces a spectral construct. **It is not ecological field validation,
and no field validation was performed in this thesis.**

**The fourth indicator is derived after inference.** Monthly fire occurrence comes from
the classification head, thresholded at the F1-optimal operating point 0.502. The
model-derived recurrence is the count of detected fire months per pixel, normalised as
`min(count / 5, 1)` (`inference_10m/polygon_10m.py`;
`supplementary_30m/compute_model_recurrence.py` at 30 m). This is distinct from the
reference recurrence, which comes from the fire-reference inventory and is used only to
stratify the modelling sample.

**Spatial units** are formed by intersecting a burned footprint with the fixed 2018
land-cover layer, subject to a minimum unit size of 36 cells (0.36 ha at 10 m). Two
footprints are carried in parallel and reported separately:

- the **reference-defined burned footprint**, from the fire-reference inventory;
- the **model-detected burned footprint**, from the thresholded classification head.

Both are summarised using the same model-derived indicators and the same formula.

**The Restoration Indicator Score** is a weighted sum of the four indicators per unit
(`inference_10m/polygon_10m.py`):

```
RIS = w_sev * severity + w_per * persistence + w_rg * recovery_gap + w_rec * recurrence
```

with three profiles published in the thesis, as (severity, persistence, recovery gap,
recurrence):

```
ecological   0.20  0.20  0.50  0.10
fire         0.35  0.35  0.20  0.10
balanced     0.25  0.25  0.25  0.25
```

Units are ranked globally and within their own land-cover class. The weights are
analytical scenarios: **no stakeholder preferences were elicited, and the rankings are
a decision-support ordering over satellite-derived indicators, not a validated
prescription of where restoration must occur.** Section 6.4 of the thesis establishes
a further constraint: because the recovery-gap indicator is normalised by the pre-fire
NBR level, within-class comparison is the more defensible basis for reading the output.

**Generating the scientific maps.** `inference_10m/figures_10m.py` (Figures 12 and 13)
and `figure15_10m.py` (Figure 16) render the indicator and prioritization maps from the
10 m surfaces; `supplementary_30m/compose_fig_4_2_and_4_3.ipynb` renders the RIS
weighting maps of Appendix Figure A.2 at 30 m; `figures/22_study_area_environment.py`
runs on public data alone. All but the last need the restricted inputs.

## 7. Required external datasets

Compiled by inspecting every path the code opens and every Earth Engine asset key in
`config.py`, not from a general description of the method. Nine inputs are required
across the pipeline; four are public, two are restricted third-party research data,
one is included here, and two are intermediate products built by stage 1.

| # | Input | Included? | Restricted? |
|---|---|---|---|
| 1 | Sentinel-2 L2A surface reflectance | no - retrieved through Earth Engine | no |
| 2 | Cloud Score+ cloud and shadow scores | no - retrieved through Earth Engine | no |
| 3 | Fire-reference inventory, **with dates** | no | **yes** |
| 4 | Yearly land-cover product | no | **yes** |
| 5 | Study-area boundary (ROI polygon) | **yes** | no |
| 6 | Human-infrastructure mask | no | no, but not redistributed here |
| 7 | WorldClim 2.1 bio_12 | no - downloaded automatically | no |
| 8 | SRTM GL1 elevation | no - downloaded automatically | no |
| 9 | Fixed 2018 land-cover raster, 30 m | no - exported by `gee/export_landcover_2018_30m.py` | derived from 4 |

### 1-2. Sentinel-2 Level-2A and Cloud Score+

| | |
|---|---|
| Purpose | the entire feature time series: 5 raw bands, 7 spectral indices, 14 anomaly features, 4 month-to-month deltas, 2 harmonic seasonality terms |
| Obtained | programmatically through Earth Engine, by `pipeline/01_data_to_gee.ipynb` |
| Collections | `COPERNICUS/S2_SR_HARMONIZED` and `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`, linked per image |
| Masking | a pixel is kept where `cs_cdf >= 0.60`; monthly composites are the mean of the kept observations |
| Temporal coverage | **2017-01 to 2025-12** for the analysis (108 monthly steps), plus **2016-2017** for the seasonal baseline used by the anomaly features |
| Spatial coverage | the ROI of input 5 |
| CRS and resolution | EPSG:32636, 10 m; the supplementary chain also uses a 30 m grid with transform `[30, 0, 620610, 0, -30, 3514560]`, 1625 x 2598 |
| Stages | 1, and through the derived assets every later stage |
| Products written | `config.GEE["ic_s2"]`, `ic_baseline_mean`, `ic_baseline_median`, `ic_baseline_std` |

Earth Engine authentication is required. See section 10.

### 3. Fire-reference inventory

| | |
|---|---|
| Purpose | monthly fire-occurrence labels for training and evaluation; the reference recurrence that stratifies the sample; the reference-defined burned footprint used in the prioritization |
| Format | polygon vector layer. In the thesis it is two Earth Engine `FeatureCollection` assets, `config.GEE["fire_truth_l1"]` and `fire_truth_l2`, merged into `fire_truth_merged`; the local copy used by the 30 m analyses is `data/thesis_maps/fig_gt_fires.shp` (3,596 polygons) |
| **Required attributes** | **an ignition date per polygon.** The labelling step assigns each polygon to the month containing its ignition date and writes `config.GEE["ic_labels"]`, one burned/not-burned value per pixel per month |
| Temporal coverage | 2017-2025, matching the analysis window |
| CRS | WGS84 for the shapefile; reprojected to EPSG:32636 for rasterisation |
| Configuration keys | `config.GEE["fire_truth_l1"]`, `fire_truth_l2`, `fire_truth_merged`; `data/thesis_maps/fig_gt_fires.shp` |
| Stages | 1, 2, 4, and the reference-footprint branch of 7 |
| Restricted | **yes.** Merges KKL fire records with Planet- and Sentinel-2-derived burn-scar polygons, and carries internal operational annotations. Not redistributed |

**Undated fire polygons are not sufficient.** Without a date per polygon the monthly
labels cannot be built, and with them go the classification head, the severity and
persistence indicators, and the recurrence strata. A fire perimeter layer carrying only
geometry can support a burned/not-burned mask, nothing in this pipeline that is
monthly.

**Exact reproduction of the thesis results requires this inventory and its
preprocessing.** A different fire inventory can be used to apply the methodology,
provided it carries dates at the same granularity, but it produces different training
labels, a different sample, and therefore different numbers. Results obtained that way
are an application of the method, not a reproduction of the thesis.

### 4. Yearly land-cover product

| | |
|---|---|
| Purpose | 20 of the 52 input features (the previous-year and current-year one-hot class indicators); the class dimension of every per-class result; the land-cover component of spatial-unit construction |
| Format | Earth Engine `ImageCollection`, one single-band integer image per year, `system:index` equal to the year |
| Classes | ten codes, mapped in `config.LC_CATEGORIES`: 10 forest, 11 orchards, 20 shrubland, 30 grassland, 40 cropland, 41 covered agriculture, 50 built-up, 60 bare, 61 fallow, 80 water. A substitute layer must be remapped to these codes, or `LC_CATEGORIES` edited, or the one-hot feature block will not align with the trained weights |
| Temporal coverage | 2018-2025 yearly. The thesis fixes **2018** as the reference year for spatial-unit construction and for every per-class comparison |
| CRS and resolution | native 10 m in EPSG:32636; the supplementary chain uses a 30 m area-weighted mode aggregation |
| Configuration keys | `config.GEE["landcover_yearly"]`; exported locally to `data/raw/landcover_2018_30m.tif` |
| Stages | 1, 2, the per-class analyses, and 7 |
| Restricted | **yes.** Developed within the research group by Nitzan Hahamof. Not redistributed |

A substituted land-cover layer changes the feature block the model was trained on, so
the published checkpoints would be applied to inputs they were not fitted to. Any
substitution should be reported as such.

### 5. Study-area boundary

| | |
|---|---|
| Purpose | defines the region of interest for sampling, region-wide inference and every spatial summary |
| Format | polygon shapefile, a single dissolved feature, about 1,938 km2 |
| Location | **included: `data/thesis_maps/fig_roi_polygon.{shp,shx,dbf,prj,cpg}`**, and as the Earth Engine asset `config.GEE["roi"]` |
| CRS | WGS84 in the shapefile; EPSG:32636 for raster work |
| Stages | 1, 2, 7, and `figures/22_study_area_environment.py` |

### 6. Human-infrastructure mask

| | |
|---|---|
| Purpose | the eligibility mask: pixels inside mapped human infrastructure are excluded from sampling and from region-wide inference, so eligibility depends on data availability and this mask rather than on fire |
| Format | Earth Engine `FeatureCollection`, rasterised and inverted |
| Location | `${GEE_ASSET_ROOT}/human_infrastructure_mask_utm36n` |
| Stages | 1, 2, 7 |
| Not included | it is an asset in the author's Earth Engine project and is not redistributed here |

### 7-8. WorldClim 2.1 and SRTM GL1

Used only by `figures/22_study_area_environment.py` for the environmental-context
figure, and by `verification/09_revision_verification.py` for the climate and
topography summary tables. Both are public and are **downloaded automatically** into
`data/external/` on first run: the WorldClim 2.1 Israel tile (30 arc-second, band 12 =
mean annual precipitation) and the SRTM GL1 1 arc-second tile N31E034. No manual step.

### 9. Fixed 2018 land-cover raster

`gee/export_landcover_2018_30m.py` exports the 2018 image of input 4, aggregated to
30 m by area-weighted mode on the exact inference grid (EPSG:32636, upper-left
620610 / 3514560, 1625 x 2598), to `data/raw/landcover_2018_30m.tif`. Required by the
30 m analyses and by the land-cover diagnostics. Derived from a restricted input, so
not redistributed.

### Intermediate products, not external inputs

`data/raw/full_pixels_10m.csv`, `pixel_coords_10m.csv`, `burn_count_with_mask_10m.tif`,
`nbr_2017_2025_30m.tif`, the feature stacks, and everything under `data/processed/`
(`X_scaled.npy`, `y_burned.npy`, `targets_aux.npz`, `nbr_raw.npy`,
`X_scaled_meta.csv`, ...) are produced by stages 1-4 from the inputs above. They are
not published: they are large and derived from the restricted inputs. `data/README.md`
gives their expected shapes and dtypes.

### Included in this repository

`data/thesis_maps/fig_roi_polygon.*`; `configs/splits.json` (608,922 integer row
indices, 70/15/15, no coordinates and no labels); `configs/scaler.pkl`;
`configs/feature_names.json`; `configs/selected_features.json`; every per-experiment
`results.json` and `history.json` under `models/`; and every aggregate result table
under `outputs/`. **The trained weight files are not included in this release**; see
section 11.

## 8. Reproducibility: three different things

**A. Reproducing the exact thesis results.** Requires the restricted fire-reference
inventory and land-cover product, the Earth Engine assets built from them, the
preprocessing in stages 1-4, and the configuration preserved here (operating threshold
0.502, 52 features, 108-month sequence, 608,922-pixel sample, seed 42, the splits in
`configs/splits.json`). With those, the published checkpoints and result files allow
the reported numbers to be checked directly. **This has not been demonstrated from the
published files alone, because the restricted inputs cannot be included.**

**B. Applying the methodology to compatible data.** Possible with a different fire
inventory and land-cover layer, provided they meet the structural requirements above -
dated fire polygons, land-cover codes mapped to `config.LC_CATEGORIES`. The code runs,
the method is the same, and **the outputs will differ**: different labels give a
different sample, different training data and different indicator values. Such results
should be reported as an application of the method, not as a reproduction.

**C. Running inference from trained checkpoints.** The 16 checkpoints are **not part
of this release** (section 11). Were they available, they would still need model inputs
built by the same preprocessing: 52 features in the order given by
`configs/feature_names.json`, scaled with `configs/scaler.pkl`, as a 108-step monthly
sequence. Weights alone would not make the evaluations reproducible either, because the
evaluation sets themselves are built from the restricted inputs.

No end-to-end reproduction from the published files alone is claimed, because none is
possible without the restricted inputs.

## 9. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch, numpy>=2.0, scikit-learn, scipy, pandas
pip install -r requirements-geo.txt    # rasterio, geopandas, earthengine-api, contextily
```

`configs/scaler.pkl` was pickled under numpy 2.x and will not unpickle under numpy 1.x.

Most notebooks were run in Google Colab and detect that automatically; `config.py`
resolves the workspace root from `THESIS_ROOT`, falling back to
`/content/drive/MyDrive/thesisv4` in Colab and `~/thesisv4` locally. Set
`THESIS_ROOT` to this repository to run from a clone.

## 10. Earth Engine configuration

The project id is personal and is not committed. `gee_project.py` resolves it from, in
order: the `GEE_PROJECT` environment variable; `configs/gee_config.json`; otherwise a
placeholder that is not a usable project.

```bash
export GEE_PROJECT=your-project-id
# or
cp configs/gee_config.example.json configs/gee_config.json   # then edit it
```

`configs/gee_config.json` is listed in `.gitignore`. `gee_project.resolve()` never
raises, so importing `config.py` always works; `gee_project.require()` raises with
these instructions immediately before an Earth Engine call.

## 11. Model and checkpoints

`model_defs.py` holds `FireRecoveryModel` (input projection, L-TAE encoder, one
classification head and three regression heads), `FocalLoss`, `MultiTaskLoss` and
`MaskedMultiTaskLoss`.

Main configuration: 52 features, sequence length 108, `d_model` 128, 4 attention heads,
`d_k` 32, sinusoidal positional encoding with tau = 10,000, dropout 0.1; AdamW at
5e-4 with weight decay 1e-4, cosine annealing over 50 epochs, gradient clipping at 1.0,
batch 1024, seed 42, checkpoint selected on best validation AUC.

### Trained weights are not in this release

Sixteen checkpoints exist - the main model, the five cross-validation folds, the
spatial, temporal and combined splits, the PCA variant, the four architectural
ablations and the two neural comparison models, 5.5 MB in total. They have been
audited: each is a bare `state_dict` containing model parameters only, with no
optimizer state, no training data and no identifier of any kind, and each loads
`strict=True` into the definition in `model_defs.py` and runs.

They are **withheld from this release** for one reason only: they were trained on the
two restricted third-party datasets, and permission to redistribute weights derived
from them has not yet been confirmed. This is a rights question, not a technical or a
safety one. If permission is granted they will be added at `models/main/model_main.pt`
and `models/experiments/**/model.pt`, and this section will be updated.

What **is** published for every experiment: its `results.json` and `history.json`, so
the reported metrics, the training curves and the configuration can be read and checked
without the weights.

Were a checkpoint available, it would load with
`torch.load(path, map_location="cpu", weights_only=True)`:

```python
import torch, model_defs
m = model_defs.FireRecoveryModel(n_features=52, d_model=128, n_head=4, d_k=32,
                                 max_len=108, tau=10000, dropout=0.1)
m.load_state_dict(torch.load("models/main/model_main.pt", map_location="cpu",
                             weights_only=True), strict=True)
m.eval()
```

Four checkpoints need the variant construction published in
`pipeline/06_experiments.ipynb`: `exp3_temporal` and `exp4_combined` use
`max_len=36, tau=1000` (cells 20 and 24); `exp7_ablations/no_positional_encoding`
replaces `model.ltae.pos_enc` with a pass-through and
`exp7_ablations/no_final_mlp` replaces `model.ltae.mlp` with `nn.Identity()`
(cell 32, `run_ablation`); `exp6_ltae_pca` uses `n_features=16`. Built without the
patch, `strict=True` fails for the two ablations because those keys are genuinely
absent.

The sequence-length difference matters: the positional-encoding table is a registered
buffer, so a checkpoint's `state_dict` holds more entries than the model has trainable
parameters. The thesis reports trainable parameters (66,436 for the L-TAE).

## 12. Known limitations

**Scientific**

- The three regression heads are evaluated against analytical indicators computed from
  the same spectral record, not against independently observed ecological variables.
  No field validation was carried out, and the rankings are not validated restoration
  outcomes.
- Generalization degrades under spatial and temporal transfer. Under the temporal split,
  which excludes the 2023-2025 conflict period from training, the severity correlation
  collapses to 0.010 while recovery gap is partly preserved at 0.535. The model's
  applicability is bounded by the fire regimes represented in its training window.
- Cross-class differences in the recovery-gap indicator are strongly affected by its
  normalisation by the pre-fire NBR level, so within-class comparison is the more
  defensible reading. See thesis section 6.4 and `diagnostics/nb16_normalization_audit/`.
- The spectral persistence indicator is sensitive to the sparse 2016-2017 baseline; the
  thesis sets out how a longer pre-disturbance record would improve it.

**Practical**

- Trained weights are not distributed in this release, pending confirmation of
  redistribution rights for models trained on the restricted inputs.
- Retraining and all land-cover-conditioned results need the restricted inputs.
- Figures 2 and 4 have no surviving generating code. Figures 1 and 6 are a separate
  case: their generators exist but draw diagrams of the text and are deliberately not
  published.
- `models/main/predictions_test.npz` (90 MB) is not published; a few diagnostics read it.
- `pipeline/01_data_to_gee.ipynb` builds Earth Engine assets in the runner's own
  project, so asset ids differ between runners.
- Package versions for the training stage were not recorded at run time; only the
  geospatial stage is version-pinned.
- No general software licence is granted yet; see [NOTICE.md](NOTICE.md).

## 13. Attribution

See [NOTICE.md](NOTICE.md) for the architecture provenance, the basemap attribution
carried by Appendix Figure A.2, the public dataset citations and the licence status.
