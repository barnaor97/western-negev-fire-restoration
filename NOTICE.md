# NOTICE

Scientific provenance, third-party attribution and licence status for this repository.

---

## 1. Model architecture - published, not invented here

The temporal encoder is the **Lightweight Temporal Attention Encoder (L-TAE)**:

> Sainte Fare Garnot, V. and Landrieu, L. (2020). Lightweight Temporal Self-Attention
> for Classifying Satellite Images Time Series.

The implementation in `model_defs.py` is this project's own, written from the
equations of that paper (its `LTAE` docstring maps the code to equations 1-6). No
source code from the authors' reference implementation is present in this repository.

What belongs to the thesis and what does not:

| | |
|---|---|
| **Published architecture** | The L-TAE encoder: channel grouping, learned master query per head, scaled dot-product temporal attention, the final MLP |
| **Thesis adaptations** | Pixel-level multi-task regression in place of parcel-level classification; one per-month fire-classification head plus three regression heads over a shared temporal representation; `MaskedMultiTaskLoss`, which gives the regression heads gradient only from pixels with at least one documented fire month; a positional-encoding period of tau = 10,000 chosen for a ~3,000-day sequence |
| **Original to the thesis** | The three analytical indicators (severity, spectral persistence, recovery gap) and their definitions; the post-hoc derivation of recurrence from the classification head; the integration of all four into the Restoration Indicator Score; the land-cover-aware spatial-unit prioritization framework and its within-class ranking constraint |

The sinusoidal positional-encoding formula in `PositionalEncoding` is equation 3.5 of
Vaswani, A. et al. (2017), *Attention Is All You Need*.

## 2. Basemap attribution

Appendix Figure A.2 of the thesis is drawn over CARTO Positron tiles, built on
OpenStreetMap data, through `contextily`
(`supplementary_30m/compose_fig_4_2_and_4_3.ipynb`):

> (C) OpenStreetMap contributors  (C) CARTO

That credit is rendered into the figure itself, in all four panels. **It must be
preserved in any re-render or re-crop of that figure.** No basemap-derived image is
redistributed through this repository.

## 3. Datasets

Accessed programmatically and not redistributed, each cited in the thesis:

- Sentinel-2 Level-2A surface reflectance (Copernicus / ESA)
- Cloud Score+ `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`
- WorldClim 2.1 annual precipitation (Fick and Hijmans, 2017)
- SRTM GL1 digital elevation model (Farr et al., 2007)
- ESA WorldCover 2021

Restricted third-party research data, documented in `data/README.md` and
`data/restricted/README.md` but **not** redistributed:

- the fire-reference inventory merging the KKL fire records with Planet- and
  Sentinel-2-derived burn-scar polygons, which carries internal operational annotations
- the yearly land-cover product developed within the research group by Nitzan Hahamof

Trained model weights were produced by training on those restricted inputs and are
**not distributed in this release**, pending confirmation that weights derived from
them may be redistributed. The per-experiment metrics and training histories under
`models/` are published.

## 4. Dependencies

Third-party packages are declared in `requirements.txt`, `requirements-geo.txt` and
`requirements-geo-lock.txt` and are not vendored into this repository. Their own
licences govern their use.

## 5. Licence status

**No general software licence is granted at this time.** The code is published so the
thesis can be read, checked and reproduced; rights are otherwise reserved. For reuse,
modification or redistribution beyond that, please contact the author.

A licence may be added later. This notice will be updated if it is.

## 6. Citation

> Naor, B. (2026). *Prioritizing Post-Fire Ecological Restoration in the Western Negev
> (2017-2025) using Sentinel-2 Time Series and Deep Learning.* MSc thesis, Department
> of Industrial Engineering and Management, Ben-Gurion University of the Negev.
