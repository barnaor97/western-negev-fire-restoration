"""
nb16_normalization_audit.py — AUDIT ONLY. Nothing is changed, retrained or redefined.

Answers the one question the existing diagnostics could not: all of Notebook 13 used
X_scaled_meta['lc_dominant'] (the modal class over 2018-2025), but the revised polygon workflow
uses the FIXED 2018 baseline class from data/raw/landcover_2018_30m.tif. This script:

  t01  how much the two land-cover definitions actually disagree on burned pixels
  t02  baseline (2017) NBR distribution by the fixed 2018 class
  t03  the recovery-gap target decomposed into numerator and denominator by the fixed 2018 class
  t04  a counterfactual isolating the denominator: what each class's recovery gap would be if
       every class had suffered the SAME absolute NBR loss

Writes diagnostics/nb16_normalization_audit/tables/. Nothing is written to outputs/.
"""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(ROOT, "data", "processed")
TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
os.makedirs(TAB, exist_ok=True)
LC_NAMES = {10: "forest", 11: "orchards", 20: "shrubland", 30: "grassland", 40: "cropland",
            41: "covered agriculture", 50: "built-up", 60: "bare", 61: "fallow", 80: "water"}


def save(stem, df, caption, sources):
    df.to_csv(os.path.join(TAB, stem + ".csv"), index=False)
    lines = [caption, "", "**Sources.**", ""] + ["- `%s` — %s" % (k, v) for k, v in sources.items()] + [""]
    cols = list(df.columns)
    lines += ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(
            ("%.4f" % r[c]) if isinstance(r[c], (float, np.floating)) and np.isfinite(r[c])
            else str(r[c]) for c in cols) + " |")
    open(os.path.join(TAB, stem + ".md"), "w").write("\n".join(lines) + "\n")
    print("\n===== %s" % stem)
    print(df.to_string(index=False, float_format=lambda v: "%.4f" % v))


meta = pd.read_csv(os.path.join(P, "X_scaled_meta.csv"))
coords = pd.read_csv(os.path.join(ROOT, "data", "raw", "pixel_coords_10m.csv"))
assert (meta.pixel_id.to_numpy() == coords.pixel_id.to_numpy()).all(), "pixel_id order differs"
y = np.load(os.path.join(P, "y_burned.npy"), mmap_mode="r")
burned = np.asarray(y).astype(bool).any(axis=1)
nbr = np.load(os.path.join(P, "nbr_raw.npy"), mmap_mode="r")
rg = np.load(os.path.join(P, "targets_aux.npz"))["recovery_gap"]
n = len(rg)
burned = burned[:n]
_lcd_raw = meta.lc_dominant.to_numpy()[:n]
# lc_dominant is stored as the numeric land-cover code, not as a name
lcd = np.array([LC_NAMES.get(int(v), "other") if str(v).strip().lstrip("-").isdigit() else str(v)
                for v in _lcd_raw])

# --- fixed 2018 baseline class for every sampled 10 m pixel, read off the 30 m raster
with rasterio.open(os.path.join(ROOT, "data", "raw", "landcover_2018_30m.tif")) as s:
    lc2018_grid, tr, crs = s.read(1), s.transform, s.crs
xs, ys = warp_transform("EPSG:4326", crs, coords.lon.to_numpy()[:n], coords.lat.to_numpy()[:n])
col = np.floor((np.array(xs) - tr.c) / tr.a).astype(int)
row = np.floor((np.array(ys) - tr.f) / tr.e).astype(int)
inside = (row >= 0) & (row < lc2018_grid.shape[0]) & (col >= 0) & (col < lc2018_grid.shape[1])
lc18 = np.where(inside, lc2018_grid[np.clip(row, 0, lc2018_grid.shape[0] - 1),
                                    np.clip(col, 0, lc2018_grid.shape[1] - 1)], -1)
name18 = np.array([LC_NAMES.get(int(v), "outside / other") for v in lc18])
print("sampled pixels: %d | burned: %d | falling on the 30 m grid: %d" % (n, burned.sum(), inside.sum()))

# --- t01 do the two land-cover definitions agree?
b = burned & inside
rows = [dict(comparison="burned sampled pixels on the grid", n=int(b.sum()),
             pct_same_class=100.0 * float((name18[b] == lcd[b]).mean()))]
for cname in sorted(set(lcd[b])):
    m = b & (lcd == cname)
    if m.sum() < 100:
        continue
    same = float((name18[m] == cname).mean())
    top = pd.Series(name18[m]).value_counts()
    other = top[top.index != cname]
    rows.append(dict(comparison="lc_dominant = %s" % cname, n=int(m.sum()), pct_same_class=100.0 * same,
                     largest_disagreement="%s (%.1f%%)" % (other.index[0], 100.0 * other.iloc[0] / m.sum())
                     if len(other) else ""))
save("nb16_t01_landcover_definition_agreement", pd.DataFrame(rows),
     "How far the two land-cover definitions agree on burned sampled pixels. Notebook 13 used "
     "`lc_dominant` (modal class 2018-2025); the polygon workflow uses the fixed 2018 baseline raster. "
     "If agreement is high, Notebook 13's conclusions carry over to the 2018 definition.",
     {"lc_dominant": "data/processed/X_scaled_meta.csv",
      "2018 class": "data/raw/landcover_2018_30m.tif sampled at each pixel's coordinate"})

# --- t02/t03 baseline NBR and the recovery-gap decomposition, by the FIXED 2018 class
base = np.asarray(nbr[:, :12], dtype=np.float64)
recent = np.asarray(nbr[:, 96:108], dtype=np.float64)
B = np.nanmean(base, axis=1)
E = np.nanmean(recent, axis=1)
NUM, DEN = B - E, np.abs(B) + 1e-6
raw_gap = NUM / DEN
rows2, rows3 = [], []
classes = [c for c in LC_NAMES.values() if (b & (name18 == c)).sum() >= 100]
for cname in classes:
    m = b & (name18 == cname)
    rows2.append(dict(land_cover=cname, n=int(m.sum()), baseline_NBR_2017_mean=float(np.nanmean(B[m])),
                      baseline_NBR_2017_median=float(np.nanmedian(B[m])), p25=float(np.nanpercentile(B[m], 25)),
                      p75=float(np.nanpercentile(B[m], 75)), sd=float(np.nanstd(B[m])),
                      pct_baseline_below_0_05=100.0 * float((B[m] < 0.05).mean())))
    rows3.append(dict(land_cover=cname, n=int(m.sum()),
                      numerator_absolute_NBR_loss=float(np.nanmean(NUM[m])),
                      denominator_baseline_NBR=float(np.nanmean(B[m])),
                      unclipped_ratio_mean=float(np.nanmean(raw_gap[m])),
                      recovery_gap_target_mean=float(np.nanmean(rg[m])),
                      recovery_gap_target_median=float(np.nanmedian(rg[m])),
                      pct_clipped_at_0=100.0 * float((rg[m] == 0).mean()),
                      pct_clipped_at_1=100.0 * float((rg[m] >= 1).mean())))
t2 = pd.DataFrame(rows2).sort_values("baseline_NBR_2017_mean", ascending=False)
t3 = pd.DataFrame(rows3).sort_values("recovery_gap_target_mean", ascending=False)
save("nb16_t02_baseline_nbr_by_2018_class", t2,
     "Baseline (2017) NBR by the fixed 2018 land-cover class, over burned sampled pixels. This is the "
     "DENOMINATOR of the recovery-gap target. A class whose pre-fire NBR is low divides by a small "
     "number, so the same absolute spectral loss produces a larger normalized value.",
     {"baseline NBR": "nbr_raw.npy[:, 0:12] mean, exactly the window compute_targets uses",
      "land cover": "data/raw/landcover_2018_30m.tif (2018 baseline)"})
save("nb16_t03_recovery_gap_decomposition_by_2018_class", t3,
     "The recovery-gap target decomposed by the fixed 2018 class: the absolute NBR loss it is built from "
     "(numerator), the pre-fire level it is divided by (denominator), the unclipped ratio, and how much of "
     "each class is destroyed by the clip to [0, 1].",
     {"numerator": "mean(nbr[:, 0:12]) - mean(nbr[:, 96:108])",
      "denominator": "abs(mean(nbr[:, 0:12])) + 1e-6",
      "recovery_gap_target": "targets_aux.npz['recovery_gap']"})

# --- t04 counterfactual: hold the absolute loss constant, vary only the denominator
common = float(np.nanmedian(NUM[b]))
rows4 = []
for cname in classes:
    m = b & (name18 == cname)
    d = float(np.nanmean(B[m]))
    rows4.append(dict(land_cover=cname, n=int(m.sum()), denominator_baseline_NBR=d,
                      actual_absolute_loss=float(np.nanmean(NUM[m])),
                      actual_recovery_gap=float(np.nanmean(rg[m])),
                      counterfactual_absolute_loss=common,
                      recovery_gap_if_all_classes_lost_the_same=float(np.clip(common / (abs(d) + 1e-6), 0, 1)))) 
t4 = pd.DataFrame(rows4).sort_values("recovery_gap_if_all_classes_lost_the_same", ascending=False)
from scipy import stats as _st
MAIN = ["forest", "orchards", "shrubland", "grassland", "cropland", "fallow"]
_all = _st.spearmanr(t4.actual_recovery_gap, t4.recovery_gap_if_all_classes_lost_the_same).statistic
_m = t4[t4.land_cover.isin(MAIN)]
_main = _st.spearmanr(_m.actual_recovery_gap, _m.recovery_gap_if_all_classes_lost_the_same).statistic
print("\nSpearman(actual recovery gap, denominator-only counterfactual): all %d classes = %.4f | "
      "6 main vegetated classes = %.4f" % (len(t4), _all, _main))
t4 = pd.concat([t4, pd.DataFrame([dict(land_cover="SPEARMAN actual vs counterfactual (all classes)",
                                       n=len(t4), recovery_gap_if_all_classes_lost_the_same=_all),
                                  dict(land_cover="SPEARMAN actual vs counterfactual (6 main vegetated)",
                                       n=len(_m), recovery_gap_if_all_classes_lost_the_same=_main)])],
               ignore_index=True)
save("nb16_t04_denominator_counterfactual", t4,
     "Isolating the denominator. Every class is given the SAME absolute NBR loss (the pooled median, "
     "%.4f) and only the class baseline is allowed to vary. The resulting spread is produced entirely by "
     "the normalization, with no difference in observed spectral change whatsoever." % common,
     {"counterfactual": "clip(pooled median absolute loss / class mean baseline NBR, 0, 1)",
      "all inputs": "nbr_raw.npy and data/raw/landcover_2018_30m.tif"})
print("\nAUDIT COMPLETE — nothing outside diagnostics/nb16_normalization_audit/ was written.")
