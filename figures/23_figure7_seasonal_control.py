"""
23_figure7_seasonal_control.py — Figure 7: event-aligned NBR around documented fire months, with a
land-cover- and calendar-aligned no-fire comparison.

Why. The original Figure 7 (notebooks/10_revision_figures.py, fig06) shows mean raw NBR in a +/-12-month
window around documented fire months. Raw NBR in the study area has a strong annual cycle, and most
documented fires fall in late spring or autumn, so part of the decline before month 0 and of the rise after
it is seasonal. This script adds a comparison trajectory from the sampled pixels with no documented fire,
evaluated on the same calendar dates and in the same land-cover classes as the burned events, so the
seasonal component is visible. The comparison is descriptive; it is not a causal or counterfactual estimate.

Burned events (identical to fig06). Every pixel-month labelled burned in y_burned.npy is one event at lag 0.
An event is kept if its full window [t-12, t+12] lies inside the 108-month record. Mean and population SD
(np.nanmean / np.nanstd, ddof=0) of nbr_raw.npy are taken across events at each lag. The first-fire variant
keeps each pixel's first labelled fire month and applies the same window rule.

No-fire comparison (construction). Let C_c(t) be the mean nbr_raw over all sampled pixels with
pixel_type == "no_fires" and dominant land-cover class c (X_scaled_meta.csv, lc_dominant) at month t.
For each burned event e with land-cover class c_e and month t_e, the comparison value at lag l is
C_{c_e}(t_e + l); the comparison trajectory is the mean of these values over all events. The comparison
therefore has exactly the land-cover and calendar-date composition of the burned-event population.
No pixel-level matching is performed. Both trajectories use the same gap-filled NBR series that enters
target construction (Notebook 04).

Inputs (read-only): data/processed/y_burned.npy, nbr_raw.npy, X_scaled_meta.csv
Outputs:
    outputs/figures/fig07_event_aligned_nbr_seasonal_comparison.png
    outputs/tables/nb23_t01_fig07_curves_by_lag.csv
    outputs/tables/nb23_t02_fig07_event_landcover_composition.csv
    outputs/tables/nb23_t03_fig07_event_calendar_composition.csv
    outputs/tables/nb23_fig07_provenance.json
Run: python notebooks/23_figure7_seasonal_control.py
"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, 'data', 'processed')
FIG = os.path.join(ROOT, 'outputs', 'figures')
TAB = os.path.join(ROOT, 'outputs', 'tables')
W = 12
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
LC_NAMES = {10: 'forest', 11: 'orchards', 20: 'shrubland', 30: 'grassland', 40: 'cropland',
            41: 'covered_agriculture', 50: 'built_up', 60: 'bare', 61: 'fallow', 80: 'water'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.edgecolor': '#444444', 'axes.linewidth': 0.8})


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def events(yb, first_only):
    """(pixel, month) of every eligible event, exactly as fig06() selects them."""
    T = yb.shape[1]
    pix, t = np.nonzero(yb)                         # row-major: sorted by pixel, then month
    if first_only:
        first = np.r_[True, pix[1:] != pix[:-1]]
        pix, t = pix[first], t[first]
    keep = (t - W >= 0) & (t + W + 1 <= T)
    return pix[keep], t[keep]


def original_fig06_curves():
    """Run the unmodified fig06() from 10_revision_figures.py into a temporary directory and read back the
    plotted mean curves, so the reconstruction can be checked against the original implementation."""
    spec = importlib.util.spec_from_file_location(
        'rev10', os.path.join(ROOT, 'notebooks', '10_revision_figures.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    captured = {}
    real_close = mod.plt.close

    def grab(fig=None):
        captured['fig'] = fig
        return real_close(fig)
    with tempfile.TemporaryDirectory() as tmp:
        mod.OUT = tmp
        mod.plt.close = grab
        try:
            n_ev, n_pix, n_ff = mod.fig06()
        finally:
            mod.plt.close = real_close
    ax = captured['fig'].axes[0]
    lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) == 2 * W + 1]
    return dict(n_events=n_ev, n_pixels=n_pix, n_first=n_ff,
                mean=np.asarray(lines[0].get_ydata(), float), mean_first=np.asarray(lines[1].get_ydata(), float))


def main():
    files = {k: os.path.join(P, v) for k, v in
             dict(y_burned='y_burned.npy', nbr_raw='nbr_raw.npy', meta='X_scaled_meta.csv').items()}
    yb = np.asarray(np.load(files['y_burned'], mmap_mode='r')).astype(bool)
    nbr = np.asarray(np.load(files['nbr_raw'], mmap_mode='r'), dtype=np.float64)
    meta = pd.read_csv(files['meta'])
    N, T = yb.shape
    lc = meta['lc_dominant'].to_numpy(int)
    nofire = meta['pixel_type'].to_numpy() == 'no_fires'
    month_of_t = np.arange(T) % 12
    year_of_t = 2017 + np.arange(T) // 12
    lags = np.arange(-W, W + 1)

    # burned events
    pe, te = events(yb, first_only=False)
    pf, tf = events(yb, first_only=True)
    win = nbr[pe[:, None], te[:, None] + lags[None, :]]
    m_burn, sd_burn = np.nanmean(win, 0), np.nanstd(win, 0)
    m_first = np.nanmean(nbr[pf[:, None], tf[:, None] + lags[None, :]], 0)

    # land-cover- and calendar-aligned no-fire comparison
    classes = np.unique(lc[pe])
    series = np.full((lc.max() + 1, T), np.nan)
    n_nofire_by_class = {}
    for c in classes:
        sel = nofire & (lc == c)
        assert sel.any(), 'no no-fire pixels in land-cover class %d' % c
        series[c] = nbr[sel].mean(0)
        n_nofire_by_class[int(c)] = int(sel.sum())
    comp = series[lc[pe][:, None], te[:, None] + lags[None, :]]
    assert np.isfinite(comp).all()
    m_comp = comp.mean(0)
    diff = m_burn - m_comp

    # check against the original implementation
    orig = original_fig06_curves()
    check = dict(original_n_events=orig['n_events'], original_n_pixels=orig['n_pixels'],
                 original_n_first=orig['n_first'],
                 max_abs_diff_mean=float(np.max(np.abs(orig['mean'] - m_burn))),
                 max_abs_diff_first=float(np.max(np.abs(orig['mean_first'] - m_first))))
    assert (orig['n_events'], orig['n_pixels'], orig['n_first']) == (len(pe), len(np.unique(pe)), len(pf))
    assert check['max_abs_diff_mean'] < 1e-9 and check['max_abs_diff_first'] < 1e-9

    # figure
    fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=300)
    ax.fill_between(lags, m_burn - sd_burn, m_burn + sd_burn, color='#9aa6b2', alpha=0.28, linewidth=0,
                    label='burned events, ±1 SD')
    ax.plot(lags, m_burn, color='#16324f', lw=2.0, zorder=4,
            label='burned events, mean (n = {:,} events, {:,} pixels)'.format(len(pe), len(np.unique(pe))))
    ax.plot(lags, m_comp, color='#b5651d', lw=1.4, ls=(0, (5, 2)), marker='o', ms=3.2, mfc='white',
            mew=1.0, zorder=5,
            label='no-fire pixels, land-cover- and calendar-aligned (n = {:,} pixels)'.format(int(nofire.sum())))
    ax.axvline(0, color='#c0392b', lw=1.1, ls='--', zorder=2)
    ax.axhline(0, color='#888888', lw=0.6, zorder=1)
    ax.set_xlim(-W, W)
    ax.set_xticks(np.arange(-12, 13, 3))
    ax.set_xlabel('Months relative to documented fire month')
    ax.set_ylabel('NBR')
    ax.grid(alpha=0.25, lw=0.5)
    ax.text(0.4, 0.97, 'documented\nfire month', transform=ax.get_xaxis_transform(), fontsize=7.5,
            color='#c0392b', ha='left', va='top')
    ax.legend(fontsize=6.8, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    os.makedirs(FIG, exist_ok=True)
    out_png = os.path.join(FIG, 'fig07_event_aligned_nbr_seasonal_comparison.png')
    fig.savefig(out_png, dpi=300, facecolor='white')
    plt.close(fig)

    # tables
    os.makedirs(TAB, exist_ok=True)
    curves = pd.DataFrame(dict(lag_months=lags, burned_mean=m_burn, burned_sd=sd_burn,
                               burned_first_fire_mean=m_first, nofire_aligned_mean=m_comp,
                               burned_minus_nofire=diff, first_fire_minus_all=m_first - m_burn,
                               n_events=len(pe)))
    curves.to_csv(os.path.join(TAB, 'nb23_t01_fig07_curves_by_lag.csv'), index=False)
    lc_rows = [dict(lc_code=int(c), land_cover=LC_NAMES.get(int(c), str(c)),
                    burned_events=int((lc[pe] == c).sum()), share_of_events=float((lc[pe] == c).mean()),
                    nofire_pixels_in_class=n_nofire_by_class[int(c)]) for c in classes]
    pd.DataFrame(lc_rows).to_csv(os.path.join(TAB, 'nb23_t02_fig07_event_landcover_composition.csv'), index=False)
    cal = pd.DataFrame(dict(year=year_of_t[te], month=month_of_t[te] + 1))
    cal_month = cal.groupby('month').size().rename('events').reset_index()
    cal_month['month_name'] = [MONTHS[m - 1] for m in cal_month.month]
    cal_month['share'] = cal_month.events / len(pe)
    cal_month.to_csv(os.path.join(TAB, 'nb23_t03_fig07_event_calendar_composition.csv'), index=False)

    script = os.path.abspath(__file__)
    prov = dict(
        run_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        script=os.path.relpath(script, ROOT), script_sha256=sha256(script),
        inputs={k: dict(path=os.path.relpath(v, ROOT), sha256=sha256(v)) for k, v in files.items()},
        packages=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                      matplotlib=matplotlib.__version__),
        window_months=W,
        burned_events=dict(n_events=int(len(pe)), n_unique_pixels=int(len(np.unique(pe))),
                           n_first_fire_pixels=int(len(pf)),
                           first_event_month='%d-%02d' % (year_of_t[te.min()], month_of_t[te.min()] + 1),
                           last_event_month='%d-%02d' % (year_of_t[te.max()], month_of_t[te.max()] + 1),
                           rule='every labelled burned pixel-month with a complete +/-12-month window (as fig06)'),
        nofire_comparison=dict(
            n_nofire_pixels=int(nofire.sum()), nofire_pixels_by_landcover=n_nofire_by_class,
            construction='C_c(t) = mean nbr_raw over no_fires pixels of dominant land-cover class c at month t; '
                         'comparison at lag l = mean over burned events e of C_{c_e}(t_e + l). Composition of land '
                         'cover and calendar date equals that of the burned events. No pixel-level matching.'),
        uncertainty_band='population SD (ddof=0) of burned-event NBR at each lag, as in the original figure',
        first_fire_sensitivity=dict(max_abs_mean_difference=float(np.max(np.abs(m_first - m_burn)))),
        reproduction_of_original_fig06=check,
        outputs=[os.path.relpath(out_png, ROOT)] + ['outputs/tables/' + n for n in (
            'nb23_t01_fig07_curves_by_lag.csv', 'nb23_t02_fig07_event_landcover_composition.csv',
            'nb23_t03_fig07_event_calendar_composition.csv')])
    with open(os.path.join(TAB, 'nb23_fig07_provenance.json'), 'w') as f:
        json.dump(prov, f, indent=1)

    print('events %s | pixels %s | first-fire pixels %s | %s .. %s'
          % (f"{len(pe):,}", f"{len(np.unique(pe)):,}", f"{len(pf):,}",
             prov['burned_events']['first_event_month'], prov['burned_events']['last_event_month']))
    print('reproduction of fig06: max |diff| mean %.2e, first-fire %.2e'
          % (check['max_abs_diff_mean'], check['max_abs_diff_first']))
    print('first-fire sensitivity: max |mean difference| %.4f' % prov['first_fire_sensitivity']['max_abs_mean_difference'])
    print(curves[['lag_months', 'burned_mean', 'burned_sd', 'nofire_aligned_mean', 'burned_minus_nofire']]
          .round(3).to_string(index=False))
    print('wrote', os.path.relpath(out_png, ROOT))


if __name__ == '__main__':
    main()
