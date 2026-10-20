"""Which population does the published Figure 6 actually plot?

The figure embedded in the thesis (word/media/image6.png) is the Notebook 03 cell-28 plot: its
own title reads "Mean NBR around fire events  (n = 511)". That cell averaged NBR in a +/-12-month
window around every documented fire month of `df["pixel_id"].unique()[:500]` - the 500
lowest-numbered burned pixels - which is a performance shortcut, not a sampling decision
(see the provenance note at the top of notebooks/10_revision_figures.py).

This script recomputes both populations from the stored arrays so the count written in the
thesis note is reproducible rather than copied from a figure title. The counts depend only on
y_burned.npy, not on any NBR values, so they are independent of how NBR was imputed.

Run: python verification/verify_fig06_population.py
"""
import os
import numpy as np

# This script lived at revision/landcover_integration/ in the working project, two
# levels below the root; in this repository it sits one level below, in verification/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = 12

yb = np.asarray(np.load(os.path.join(ROOT, 'data', 'processed', 'y_burned.npy'), mmap_mode='r')).astype(bool)
T = yb.shape[1]
burned = np.where(yb.any(axis=1))[0]


def count(rows):
    """Events = one (pixel, documented fire month) pair whose +/-12-month window fits the record."""
    ev = pix = 0
    for r in rows:
        f = np.where(yb[r])[0]
        f = f[(f - W >= 0) & (f + W + 1 <= T)]
        if len(f):
            pix += 1
            ev += len(f)
    return ev, pix


ev500, pix500 = count(burned[:500])
evall, pixall = count(burned)
first = np.argmax(yb, axis=1)
ff = int(((first >= W) & (first <= T - W - 1) & yb.any(axis=1)).sum())

L = [
    "months in the record (T): %d ; burned pixels: %d" % (T, len(burned)),
    "",
    "PLOTTED in the published Figure 6 (Notebook 03 cell 28, first 500 burned pixels):",
    "  pixel-fire-month events with a complete +/-12-month window : %d" % ev500,
    "  pixels contributing at least one such event                : %d" % pix500,
    "  -> matches the n = 511 printed in the figure's own title    : %s" % (ev500 == 511),
    "",
    "WHOLE burned population (regenerated figure, outputs/figures/fig06_event_aligned_nbr.png):",
    "  pixel-fire-month events with a complete +/-12-month window : %d" % evall,
    "  pixels contributing at least one such event                : %d" % pixall,
    "  first documented fire per pixel, complete window           : %d" % ff,
    "  -> matches outputs/tables/nb09_t03_event_aligned_window_counts (319893 / 280820 / 279060)",
    "",
    "The thesis note must describe the image on the page, so the count written there is %d." % ev500,
]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig06_population_check.txt')
open(out, 'w').write("\n".join(L) + "\n")
print("\n".join(L))
print("\nwrote", out)
