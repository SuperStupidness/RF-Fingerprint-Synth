#!/usr/bin/env python3
"""July vs August replication, with between-config offset removed."""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "analysis_scripts"))
import session_vs_device as S

CFG_COL = {"g77_433": "#4E79A7", "g77_915": "#7BA7CC", "g77_2400": "#A0C4E0",
           "g89_433": "#C1504D", "g89_915": "#D97C6E", "g89_2400": "#E8A798"}

# SNR REPLACES CLOCK PHASE. clock_phase_mean_samp is the mean of a
# fully-wrapped sweep (within-run sd 0.2876 against 0.2887 for a complete
# sweep), so it reports the sweep geometry -- set by the clock mismatch --
# rather than the sampling-grid offset, and radio_characterise's own
# comment calls it a wrapping artifact. The well-defined quantity is
# clock_phase_intercept_samp (cp0), which only 5 of 23 July radios carry.
# SNR is present in every row, separates devices 1.7-2.9x, and replicates
# July->August at rs 0.61-0.93.
CORE = ["cfo_ppm", "clock_mismatch_ppm", "lo_leakage_dbc",
        "iq_phase_deg", "iq_amp_db", "snr_mean_db"]

_sel = sys.argv[1] if len(sys.argv) > 1 else "core"
D = S.load()
rng = np.random.default_rng(0)
res = {}
for blk, p, unit in S.PARAMS:
    for cfg in S.CONFIGS:
        o = S.decompose(D, cfg, p, rng)
        if o is not None:
            res[(cfg, p)] = o

params = [(b, p, u) for b, p, u in S.PARAMS
          if any((c, p) in res for c in S.CONFIGS)]

if _sel == "core":
    _ord = {q: i for i, q in enumerate(CORE)}
    missing = [q for q in CORE if q not in {p for _b, p, _u in params}]
    if missing:
        raise SystemExit(f"core params absent from the decomposition: {missing}")
    params = sorted((q for q in params if q[1] in _ord), key=lambda q: _ord[q[1]])
elif _sel != "all":
    keep = set(_sel.split(","))
    params = [q for q in params if q[1] in keep]

NC, LIM = (3 if _sel == "core" else 4), 3.2
NR = int(np.ceil(len(params) / NC))

# Smaller base canvas so LaTeX doesn't shrink the fonts and panels
PANEL_SIZE = 1.15
fig_w = NC * PANEL_SIZE + 0.5
fig_h = NR * PANEL_SIZE + 1.05   # extra height buffer for bottom labels

fig = plt.figure(figsize=(fig_w, fig_h))

# NR data rows + 1 legend row
gs = fig.add_gridspec(
    NR + 1, NC,
    height_ratios=[1.0] * NR + [0.18],
    hspace=0.46,
    # 0.22 let the two-part titles ("delta ... r_s ...") run into each other at
    # this figure width -- the middle one was clipped mid-symbol. Widened rather
    # than shrinking the text, since figsize is fixed.
    wspace=0.60,
    top=0.82,
    bottom=0.10,     # lifted from 0.08 to prevent axis-label collision
    left=0.14,
    right=0.96
)

axes = np.empty((NR, NC), dtype=object)
for r in range(NR):
    for c_idx in range(NC):
        share_x = axes[0, 0] if (r > 0 or c_idx > 0) else None
        share_y = axes[0, 0] if (r > 0 or c_idx > 0) else None
        axes[r, c_idx] = fig.add_subplot(gs[r, c_idx], sharex=share_x, sharey=share_y)

for i, (blk, p, unit) in enumerate(params):
    r, c_idx = divmod(i, NC)
    a = axes[r, c_idx]
    
    a.plot([-LIM, LIM], [-LIM, LIM], "--", color="0.55", lw=0.9, zorder=1)
    a.axhline(0, color="0.88", lw=0.7, zorder=0)
    a.axvline(0, color="0.88", lw=0.7, zorder=0)

    got = [c for c in S.CONFIGS if (c, p) in res]
    offsets_raw, rs = [], []

    for c in got:
        o = res[(c, p)]
        pool = np.concatenate([o["july"], o["august"]])
        mu, sd = pool.mean(), pool.std(ddof=1)
        if sd <= 0:
            continue

        zj, za = (o["july"] - mu) / sd, (o["august"] - mu) / sd

        # Track physical offset: August mean - July mean
        offsets_raw.append(o["august"].mean() - o["july"].mean())

        out = (np.abs(zj) > LIM) | (np.abs(za) > LIM)
        zjc, zac = np.clip(zj, -LIM, LIM), np.clip(za, -LIM, LIM)

        a.plot(zjc[~out], zac[~out], "o", ms=2.8, alpha=0.75,
               color=CFG_COL[c], mew=0, zorder=3)
        a.plot(zjc[out], zac[out], "o", ms=3.0, mfc="none", mec=CFG_COL[c],
               mew=0.8, alpha=0.9, zorder=3)

        A, B = o["july"], o["august"]
        if A.std() > 0 and B.std() > 0:
            rs.append(spearmanr(A, B).statistic)

    # Median physical shift across configs
    med_offset_raw = float(np.median(offsets_raw)) if offsets_raw else 0.0
    rrs = float(np.median(rs)) if rs else float("nan")

    # Strip redundant unit suffixes from the display name
    p_display = (p.replace("_mean_samp", "")
                  .replace("_ppm", "")
                  .replace("_deg", "")
                  .replace("_dbc", "")
                  .replace("_db", ""))
    
    # Second line: delta with physical unit + Spearman rank
    a.set_title(f"{p_display}\n"
                f"$\\Delta${med_offset_raw:+.2f} {unit}  $r_s${rrs:+.2f}",
                fontsize=7.8, linespacing=1.1, pad=3)

    TICKS = [-3, 0, 3]
    a.set_xlim(-LIM, LIM); a.set_ylim(-LIM, LIM)
    a.set_aspect("equal")
    a.set_xticks(TICKS); a.set_yticks(TICKS)
    a.tick_params(labelsize=7.5)
    a.grid(True, alpha=0.25)

    if c_idx == 0:
        a.set_ylabel("August (z)", fontsize=8.5)
    else:
        plt.setp(a.get_yticklabels(), visible=False)

    if r == NR - 1:
        a.set_xlabel("July (z)", fontsize=8.5)
    else:
        plt.setp(a.get_xticklabels(), visible=False)

# Hide any empty panels in data rows
for j in range(len(params), NR * NC):
    axes[j // NC, j % NC].axis("off")

# Legend placed in the dedicated bottom row (spanning all columns)
h = [plt.Line2D([], [], marker="o", ls="", color=CFG_COL[c], label=c, markersize=4.0)
     for c in S.CONFIGS]

# Suptitle & Legend
fig.suptitle("July vs August re-capture, per-config z-score",
             fontsize=9.0, fontweight="bold", y=0.96)

# Remove any 'if spare is not None' branch so legend is ALWAYS at the very bottom
leg_ax = fig.add_subplot(gs[NR, :])
leg_ax.axis("off")
leg_ax.legend(
    handles=h,
    loc="upper center",
    ncol=6,
    fontsize=7.2,
    frameon=False,            # <--- Removes the box completely
    handletextpad=0.2,
    columnspacing=0.8,
    bbox_to_anchor=(0.5, -0.1) # <--- Clears well below the "July (z)" text
)

out = (BASE / "figures"
       / f"A4s_session_replication_z{'' if _sel == 'all' else '_core'}.png")
fig.savefig(out, dpi=250, bbox_inches="tight")
print(f"  -> {out}")