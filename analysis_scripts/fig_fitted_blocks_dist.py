"""Do the fitted blocks reproduce the measured deviation distribution?

Left column: the per-symbol deviation of the REAL capture, one thin curve per
run, so run-to-run consistency is visible. Right column: the pooled real
distribution against the generator's output with all fitted blocks active, and
against the ablation with the three burst-average-fitted blocks removed.

PLOT ONLY. Producing the arrays needs the raw captures, which are not shipped
with this repository, plus roughly half an hour of synthesis. They are stored
instead in fitted_blocks_<radio>_<config>.npz:

    A, P        (runs, bursts, symbols) real amplitude % and phase deg
    s_0_a/p     synthetic, all fitted blocks active
    s_1_a/p     synthetic, ISI taps + PA + burst transient switched off

Real and synthetic went through the SAME extractor -- matched filter, sample at
the symbol instant, deviation in each symbol's own rotated frame. Comparing
distributions produced by two different extraction paths would confound the
model with the measurement.

    python analysis_scripts/fig_fitted_blocks_dist.py [RADIO] [CONFIG]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RAD = sys.argv[1] if len(sys.argv) > 1 else "30BF779"
CFG = sys.argv[2] if len(sys.argv) > 2 else "89_433"

src = BASE / "data" / f"fitted_blocks_{RAD}_{CFG}.npz"
if not src.exists():
    raise SystemExit(f"{src.name} not found -- this figure needs the packaged "
                     f"deviation arrays, which ship only for 30BF779 / 89_433")
z = np.load(src, allow_pickle=True)

CASES = ["all fitted blocks", "AWGN + PN"]
COL = {"all fitted blocks": "#C1504D", "AWGN + PN": "#7F7F7F"}
ROWS = [("amplitude deviation", "%", "a"), ("phase deviation", "deg", "p")]

real_runs = {"a": z["A"], "p": z["P"]}
pooled = {k: v.ravel() for k, v in real_runs.items()}

print(f"  {RAD} / {CFG}   {z['A'].shape[0]} runs x {z['A'].shape[1]} bursts")
for lab, unit, key in ROWS:
    r = pooled[key]
    print(f"    REAL {lab:<20} sd {r.std():6.3f} {unit}")
    for ci, name in enumerate(CASES):
        v = z[f"s_{ci}_{key}"].ravel()
        print(f"      {name:<20} sd {v.std():6.3f} {unit}"
              f"   {v.std() / r.std() * 100:5.1f} % of real")
    miss = np.sqrt(max(r.std() ** 2 - z[f"s_0_{key}"].ravel().std() ** 2, 0.0))
    print(f"      missing term {miss:.3f} {unit}")

fig = plt.figure(figsize=(5.6, 5.6))
gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.30,
                      top=0.835, bottom=0.095, left=0.14, right=0.96)
handles = {}
for row, (lab, unit, key) in enumerate(ROWS):
    real_all = pooled[key]
    lim = np.percentile(np.abs(real_all), 99.5) * 1.35
    bins = np.linspace(-lim, lim, 90)

    ax = fig.add_subplot(gs[row, 0])
    for run in real_runs[key]:
        ax.hist(run.ravel(), bins=bins, histtype="step", density=True,
                lw=0.55, color="#2c6fbb", alpha=0.20)
    handles["pooled real"] = ax.hist(
        real_all, bins=bins, histtype="step", density=True, lw=1.6,
        color="#08306b", label="pooled real")[2][0]
    handles[f"{len(real_runs[key])} runs overlaid"] = ax.plot(
        [], [], color="#2c6fbb", lw=0.9, alpha=0.75,
        label=f"{len(real_runs[key])} runs overlaid")[0]
    ax.set_title(lab, fontsize=8.2, pad=5)
    ax.set_xlabel(f"{lab} ({unit})", fontsize=8.0)
    ax.set_ylabel("density", fontsize=8.0)

    ax = fig.add_subplot(gs[row, 1])
    handles["real"] = ax.hist(real_all, bins=bins, density=True,
                              color="#9ecae1", alpha=0.85, label="real")[2][0]
    for ci, name in enumerate(CASES):
        handles[name] = ax.hist(
            z[f"s_{ci}_{key}"].ravel(), bins=bins, histtype="step",
            density=True, lw=1.6 if ci == 0 else 1.2, color=COL[name],
            label=name)[2][0]
    ax.set_title(f"{lab} (fit vs real)", fontsize=8.2, pad=5)
    ax.set_xlabel(f"{lab} ({unit})", fontsize=8.0)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

fig.legend(handles=list(handles.values()), labels=list(handles),
           loc="upper center", bbox_to_anchor=(0.55, 0.945), ncol=5,
           fontsize=9, frameon=False, handletextpad=0.3, columnspacing=0.8)
fig.suptitle(f"Fitted blocks vs measured deviation distribution — {RAD} / {CFG}",
             fontsize=8.8, fontweight="bold", y=0.995)
for ax in fig.get_axes():
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7.0)

out = BASE / "figures" / f"A4s_fitted_blocks_dist_{RAD}_{CFG}.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=220, bbox_inches="tight")
print(f"  -> {out}")
