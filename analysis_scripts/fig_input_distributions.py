"""Measured per-run distributions that parameterise the generator.

One radio, one configuration, every run of a single capture session. These are
the quantities the variation spec draws from: the generator holds the profile
centre fixed per radio and redraws these once per synthetic run.

ONE CRYSTAL, TWO ESTIMATORS. The sampling clock mismatch IS the reference
oscillator here, and CFO is derived from it by a fixed offset. They are not
independent measurements: across all six configs they correlate at pearson
r = 1.0000, ratio ~1.01, sd(difference) 0.0008-0.0017 ppm. The clock estimator
is taken as the reference because it is 5.8x the more precise of the two.

(synth_dataset internally names the primary variable ref_ppm and centres it on
the CFO mean, then adds clock_offset_ppm to get the clock. That is the same
model up to a constant -- the presentation here just puts the precise estimator
first, which is what the numbers support.)

WHY THE CFO PANEL'S NARROW CURVE DOES NOT MATCH ITS HISTOGRAM. Three estimators
of that one oscillator differ enormously in precision -- clock mismatch 0.403 Hz,
LO leakage tone 0.895 Hz, CFO 2.326 Hz -- so ~97% of the CFO histogram's variance
is ESTIMATOR ERROR rather than oscillator wander. It is NOT thermal noise: the
CRLB over the 1 ms data segment at 49 dB SNR is 0.027 Hz per burst, so the excess
runs 84x that (729x if CFO averages 76 bursts). The leakage tone is the check
that settles it -- the same LO seen a second way moves only 0.895 Hz, so at most
that is real wander and the remaining 2.15 Hz is error in the CFO estimator,
whatever its mechanism (modulation-dependent bias, residual timing, ISI). The
generator therefore takes ref_ppm's spread from the clock estimator. Plotting
that injected marginal straight onto the CFO histogram compares an injected
truth against a noisy measurement, which is why the CFO panel shows BOTH the
injected marginal and that marginal broadened by the CFO estimator noise -- the
latter is what a CFO measurement of synthetic data would actually look like.

For the Gaussian parameters the generator uses the measured sd directly, so on
those panels the estimator noise is already inside the model and the curve sits
on the histogram by construction.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RAD = sys.argv[1] if len(sys.argv) > 1 else "30BF779"
CFG = sys.argv[2] if len(sys.argv) > 2 else "89_433"

# The generator's OWN mixture selection is imported rather than refitted here,
# so this panel cannot disagree with what synth_dataset will actually sample.
# argv is blanked first: synth_dataset parses environment and paths at import.
_argv, sys.argv = sys.argv[1:], ["x"]
for _q in ("", "PA_modelling_with_GMP", "analysis_scripts", "analysis_scripts/bottom_up"):
    sys.path.insert(0, str(BASE / _q) if _q else str(BASE))
import synth_dataset as SD          # noqa: E402
sys.argv = ["x"] + _argv

log = json.loads((BASE / "data" / "radio_characterisation.json").read_text())
pat = re.compile(rf"^TX{RAD}_RXBB60_(\d+)/g{CFG}$")
rows_all = [v for v in log.values()
            if isinstance(v, dict) and pat.match(str(v.get("session", "")))]
if not rows_all:
    raise SystemExit(f"no runs for {RAD}/{CFG}")
# ONE SESSION ONLY. A radio can have more than one session in the log (five do),
# and matching on radio+config alone silently pools them -- mixing capture days,
# and for the five old-format sessions mixing rows that carry iq_amp_db with
# rows that do not. Pick one: prefer current-format, then most runs, then the
# lowest session id so the choice is deterministic.
_by_sess = {}
for v in rows_all:
    _by_sess.setdefault(pat.match(str(v["session"])).group(1), []).append(v)
sess = max(_by_sess, key=lambda s: (
    any(x.get("iq_amp_db") is not None for x in _by_sess[s]),
    len(_by_sess[s]), "0" + s))
rows = _by_sess[sess]
if len(_by_sess) > 1:
    print(f"  {RAD}/{CFG}: {len(_by_sess)} sessions "
          f"({', '.join(sorted(_by_sess))}) -- using {sess}")


def col(k):
    return np.array([r[k] for r in rows if r.get(k) is not None], float)


fc = col("fc_hz").mean()

# MODEL PARAMETERS ARE TAKEN THE WAY synth_dataset DERIVES THEM, not refitted
# here, so each overlay is the distribution the generator actually samples.
_cfo = col("cfo_hz") / (fc * 1e-6)
_clk = col("clock_mismatch_ppm")
_x = _cfo - _cfo.mean()
PHI = (float(np.clip(np.corrcoef(_x[:-1], _x[1:])[0, 1], 0.0, 0.98))
       if _x.size > 2 and _x.std() > 0 else 0.0)
REF_MU, REF_SD = float(_cfo.mean()), float(_clk.std())
CLK_OFF = float(np.median(_clk - _cfo))
# The clock estimator viewed as a direct reading of ref_ppm: same quantity,
# offset removed. This is the panel that shows what "ref osc" actually is.
_ref = _clk - CLK_OFF
# CFO estimator noise, backed out in quadrature. Guarded: if the CFO estimate
# were ever the quieter one this would go imaginary, and a zero is the honest
# answer rather than a nan.
CFO_NOISE = float(np.sqrt(max(_cfo.var(ddof=1) - _clk.var(ddof=1), 0.0)))
CFO_TOT = float(np.hypot(REF_SD, CFO_NOISE))
# LO leakage: the generator uses the MEDIAN and a ROBUST (IQR/1.349) sd after
# trimming, not the sample mean and sd -- the distribution is bimodal, so a
# plain Gaussian fit would misstate the model.
_lk_all = col("lo_leakage_dbc")
_q1, _q3 = np.percentile(_lk_all, [25, 75])
_lk = _lk_all[_lk_all > _q1 - 3.0 * max(_q3 - _q1, 0.5)]
LEAK_MU = float(np.median(_lk))
LEAK_SD = float((np.percentile(_lk, 75) - np.percentile(_lk, 25)) / 1.349)

GAUSS_COL, AR_COL, UNI_COL, NOISE_COL = "#C1504D", "#7B5EA7", "#3F8F5B", "#B07AA1"
MIX_COL = "#E8A33D"


# TX LO leakage is bimodal on most radio/configs (95 of 138 in the 100-run log).
# synth_dataset.fit_mixture_bic decides -- BIC margin -10 plus a degeneracy
# guard -- and returns None where a single Gaussian is the better model, so this
# panel shows a mixture exactly when the generator will draw one.
MIX = SD.fit_mixture_bic(_lk)
PHI_LAB = "AR(1)  $\\phi$=" + f"{PHI:.2f}"

# (name, samples, unit, [curves]). Each curve is
#   ("gauss", mu, sd, colour, linestyle, label)  or  ("uniform", lo, hi, ...).
# A panel with no curve given falls back to a Gaussian at the sample mean/sd,
# which is exactly what the generator uses for those three.
# THE CLOCK MISMATCH IS THE REFERENCE OSCILLATOR. There is one crystal and two
# estimators of it; the clock estimator is 5.8x the more precise, so it is the
# reference and CFO is derived from it by a fixed offset. An earlier version
# carried a separate "ref_ppm" panel, but that was the clock panel shifted by a
# constant -- it invited the reader to count three oscillator measurements where
# the data holds two, one of which is mostly estimator error.
# SOLID = a distribution the generator samples. DASHED is reserved for the one
# curve that is NOT a generator model: the CFO marginal broadened by that
# estimator's error, which exists only to explain why the CFO histogram is wide.
PANELS = [
    # CENTRE AS THE GENERATOR DOES IT. It centres the reference on the CFO
    # MEAN and adds the MEDIAN of (clock - CFO); centring on the clock's own
    # sample mean instead put this curve 0.49 sigma off what is sampled.
    ("Sampling clock mismatch", _clk, "ppm",
     [("gauss", REF_MU + CLK_OFF, REF_SD, AR_COL, "-", PHI_LAB)]),
    ("CFO  = clock mismatch " + f"{-CLK_OFF:+.4f}" + " ppm", _cfo, "ppm",
     [("gauss", REF_MU, REF_SD, AR_COL, "-", "derived"),
      ("gauss", REF_MU, CFO_TOT, NOISE_COL, "--", "+ est. error")]),
    # WRAP ARTIFACT, not a generator input. radio_characterise stores this as
    # clock_phase_mean_samp and its own comment calls it a wrapping artifact,
    # preferring clock_phase_intercept_samp (the burst-0 phase = cp0, the actual
    # run-level draw). That field does not exist in the July characterisation,
    # so it cannot be plotted here. The uniform curve is the model for cp0, NOT
    # for this statistic -- kept only to show what the generator draws.
    # cp0's uniformity is verified on the August repeat set, where the intercept
    # does exist: 30BF779/g89_433 gives R=0.250, Rayleigh p=0.548, KS p=0.159,
    # sd 0.2548 against 0.2887 predicted by uniform.
    # SNR REPLACES CLOCK PHASE. clock_phase_mean_samp is the mean of a
    # fully-wrapped sweep (within-run sd 0.2876 against 0.2887 for a complete
    # sweep), so it reports the sweep geometry -- set by the clock mismatch --
    # rather than the sampling-grid offset, and radio_characterise's own
    # comment calls it a wrapping artifact. The well-defined quantity is
    # clock_phase_intercept_samp (cp0), which only 5 of 23 July radios carry.
    # SNR is present in every row, separates devices 1.7-2.9x, and replicates
    # July->August at rs 0.61-0.93.
    ("AWGN (SNR)", col("snr_mean_db"), "dB", None),
    ("IQ amplitude imbalance", col("iq_amp_db"), "dB", None),
    ("IQ phase imbalance", col("iq_phase_deg"), "deg", None),
    # Only the model the generator actually draws is shown. The superseded
    # single Gaussian is still computed above because it remains the model
    # wherever fit_mixture_bic declines the mixture (43 of 138 radio/configs).
    ("TX LO leakage", _lk_all, "dBc",
     ([("mix", (MIX["w"], MIX["mu"], MIX["sd"]), None, MIX_COL, "-",
        "mixture")]
      if MIX is not None else
      [("gauss", LEAK_MU, LEAK_SD, GAUSS_COL, "-", "Gaussian")])),
]

# SIZED FOR PRINT, NOT FOR SCREEN. IEEE two-column full width is 7.16 in;
# setting figsize to the final width makes LaTeX scale by 1.0, so the point
# sizes below are the point sizes on the page. A 10.4 in figure dropped into
# \includegraphics[width=	extwidth] was being shrunk by 0.69, turning 8.4 pt
# titles into 5.8 pt.
fig, axes = plt.subplots(3, 2, figsize=(5.5, 6))
for ax, (name, v, unit, curves) in zip(axes.ravel(), PANELS):
    mu, sd = float(v.mean()), float(v.std(ddof=1))
    ax.hist(v, bins=18, density=True, color="#4E79A7", alpha=0.75,
            edgecolor="white", linewidth=0.6, label="measured")
    if curves is None:
        curves = [("gauss", mu, sd, GAUSS_COL, "-", "Gaussian")]
    lo_x, hi_x = v.min(), v.max()
    for kind, a, b, c, ls, lab in curves:
        if kind == "mix":
            # fit_mixture_bic returns LISTS (the spec must be JSON-safe), so
            # they are coerced here rather than assumed to be arrays.
            wts = np.asarray(a[0], float)
            mus = np.asarray(a[1], float)
            sds = np.asarray(a[2], float)
            xg = np.linspace(min(v.min(), (mus - 3.4 * sds).min()),
                             max(v.max(), (mus + 3.4 * sds).max()), 800)
            yg = sum(wi * np.exp(-0.5 * ((xg - mi) / si) ** 2)
                     / (si * np.sqrt(2 * np.pi))
                     for wi, mi, si in zip(wts, mus, sds))
            ax.plot(xg, yg, ls, color=c, lw=1.6, label=lab)
            lo_x, hi_x = min(lo_x, xg[0]), max(hi_x, xg[-1])
        elif kind == "uniform":
            pad = 0.02 * (b - a)
            xs = np.array([a - pad, a, a, b, b, b + pad])
            ys = np.array([0.0, 0.0, 1.0 / (b - a), 1.0 / (b - a), 0.0, 0.0])
            ax.plot(xs, ys, ls, color=c, lw=1.6, label=lab)
            lo_x, hi_x = min(lo_x, a), max(hi_x, b)
        else:
            w = 3.4 * b
            x = np.linspace(a - w, a + w, 600)
            ax.plot(x, np.exp(-0.5 * ((x - a) / b) ** 2)
                    / (b * np.sqrt(2 * np.pi)), ls, color=c, lw=1.6, label=lab)
            lo_x, hi_x = min(lo_x, x[0]), max(hi_x, x[-1])
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    # KIND MOVED OUT OF THE TITLE and into the legend, beside the curve it
    # describes; the title now carries only the measurement.
    ax.set_title(name,
                 fontsize=10, linespacing=1.25)
    ax.set_xlabel(unit, fontsize=10.0)
    ax.set_ylabel("density", fontsize=10.0)
    ax.legend(fontsize=8.0, loc="upper left", framealpha=0.9,
              handlelength=1.4, handletextpad=0.5, borderpad=0.3,
              labelspacing=0.3)
    ax.tick_params(labelsize=10.0)
    ax.grid(True, alpha=0.25)
for k in range(len(PANELS), axes.size):
    axes.ravel()[k].axis("off")

fig.suptitle(f"Per-run input distributions   {RAD}, {CFG}, {len(rows)} runs "
             f"(session {sess})", fontsize=11, y=0.94)
fig.tight_layout(rect=[0, 0, 1, 0.972])
out = BASE / "figures" / f"A4s_input_distributions_{RAD}_{CFG}.png"
fig.savefig(out, dpi=190, bbox_inches="tight")
print(f"  -> {out}")
for name, v, unit, _c in PANELS:
    print(f"  {name:<40}{v.mean():>12.5g} +/- {v.std(ddof=1):<11.4g}{unit:<6}"
          f"n={len(v)}")
print(f"\n  ref osc sd {REF_SD:.5f} ppm (clock estimator)")
print(f"  CFO estimator noise {CFO_NOISE:.5f} ppm -> total {CFO_TOT:.5f} ppm"
      f"   (measured CFO sd {_cfo.std(ddof=1):.5f})")
