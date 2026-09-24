#!/usr/bin/env python3
"""
radio_population_bars.py — same population view as radio_population_compare.py
but every panel is a BAR chart (grouped by band) with run-to-run std error bars.
Faceted by gain (g77 | g89), rows = metrics, sorted by oscillator ppm.

    python analysis_scripts/radio_population_bars.py
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CALM_SPAN_HZ = 3.0
BANDS = [433, 915, 2400]
GAINS = [77, 89]

# FOUR JULY SESSIONS ARE UNUSABLE. They were recorded at 2400.0/433.0 instead
# of 2450.0/433.92 -- both bands wrong together, apparently without a GPSDO.
# Dropping them entirely, not just at 2400: the earlier fc filter only covered
# that band, so their 433.0 and 915 runs were still feeding the bars, and for
# 30EAE27/30ECB6B were being averaged in with a good August session.
BAD_JULY_SESSIONS = {"172217",   # 30EAE27
                     "111617",   # 30EAE54
                     "181829",   # 30ECB6B
                     "193436"}   # 30ECBAD

d = json.loads((BASE / "data" / "radio_characterisation.json").read_text())
groups = defaultdict(list)
_sess_re = re.compile(r"^TX\w+_RXBB60_(\d+)/g")
for k, v in d.items():
    m = _sess_re.match(k)
    if m and m.group(1) in BAD_JULY_SESSIONS:
        continue
    sl, _run = k.rsplit("/", 1)
    groups[sl].append(v)

# (A repeat_log.json backfill used to sit here for 30EAE54 and 30ECBAD,
#  which had only their bad-carrier July session. Both now have a proper
#  100-run entry in radio_characterisation.json -- their corrected 18/19 Aug
#  captures, characterised in full -- so the backfill is no longer needed.
#  They remain AUGUST captures standing in for July: on the leakage row they
#  sit on the other side of a ~7-9 dB session offset from the other radios.)

def is_dup_flat(k):
    if "/" in k:
        return False
    m = re.search(r"_(g\d+_\d+)$", k)
    return bool(m) and (k[:m.start()] + "/" + m.group(1)) in groups
groups = {k: v for k, v in groups.items() if not is_dup_flat(k)}

pat = re.compile(r"TX([0-9A-Za-z]+)_RXBB60_\d+/g(\d+)_(\d+)$")

def col(rv, key):
    vals = [x[key] for x in rv if isinstance(x, dict) and x.get(key) is not None]
    return np.array(vals, float) if vals else None

def mstd(rv, key):
    a = col(rv, key)
    return (float(a.mean()), float(a.std())) if a is not None else (np.nan, np.nan)

def medq(rv, key):                          # median + half-IQR (robust; for skewed var)
    a = col(rv, key)
    return ((float(np.median(a)), float((np.percentile(a, 75) - np.percentile(a, 25)) / 2))
            if a is not None else (np.nan, np.nan))

# ONE SESSION PER RADIO. This was
#     rad[ser][(g, b)] = dict(...)
# which silently kept whichever session came LAST for the radios that have
# more than one, so their bar was an arbitrary session rather than a choice.
# Sessions are collected here and exactly one is selected below; they are
# deliberately NOT averaged, because averaging mixes capture days and the
# error bar would stop meaning run-to-run spread within a session.
_per_session = defaultdict(list)
for sl, rv in groups.items():
    m = pat.match(sl)
    if not m:
        continue
    ser, g, b = m.group(1), int(m.group(2)), int(m.group(3))
    # DROP THE OLD 2.40 GHz CAPTURES. The "2400" directory holds two carriers:
    # 2450.0 MHz on 26 of the 30 radio-sessions, and 2400.0 MHz on the four
    # earliest radios (30EAE27, 30EAE54, 30ECB6B, 30ECBAD) recorded before the
    # campaign changed frequency. A 50 MHz step is not a device difference, and
    # LO leakage in particular is strongly frequency dependent.
    #
    # Filtering by CARRIER rather than by radio matters: 30EAE27 and 30ECB6B
    # each have a second session at 2450, which this keeps. Only 30EAE54 and
    # 30ECBAD drop out of the 2400 group entirely, having no 2450 session.
    # ("433" is split the same way, 433.0 vs 433.92, but that is a 0.2 %
    # step against 2 % here, so it is left alone.)
    if b == 2400:
        rv = [x for x in rv if isinstance(x, dict)
              and x.get("fc_hz") is not None and float(x["fc_hz"]) > 2.42e9]
        if not rv:
            continue
    fc, cfo = col(rv, "fc_hz"), col(rv, "cfo_hz")
    ppm = (cfo / fc * 1e6) if (fc is not None and cfo is not None) else None
    leak, span = col(rv, "lo_leakage_dbc"), col(rv, "lo_drift_span_hz")
    if leak is not None and span is not None and (span < CALM_SPAN_HZ).any():
        lc = leak[span < CALM_SPAN_HZ]; leak_ms = (float(np.median(lc)), float(lc.std()))
    else:
        leak_ms = (np.nan, np.nan)
    _sid = sl.split("_RXBB60_")[1].split("/")[0]
    _has_iq = any(isinstance(x, dict) and x.get("iq_amp_db") is not None
                  for x in rv)
    _per_session[(ser, (g, b))].append((_sid, len(rv), _has_iq, dict(
        osc_ppm = (float(ppm.mean()), float(ppm.std())) if ppm is not None else (np.nan, np.nan),
        clk_mis = mstd(rv, "clock_mismatch_ppm"),
        iq_ph   = mstd(rv, "iq_phase_deg"),
        iq_amp  = mstd(rv, "iq_amp_db"),
        amp_var = medq(rv, "amp_var_pct"),
        ph_var  = medq(rv, "phase_var_deg"),
        clk_ph  = mstd(rv, "clock_phase_mean_samp"),
        snr     = mstd(rv, "snr_mean_db"),
        # RX DC was absent while the cross-day replication figure carries it,
        # so the two figures covered different parameter sets. Median + half-IQR
        # like the other skewed quantities: one radio sits far off the pack.
        rx_dc   = medq(rv, "rx_dc_frac"),
        leak    = leak_ms,
    )))

# ONE SESSION PER RADIO, NOT A COMBINATION. Averaging two sessions would mix
# capture days, and the error bar would stop meaning run-to-run spread within a
# session. Where a radio has more than one, the chosen session is:
#   1. one whose rows carry iq_amp_db -- five radios have a second session in
#      the OLD characteriser format (iq_amp_mean_db instead), which yields NaN
#      for every IQ panel;
#   2. then the one with the most runs;
#   3. then the lowest session id, purely so the choice is deterministic.
rad = defaultdict(dict)
_chosen = {}
for (ser, gb), entries in _per_session.items():
    sid, nruns, _has_iq, stats = max(entries, key=lambda e: (e[2], e[1], "0" + e[0]))
    rad[ser][gb] = stats
    _chosen[ser] = (sid, nruns)
_multi = sorted({ser for (ser, _), e in _per_session.items() if len(e) > 1})
if _multi:
    print(f"{len(_multi)} radios have >1 session; one chosen for each:")
    for ser in _multi:
        sid, nruns = _chosen[ser]
        print(f"    {ser}  -> session {sid}  ({nruns} runs per config)")

def fam(ser):
    return "BF7" if ser.startswith("30BF7") else "EAE" if ser.startswith("30EAE") \
        else "ECB" if ser.startswith("30ECB") else "?"
FAMCOL = {"BF7": "C0", "EAE": "C1", "ECB": "C2", "?": "0.5"}
BANDCOL = {433: "C3", 915: "C4", 2400: "C5"}
# THE CONFIG NAME IS NOT THE CARRIER. "2400" is 2450.0 MHz on 26 of the 30
# radio-sessions and "433" is 433.92 MHz on the same 26; only the four earliest
# radios (30EAE27, 30EAE54, 30ECB6B, 30ECBAD) were recorded at 2400.0/433.0
# before the campaign changed frequency. The legend shows the carrier actually
# used by the majority rather than the directory label.
BANDLBL = {433: "433.92 MHz", 915: "915 MHz", 2400: "2450 MHz"}

order = sorted(rad, key=lambda s: np.nanmean([rad[s][c]["osc_ppm"][0] for c in rad[s]]))
x = np.arange(len(order))
print(f"{len(order)} radios: " + ", ".join(order))

ALL_ROWS = [("osc_ppm", "osc ppm"), ("clk_mis", "clock mismatch (ppm)"),
            ("clk_ph", "clock phase (samp)"), ("leak", "LO leakage (dBc)"),
            ("rx_dc", "RX DC (frac)"), ("iq_ph", "IQ phase (deg)"),
            ("iq_amp", "IQ amplitude (dB)"), ("amp_var", "amp variance (%)"),
            ("ph_var", "phase variance (deg)"), ("snr", "SNR (dB)")]
# ORDERED to match the cross-day figure's story: the three crystal-derived
# parameters that survive re-capture, then the partial ones, then those that do
# not. Ten metrics x two gains is 20 panels of 69 bars; at four pages that is
# unreadable, so --rows selects a subset. "core" is the set the text discusses.
# SNR REPLACES CLOCK PHASE. clock_phase_mean_samp is the mean of a
# fully-wrapped sweep (within-run sd 0.2876 against 0.2887 for a complete
# sweep), so it reports the sweep geometry -- set by the clock mismatch --
# rather than the sampling-grid offset, and radio_characterise's own
# comment calls it a wrapping artifact. The well-defined quantity is
# clock_phase_intercept_samp (cp0), which only 5 of 23 July radios carry.
# SNR is present in every row, separates devices 1.7-2.9x, and replicates
# July->August at rs 0.61-0.93.
CORE = {"osc_ppm", "clk_mis", "leak", "iq_ph", "iq_amp", "snr"}
_sel = sys.argv[1] if len(sys.argv) > 1 else "core"
ROWS = ([r for r in ALL_ROWS if r[0] in CORE] if _sel == "core"
        else ALL_ROWS if _sel == "all"
        else [r for r in ALL_ROWS if r[0] in set(_sel.split(","))])
# THE BAND LEGEND MUST NOT BE PINNED TO ROW 0. It is drawn inside the grouped-
# bar branch, but row 0 is osc_ppm, which is band-invariant and takes the other
# branch -- so "gi == 1 and ri == 0" could never fire and the legend vanished
# from every layout. Anchor it to the first row that actually draws bands.
BAND_ROW = next((i for i, (k, _) in enumerate(ROWS) if k != "osc_ppm"), None)
W = 0.27
OFF = {433: -W, 915: 0.0, 2400: +W}
EK = dict(lw=0.5, capsize=1.2)

fig, ax = plt.subplots(len(ROWS), 2, figsize=(9.5, 1.1 * len(ROWS) + 1.0),
                       sharex="col")
for gi, g in enumerate(GAINS):
    ax[0, gi].set_title(f"gain {g}", fontsize=12, fontweight="bold")
    for ri, (key, ylab) in enumerate(ROWS):
        a = ax[ri, gi]
        if key == "osc_ppm":                     # one bar per radio (band-invariant)
            # osc_ppm averages over bands, so it must also tolerate a radio
            # that lost a band to the old-carrier drop.
            miss = (np.nan, np.nan)
            mean = [np.nanmean([rad[s].get((g, b), {}).get(key, miss)[0]
                                for b in BANDS]) for s in order]
            err  = [np.nanmean([rad[s].get((g, b), {}).get(key, miss)[1]
                                for b in BANDS]) for s in order]
            a.bar(x, mean, yerr=err, color=[FAMCOL[fam(s)] for s in order], alpha=0.85,
                  error_kw=EK)
            a.axhline(0, color="k", lw=0.6)
            if gi == 0:
                for f_, c in FAMCOL.items():
                    if any(fam(s) == f_ for s in order):
                        a.bar(np.nan, 0, color=c, label=f_)
                a.legend(title="family", fontsize=7)
        else:                                    # grouped bars per band
            for b in BANDS:
                # a radio can now be absent from a band (old-carrier drop), so
                # miss -> NaN and matplotlib simply leaves that bar out
                miss = (np.nan, np.nan)
                mean = [rad[s].get((g, b), {}).get(key, miss)[0] for s in order]
                err  = [rad[s].get((g, b), {}).get(key, miss)[1] for s in order]
                a.bar(x + OFF[b], mean, W, yerr=err, color=BANDCOL[b], alpha=0.85,
                      label=BANDLBL[b], error_kw=EK)
            if gi == 1 and ri == BAND_ROW:   # one band legend, not one per row
                a.legend(fontsize=6.5, title="band", loc="best")
        if gi == 0:
            a.set_ylabel(ylab.replace(' (', chr(10) + '('),
                         fontsize=10, linespacing=1.15)
        a.grid(True, alpha=0.3, axis="y")
    ax[-1, gi].set_xticks(x)
    ax[-1, gi].set_xticklabels([str(i + 1) for i in x], rotation=90, fontsize=11)
    ax[-1, gi].set_xlabel("Radio Index", fontsize=11)

plt.suptitle(f"Radio population (bars) — {len(order)} radios, faceted by gain "
             f"(error bars = run-to-run std)", fontsize=14)
plt.tight_layout()
(BASE / "figures").mkdir(exist_ok=True)
out = BASE / "figures" / (f"A4s_radio_population_bars"
                         f"{'' if _sel == 'all' else '_core'}.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print("saved ->", out)
