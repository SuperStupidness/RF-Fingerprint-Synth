"""Multi-run synthetic capture generator: device profile + variation spec -> SigMF.

Converges the two halves of this project.

  BOTTOM-UP (mimic_radio.py) gives the MODEL FORM and the per-device means:
  complex symbol-spaced ISI taps, a memoryless PA, a 1/f^a phase-noise mask,
  IQ imbalance, SNR. One profile dict, every number measured.

  TOP-DOWN (radio_characterisation.json, ~22k runs) gives RUN-TO-RUN VARIATION:
  which parameters actually move between runs of one session, by how much, and
  with what correlation structure.

mimic_radio.synthesise() builds ONE burst train from ONE profile. This builds a
DATASET: many runs, each with its own draw of the things that genuinely vary,
written in the real SigMF layout so radio_characterise.py reads it unchanged and
the whole thing can be validated end to end.

WHAT VARIES IS A KNOB, NOT A DECISION
-------------------------------------
Whether a parameter is a population constant or a device fingerprint depends on
the fleet, the band and the hardware -- it is the USER'S experiment, not ours to
hardcode. Every parameter therefore carries a variation spec:

    {'kind': 'fixed'}                      hold at the profile value
    {'kind': 'gauss',   'sd': x}           i.i.d. Normal about it
    {'kind': 'uniform', 'lo': a, 'hi': b}  i.i.d. Uniform
    {'kind': 'ar1',     'sd': x, 'phi': p} Gaussian that WANDERS across runs

DEFAULT_VARIATION below is what we measured on four B210s at 77_433. Override
any entry to run a different experiment -- e.g. hold `isi_taps` fixed across two
radios to test whether the taps carry identity, or give a parameter an AR(1) to
model a drifting rig.

Defaults are 'fixed' for the transfer-function parameters (ISI, PA, phase noise)
because their run-to-run scatter measured BELOW their own estimation error --
sampling them would inject our measurement noise as if it were device variation.
The state parameters (clock phase, carrier phase, CFO) genuinely vary and are
sampled.

Run:  python synth_dataset.py --radio 30BF7B6 --config 77_433 --runs 50
      python synth_dataset.py --radio 30BF7B6 --dry-run      # profile only
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.signal

BASE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "PA_modelling_with_GMP"))
from cel_signal_gen_lib.core.filter_design import srrc_design
from cel_signal_gen_lib.impairments.hardware import (
    add_cfo, add_iq_imbalance, add_symbol_clock_phase, add_phase_noise,
    add_decimation_filter, _decimation_taps)
from cel_signal_gen_lib.impairments.channel import add_awgn_snr

# ── waveform geometry (matches the capture campaign) ────────────────────────
FS, SPS, BETA, SPAN = 5e6, 5, 0.35, 10
N_PRE, N_DATA, N_TAIL = 100, 1000, 200
BURST_LEN = (N_PRE + N_DATA + N_TAIL) * SPS
BURST_SPACING = 6525
# BB60 anti-alias / decimation filter. The real receiver rolls off at 3.75 MHz,
# confirmed EXACTLY against the bbConfigureIQ max-bandwidth table: decimation 8
# gives 5 MS/s and 3.75 MHz. That happens to equal 0.75 x fs, but 0.75 is NOT a
# rule -- the table reads 0.675 at 40 MS/s, 0.890 at 20, 0.800 at 10/2.5/1.25. If
# this generator ever runs at another rate, look the row up rather than scaling.
# Without the filter a synthetic capture has NO band
# edge. Measured on the generator before this was added: +45 dB of excess power
# in 2.10-2.50 MHz against the real capture, on the HEALTHY radios too -- a
# one-line discriminator for anyone training on raw IQ. It does not show up in
# snr_mean_db because that reads the null tail in-band.
# DEAD as of the measured-FIR switch: nothing reads this. Kept only as the
# record of what the Kaiser design used. Do NOT quote it as the receiver
# passband -- the FIR actually in use measures -3 dB at 1.951 MHz single-sided
# (3.90 MHz two-sided). This constant is where the wrong 3.75 MHz in
# paper/tab_recon_impairments.tex came from (caught 2026-09-11).
RX_BANDWIDTH = 3.75e6      # UNUSED
# Stopband depth is CALIBRATED, not taken from the datasheet. The library
# default of 100 dB is far cleaner than the real BB60: with it the synthetic
# far band (2.10-2.50 MHz) came out 44.5 dB BELOW the real capture, having been
# 45.2 dB ABOVE it with no filter at all. The real receiver suppresses by only
# about 45 dB out there, and that shelf is NOT the filter's stopband. Modelling
# it by flattening the filter instead would match the far-band number while
# destroying the band EDGE, which is the feature that actually matters. So: a
# realistic filter, plus a calibrated post-filter floor.
#
# MECHANISM RETRACTED 2026-09-10. This used to say the shelf "is the ADC and
# quantisation noise, which enters AFTER the anti-alias filter and is therefore
# flat across the whole band". That is almost certainly wrong: the BB60C
# digitises at its native rate and decimates DIGITALLY, so the converter sits
# UPSTREAM of the decimation filter and its noise is shaped by the same stopband
# as everything else. It cannot arrive flat at the output.
#
# Two further reasons to treat -45 as a FIT CONSTANT, not a physical quantity:
#   - it does not transfer. Implemented in THIS convention on a 40 MS/s wifi
#     capture, -45 came out +25.9 / +25.6 / +25.4 dB hot in the far band at 58 /
#     80 / 100 dB stopband. Near-independence of stopband is the signature of an
#     additive term that is simply too large, not of a shaping mismatch.
#     (I briefly retracted this as a units error and was WRONG: I compared our
#     value converted to dBc, -83.3, against the peer's -73.4 read as if it were
#     dBc. It is not -- theirs is also dB-below-thermal, -101.0 dBc in their
#     config. Like for like the two constants are 28.4 dB apart, and "dB below
#     thermal" is the config-independent axis because the dBc row mixes their
#     snr 27.6 with our 38.3. ALWAYS state which reference when quoting this.)
#     One real but much smaller difference: we pre-scale AWGN by 1/sum(h^2) so
#     in-band noise lands on target AFTER filtering (sum(h^2) = 0.7413, -1.30 dB)
#     and they do not, so ~1.5 dB of the 28.4 is noise-bandwidth convention.
#   - [ALSO WITHDRAWN, by its author.] "The far-band shelf is fully explained
#     without any extra floor" was an artefact of fitting with a KAISER stopband,
#     which keeps decaying where the real BB60 shelf is FLAT. Re-measured with the
#     receiver's own response (601-tap zero-phase FIR from pooled quiet windows,
#     1.05 dB rms vs 8.62 for the best Kaiser), the chain UNDERSHOOTS the far band
#     by 4.7 dB -- so a real flat excess does sit above shaped thermal, and a floor
#     is modelling something that exists. Note OUR filter is still a Kaiser, so the
#     same shape mismatch applies here and some of -45 may be absorbing it.
# Whether a genuine shelf survives at THIS config's 5 MS/s is unmeasured -- the
# quiet-window measurement has not been run on an SRRC capture. Until it is,
# RX_FLOOR_DBC is an empirical correction that reproduces one number on one
# configuration, and the mechanism behind it is open.
#
# Note this is an ADDITIVE GAUSSIAN floor, not a quantiser: the library has
# add_quantization() but nothing here calls it. See paper_figure_chain.py, which
# labels the box "floor" for the same reason -- naming it quantisation would
# assert a mechanism the code does not run. True quantisation error is uniform
# and, at low bit depth, signal-correlated; this is neither.
# DEAD as of the measured-FIR switch: nothing reads this either. The FIR in use
# was designed with rs=70 and measures about -69 dB in the far band, not 80.
RX_STOP_DB = 80.0          # UNUSED
# MEASURED RECEIVE RESPONSE (2026-09-11), replacing the Kaiser. Derived from
# 1,171,456 pooled samples of the quiet region -- the SECOND HALF of each null
# tail, verified at -38.34 dB below data against a measured SNR of 38.31, i.e.
# sitting exactly on thermal, so its PSD is |H(f)|^2 up to a constant.
#
# The Kaiser was wrong in a way no stop_db could fix, because the real filter is
# not a parametric shape:
#                      measured    Kaiser 80
#     passband           -0.04       0.00
#     1.7-1.95 (edge)    -4.32      -5.61
#     1.95-2.10         -32.93     -79.30
#     2.10-2.50 (far)   -74.89     -96.26
# The real stopband is a FLAT shelf near -75 dB with a gradual transition; the
# Kaiser plunges and keeps decaying. 21.4 dB too deep in the far band, which is
# what the gated floor exposed once it stopped filling the quiet regions.
#
# The 129-tap FIR reproduces the measurement to ~1 dB in every band.
# EVEN-ORDER ELLIPTIC, fitted to that measurement. The key property is the
# EQUIRIPPLE STOPBAND: an elliptic holds a FLAT shelf, which is what the real
# receiver shows, whereas a Kaiser's stopband keeps decaying and a windowed-sinc
# FIR built from the measured PSD cannot hold the depth at any tap count
# (129/257/513/1025 taps all plateau near -62 dB against a -72.6 dB target,
# because the limit is the 256-point frequency resolution of the measurement,
# not filter length).
#
# Fitted for ZERO-PHASE application: sosfiltfilt applies |H|^2, so the design
# targets half the attenuation in dB. Zero phase matters -- an IIR's group delay
# would add in-band distortion the fitted ISI taps do not model, and those taps
# were fitted against real captures without it.
#
# Fit quality against the measured response, per band:
#   passband -0.00, 1.0-1.7 +0.11, edge +0.47, 1.95-2.10 -3.77, far +2.25 dB
#   rms 1.97 dB   (Kaiser 80 was 23.7 dB too deep in the far band). Fitted on
#   mean(|H|^4), NOT by doubling the dB of mean(|H|^2) -- those differ whenever
#   the stopband ripples, which an elliptic does by construction, and the naive
#   version overstated the depth by 10 dB.
# Order-10 elliptic, realised as a 601-tap ZERO-PHASE FIR and applied ONCE.
#
# SINGLE APPLICATION is the physically faithful operation: the BB60C applies its
# filter once. sosfiltfilt applying it forward and backward was a convenience
# that bought zero phase, and it made the design parameters uninterpretable --
# rs=44 meant an applied 88 dB, which nobody reading the number would expect.
# With single application rs is (modulo the equiripple caveat below) the depth.
#
# ORDER 10 because order 4 collapses under single application: refitted on our
# own measured response, rms 8.17 dB at order 4 against 1.82 at order 10. The
# old order 4 only ever worked BECAUSE of the squaring. Designs are tied to the
# realisation they were fitted under, in both directions.
#
# Realised as an FIR rather than run as an IIR so the phase stays zero -- an
# IIR's group delay would be in-band distortion the fitted ISI taps do not model.
# 601 taps reproduces the design to 0.03 dB; note this only works because the
# source is a SMOOTH PARAMETRIC magnitude. An earlier FIR built from the raw
# measured PSD plateaued 10 dB short, because 256 frequency points carry their
# own estimator noise.
#
# rs = 70 IS NOT THE STOPBAND DEPTH. A least-squares elliptic sits systematically
# SHALLOW of a structured stopband -- 3.2 dB here against our measured -72.67.
# The reconstruction chain fits rs = 70 too, but from a 75.7 dB measured shelf,
# i.e. 5.7 dB shallow. Two different depths and two different offsets landing
# near the same fitted value: a shared geometry effect, NOT two independent
# recoveries of a hardware constant. Do not report it as agreement.
RX_FIR = np.load(BASE / "bb60_rx_fir_5msps_n10.npy")
# post-filter noise floor, dB relative to the in-band noise power
RX_FLOOR_DBC = -45.0
CHUNK_SAMPLES = 250000       # the real BB60 writer's annotation chunk
DATA_START, DATA_END = N_PRE * SPS, (N_PRE + N_DATA) * SPS
IDEAL = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4)))

# ZADOFF-CHU PREAMBLE, matching the real chain exactly. The TX is not random
# here: data_collection/Utils/preamble_generator.py builds a CAZAC ZC sequence,
# and colleagues run preamble-based classifiers, for which the preamble IS the
# feature -- a random-QPSK stand-in would be a clean synthetic-vs-real separator.
#
# The parameters are NOT the ones the YAML asks for. It requests zc_root=25, but
# gcd(25, 100) = 25, so _zadoff_chu falls back to the first coprime root (1);
# and N=100 is EVEN, so it takes the n^2 branch, not the standard n(n+1). Both
# were confirmed against a real capture rather than read off the config:
# this sequence correlates 0.9970 with the real TX preamble waveform, where the
# odd branch scores 0.64, root=25 scores 0.20, and random QPSK scores 0.056.
_ZC_N, _ZC_ROOT = 100, 1
_zc_n = np.arange(_ZC_N)
PREAMBLE = np.exp(-1j * np.pi * _ZC_ROOT * _zc_n * _zc_n / _ZC_N)
SRRC = srrc_design(SPS, SPAN, BETA)
MF_DELAY = len(SRRC) // 2

LOG = BASE / "radio_characterisation.json"
TAPS = None      # set below from SG_TAPS_FILE

# ── optional RUN SUBSET / alternate taps file ────────────────────────────────
# Set by env var so all three scripts (fit_isi_taps, fit_joint, synth_dataset)
# honour the same restriction without threading an argument through each one.
#   SG_RUN_SUBSET  path to a JSON list of run numbers -- every stage then uses
#                  ONLY those real runs (profile stats, tap/PA fit, pn mask)
#   SG_TAPS_FILE   path to the taps JSON, so a subset fit does not overwrite
#                  the main isi_taps.json
def _sg_subset():
    import os, json as _j
    p = os.environ.get("SG_RUN_SUBSET")
    if not p:
        return None
    return {f"run_{int(n):03d}" for n in _j.loads(Path(p).read_text())}


def _sg_taps_path(base):
    import os
    return Path(os.environ.get("SG_TAPS_FILE", str(base / "isi_taps.json")))

TAPS = _sg_taps_path(BASE)

# MEASURED SETTLING SHAPE. Sampled profile from the bottom-up direct estimator
# (tap-referenced, whole-span aligned, validated against injected ground truth
# at 0.99x amplitude / +0.9937 shape correlation at 89_2400).
#
# SCOPED DELIBERATELY: applied ONLY to the (radio, config) pairs it was measured
# on. It is NOT generalised to other configs or radios. Two reasons. The
# measurement exists only at 89_2400; and applying one shape everywhere would
# flatten exactly the carrier-frequency and gain differences these datasets are
# meant to carry. Everywhere else the fitted complex cubic below is used
# unchanged, so those configs keep their own per-config amplitude.
#
# Conventions, all from the file's own "convention" block:
#   phase in DEGREES, applied as multiply by exp(1j*deg2rad(phi))
#   PHASE ONLY -- no amplitude term. The radial settling profile was measured
#     insignificant (|mean|/sd 0.1-1.4), and the earlier complex cubic was free
#     to re-absorb a resampler artefact that had already been retracted.
#   zero point is mean-removed over the DATA portion; the residual constant is
#     absorbed by the per-burst carrier-phase alignment downstream.
#   measured on the DELAY-CORRECTED burst average -- do not re-apply a delay
#     correction anywhere downstream of this.
#   linear interpolation between points, HOLD the end values outside the
#     measured span. np.interp does exactly this, so the null tail past t=0.846
#     holds the last value: unobservable (no signal there) and avoids a
#     discontinuity if the envelope is not exactly zero.
# SG_SHAPE_FILE overrides the path (point it at a nonexistent file to force
# the fitted-cubic route -- needed when generating from a DIFFERENT capture
# session than the shape was measured on, where using it would mix sessions).
import os as _os_sh
SHAPE_FILE = Path(_os_sh.environ.get("SG_SHAPE_FILE",
                  str(BASE / "srrc_settling_shape_for_generator.json")))
# The format this loader was written against. Bump ONLY after re-reading the
# file and re-verifying the routing, because the failure mode of a format change
# is silent: v1 carried the scope as free text in convention.config, v2 moved it
# to applies_to, and the revision broke an installed parse that then fell back to
# the fitted cubic and emitted wrong output while reporting success.
SHAPE_FORMAT = 4

# ── COMMON-MODE BAND RIPPLE, per config ─────────────────────────────────────
# The part of the tx->rx response that taps + PA cannot represent, measured from
# their RESIDUAL (analysis_scripts/fit_ripple.py) and pooled across radios, so it
# is common-mode by construction and carries no per-radio identity. Adding it cut
# residual 9.3% at 89_433 and 21.8% at 89_2400, better on 40/40 fits at both,
# with tap discriminability unchanged (3.1x -> 3.1x and 5.3x -> 5.8x).
#
# IT TRAVELS WITH ITS OWN TAPS. The shipped c_inject was fitted with no ripple
# term and has therefore already absorbed whatever band shape +-1 symbol-spaced
# taps can represent. Injecting this curve on top of those taps DOUBLE-COUNTS it:
# measured +30% residual, better on 0/40 fits. So a config with a ripple curve
# must also have a 'ripple_fit' block, and it raises if it does not -- the same
# pairing guard the measured settling shape uses, for the same reason.
RIPPLE_FILE = Path(_os_sh.environ.get(
    "SG_RIPPLE_FILE", str(BASE / "srrc_ripple_per_config.json")))
RIPPLE_FORMAT = 1


def _ripple_fir(cfg_dir):
    """Complex FIR for this config's ripple, or None if none is in scope."""
    if not RIPPLE_FILE.exists():
        return None
    d = json.loads(RIPPLE_FILE.read_text())
    _v = int(d.get("_format_version", -1))
    if _v != RIPPLE_FORMAT:
        raise SystemExit(
            f"{RIPPLE_FILE.name}: format_version {_v}, this generator expects "
            f"{RIPPLE_FORMAT}. The FIR convention or band may have changed; "
            "refusing to inject a curve written for a different contract.")
    c = d["configs"].get(cfg_dir)
    if c is None:
        return None
    return np.array([complex(a, b) for a, b in c["fir"]], dtype=np.complex128)

# v4 adds radial_* keys (a measured amplitude profile). They are NOT consumed
# yet -- the loader reads only the phase keys, which v4 leaves byte-identical to
# v3. Verified before bumping: phase data unchanged, routing still MEASURED for
# all four radios. Injecting the radial is a separate, deliberate change.


def _measured_shape(radio, cfg_dir):
    """(t, phi_deg) for this radio/config, or None if out of scope.

    Scope comes from the file's machine-readable `applies_to` block, NEVER from
    `convention.config`, which is free text. An earlier version parsed the free
    text, matched nothing, returned None, and the generator fell back to the
    cubic and produced wrong output while reporting success. So an unrecognised
    format now RAISES rather than returning None: the only silent path out of
    here is a genuine, deliberate out-of-scope pair.
    """
    if not SHAPE_FILE.exists():
        return None
    d = json.loads(SHAPE_FILE.read_text())
    _v = d.get("format_version")
    if _v != SHAPE_FORMAT:
        raise SystemExit(
            f"{SHAPE_FILE.name}: format_version is {_v!r}, this loader was "
            f"written for {SHAPE_FORMAT}. Refusing to load. A WELL-FORMED file "
            "in a changed format is the case that bit us -- the malformed check "
            "below would not catch it. Re-read the file, re-verify the routing "
            "actually selects MEASURED, then bump SHAPE_FORMAT.")
    ap = d.get("applies_to")
    if not isinstance(ap, dict) or "radio" not in ap or "config" not in ap:
        raise SystemExit(
            f"{SHAPE_FILE.name}: missing or malformed 'applies_to' block. "
            "Refusing to guess the scope -- an unrecognised format here would "
            "fall back to the fitted cubic SILENTLY and emit wrong output that "
            "still looks reasonable.")
    # applies_to.radio is a single name or a list of names -- the shape was
    # measured across 4 radios (see measured_on), so widening it to those 4 is
    # using the measurement, NOT generalising it. Anything other than str/list
    # raises rather than quietly failing to match.
    _ar = ap["radio"]
    if isinstance(_ar, str):
        _ar = [_ar]
    elif not (isinstance(_ar, list) and all(isinstance(x, str) for x in _ar)):
        raise SystemExit(
            f"{SHAPE_FILE.name}: applies_to.radio must be a string or a list "
            f"of strings, got {type(_ar).__name__}.")
    if radio not in _ar or cfg_dir != ap["config"]:
        return None                      # deliberately out of scope
    A = d.get("amplitude_deg_per_radio", {}).get(radio)
    if A is None:
        raise SystemExit(
            f"{SHAPE_FILE.name}: applies_to names {radio}/{cfg_dir} but "
            "amplitude_deg_per_radio has no entry for that radio.")
    t = np.asarray(d["t"], dtype=float)
    return t, float(A) * np.asarray(d["shape_unit_rms"], dtype=float)

SG_SUBSET = _sg_subset()

# Minimum residual improvement for the cubic PA to be believed at all.
# Below this the coefficient is fit noise and b is forced to zero.
PA_GAIN_MIN = 1.30
# Minimum (cross-radio spread)/(run-to-run noise) for the per-radio phase-noise
# LEVEL to be believed. Below it every radio gets the FLEET MEAN level, because
# injecting differences the measurement cannot resolve hands a classifier a
# noise-free per-radio constant with no physical content.
PN_LEVEL_GAIN_MIN = 1.5

# add_iq_imbalance puts the +gain on Q; the estimator's 2x2 fit reports
# 20log10(|col_I|/|col_Q|). Amplitude is negated on the way in, phase is not --
# the skew term I*sin(phi)+Q*cos(phi) already matches the fit.
IQ_SIGN = -1.0


# ════════════════════════════════════════════════════════════════════════════
# variation spec
# ════════════════════════════════════════════════════════════════════════════
DEFAULT_VARIATION = {
    # ── state: genuinely varies run to run ──────────────────────────────────
    # One reference drives BOTH the CFO and the clock mismatch on THESE radios
    # (verified 23/23). That is a property of this fleet, not a law -- a radio
    # with independent LO and sampling references would need
    #   "clock_mismatch_ppm": {"kind": "gauss", "sd": ...}
    # which samples it separately instead of deriving it. See below.
    "ref_ppm":       {"kind": "ar1"},
    "clock_mismatch_ppm": {"kind": "derived"},   # or gauss / ar1 / uniform
    # Unsynchronised RX sampling grid -- measured uniform, not assumed
    # (Rayleigh p=0.148, KS p=0.277, lag-1 +0.005).
    "cp0_samp":      {"kind": "uniform", "lo": -0.5, "hi": 0.5},
    "phi0_rad":      {"kind": "uniform", "lo": 0.0, "hi": 2 * np.pi},
    "leak_phase_rad": {"kind": "uniform", "lo": 0.0, "hi": 2 * np.pi},
    "dc_phase_rad":  {"kind": "uniform", "lo": 0.0, "hi": 2 * np.pi},
    # Measured Gaussian across runs (excess kurtosis +0.16 / +0.06).
    "iq_amp_db":     {"kind": "gauss"},
    "iq_phase_deg":  {"kind": "gauss"},
    "snr_db":        {"kind": "gauss"},
    "lo_leak_dbc":   {"kind": "gauss"},

    # ── transfer function: 'fixed' HERE, but drawable anywhere ──────────────
    # On this fleet the run-to-run scatter of these sat BELOW their own
    # estimation error (rr/est 0.2-1.1), so they are constants measured noisily
    # and sampling them would inject measurement noise as device variation.
    # That is a MEASUREMENT about these radios, not an assumption to inherit:
    # give any of them a 'gauss' spec and it is drawn per run.
    #
    # Note pa_b_re and pa_b_im are separate knobs on purpose. They are not
    # equally determined -- compression is a 264-sigma measurement at 0.38 % CV
    # while AM/PM is 7.4-sigma at 13.5 %, a ~35x difference in precision. A
    # single complex knob would hide that; someone may reasonably want to hold
    # compression and sample AM/PM.
    "pa_b_re":            {"kind": "fixed"},
    "pa_b_im":            {"kind": "fixed"},
    "pn_level_db":        {"kind": "fixed"},   # dB offset on the mask
    "pn_tangential_deg":  {"kind": "fixed"},   # diagnostic only, not injected
    "pn_radial_pct":      {"kind": "fixed"},
    "rx_dc_frac":         {"kind": "fixed"},
    # isi_taps takes a RELATIVE sd applied to every tap (real and imaginary,
    # scaled by |c|), because the taps are a vector rather than a scalar:
    #   "isi_taps": {"kind": "gauss", "rel_sd": 0.01}
    "isi_taps":           {"kind": "fixed"},
}


def draw(spec, centre, rng, state=None):
    """One run's value for a parameter. `state` carries AR(1) memory."""
    kind = spec.get("kind", "fixed")
    if kind in ("fixed", "derived"):
        return centre, state
    new_state = state
    if kind == "uniform":
        # lo/hi are the SUPPORT here, not clamps, so the bounds step below is a
        # no-op for this kind and is skipped rather than relied on.
        v = rng.uniform(spec["lo"], spec["hi"])
    elif kind == "gauss":
        v = centre + rng.normal(0.0, spec["sd"])
    elif kind == "ar1":
        phi, sd = spec.get("phi", 0.0), spec["sd"]
        if state is None:
            # START STATIONARY. Seeding s=0 and applying the sqrt(1-phi^2)
            # innovation makes run 1 under-dispersed by exactly that factor
            # (measured 0.757 vs 0.763 predicted at phi=0.646) -- run_001 of
            # every config would sit closer to the mean than every other run.
            s = rng.normal(0.0, sd)
        else:
            s = phi * state + rng.normal(0.0, sd * np.sqrt(max(1 - phi ** 2, 1e-9)))
        new_state = s
        v = centre + s
    elif kind == "mixture":
        # Finite Gaussian mixture. mu/sd are ABSOLUTE unless "relative" is set,
        # because a mixture is normally fitted to the quantity itself (TX LO
        # leakage) rather than to a residual about the profile centre.
        w = np.asarray(spec["w"], float)
        w = w / w.sum()
        i = int(rng.choice(len(w), p=w))
        v = rng.normal(spec["mu"][i], spec["sd"][i])
        if spec.get("relative"):
            v = centre + v
    elif kind == "scipy":
        # ANY scipy.stats distribution, addressed by name so the spec stays
        # JSON-serialisable -- run_spec writes {"variation": var} to
        # profile.json, and a Python callable would raise TypeError there and
        # destroy the provenance record every synthetic run carries.
        import scipy.stats as _st
        _d = getattr(_st, spec["dist"], None)
        if _d is None:
            raise ValueError(f"unknown scipy.stats distribution {spec['dist']!r}")
        v = _d.rvs(**spec.get("params", {}), random_state=rng)
        if spec.get("relative"):
            v = centre + v
    else:
        raise ValueError(f"unknown variation kind {kind!r}")
    # PHYSICAL bounds, hoisted out of the gauss branch so every unbounded kind
    # gets them. The reason they exist: LO leakage at 89_915 has an IQR sigma of
    # 12 dB, so +3 sigma lands at -21 dBc where the hardware never exceeded -51.
    # That fed an unphysical run AND broke the AWGN budget downstream.
    if kind != "uniform":
        if "lo" in spec:
            v = max(v, spec["lo"])
        if "hi" in spec:
            v = min(v, spec["hi"])
    return float(v), new_state


def fit_mixture_bic(x, dbic=-10.0, iters=400):
    """Two-component Gaussian mixture, returned ONLY if BIC clearly prefers it.

    Returns {"w": [...], "mu": [...], "sd": [...]} or None. The -10 threshold is
    deliberate: a marginal preference would make the model kind flip between
    radios on noise, and a mixture that is really unimodal is strictly worse
    than the single Gaussian it replaces.
    """
    x = np.asarray(x, float)
    n = x.size
    if n < 20:
        return None
    lo, hi = np.percentile(x, [15, 85])
    mu = np.array([lo, hi], float)
    sd = np.full(2, max(x.std(), 1e-9) / 2.0)
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        r = w * np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        r = r / np.maximum(r.sum(1, keepdims=True), 1e-300)
        nk = np.maximum(r.sum(0), 1e-12)
        w = nk / n
        mu = (r * x[:, None]).sum(0) / nk
        sd = np.maximum(np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk), 1e-6)
    ll2 = float(np.log(np.maximum(
        (w * np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2)
         / (sd * np.sqrt(2 * np.pi))).sum(1), 1e-300)).sum())
    m1, s1 = x.mean(), x.std(ddof=1)
    if not np.isfinite(s1) or s1 <= 0:
        return None
    ll1 = float(np.log(np.exp(-0.5 * ((x - m1) / s1) ** 2)
                       / (s1 * np.sqrt(2 * np.pi))).sum())
    if (-2 * ll2 + 5 * np.log(n)) - (-2 * ll1 + 2 * np.log(n)) > dbic:
        return None
    # DEGENERACY GUARD. EM on a Gaussian mixture has a well-known singularity:
    # a component can collapse onto a single point, sd -> 0, likelihood -> inf,
    # and BIC then "prefers" it. Measured on this fleet: 30BF7C1/77_433 and
    # 30EAE76/77_2400 both produced a component with sd exactly 0.0 carrying
    # ~1 point, which would make the generator emit a constant value on ~1% of
    # runs. Reject on too-few effective points or a collapsed width; the caller
    # then keeps the robust single Gaussian. Genuine narrow lobes survive --
    # 30BF779/77_915 has 23 values inside 0.8 dB and its sd 0.19 passes.
    if float(np.min(w)) * n < 5.0 or float(np.min(sd)) < 0.01 * float(x.std()):
        return None
    o = np.argsort(mu)
    return {"w": [float(q) for q in w[o]], "mu": [float(q) for q in mu[o]],
            "sd": [float(q) for q in sd[o]]}


# ════════════════════════════════════════════════════════════════════════════
# profile
# ════════════════════════════════════════════════════════════════════════════
def profile_from_log(radio, config="77_433", radial_filler=False,
                     pa_gain_min=PA_GAIN_MIN,
                     pn_level_mode="auto"):
    """Build a device profile + variation spec from the characterisation log.

    Everything here is measured. Values that need the raw IQ (ISI taps, PA) are
    read from isi_taps.json, written by fit_isi_taps.py / fit_pa.py.
    """
    cfg_dir = config.lstrip("g")
    cfg_key = f"g{cfg_dir}"
    log = json.loads(LOG.read_text())
    rows = [v for k, v in log.items()
            if len(k.split("/")) == 3 and k.split("/")[1] == cfg_key
            and re.match(rf"TX{radio}_RX", k.split("/")[0])
            and (SG_SUBSET is None or k.split("/")[2] in SG_SUBSET)]
    if not rows:
        raise SystemExit(f"no runs for {radio} / {cfg_key} in {LOG.name}")
    # ── ONE SESSION, NEVER A POOL ───────────────────────────────────────────
    # 36 of 138 radio/config pairs in the log appear in more than one capture
    # session, and pooling them was silently reading a BETWEEN-SESSION
    # difference as run-to-run variation. On 30BF795/g89_2400 the two sessions
    # sit at 38.30 and 23.66 dB SNR; pooled that is 30.98 +- 7.35 against a
    # within-session 0.02, a 20x inflated spread AND a centre matching neither
    # session. The generator then injected ~7 dB too little SNR, which inflated
    # every amplitude statistic downstream.
    #
    # It is also internally inconsistent: the ISI taps, PA and settling are all
    # fitted on ONE session's raw IQ (the capture on disk), so pooling the
    # scalars describes a different capture than the waveform-domain terms do.
    # Default therefore to the session whose capture is on disk, since that is
    # the one everything else was fitted on. SG_SESSION overrides. If neither
    # resolves it, RAISE rather than pool -- silently averaging two populations
    # is exactly the failure this comment exists to prevent.
    _sess = {k.split("/")[0] for k in log
             if len(k.split("/")) == 3 and k.split("/")[1] == cfg_key
             and re.match(rf"TX{radio}_RX", k.split("/")[0])}
    if len(_sess) > 1:
        import os as _os
        _want = _os.environ.get("SG_SESSION")
        if not _want:
            _d = sorted((BASE / "New_captures").rglob(f"4QAM_{radio}_sweep_*"))
            _d = [x for x in _d if x.is_dir() and not x.name.endswith("_syn")]
            _want = _d[0].name.rsplit("_", 1)[-1] if _d else None
        _hit = sorted(x for x in _sess if _want and x.endswith(f"_{_want}"))
        if len(_hit) != 1:
            raise SystemExit(
                f"{radio}/{cfg_key} appears in {len(_sess)} capture sessions: "
                f"{sorted(_sess)}. Pooling them would read a between-session "
                f"difference as run-to-run variation. Set SG_SESSION to the "
                f"session token you want (e.g. SG_SESSION=104619).")
        rows = [v for k, v in log.items()
                if k.split("/")[0] == _hit[0] and k.split("/")[1] == cfg_key
                and (SG_SUBSET is None or k.split("/")[2] in SG_SUBSET)]
        print(f"  session: {_hit[0]}  ({len(rows)} runs; "
              f"{len(_sess)} sessions exist, NOT pooled)")
    fc = rows[0]["fc_hz"]
    col = lambda k: np.array([v[k] for v in rows if v.get(k) is not None], float)

    cfo_ppm = col("cfo_hz") / (fc * 1e-6)
    clk_ppm = col("clock_mismatch_ppm")

    # LO leakage needs a ROBUST centre: failed estimates form a long low tail
    # (one radio: median -42.2 but mean -45.9, sd 7.2, p5 -62.7). Reject it,
    # then use median and an IQR sigma, which the tail cannot move.
    lk = col("lo_leakage_dbc")
    q1, q3 = np.percentile(lk, [25, 75])
    lk = lk[lk > q1 - 3.0 * max(q3 - q1, 0.5)]

    x = cfo_ppm - cfo_ppm.mean()
    phi = float(np.clip(np.corrcoef(x[:-1], x[1:])[0, 1], 0.0, 0.98)) \
        if x.size > 2 and x.std() > 0 else 0.0

    tp = json.loads(TAPS.read_text()).get(f"{radio}/{cfg_dir}", {})
    # Measured settling shape for this radio/config, and the Hammerstein taps/PA
    # fitted with that profile DIVIDED OUT. The two must travel together: using
    # j3's jointly-fitted taps while injecting the measured settling would mean
    # the taps were fitted against a different settling than the one applied.
    # That mismatch is silent and produces plausible-looking output, so it raises.
    _mshape = _measured_shape(radio, cfg_dir)
    _hamm = _mshape is not None and isinstance(tp.get("hammerstein"), dict)
    if _mshape is not None and not _hamm:
        raise SystemExit(
            f"{radio}/{cfg_dir}: a MEASURED settling shape is in scope but no "
            "'hammerstein' fit exists for it. Using j3's jointly-fitted taps "
            "with the measured profile would inject one settling against taps "
            "fitted for another. Run: "
            f"python analysis_scripts/fit_hammerstein.py {radio} {cfg_dir}")
    _htp = tp["hammerstein"] if _hamm else tp
    # RIPPLE IN SCOPE supersedes both routes above. fit_with_ripple re-estimated
    # taps, PA and settling with the ripple FIR folded into the design, using the
    # FITTED settling cubic -- so the measured shape is forced off here, exactly
    # as build_generalisation_set already does for a different reason. Mixing the
    # measured shape with ripple_fit's taps would repeat the settling mismatch
    # the _hamm guard above exists to prevent.
    _rip = _ripple_fir(cfg_dir)
    if _rip is not None:
        if not isinstance(tp.get("ripple_fit"), dict):
            raise SystemExit(
                f"{radio}/{cfg_dir}: a ripple curve is in scope for this config "
                "but no 'ripple_fit' block exists. Injecting it with taps fitted "
                "WITHOUT it double-counts the band shape (+30% residual, better "
                "on 0/40 fits). Run: python analysis_scripts/fit_with_ripple.py "
                f"{cfg_dir}")
        _mshape = None
        _hamm = False
        _htp = tp["ripple_fit"]
    _pak = "cubic" if tp.get("pa_kind") == "cubic" else "rapp"
    # take b from the Hammerstein fit where it is in use, BEFORE the
    # resolvability gate below -- selecting it after the gate silently
    # bypasses it and re-enables a cubic the gate exists to suppress.
    _bsrc = _htp if (_hamm or _rip is not None) else tp
    _bre = float(_bsrc.get("pa_b_re", 0.0)); _bim = float(_bsrc.get("pa_b_im", 0.0))
    # Gate the PA on whether the cubic EXPLAINS anything, not on |b|. At 2400 the
    # drive sits far enough below saturation that b estimates ~zero, so its value
    # is whatever else happens to project onto x*|x|^2 -- fit noise. Injecting it
    # would hand the classifier a per-radio constant with no RF meaning.
    # The measured split is unambiguous: residual(linear)/residual(cubic) is
    # perfectly bimodal over 48 radio/config fits: 32 values in 1.000-1.138
    # and 16 in 1.453-6.607, with NOTHING in between. The 16 are exactly
    # 89_433 and 89_915 on all 8 radios -- the two compressed configs.
    # PA_GAIN_MIN sits at the midpoint of that empty gap.
    _cg = (tp.get("pa_cubic_vs_linear", 0.0) / tp["pa_cubic_residual"]
           if tp.get("pa_cubic_residual") else 0.0)
    if _pak == "cubic" and _cg < pa_gain_min:
        _bre = _bim = 0.0
    # ── phase-noise level: per-radio only where it is RESOLVABLE ────────────
    # Bottom-up's CW measurement (four decades, GPSDO) found the level is a real
    # device feature at 2400 but sits ON the floor at 433 -- 0.55 dB across four
    # radios, one with zero resolvable excess. My own between/within test on the
    # modulated fits agrees: 1.12 and 0.84 at the two 433 configs against
    # 1.71-3.05 at 915/2400. Where it is not resolvable every radio takes the
    # FLEET MEAN, so the generator does not manufacture a per-radio constant.
    _pn_mask = [list(m) for m in tp["pn_mask"]] if tp.get("pn_mask") else None
    _pn_shared = False
    if _pn_mask is not None and pn_level_mode != "per_radio":
        _all = json.loads(TAPS.read_text())
        _lv = [v["pn_level_db_10k"] for k, v in _all.items()
               if k.endswith(f"/{cfg_dir}") and v.get("pn_level_db_10k") is not None]
        _sd = [v["pn_level_sd_db"] for k, v in _all.items()
               if k.endswith(f"/{cfg_dir}") and v.get("pn_level_sd_db")]
        if len(_lv) >= 3 and _sd:
            _ratio = float(np.std(_lv, ddof=1) / max(np.median(_sd), 1e-9))
            if pn_level_mode == "fleet" or _ratio < PN_LEVEL_GAIN_MIN:
                _shift = float(np.mean(_lv) - tp["pn_level_db_10k"])
                _pn_mask = [[f, lv + _shift] for f, lv in _pn_mask]
                _pn_shared = True

    if "c_inject" not in tp:
        raise SystemExit(f"no ISI taps for {radio}/{cfg_dir} — run "
                         f"fit_pa.py then fit_isi_taps.py first")

    prof = {
        "radio": radio, "config": cfg_dir, "fc_hz": fc, "n_real_runs": len(rows),
        "ref_ppm": float(cfo_ppm.mean()),
        # the two estimators sit a fixed distance apart (median +0.0057 ppm over
        # 12 sessions, always positive); carried as a measured constant
        "clock_offset_ppm": float(np.median(clk_ppm - cfo_ppm)),
        "cp0_samp": 0.0, "phi0_rad": 0.0,
        "leak_phase_rad": 0.0, "dc_phase_rad": 0.0,
        "iq_amp_db": float(col("iq_amp_db").mean()),
        "iq_phase_deg": float(col("iq_phase_deg").mean()),
        "snr_db": float(col("snr_mean_db").mean()),
        "lo_leak_dbc": float(np.median(lk)),
        "rx_dc_frac": float(np.median(col("rx_dc_frac"))),
        # HAMMERSTEIN taps/PA where the settling is MEASURED rather than fitted.
        # fit_joint's c_inject and pa_b were estimated in one least squares WITH
        # a settling cubic, which is what stops the model being a Hammerstein
        # system -- a time-varying gain sharing a solve with the linear filter.
        # Worse, the generator injects the MEASURED settling, so those taps were
        # fitted against a DIFFERENT settling than the one applied. fit_hammerstein
        # divides the known profile out first and fits taps + PA with no settling
        # term, at 0-10% of j3's residual (j3 has 3 more free parameters on the
        # same data, so it should win slightly). See analysis_scripts/fit_hammerstein.py.
        "isi_taps": {"lags": _htp["lags"],
                     "c": [[c[0], c[1]] for c in _htp["c_inject"]]},
        # CUBIC preferred: y = x + b*x*|x|^2 with b COMPLEX, so compression
        # (real part) and AM/PM (imaginary part) are free to differ. Rapp is
        # amplitude-only and cannot produce AM/PM at all; Saleh ties the two
        # together on more parameters. Falls back to the Rapp fit if the cubic
        # has not been run for this radio/config.
        # Flattened into SCALAR knobs so the variation machinery can draw any of
        # them. pa_b_re / pa_b_im are separate because they are not equally
        # determined (~35x apart in precision), and pn_radial_pct defaults to 0:
        # it is the quadrature remainder left when the ISI model fails to explain
        # the radial blob (0.000 % at 77_433, up to 2.6 % elsewhere), so injecting
        # it would match a number while hiding the defect. The fitted value is
        # kept alongside so the gap stays visible.
        # LO SETTLING: a burst-repetitive transient, a complex cubic in position
        # through the burst (powers 1..3; a constant would be the main tap).
        # Fitted jointly with the taps and the cubic -- at 2400 it is the
        # DOMINANT deterministic error (residual 0.0362 -> 0.0109) because there
        # is almost no compression there to disentangle, while at 433 it is
        # nearly a no-op on end metrics. It is burst-repetitive, so the generator
        # folds it into the per-run base waveform at zero per-burst cost.
        # Reconciled with the direct estimator (srrc_transient_figure.run_profile,
        # 2026-09-08) by DIRECT comparison on 30BF795, data-portion axis, mean
        # removed: this fit reads 0.315 / 1.934 deg rms at 89_433 / 89_2400
        # against the direct estimator's 0.411 / 2.130, i.e. 0.77x and 0.91x --
        # NOT the "about half" the old caveat claimed. Shape agrees too (profile
        # minimum 0.384-0.389 here vs 0.365-0.404 there).
        # INJECTION CAVEAT: the cubic is FITTED on t in [0.077, 0.846] (the data
        # portion) and applied below over the WHOLE burst, so t < 0.077 is
        # extrapolation. Across all 30 radio/config pairs that inflates the peak
        # excursion by a uniform 1.80x (1.69-1.96), and every bit of the excess
        # lands in the PREAMBLE -- 11.1 deg there vs 5.3 deg in-band at 89_2400.
        # See scratchpad/extrap_check.py. This is the leading explanation for the
        # sigma_t overshoot bottom-up saw on injection, because the pipeline
        # phase-aligns each burst ON the preamble. Left as-is pending a joint
        # decision: clamping to the fitted window is not obviously right either,
        # since the true transient should be LARGER early in the burst.
        "settling": [list(v) for v in _htp.get("settling",
                                               tp.get("settling", []))],
        "ripple_fir": (None if _rip is None else
                       [[v.real, v.imag] for v in _rip]),
        # measured sampled profile where one exists; None falls back to the cubic
        "settling_shape": (None if _mshape is None else
                           {"t": _mshape[0].tolist(),
                            "phi_deg": _mshape[1].tolist()}),
        "pa_kind": _pak, "pa_cubic_gain": _cg,
        "pa_b_re": _bre, "pa_b_im": _bim,
        "pa_A": float(tp.get("pa_A", 1e6)),
        # PHASE NOISE is a fitted 1/f^a MASK, not a Gaussian scalar. The old
        # pn_tangential_deg was a quadrature top-up -- it matched the tangential
        # variance by construction but put the symbol-to-symbol correlation at
        # zero, where the real residual measures lag-1 0.56-0.68 (the radial
        # axis, carrying the same AWGN, is white at 0.01-0.09). The mask fits
        # -8.3 to -9.6 dB/decade with a 1.1-1.6 dB residual. pn_tangential_deg is
        # KEPT as a diagnostic: it is still the honest size of the tangential
        # gap the ISI taps leave.
        "pn_mask": _pn_mask,
        "pn_level_shared": _pn_shared,
        "pn_level_db": 0.0,        # shifts the whole mask; a knob like any other
        "pn_slope_db_dec": float(tp.get("pn_slope_db_dec", 0.0)),
        "pn_tangential_deg": float(tp.get("extra_phase_deg", 0.0)),
        "pn_radial_pct": (float(tp.get("extra_radial_pct", 0.0))
                          if radial_filler else 0.0),
        "pn_radial_pct_fitted": float(tp.get("extra_radial_pct", 0.0)),
        # centre for the case where the clock is NOT derived from ref_ppm
        "clock_mismatch_ppm": float(clk_ppm.mean()),
    }

    var = {k: dict(v) for k, v in DEFAULT_VARIATION.items()}
    # spread from the CLEANER estimator: clock mismatch is a slope fit over ~762
    # bursts where CFO is a mean of ~76 per-burst estimates. Their std ratio
    # -> 1.00 whenever true spread exceeds estimator noise.
    var["ref_ppm"].update(sd=float(clk_ppm.std()), phi=phi)
    var["iq_amp_db"].update(sd=float(col("iq_amp_db").std()))
    var["iq_phase_deg"].update(sd=float(col("iq_phase_deg").std()))
    var["snr_db"].update(sd=float(col("snr_mean_db").std()))
    # TX LO leakage is bimodal on most radio/configs (30BF779/89_433: two modes
    # 12.1 dB apart, 6.0x the component sd, dBIC -79.5). A single Gaussian puts
    # its peak in the valley BETWEEN the lobes and assigns real mass to values
    # the hardware never produces, so a mixture is used where BIC clearly
    # prefers one and the robust Gaussian is kept where it does not.
    _mix = fit_mixture_bic(lk)
    if _mix is not None:
        var["lo_leak_dbc"] = {"kind": "mixture", **_mix,
                              "lo": float(lk.min()), "hi": float(lk.max())}
    else:
        var["lo_leak_dbc"].update(sd=float((np.percentile(lk, 75)
                                            - np.percentile(lk, 25)) / 1.349),
                                  lo=float(lk.min()), hi=float(lk.max()))
    # centre for the independent-clock case; sd from the log, so switching
    # clock_mismatch_ppm to 'gauss' or 'ar1' works without further edits
    var["clock_mismatch_ppm"].setdefault("sd", float(clk_ppm.std()))
    return prof, var


# ════════════════════════════════════════════════════════════════════════════
# synthesis
# ════════════════════════════════════════════════════════════════════════════
def _pn_amplitude(mask, n, fs):
    """The LOOP-INVARIANT half of add_phase_noise: the shaped amplitude
    spectrum. Splitting it out is purely a speed change -- the mask's
    log-frequency interpolation and the dB->linear conversion depend only on
    (mask, n, fs), so recomputing them for all 762 bursts was pure waste.

    This mirrors cel_signal_gen_lib.impairments.hardware.add_phase_noise
    exactly, including its irfft normalisation PSD = 2|X|^2/(N*fs); _pn_draw
    below consumes the rng in the same order, so the two produce identical
    output for the same seed (asserted in the tests).
    """
    m = sorted(mask, key=lambda x: x[0])
    mf = np.array([x[0] for x in m], float)
    ml = np.array([x[1] for x in m], float)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = np.zeros(len(freqs))
    psd[1:] = np.interp(np.log10(freqs[1:]), np.log10(mf), ml)
    S = np.zeros(len(freqs))
    S[1:] = 10.0 ** (psd[1:] / 10.0)
    amp = np.sqrt(S * n * fs / 2.0)
    amp[0] = 0.0
    return amp


def _pn_draw(amp, n, rng):
    """The per-burst half: one white draw, shaped, back to the time domain."""
    noise = (rng.standard_normal(len(amp))
             + 1j * rng.standard_normal(len(amp))) / np.sqrt(2.0)
    noise[0] = 0.0
    if n % 2 == 0:
        noise[-1] = noise[-1].real
    return np.fft.irfft(amp * noise, n=n)


def _shift_lin(v, k):
    """v[n+k] with ZERO fill -- LINEAR, not circular (np.roll wraps)."""
    out = np.zeros_like(v)
    if k > 0:
        out[:-k] = v[k:]
    elif k < 0:
        out[-k:] = v[:k]
    else:
        out[:] = v
    return out


def _isi_wave(w, taps, lags):
    """Symbol-spaced channel ISI applied to the WAVEFORM, after the PA.

    out[n] = w[n] + sum_d c_d w[n - d*SPS]

    DELAY convention, matching fit_joint.py's design matrix np.roll(x, d*SPS).
    The old symbol-domain code used an ADVANCE convention (s[n+k]) -- the lag
    sign flips between them, so the two are not interchangeable.

    Linear (zero-filled) shift: bursts sit BURST_SPACING apart with a 25-sample
    gap, and the ISI span is +-1 symbol = +-5 samples, so nothing leaks between
    bursts and a wrap would be wrong.
    """
    out = w.copy()
    for c, d in zip(taps, lags):
        out = out + c * _shift_lin(w, -int(d) * SPS)
    return out


def _pa(x, pa):
    """Memoryless PA. Normalised on the DATA portion, matching how it is fitted
    -- a whole-burst rms pulls in the null tail and over-drives it."""
    if pa["kind"] == "rapp" and pa["A"] < 100:
        s0 = np.sqrt(np.mean(np.abs(x[DATA_START:DATA_END]) ** 2))
        u = x / s0
        u = u / np.power(1 + np.power(np.abs(u) / pa["A"], 2 * pa["p"]),
                         1 / (2 * pa["p"]))
        return u * (s0 / np.sqrt(np.mean(np.abs(u[DATA_START:DATA_END]) ** 2)))
    if pa["kind"] == "cubic":
        b = complex(pa["b_re"], pa["b_im"])
        s0 = np.sqrt(np.mean(np.abs(x[DATA_START:DATA_END]) ** 2))
        u = x / s0
        u = u + b * u * np.abs(u) ** 2
        return u * (s0 / np.sqrt(np.mean(np.abs(u[DATA_START:DATA_END]) ** 2)))
    return x


def synth_run(prof, var, rng, state, n_bursts):
    """One run: draw its parameters, then build the burst train.

    IMPAIRMENT ORDER (corrected 2026-09-10 -- the old docstring here described a
    chain this function stopped running some time ago, claiming symbol-level ISI
    ahead of the pulse shaping: "ISI -> phase noise -> SRRC -> PA"):

      once per run   SRRC(ZC preamble + random QPSK)  [x settling, cubic route]
      per burst      phase noise -> PA -> ISI -> [settling, measured route]
                     -> [ripple FIR] -> IQ -> TX leakage -> CFO -> RX DC
                     -> timing -> AWGN
      once per run   BB60 decimation filter -> additive floor

    Everything downstream of the SRRC is at SAMPLE rate; there is no symbol-level
    channel. PA BEFORE ISI because the PA sits in the transmitter and the ISI is
    everything after it -- feeding the jointly fitted coefficients through
    ISI->PA gives residual 0.158 against 0.015 the other way, a 12x penalty.

    The ORDER above is shared with the reconstruction chain, which is what a
    merged block figure needs -- one ISI box, not two. The CONTENTS are not:
    reconciled against bottom-up 2026-09-11 and the two chains genuinely differ
    in five places, so do not read the shared ordering as a shared model.
      TX source      here: synthesised, fixed ZC preamble + per-run random QPSK,
                     N_TAIL=200 null tail.  there: reads the real TX record and
                     resamples it, no null tail (so SNR is measured differently).
      phase noise    NOT a difference after all (checked 2026-09-11): the
                     radial/AM path here is INERT -- pn_radial_pct is 0.0 in all
                     131 shipped profiles, so `if pn["radial_pct"] > 0` never
                     fires and no AM is injected. The b *= (1 + am) capability
                     exists but is unused, matching bottom-up, which measures a
                     radial term and deliberately does not inject it either.
      CFO            here: one AR(1) draw per run + a deterministic ramp across
                     bursts (see the no-per-burst-dither note below).  there:
                     redrawn per burst.
      settling       here: complex multiplicative cubic folded into `base`
                     pre-PA, or the measured phase-only profile post-ISI where
                     one exists.  there: measured phase-only, post-ISI, always.
      PA             three routes between us, not two -- see the settling note.
    Everything else (per-RUN draws for IQ, leakage, RX DC and SNR included)
    agrees.

    Settling has two routes and they sit in different places, because each has to
    match the fit that produced it: the MEASURED profile is divided out of the
    burst average before the taps and PA are fitted, so it is re-applied after
    ISI; the older fitted CUBIC was estimated jointly with the taps in one lstsq
    and is folded into `base` ahead of them. Moving either would break its pairing.

    Leakage sits BEFORE the CFO so it lands at the CFO offset in the RX frame,
    which is where the estimator finds it; RX DC sits AFTER, at 0 Hz.
    """
    p = {}
    for k, spec in var.items():
        if k == "isi_taps":
            continue
        p[k], state[k] = draw(spec, prof.get(k, 0.0), rng, state.get(k))
    p["cp0_samp"] = (p["cp0_samp"] + 0.5) % 1.0 - 0.5      # cyclic

    cfo_hz = p["ref_ppm"] * 1e-6 * prof["fc_hz"]
    # DERIVED (default) ties the sampling clock to the same reference as the LO.
    # Any other kind samples it independently -- for hardware where the two are
    # not locked together.
    if var.get("clock_mismatch_ppm", {}).get("kind", "derived") == "derived":
        mismatch = p["ref_ppm"] + prof["clock_offset_ppm"]
    else:
        mismatch = p["clock_mismatch_ppm"]
    p["clock_mismatch_ppm"] = mismatch     # record what was ACTUALLY used
    advance = -mismatch * 1e-6 * BURST_SPACING     # +ppm => FALLING ramp
    lags = prof["isi_taps"]["lags"]
    taps = [complex(a, b) for a, b in prof["isi_taps"]["c"]]
    _ts = var.get("isi_taps", {})
    if _ts.get("kind") == "gauss" and _ts.get("rel_sd", 0) > 0:
        r = _ts["rel_sd"]
        taps = [c + complex(rng.normal(0, r * abs(c)), rng.normal(0, r * abs(c)))
                for c in taps]
    pn = {"tangential_deg": p["pn_tangential_deg"],
          "radial_pct": p["pn_radial_pct"]}
    # mask levels are dB, so the knob is an offset
    pn_mask = ([(f, lv + p["pn_level_db"]) for f, lv in prof["pn_mask"]]
               if prof.get("pn_mask") else None)
    dc = p["rx_dc_frac"]

    # Fixed system preamble -- NOT drawn per run, exactly as the real TX repeats
    # the same ZC every burst of every run.
    pre = PREAMBLE
    data = IDEAL[rng.integers(0, 4, N_DATA)]
    # ISI is applied per burst now, AFTER the PA (see below). It is still
    # deterministic given this run's symbols, so the deviation still repeats.
    #
    # LINEAR, over the WHOLE burst grid. np.roll on the data block alone wrapped
    # data[997:999] into data[0:2], asserting that the symbol before the first
    # data symbol was the last data symbol. It is not -- it is the last preamble
    # symbol, and the symbol after the last data symbol is the null tail. Both
    # are known here, so the edges get their TRUE neighbours rather than a
    # zero-fill. Applying the taps across the full grid also puts the channel's
    # ISI on the PREAMBLE, which is physically right (a linear channel filters
    # the whole burst) and matters now that the preamble is the feature for
    # preamble-based classifiers. The TX reference below stays CLEAN -- the ISI
    # happens between TX and RX, so the tx file must not carry it.
    grid = np.concatenate([pre, data, np.zeros(N_TAIL, complex)])
    tx = scipy.signal.upfirdn(SRRC, grid, up=SPS)[MF_DELAY:MF_DELAY + BURST_LEN]
    tx = (tx / np.sqrt(np.mean(np.abs(tx[DATA_START:DATA_END]) ** 2))
          ).astype(np.complex64)

    # SRRC ONCE PER RUN. isi_full is identical for every burst, so shaping it
    # inside the loop repeated the same 80 us filter 762 times. Phase noise is
    # what differs per burst, and it belongs on the WAVEFORM anyway: an
    # oscillator multiplies the continuous signal, it does not modulate the
    # symbols before pulse shaping. Applying it pre-SRRC let the filter smooth
    # the phase process, altering its spectrum.
    base = scipy.signal.upfirdn(SRRC, grid, up=SPS)[MF_DELAY:MF_DELAY + BURST_LEN]
    # burst-repetitive settling profile, applied ONCE per run.
    # MEASURED sampled profile where one exists for this radio/config, else the
    # fitted complex cubic. The measured route is PHASE ONLY and is defined on
    # t in [0.0035, 0.8265]; np.interp holds the end values outside that, which
    # is the documented null-tail convention. The cubic route is retained
    # unchanged for every config without a measurement -- note it EXTRAPOLATES
    # into the preamble, where it runs inverted against the measured profile
    # (corr -0.732 at 89_2400) as well as 1.83x too large. That is a known
    # defect of the cubic route, not of the measured one.
    #
    # ORDER. The MEASURED profile is applied AFTER ISI, per burst, because that
    # is how fit_hammerstein removes it: it divides the known phase out of the
    # burst average and then fits taps + PA on what is left. Settling is a pure
    # phase and the PA is memoryless, so it commutes with the PA exactly
    # (PA(x*e^jp) = PA(x)*e^jp); it does NOT commute with the ISI convolution,
    # so settling-vs-ISI is the only ordering that matters and it must match the
    # fit. The old fitted-cubic route stays where it was, folded into `base`
    # pre-ISI, because j3's taps were estimated with the cubic in the same solve
    # and moving it would break the pairing those taps were fitted under.
    _t = np.arange(BURST_LEN, dtype=float) / BURST_LEN
    _shape = prof.get("settling_shape")
    _set = [complex(a, b_) for a, b_ in prof.get("settling", [])]
    # common-mode band ripple for this config, applied per burst after the ISI.
    # LINEAR convolution, centred, matching fit_with_ripple's
    # np.convolve(col, fir, "same") exactly -- the taps were estimated against
    # this operator, so any difference here breaks the pairing. Applying it by
    # FFT multiply instead would be CIRCULAR on a non-periodic burst and would
    # fold preamble energy into the null tail, which is where the SNR estimator
    # reads its floor.
    _rip_fir = (np.array([complex(a, b_) for a, b_ in prof["ripple_fir"]],
                         dtype=np.complex128)
                if prof.get("ripple_fir") else None)
    _set_phase = (np.exp(1j * np.deg2rad(np.interp(
        _t, np.asarray(_shape["t"]), np.asarray(_shape["phi_deg"]))))
        if _shape else None)
    if _set_phase is None and _set:
        base = base * (1.0 + sum(c * _t ** (m + 1) for m, c in enumerate(_set)))
    # Placement note: applying this phase BEFORE the PA and ISI instead of
    # after was tested and changes NOTHING -- all seven end metrics identical
    # to 4 decimals. A profile this smooth commutes with a +-1 symbol ISI in
    # practice, so the physical argument (LO sits at the mixer, before both)
    # and the identification argument (removed from the received average,
    # after both) do not have to be reconciled. Kept post-ISI to match the fit.
    base = base / np.sqrt(np.mean(np.abs(base[DATA_START:DATA_END]) ** 2))
    _n_sym = N_PRE + N_DATA + N_TAIL
    _x_sym = np.arange(_n_sym) * SPS           # symbol instants in sample index
    _x_samp = np.arange(BURST_LEN)
    _pn_amp = (_pn_amplitude(pn_mask, _n_sym, FS / SPS)
               if pn_mask is not None else None)
    # white noise through a filter h keeps sum(h^2) of its power (Parseval)
    # from the MEASURED response, not the Kaiser: the two differ by 1.68 dB
    # (0.5035 vs 0.7413) and this scales the AWGN so in-band noise lands on
    # target AFTER filtering. Leaving it on the Kaiser value would put every
    # synthetic SNR 1.7 dB out.
    # sum|h|^2 of the 601-tap FIR, applied ONCE (was 0.7323 when squared)
    _RX_RETAIN = 0.8734

    rx = np.zeros(n_bursts * BURST_SPACING, dtype=np.complex64)
    # TRANSMIT-ENVELOPE GATE for the post-filter floor, built from the burst
    # waveform BEFORE AWGN so it reaches ~0 where the TX is idle. Using |rx|
    # instead would never fall below the thermal noise already in rx and the
    # gate would not close.
    gate = np.zeros(n_bursts * BURST_SPACING, dtype=np.float32)
    for j in range(n_bursts):
        b = base
        # Generate the phase process at SYMBOL rate and RESAMPLE UP, per
        # fit_phase_noise_mask: synthesising straight at sample rate would
        # extrapolate the mask far past the fitted 3k-150k band, and with a
        # shallow slope the variance integral is dominated by that extrapolation.
        if pn_mask is not None:
            ph = _pn_draw(_pn_amp, _n_sym, rng)
            b = b * np.exp(1j * np.interp(_x_samp, _x_sym, ph))
        elif pn["tangential_deg"] > 0:          # fallback: no mask fitted yet
            ph = np.deg2rad(rng.normal(0, pn["tangential_deg"], _n_sym))
            b = b * np.exp(1j * np.interp(_x_samp, _x_sym, ph))
        # INERT IN PRACTICE: pn_radial_pct is 0.0 in every shipped profile
        # (131/131 checked 2026-09-11), so this never fires. Bottom-up measures
        # a radial term (0.184% rms on wifi) and deliberately does not inject
        # it, attributing it to the clock resampler. Leaving the capability in
        # place, but do not describe the chain as injecting AM -- it does not.
        if pn["radial_pct"] > 0:
            am = rng.normal(0, pn["radial_pct"] / 100.0, _n_sym)
            b = b * (1 + np.interp(_x_samp, _x_sym, am))
        b = b / np.sqrt(np.mean(np.abs(b[DATA_START:DATA_END]) ** 2))
        # PA BEFORE ISI. The PA sits in the transmitter; the ISI is everything
        # downstream of it -- cable, filters, receive chain. Applying ISI first
        # (as this did) is physically backwards, and it shows: feeding the
        # jointly-fitted coefficients through ISI->PA gives residual 0.158
        # against 0.015 for PA->ISI and 0.013 for the fit itself, a 12x penalty.
        b = _pa(b, {"kind": prof["pa_kind"], "b_re": p["pa_b_re"],
                    "b_im": p["pa_b_im"], "A": prof.get("pa_A", 1e6),
                    "p": 3.0})
        b = _isi_wave(b, taps, lags)
        # measured settling: AFTER ISI, matching fit_hammerstein's removal
        if _set_phase is not None:
            b = b * _set_phase
        if _rip_fir is not None:
            b = np.convolve(b, _rip_fir, mode="same")
        b = b / np.sqrt(np.mean(np.abs(b[DATA_START:DATA_END]) ** 2))
        b = add_iq_imbalance(b, IQ_SIGN * p["iq_amp_db"], p["iq_phase_deg"])
        b = b + 10 ** (p["lo_leak_dbc"] / 20.0) * np.exp(1j * p["leak_phase_rad"])
        # No per-burst CFO dither. It was added to break the estimator's 9.5367 Hz
        # quantisation, which parabolic interpolation in estimate_cfo now removes at
        # source -- verified: run-level sigma is identical with it on and off
        # (0.003550 vs 0.003549). Nor was it worth keeping as physics: the scatter
        # is ~14.95 Hz at 2400 vs 1.26 at 433, and at 2400 it is the RECEIVER's
        # noise floor (lag-1 white, common-mode across radios, tracking 1/sqrt(SNR)),
        # not the radio. At 433 part of it IS real LO wander but it is CORRELATED
        # burst to burst (lag-1 0.59 / 0.32 on two units), so an i.i.d. Gaussian has
        # the wrong shape anyway. Extraction estimates and removes each burst's CFO
        # and _deramp strips residual per-burst linear phase, so injected scatter is
        # cancelled twice before reaching any metric. Modelling it properly would be
        # add_carrier_frequency_drift's OU process, for no measurable gain.
        b = add_cfo(b, cfo_hz / FS,
                    phase_offset=p["phi0_rad"] + 2 * np.pi * cfo_hz / FS
                    * (j * BURST_SPACING))
        b = b + dc * np.exp(1j * p["dc_phase_rad"])
        # Split the accumulated clock offset into an INTEGER sample shift, which
        # MOVES THE BURST in the capture, and the fractional remainder, which is
        # the symbol clock phase. Wrapping the whole thing to [-0.5, 0.5) and
        # laying every burst on an exact BURST_SPACING grid throws the integer
        # part away -- but it is physically real: the RX clock differs from the
        # TX clock, so burst POSITIONS drift. Measured on real captures, mean gap
        # 6525.0105 (+8.0 samples over 763 bursts) on 30BF779/89_433 and 6524.9948
        # (-4.0) on 30ECBC5/77_2400, against an exact 6525.0000 before this fix.
        # The mean gap IS a clock-ppm estimate, so dropping the drift erased a
        # per-device signature and left a constant separating synthetic from real.
        # Max |drift| is 12 samples over 762 bursts across all 8 radios, inside
        # the 25-sample slack (BURST_SPACING - BURST_LEN), so length is unchanged.
        tot = p["cp0_samp"] + advance * j
        frac = (tot + 0.5) % 1.0 - 0.5
        shift = int(round(tot - frac))
        b = add_symbol_clock_phase(b, frac, method="fir", N_taps=25)
        # AWGN last. Its budget is what is LEFT after the leakage and DC, which
        # also land in the null tail where the estimator reads its noise floor.
        # And add_awgn_snr references mean(|x|^2) over the WHOLE array including
        # that tail, ~0.7 dB below the data-portion power -- shift the argument
        # by the measured ratio or the recovered SNR reads high by exactly that.
        # The decimation filter at the end of the run removes the out-of-band
        # noise, which would raise the recovered SNR by 10log10(1/sum(h^2)).
        # Inject that much extra so the post-filter in-band SNR is the target.
        tot = 10 ** (-p["snr_db"] / 10.0) / _RX_RETAIN
        eff = tot - 10 ** (p["lo_leak_dbc"] / 10.0) - dc ** 2
        ratio = np.mean(np.abs(b) ** 2) / np.mean(
            np.abs(b[DATA_START:DATA_END]) ** 2)
        # gate from the CLEAN envelope, smoothed to burst scale (~1 symbol).
        # Deliberately not the instantaneous |b|: what was measured is a
        # difference between transmitting and idle REGIONS, not symbol-to-symbol
        # structure, and tracking individual symbols would assert more than that.
        _e = np.convolve(np.abs(b) ** 2, np.ones(SPS) / SPS, mode="same")
        b = add_awgn_snr(b, -10 * np.log10(max(eff, tot * 1e-3))
                         + 10 * np.log10(ratio), db=True, rng=rng)
        pos = j * BURST_SPACING + shift
        rx[pos:pos + BURST_LEN] = b.astype(np.complex64)
        gate[pos:pos + BURST_LEN] = np.sqrt(
            _e / max(float(np.mean(_e[DATA_START:DATA_END])), 1e-30))

    # RECEIVER anti-alias filter, applied ONCE to the whole stream. Per burst
    # would ring across the burst boundaries; the real BB60 filters continuously.
    # AWGN above was scaled up by 1/sum(h^2) so the IN-BAND noise -- which is
    # what snr_mean_db reads -- lands on target after the filter removes the
    # out-of-band part.
    rx = scipy.signal.oaconvolve(rx, RX_FIR, mode="same").astype(np.complex64)
    # POST-FILTER floor, flat across the full band. Level is relative to the
    # IN-BAND NOISE, not to the signal: variance = thermal * 10^(FLOOR/10). See
    # the RX_FLOOR_DBC block for why the mechanism is NOT the converter.
    #
    # GATED BY THE TRANSMIT ENVELOPE (2026-09-10). This term used to be added
    # flat across the whole stream, including the null tails and inter-burst
    # gaps. Measured consequence on real vs synthetic 89_2400: synthetic quiet
    # regions came out +26.2 dB hot in the 2.10-2.50 MHz band, and one scalar --
    # far-band power in the quiet region -- separated 15 real from 30 synthetic
    # files with a 24 dB margin and no overlap. A classifier reaches 100% on
    # real-vs-synthetic without learning anything about radios.
    #
    # Gating is the model FORM the measurement supports, not a patch:
    #   present only while transmitting  real data - quiet = +17.8 dB in the far
    #     band, synthetic (flat floor) = +0.03 dB
    #   incoherent                       far band falls 17.5 dB over 80 aligned
    #     burst averages against a 19.0 dB noise prediction, while in-band stays
    #     flat to 0.05 dB. So no deterministic PA model can produce it at any
    #     order -- confirmed independently on the reconstruction chain, where
    #     parallel Hammerstein, order 5, order 7 and spectral whitening all left
    #     the far band unmoved.
    # A constant gets "noise-like" right and "only while transmitting" wrong.
    #
    # MECHANISM STILL NOT CLAIMED. Signal-dependence alone would fit amplified
    # TX-chain noise, but that would be radiated, pass the receiver anti-alias
    # filter and be suppressed ~80 dB out here, so it could not appear flat at
    # this level. Flat AND unfiltered points instead at something generated
    # downstream of the filter but driven by input level -- receiver
    # nonlinearity or ADC products. Both readings imply the same implementation
    # (post-filter, envelope-gated), which is why the code does not depend on
    # settling it. One mechanism claim about this term has already been
    # retracted; this comment does not add another.
    _fl = 10 ** (-p["snr_db"] / 10.0) * 10 ** (RX_FLOOR_DBC / 10.0)
    rx = (rx + np.sqrt(_fl / 2.0) * gate
          * (rng.standard_normal(rx.size)
             + 1j * rng.standard_normal(rx.size))).astype(np.complex64)
    return tx, rx, p


def write_run(root, idx, tx, rx, prof, stamp):
    """Real SigMF layout, so radio_characterise.py reads it unchanged."""
    vd = root / f"run_{idx:03d}" / "valid"
    vd.mkdir(parents=True, exist_ok=True)
    ts = f"{stamp:%Y%m%dT%H%M%SZ}"
    tx.tofile(vd / f"4qam_srrc_syn_tx_{ts}.sigmf-data")
    rx.tofile(vd / f"capture_{ts}_combined.sigmf-data")
    gain = prof["config"].split("_")[0]
    meta = lambda hw_, sn: {
        "global": {"core:datatype": "cf32_le", "core:sample_rate": FS,
                   "core:version": "1.0.0", "core:hw": hw_,
                   "device:serial_number": sn,
                   "tx:frequency": prof["fc_hz"], "tx:gain": float(gain)},
        "captures": [{"core:sample_start": 0, "core:frequency": prof["fc_hz"],
                      "core:datetime": stamp.isoformat(),
                      "device:temperature": 40.0}],
        "annotations": []}
    (vd / f"4qam_srrc_syn_tx_{ts}.sigmf-meta").write_text(
        json.dumps(meta(f"USRP B210 SN:{prof['radio']}syn",
                        f"{prof['radio']}syn"), indent=2))
    mr = meta("BB60C SN:synthetic", "synthetic")
    mr["global"].update({"core:description": "capture chunk 0",
                         "device:name": "BB60C", "device:type": "BB60C",
                         "device:firmware_version": "8",
                         "device:temperature": 40.0,
                         "stream:sample_rate": FS,
                         "stream:center_freq": prof["fc_hz"],
                         "stream:bandwidth": 3.75e6})
    # MIRROR the real annotation: one entry of CHUNK_SAMPLES, not one spanning
    # the whole buffer. The real BB60 metas record core:sample_count = 250000 on
    # a ~5M-sample file (the writer's chunk size); writing rx.size instead made
    # the synthetic structurally different from real under any standard SigMF
    # reader, even though nothing in radio_characterise reads this field.
    mr["annotations"] = [{"core:sample_start": 0,
                          "core:sample_count": CHUNK_SAMPLES,
                          "capture:bandwidth": 3.75e6,
                          "capture:firmware_version": "8",
                          "capture:temperature": 40.0}]
    (vd / f"capture_{ts}_combined.sigmf-meta").write_text(json.dumps(mr, indent=2))


def generate(prof, var, n_runs, n_bursts, out, seed=0, verbose=True,
             write_data=True):
    """write_data=False replays the parameter draws and writes ground truth
    WITHOUT the waveforms -- for recovering provenance of an existing set."""
    rng = np.random.default_rng(seed)
    stamp = datetime.now(timezone.utc)
    root = Path(out)
    if not root.is_absolute():
        root = BASE / root
    root = root / prof["config"]
    state, truth = {}, []
    for r in range(n_runs):
        tx, rx, p = synth_run(prof, var, rng, state, n_bursts)
        if write_data:
            write_run(root, r + 1, tx, rx, prof, stamp)
        truth.append(dict(run=r + 1, **{k: float(v) for k, v in p.items()}))
        if verbose and ((r + 1) % 10 == 0 or r == 0):
            print(f"  run {r + 1}/{n_runs}  ({rx.nbytes / 2**20:.0f} MB)")
    root.mkdir(parents=True, exist_ok=True)
    (root / "profile.json").write_text(
        json.dumps({"profile": prof, "variation": var, "seed": seed}, indent=2))
    (root / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    return root


def describe(prof, var):
    print(f"\n{prof['radio']} / {prof['config']}   "
          # A HAND-BUILT PROFILE HAS NO n_real_runs. That key is provenance from
        # profile_from_log; requiring it made describe() crash on exactly the
        # profiles a new user writes themselves.
        f"({str(prof['n_real_runs']) + ' real runs' if 'n_real_runs' in prof else 'hand-built'}"
        f", fc {prof['fc_hz']/1e6:.2f} MHz)")
    print(f"  {'parameter':<18}{'value':>14}   variation")
    for k in ["ref_ppm", "clock_mismatch_ppm", "iq_amp_db", "iq_phase_deg",
              "snr_db", "lo_leak_dbc", "cp0_samp", "phi0_rad", "rx_dc_frac",
              "pa_b_re", "pa_b_im", "pn_level_db", "pn_radial_pct"]:
        s = var.get(k, {"kind": "fixed"})
        # LAZY, NOT A DICT LITERAL. Every f-string in a dict literal is
        # evaluated before .get() picks one, so a mixture spec -- whose "sd" is
        # a LIST -- crashed the gauss branch it never selects. --dry-run died on
        # any radio/config using the mixture.
        _k = s["kind"]
        if _k == "fixed":
            d = "held at the profile value"
        elif _k == "derived":
            d = "derived from ref_ppm (shared reference)"
        elif _k == "gauss":
            d = f"Gaussian sd={s.get('sd', 0):.5g}"
        elif _k == "uniform":
            d = f"Uniform [{s.get('lo', 0):.4g}, {s.get('hi', 0):.4g}]"
        elif _k == "ar1":
            d = f"AR(1) sd={s.get('sd', 0):.5g} phi={s.get('phi', 0):.3f}"
        elif _k == "mixture":
            d = ("mixture " + ", ".join(
                f"{w:.2f}@{m:.4g}+/-{sd:.3g}"
                for w, m, sd in zip(s["w"], s["mu"], s["sd"])))
        elif _k == "scipy":
            d = f"scipy.stats.{s.get('dist')} {s.get('params', {})}"
        else:
            d = _k
        print(f"  {k:<18}{prof.get(k, 0.0):>14.5g}   {d}")
    if prof.get("ripple_fir"):
        _rf = np.array([complex(a, b) for a, b in prof["ripple_fir"]])
        _H = np.fft.fft(_rf, 5000)
        _f = np.fft.fftfreq(5000, 1.0 / 5e6)
        _m = np.abs(_f) <= 0.60e6
        _db = 20 * np.log10(np.abs(_H[_m]) / np.abs(_H[_m]).mean())
        print(f"  {'ripple':<18}{'':>14}   common-mode FIR, {len(_rf)} taps, "
              f"{_db.std():.4f} dB rms in band")
    t = prof["isi_taps"]
    print(f"  {'isi_taps':<18}{'':>14}   fixed, {len(t['lags'])} complex taps")
    for k, c in zip(t["lags"], t["c"]):
        z = complex(c[0], c[1])
        print(f"      lag {k:+d}   {20*np.log10(max(abs(z),1e-12)):+7.2f} dBc"
              f"   {np.degrees(np.angle(z)):+7.1f} deg")
    if prof["pa_kind"] == "cubic":
        b = complex(prof["pa_b_re"], prof["pa_b_im"])
        print(f"  {'pa':<18}{'':>14}   cubic  b = {b.real:+.5f} {b.imag:+.5f}j"
              f"   |b| = {abs(b):.5f}")
    else:
        _A = prof.get("pa_A", 0)
        print(f"  {'pa':<18}{'':>14}   rapp"
              + (f", A={_A:.3g} (linear)" if _A > 100 else f", A={_A:.3g} p=3"))
    _rf = prof.get('pn_radial_pct_fitted', 0.0)
    if prof.get("settling_shape"):
        _p = np.asarray(prof["settling_shape"]["phi_deg"])
        print(f"  {'settling':<18}{'':>14}   MEASURED shape, phase only, "
              f"{_p.max() - _p.min():.2f} deg span, {len(_p)} pts  [this "
              f"radio/config only]")
    elif prof.get("settling"):
        _s = [complex(a, b_) for a, b_ in prof["settling"]]
        print(f"  {'settling':<18}{'':>14}   fitted complex cubic, |profile| "
              f"span {abs(sum(_s)):.4f}  [extrapolates into the preamble]")
    if prof.get("pn_mask"):
        print(f"  {'pn':<18}{'':>14}   1/f^a mask, slope "
              f"{prof.get('pn_slope_db_dec', 0):+.2f} dB/dec, level "
              f"{prof['pn_mask'][2][1]:.1f} dB @10 kHz"
              + ("   [FLEET MEAN -- per-radio level not resolvable]"
                 if prof.get("pn_level_shared") else "   [per-radio]"))
    print(f"  {'pn (diag)':<18}{'':>14}   {prof['pn_tangential_deg']:.3f} deg "
          f"tangential gap left by the ISI taps (NOT injected)")
    print(f"  {'':<18}{'':>14}   {prof['pn_radial_pct']:.3f} % radial injected"
          + (f"  [filler {_rf:.3f} % NOT injected]" if _rf > 0.001
             and prof['pn_radial_pct'] == 0.0 else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate a multi-run synthetic capture set from a measured "
                    "device profile plus a run-to-run variation spec.")
    ap.add_argument("--radio", default="30BF7B6")
    ap.add_argument("--config", default="77_433", help="77_433 or g77_433")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--bursts", type=int, default=762)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output sweep directory")
    ap.add_argument("--radial-filler", action="store_true",
                    help="inject the unmodelled radial remainder (off by default)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the profile and variation spec, write nothing")
    a = ap.parse_args(argv)

    prof, var = profile_from_log(a.radio, a.config, a.radial_filler)
    describe(prof, var)
    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return
    stamp = datetime.now(timezone.utc)
    out = a.out or (f"New_captures/4QAM_{a.radio}syn_sweep_"
                    f"{stamp:%Y%m%d}_{stamp:%H%M%S}_syn")
    print(f"\nwriting -> {out}")
    root = generate(prof, var, a.runs, a.bursts, out, seed=a.seed)
    print(f"\nwrote {a.runs} runs -> {root}")
    print(f"profile + ground truth -> {root.parent}")


if __name__ == "__main__":
    main()
