"""Multi-run synthetic capture generator: device profile + variation spec -> SigMF.

A profile holds one radio's measured impairments (from the characterisation log
and the fitted data files). A variation spec says how each parameter is redrawn
per run. The output uses the real capture layout, so the same analysis code
reads real and synthetic data.

Variation kinds (see draw()):

    {'kind': 'fixed'}                      hold at the profile value
    {'kind': 'gauss',   'sd': x}           Normal about it (or about 'mu')
    {'kind': 'uniform', 'lo': a, 'hi': b}  Uniform
    {'kind': 'ar1',     'sd': x, 'phi': p} Gaussian that wanders across runs
    mixture, quantile, scipy               see draw()

By default the spec is fingerprint-only: device-fixed impairments stay per
radio and the rest are drawn from the fleet (_fleet_nuisance).

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
# Data folder. SG_DATA_DIR swaps the whole folder (e.g. a refit on training
# sessions only); each file's own SG_*_FILE override applies on top.
import os
DATA = Path(os.environ.get("SG_DATA_DIR", str(BASE / "data")))
import sys
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "PA_modelling_with_GMP"))
from cel_signal_gen_lib.core.filter_design import srrc_design
from cel_signal_gen_lib.impairments.hardware import (
    add_cfo, add_iq_imbalance, add_symbol_clock_phase, add_phase_noise,
    add_decimation_filter, _decimation_taps, add_pa_nonlinearity)
from cel_signal_gen_lib.impairments.channel import add_awgn_snr

# ── waveform geometry (matches the capture campaign) ────────────────────────
FS, SPS, BETA, SPAN = 5e6, 5, 0.35, 10
N_PRE, N_DATA, N_TAIL = 100, 1000, 200
BURST_LEN = (N_PRE + N_DATA + N_TAIL) * SPS
BURST_SPACING = 6525
# Unused; kept for scripts that import them. The receive filter in use is RX_FIR.
RX_BANDWIDTH = 3.75e6
RX_STOP_DB = 80.0
# BB60 receive (anti-alias) filter: an order-10 elliptic fitted to the measured
# response (the receiver's quiet-region PSD), realised as a 601-tap zero-phase
# FIR and applied once. The elliptic keeps the flat ~-75 dB stopband shelf the
# real receiver shows; zero phase adds no in-band distortion the ISI taps would
# have to explain. Fit: rms 1.97 dB across bands.
RX_FIR = np.load(DATA / "bb60_rx_fir_5msps_n10.npy")


def _flat_passband(h, fs=5e6, f_flat=0.9e6, nfft=1 << 16):
    """The filter with its passband magnitude divided out up to f_flat (held
    beyond), so the stopband is unchanged.

    For fits that already contain the BB60 passband (+0.42 dB at 500 kHz):
    applying the full filter on top would count it twice. Same definition as
    analysis_scripts/bb60_eq.py."""
    c = (len(h) - 1) // 2
    H = np.fft.fft(np.roll(np.r_[h, np.zeros(nfft - len(h))], -c))
    f = np.fft.fftfreq(nfft, 1 / fs)
    m = np.abs(H) / np.abs(H[0])
    edge = np.interp(f_flat, f[:nfft // 2], m[:nfft // 2])
    g = np.real(np.fft.ifft(H / np.where(np.abs(f) <= f_flat, m, edge)))
    return np.roll(g, c)[:len(h)]


RX_FIR_FLAT = _flat_passband(RX_FIR)
# Additive Gaussian floor after the filter, dB relative to the in-band noise.
# An empirical constant that matches the real far-band level; not a quantiser
# model, and its mechanism is open.
RX_FLOOR_DBC = -45.0
CHUNK_SAMPLES = 250000       # the real BB60 writer's annotation chunk
DATA_START, DATA_END = N_PRE * SPS, (N_PRE + N_DATA) * SPS
IDEAL = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4)))

# Zadoff-Chu preamble, as the real transmitter sends it: root 1 (the YAML's 25
# is not coprime with 100, so the TX code falls back to 1) and the even-length
# n^2 form. Correlates 0.997 with a real TX preamble.
_ZC_N, _ZC_ROOT = 100, 1
_zc_n = np.arange(_ZC_N)
PREAMBLE = np.exp(-1j * np.pi * _ZC_ROOT * _zc_n * _zc_n / _ZC_N)
SRRC = srrc_design(SPS, SPAN, BETA)
MF_DELAY = len(SRRC) // 2

LOG = DATA / "radio_characterisation.json"

# Optional, shared with the fitting scripts:
#   SG_RUN_SUBSET  JSON list of run numbers: use only those real runs
#   SG_TAPS_FILE   alternative taps file
def _sg_subset():
    import os, json as _j
    p = os.environ.get("SG_RUN_SUBSET")
    if not p:
        return None
    return {f"run_{int(n):03d}" for n in _j.loads(Path(p).read_text())}


TAPS = Path(os.environ.get("SG_TAPS_FILE", str(DATA / "isi_taps.json")))

# Measured settling shape (phase only, degrees, mean-removed over the data
# portion), used only for the (radio, config) pairs listed in the file's
# applies_to block; everywhere else the fitted cubic is used. Interpolated
# linearly, end values held. SG_SHAPE_FILE overrides the path.
import os as _os_sh
import functools as _ft


@_ft.lru_cache(maxsize=None)
def _load_versioned(path, fmt, key="format_version", why=""):
    """A versioned JSON data file, parsed once per process, or None if it does
    not exist. A different format version raises (`why` is appended)."""
    path = Path(path)
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if d.get(key) != fmt:
        raise SystemExit(f"{path.name}: format_version is {d.get(key)!r}, this generator "
                         f"expects {fmt}.{(' ' + why) if why else ''}")
    return d


SHAPE_FILE = Path(_os_sh.environ.get("SG_SHAPE_FILE",
                  str(DATA / "srrc_settling_shape_for_generator.json")))
# Bump only after re-reading the file and re-checking the routing: a format
# change otherwise fails silently (falls back to the cubic).
SHAPE_FORMAT = 4

# ── Common-mode band ripple, per config ─────────────────────────────────────
# What taps + PA leave of the tx->rx response, pooled across radios
# (fit_ripple.py), so it carries no per-radio identity. It must be used with
# the taps fitted alongside it ('ripple_fit'); taps fitted without it already
# contain part of the band shape, so the generator raises if that block is
# missing.
RIPPLE_FILE = Path(_os_sh.environ.get(
    "SG_RIPPLE_FILE", str(DATA / "srrc_ripple_per_config.json")))
RIPPLE_FORMAT = 1


# ── Burst transient over the preamble, per config ───────────────────────────
# The settling cubic is fitted on the data portion only; over the preamble the
# measured fleet profile is used instead (measure_preamble_transient.py),
# spliced in by _splice_preamble. --cubic-preamble extrapolates the cubic.
PREAMBLE_FILE = Path(_os_sh.environ.get(
    "SG_PREAMBLE_FILE", str(DATA / "srrc_preamble_transient.json")))
PREAMBLE_FORMAT = 1


def _preamble_transient(cfg_dir):
    """{"t", "phi_deg"} for this config, or None."""
    d = _load_versioned(PREAMBLE_FILE, PREAMBLE_FORMAT)
    if d is None:
        return None
    c = d["configs"].get(cfg_dir)
    return None if c is None else {"t": d["t"], "phi_deg": c["phi_deg"]}


def _splice_preamble(g, t, pt, blend=100):
    """Replace the cubic's extrapolated preamble by the measured profile.

    Phase: the fleet profile, referenced to the data-portion mean of this
    burst's cubic (as measured), not rescaled per radio. The last `blend`
    preamble samples ramp onto the cubic, so there is no step. Magnitude: held
    at its value at the first data sample."""
    i0 = DATA_START
    phi = np.unwrap(np.angle(g))
    P = np.deg2rad(np.interp(t, np.asarray(pt["t"]), np.asarray(pt["phi_deg"])))
    new = np.mean(phi[DATA_START:DATA_END]) + P[:i0]
    w = np.clip((np.arange(i0) - (i0 - blend)) / blend, 0.0, 1.0)
    new = new + w * (phi[i0] - new[-1])
    out = g.copy()
    out[:i0] = np.abs(g[i0]) * np.exp(1j * new)
    return out


def _transient_basis():
    """Mean-removed cubic basis over the data portion, where the transient is
    fitted (the main tap absorbs a constant, so the mean is not identified)."""
    t = np.arange(DATA_START, DATA_END, dtype=float) / BURST_LEN
    P = np.stack([t, t ** 2, t ** 3], 1)
    return P - P.mean(0)


@_ft.lru_cache(maxsize=32)
def _transient_shape_cached(taps_path, cfg_dir, src):
    C = []
    for k, tp in json.loads(Path(taps_path).read_text()).items():
        b = tp.get(src) if src else tp
        if k.endswith(f"/{cfg_dir}") and isinstance(b, dict) and b.get("settling"):
            C.append([complex(a, c) for a, c in b["settling"]])
    if len(C) < 3:
        return None
    C = np.array(C)
    Pm = _transient_basis()
    _, R = np.linalg.qr(Pm)
    U, _, _ = np.linalg.svd(R @ C.T, full_matrices=False)
    s = np.linalg.solve(R, U[:, 0])
    s = s / np.sqrt(np.mean(np.abs(Pm @ s) ** 2)) * np.deg2rad(1.0)     # 1 deg rms
    A = (np.conj(Pm @ s) @ (Pm @ C.T)) / np.vdot(Pm @ s, Pm @ s)
    return tuple(s * np.exp(1j * np.angle(A.mean())))                   # fleet level real, > 0


def _transient_shape(cfg_dir, src):
    """Fleet shape of the burst transient for this config.

    The transient is g(t) = 1 + level_deg * s(t), t = sample / BURST_LEN, with
    s(t) a complex cubic shared by the fleet (normalised to 1 deg rms over the
    data portion) and level_deg the radio's size. Reproduces each radio's own
    cubic to a median 3-13 % of its size. Computed from the taps file in use.
    Returns 3 complex coefficients, or None (fewer than 3 radios).
    --full-transient uses each radio's own cubic instead."""
    return _transient_shape_cached(str(TAPS), cfg_dir, src)


def _transient_level(settling, shape):
    """The radio's level: its fitted cubic projected onto the fleet shape (deg)."""
    Pm = _transient_basis()
    gs, gr = Pm @ np.asarray(shape), Pm @ np.array([complex(a, b) for a, b in settling])
    return float((np.vdot(gs, gr) / np.vdot(gs, gs)).real)


ISI_LAGS = [-1, 1]


@_ft.lru_cache(maxsize=32)
def _isi_base_cached(taps_path, cfg_dir, src):
    C, names = [], []
    for k, tp in sorted(json.loads(Path(taps_path).read_text()).items()):
        b = tp.get(src) if src else tp
        if (k.endswith(f"/{cfg_dir}") and isinstance(b, dict) and b.get("c_inject")
                and list(b.get("lags", tp.get("lags", []))) == ISI_LAGS):
            C.append([complex(a, c) for a, c in b["c_inject"]])
            names.append(k.split("/")[0])
    if len(C) < 3:
        return None
    C, k = np.array(C), np.array(ISI_LAGS)
    A = C.mean(0)
    grid = np.deg2rad(np.linspace(-90, 90, 3601))           # 0.05 deg steps
    E = np.exp(-1j * np.outer(grid, k))                     # (grid, lags)
    for _ in range(30):                                     # alternate: shifts, then A
        d = grid[np.argmin(np.abs(C[:, None, :] - A[None, None, :] * E[None]) ** 2 @ np.ones(len(k)), axis=1)]
        A = np.mean(C * np.exp(1j * np.outer(d, k)), 0)
    khz = -np.degrees(d) / 360.0 * (FS / SPS) / 1e3         # + = response moves UP
    off = khz.mean()                                        # centre the shifts on 0
    A = A * np.exp(1j * k * 2 * np.pi * off * 1e3 / (FS / SPS))   # A at shift = off
    return tuple(A), float(np.std(khz)), dict(zip(names, (khz - off).tolist()))


def _isi_base(cfg_dir, src):
    """The post-PA linear block as a fleet response slid in frequency.

    Two taps at -1, +1 symbol give a ripple with a 1 MHz period. Each radio's
    taps are one fleet pair A shifted in frequency: c_k = A_k exp(j k 2pi f/Rs),
    f = isi_shift_khz (positive moves the response up). This explains 94-99 %
    of the radio-to-radio variance; the shift follows the capture session.
    Computed from the taps file in use. Returns (A, fleet sd of the shift in
    kHz, {radio: shift}) or None. --raw-taps uses each radio's own taps."""
    return _isi_base_cached(str(TAPS), cfg_dir, src)


def _isi_taps_from_shift(base, shift_khz):
    k = np.array(ISI_LAGS)
    return np.asarray(base) * np.exp(1j * k * 2 * np.pi * shift_khz * 1e3 / (FS / SPS))


def _ripple_fir(cfg_dir):
    """Complex FIR for this config's ripple, or None if none is in scope."""
    d = _load_versioned(RIPPLE_FILE, RIPPLE_FORMAT, key="_format_version",
                        why="The FIR convention or band may have changed; refusing to "
                            "inject a curve written for a different contract.")
    if d is None:
        return None
    c = d["configs"].get(cfg_dir)
    if c is None:
        return None
    return np.array([complex(a, b) for a, b in c["fir"]], dtype=np.complex128)

# ── Phase noise: one curve per radio/config ────────────────────────────────
# S_phi(f), 1.5 Hz to 500 kHz, drawn as one continuous process per run, so
# bursts share the slow LO wander. Fitted (fit_pn_curve.py) to the in-burst
# symbol deviation (per radio) and the burst-to-burst preamble phase (fleet-
# common below ~200 Hz). --per-burst-pn: the older per-burst mask.
PN_CURVES_FILE = Path(_os_sh.environ.get(
    "SG_PN_CURVES_FILE", str(DATA / "pn_curves.json")))
PN_CURVES_FORMAT = 1


def _pn_curve(radio, cfg_dir):
    """[(f_hz, S_phi dB)] for this radio/config, or None. radio=None returns
    every curve of the config, as a list."""
    d = _load_versioned(PN_CURVES_FILE, PN_CURVES_FORMAT,
                        why="Units or interpolation may have changed; refusing to inject "
                            "a curve written for a different contract.")
    if d is None:
        return None
    if radio is None:
        return [[list(map(float, p)) for p in c["curve"]]
                for k, c in d.get("curves", {}).items() if k.endswith(f"/{cfg_dir}")]
    c = d.get("curves", {}).get(f"{radio}/{cfg_dir}")
    if c is None:
        return None
    return [list(map(float, p)) for p in c["curve"]]


def _curve_at(curve, f):
    m = sorted(curve)
    return float(np.interp(np.log10(f), np.log10([p[0] for p in m]), [p[1] for p in m]))


def shift_pn_curve(curve, db, lo=200.0, hi=3200.0):
    """Shift the PER-RADIO (in-burst) part of a curve by db: fully at and
    above hi, not at all at or below lo (the fleet-common slow part), linear in
    log f between. Used where a per-radio level is replaced by the fleet's."""
    w = np.clip(np.log10(np.array([p[0] for p in curve]) / lo)
                / np.log10(hi / lo), 0.0, 1.0)
    return [[p[0], p[1] + db * wi] for p, wi in zip(curve, w)]

# ── Receiver phase spur (opt-in, --rx-spur) ─────────────────────────────────
# The BB60's phase modulation every 1168 samples (4280.82 Hz) plus its 2nd
# harmonic, phase fixed to the capture start (rx_spur.py). The fits used
# captures with it removed. --rx-spur adds it back, one fleet level and phase
# per config, before the receive filter; it uses no random numbers.
from rx_spur import add_rx_spur                                    # noqa: E402
RX_SPUR_FILE = Path(_os_sh.environ.get(
    "SG_RX_SPUR_FILE", str(DATA / "rx_spur.json")))
RX_SPUR_FORMAT = 1


# ── PA gain modulation at gain 89 (default; --no-pa-mod) ─────────────────────
# At 89_433 / 89_915 the data portion's complex gain rotates at a fixed rate
# seen at the burst rate (-2.15 / -2.38 Hz), about +-5 % / +-1.6 %:
# g_j = 1 + amp exp(j(2 pi f j / f_burst + phase)). Transmit-side and
# fleet-common: one entry per config, random phase per run.
PA_MOD_FILE = Path(_os_sh.environ.get(
    "SG_PA_MOD_FILE", str(DATA / "pa_gain_mod.json")))
PA_MOD_FORMAT = 1
# what the modulation acts on, '+'-joined: 'data' (the data portion's complex
# gain step) and/or 'b' (the cubic, data portion only). Default both.
_PA_MOD_MODE = set(_os_sh.environ.get("SG_PA_MOD_MODE", "data+b+taps").split("+"))


# ── TX LO leakage and precise CFO, per run (default; --old-leakage) ──────────
# Re-measured leakage level and frequency (remeasure_lo_leakage.py): the tone
# sits at the CFO plus a fleet constant per band (offset_hz). The same file
# holds the precise per-run CFO, which replaces the log's x^4 values.
LEAK_FILE = Path(_os_sh.environ.get("SG_LEAK_FILE", str(DATA / "lo_leakage.json")))
LEAK_FORMAT = 1


def _leak_table(radio, token, cfg_dir):
    """{'runs': {run: {'cfo_hz', 'leak_dbc'}}, 'offset_hz'} for one session, or None."""
    d = _load_versioned(LEAK_FILE, LEAK_FORMAT)
    runs = None if d is None else d["sessions"].get(f"{radio}/{token}/{cfg_dir}")
    if not runs or sum(x["leak_dbc"] is not None for x in runs.values()) < 10:
        return None
    return {"runs": runs, "offset_hz": float(d["offset_hz"][cfg_dir.split("_")[1]])}


def _pa_mod_params(cfg_dir):
    """{'freq_hz', 'amp', 'b_amp'} for this config, or None."""
    d = _load_versioned(PA_MOD_FILE, PA_MOD_FORMAT)
    if d is None:
        return None
    c = d["configs"].get(cfg_dir)
    if c is None:
        return None
    out = {"freq_hz": float(c["freq_hz"]), "amp": float(c["amp"]),
           "b_amp": float(c.get("b_amp", 0.0))}
    if "phase_rad" in c:             # measured start phase (measure_pa_mod_phase.py)
        out.update(phase_rad=float(c["phase_rad"]), phase_sd_rad=float(c["phase_sd_rad"]))
    if "tap_swing" in c:             # measured swing of the taps and the cubic (measure_pa_mod_cubic.py)
        out.update(tap_swing=c["tap_swing"]["c"], tap_swing_lags=c["tap_swing"]["lags"],
                   b_swing=c["tap_swing"]["b_swing_measured"])
    return out


def _rx_spur_params(cfg_dir):
    """{'amps_rad': [...], 'phases_rad': [...]} for this config, or None."""
    d = _load_versioned(RX_SPUR_FILE, RX_SPUR_FORMAT,
                        why="Refusing to inject a spur written for a different contract.")
    if d is None:
        return None
    c = d["configs"].get(cfg_dir)
    if c is None:
        return None
    return {"amps_rad": [float(v) for v in c["amps_rad"]],
            "phases_rad": [float(v) for v in c["phases_rad"]]}

# The settling-shape file's radial_* keys (format 4) are not used yet.


def _measured_shape(radio, cfg_dir):
    """(t, phi_deg) for this radio/config, or None if out of scope.

    Scope comes from the file's `applies_to` block. Anything unrecognised
    raises rather than returning None, so the only silent None is a genuinely
    out-of-scope pair."""
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
    # applies_to.radio: one name or a list of names
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

# Minimum residual improvement (linear / cubic) for the PA cubic to be used;
# below it b is fit noise and set to zero.
PA_GAIN_MIN = 1.30
# configs where the clean PA refit overrides that gate
PA_GATE_EXEMPT = ("89_2400",)
# Fingerprint-only variation (_fleet_nuisance): fleet percentile of the
# hour-scale CFO drift, and how far below the fleet's p10 SNR the draw reaches
FP_CFO_DRIFT_PCT = 90
FP_SNR_DROP_DB = 10.0
# Minimum (cross-radio spread)/(run-to-run noise) for a per-radio phase-noise
# level; below it every radio gets the fleet mean.
PN_LEVEL_GAIN_MIN = 1.5

# add_iq_imbalance puts +gain on Q while the estimator reports
# 20log10(|col_I|/|col_Q|), so amplitude is negated on the way in (phase is not).
IQ_SIGN = -1.0


# ════════════════════════════════════════════════════════════════════════════
# variation spec
# ════════════════════════════════════════════════════════════════════════════
DEFAULT_VARIATION = {
    # ── state: varies run to run ────────────────────────────────────────────
    # One reference drives both the CFO and the sampling clock on these radios,
    # so the clock is derived; give it its own spec for hardware where it isn't.
    "ref_ppm":       {"kind": "ar1"},
    "clock_mismatch_ppm": {"kind": "derived"},   # or gauss / ar1 / uniform
    # receiver sampling grid and phases: measured uniform
    "cp0_samp":      {"kind": "uniform", "lo": -0.5, "hi": 0.5},
    "phi0_rad":      {"kind": "uniform", "lo": 0.0, "hi": 2 * np.pi},
    "leak_phase_rad": {"kind": "uniform", "lo": 0.0, "hi": 2 * np.pi},
    "dc_phase_rad":  {"kind": "uniform", "lo": 0.0, "hi": 2 * np.pi},
    "iq_amp_db":     {"kind": "gauss"},
    "iq_phase_deg":  {"kind": "gauss"},
    "snr_db":        {"kind": "gauss"},
    "lo_leak_dbc":   {"kind": "gauss"},

    # ── transfer function: fixed (their run-to-run scatter is below their
    # estimation error), but any can be given a spec and drawn per run.
    # pa_b_re (compression) and pa_b_im (AM/PM) are separate knobs because they
    # are measured with very different precision.
    "pa_b_re":            {"kind": "fixed"},
    "pa_b_im":            {"kind": "fixed"},
    "pn_level_db":        {"kind": "fixed"},   # dB offset on the phase-noise curve
    "pn_tangential_deg":  {"kind": "fixed"},   # diagnostic only, not injected
    "pn_radial_pct":      {"kind": "fixed"},
    "rx_dc_frac":         {"kind": "fixed"},
    # a vector: "gauss" takes a relative sd on every tap, e.g. {"kind": "gauss",
    # "rel_sd": 0.01}; "pool" picks one of a list of tap sets per run
    "isi_taps":           {"kind": "fixed"},
}


def draw(spec, centre, rng, state=None):
    """One run's value for a parameter. `state` carries AR(1) memory."""
    kind = spec.get("kind", "fixed")
    if kind in ("fixed", "derived"):
        return centre, state
    new_state = state
    if kind == "uniform":
        # lo/hi are the support here, not clamps
        v = rng.uniform(spec["lo"], spec["hi"])
    elif kind == "gauss":
        # optional "mu": an absolute centre (e.g. a fleet mean) instead of the profile value
        v = spec.get("mu", centre) + rng.normal(0.0, spec["sd"])
    elif kind == "ar1":
        phi, sd = spec.get("phi", 0.0), spec["sd"]
        if state is None:
            # start stationary, so run 1 has the same spread as the rest
            s = rng.normal(0.0, sd)
        else:
            s = phi * state + rng.normal(0.0, sd * np.sqrt(max(1 - phi ** 2, 1e-9)))
        new_state = s
        v = centre + s
    elif kind == "mixture":
        # Gaussian mixture; mu/sd are absolute unless "relative" is set
        w = np.asarray(spec["w"], float)
        w = w / w.sum()
        i = int(rng.choice(len(w), p=w))
        v = rng.normal(spec["mu"][i], spec["sd"][i])
        if spec.get("relative"):
            v = centre + v
    elif kind == "scipy":
        # any scipy.stats distribution, by name, so the spec stays JSON
        # (it is written to profile.json with every dataset)
        import scipy.stats as _st
        _d = getattr(_st, spec["dist"], None)
        if _d is None:
            raise ValueError(f"unknown scipy.stats distribution {spec['dist']!r}")
        v = _d.rvs(**spec.get("params", {}), random_state=rng)
        if spec.get("relative"):
            v = centre + v
    elif kind == "quantile":
        # empirical distribution as evenly spaced quantiles (min ... max),
        # sampled by inverse CDF, linear between quantiles
        q = np.asarray(spec["q"], float)
        v = float(np.interp(rng.uniform(), np.linspace(0.0, 1.0, q.size), q))
    else:
        raise ValueError(f"unknown variation kind {kind!r}")
    # optional physical bounds (e.g. the measured range of the leakage level)
    if kind != "uniform":
        if "lo" in spec:
            v = max(v, spec["lo"])
        if "hi" in spec:
            v = min(v, spec["hi"])
    return float(v), new_state


def fit_mixture_bic(x, dbic=-10.0, iters=400):
    """Two-component Gaussian mixture, returned only if BIC clearly prefers it
    (by at least -dbic), else None. Returns {"w", "mu", "sd"}."""
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
    # reject a component collapsed onto a point (EM's singularity: sd -> 0
    # makes BIC "prefer" it) or carrying too few points
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
                     pn_level_mode="auto", continuous_pn=True, rx_spur=False,
                     cubic_preamble=False, pa_clean=True, pa_mod=True,
                     old_leakage=False, fingerprint_only=True,
                     pa_gate_2400=False, snr_drop_db=None, full_transient=False,
                     raw_taps=False, bb60_passband=False, random_pa_mod_phase=False,
                     old_pa_mod_swing=False):
    """Device profile + variation spec for one radio/config.

    Run-level values come from the characterisation log, waveform-level fits
    (ISI, PA, transient, phase noise) from the data files. fingerprint_only
    (default): only device-fixed impairments stay per radio, the rest are
    drawn from the fleet (_fleet_nuisance). The other flags select earlier
    models; see the CLI help."""
    cfg_dir = config.lstrip("g")
    cfg_key = f"g{cfg_dir}"
    log = json.loads(LOG.read_text())
    rows = [v for k, v in log.items()
            if len(k.split("/")) == 3 and k.split("/")[1] == cfg_key
            and re.match(rf"TX{radio}_RX", k.split("/")[0])
            and (SG_SUBSET is None or k.split("/")[2] in SG_SUBSET)]
    if not rows:
        raise SystemExit(f"no runs for {radio} / {cfg_key} in {LOG.name}")
    # ── one session, never a pool ───────────────────────────────────────────
    # Pooling a radio's sessions reads the between-session difference as
    # run-to-run spread (e.g. 38 vs 24 dB SNR on 30BF795/89_2400), and the
    # waveform fits come from one session anyway. Use SG_SESSION, else the
    # session on disk under New_captures/, else raise.
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

    # re-measured leakage and precise CFO for this session (LEAK_FILE)
    _skey = (_hit[0] if len(_sess) > 1 else next(iter(_sess)))
    _leak = None if old_leakage else _leak_table(radio, _skey.rsplit("_", 1)[-1], cfg_dir)
    _cfo_rows = [v for v in rows if v.get("cfo_hz") is not None]
    cfo_ppm = np.array([v["cfo_hz"] for v in _cfo_rows], float) / (fc * 1e-6)
    if _leak is not None:
        # only if it covers every run, so two estimators are never mixed
        _by_run = {r: x["cfo_hz"] for r, x in _leak["runs"].items()}
        if all(v.get("run") in _by_run for v in _cfo_rows):
            cfo_ppm = np.array([_by_run[v["run"]] for v in _cfo_rows], float) / (fc * 1e-6)
            print(f"  CFO: precise per-run values for all {len(_cfo_rows)} runs (x^4 bias removed)")
        else:
            print(f"  CFO: precise values cover only {sum(v.get('run') in _by_run for v in _cfo_rows)}"
                  f"/{len(_cfo_rows)} runs -- keeping the log's x^4 CFO for all of them")
    elif not old_leakage:
        print(f"  LEAKAGE: no entry for {radio}/{_skey}/{cfg_dir} in {LEAK_FILE.name} -- "
              "using the log's leakage and x^4 CFO")
    clk_ppm = col("clock_mismatch_ppm")

    if _leak is not None:
        # re-measured levels; no tail rejection, since a gain-89 session's
        # second state sits 10-17 dB below the first
        lk = np.array([x["leak_dbc"] for x in _leak["runs"].values()
                       if x["leak_dbc"] is not None], float)
    else:
        # the log's levels: drop the long low tail of failed estimates
        lk = col("lo_leakage_dbc")
        q1, q3 = np.percentile(lk, [25, 75])
        lk = lk[lk > q1 - 3.0 * max(q3 - q1, 0.5)]

    x = cfo_ppm - cfo_ppm.mean()
    phi = float(np.clip(np.corrcoef(x[:-1], x[1:])[0, 1], 0.0, 0.98)) \
        if x.size > 2 and x.std() > 0 else 0.0

    tp = json.loads(TAPS.read_text()).get(f"{radio}/{cfg_dir}", {})
    # Settling route, resolved once in precedence order. Taps and settling must
    # come from the same fit, so each route checks its pairing and raises if
    # it is missing:
    #   ripple       ripple_fit's taps, PA and fitted settling cubic
    #   measured     measured settling shape + the 'hammerstein' taps/PA fitted
    #                with that shape divided out
    #   cubic        the joint fit's taps, PA and settling cubic
    _mshape = _measured_shape(radio, cfg_dir)
    _rip = _ripple_fir(cfg_dir)
    if _rip is not None:
        if not isinstance(tp.get("ripple_fit"), dict):
            raise SystemExit(
                f"{radio}/{cfg_dir}: a ripple curve is in scope for this config "
                "but no 'ripple_fit' block exists. Injecting it with taps fitted "
                "WITHOUT it double-counts the band shape (+30% residual, better "
                "on 0/40 fits). Run: python analysis_scripts/fit_with_ripple.py "
                f"{cfg_dir}")
        _mshape, _hamm, _htp = None, False, tp["ripple_fit"]
    else:
        _hamm = _mshape is not None and isinstance(tp.get("hammerstein"), dict)
        if _mshape is not None and not _hamm:
            raise SystemExit(
                f"{radio}/{cfg_dir}: a MEASURED settling shape is in scope but no "
                "'hammerstein' fit exists for it. Using j3's jointly-fitted taps "
                "with the measured profile would inject one settling against taps "
                "fitted for another. Run: "
                f"python analysis_scripts/fit_hammerstein.py {radio} {cfg_dir}")
        _htp = tp["hammerstein"] if _hamm else tp
    _pak = "cubic" if tp.get("pa_kind") == "cubic" else "rapp"
    # b comes from the route's own fit, chosen before the gate below
    _bsrc = _htp if (_hamm or _rip is not None) else tp
    # Clean PA refit (default; --no-pa-clean = ripple_fit): PA cubic and taps
    # refitted through the generator's own chain, settling held at ripple_fit's.
    # 2-6 dB closer, with the full gain-89 AM/PM span (refit_pa_clean_fleet.py).
    if pa_clean and not isinstance(tp.get("pa_clean"), dict):
        print(f"  PA: {radio}/{cfg_dir} has no 'pa_clean' block -- using ripple_fit "
              "(analysis_scripts/refit_pa_clean_fleet.py)")
        pa_clean = False
    if pa_clean:
        _bsrc = tp["pa_clean"]
        _htp = dict(_htp, lags=tp["pa_clean"]["lags"], c_inject=tp["pa_clean"]["c_inject"])
    _bre = float(_bsrc.get("pa_b_re", 0.0)); _bim = float(_bsrc.get("pa_b_im", 0.0))
    # Gate the PA on whether the cubic explains anything (residual ratio
    # linear / cubic, PA_GAIN_MIN), not on |b|: below compression b is fit
    # noise. The ratio splits cleanly: only 89_433 and 89_915 pass.
    _cg = (tp.get("pa_cubic_vs_linear", 0.0) / tp["pa_cubic_residual"]
           if tp.get("pa_cubic_residual") else 0.0)
    # Exception, 89_2400 with the clean refit (--gate-pa-2400 = gated): that
    # refit pins a small common cubic on every radio (+1.8 dB on held-out runs).
    _exempt = pa_clean and not pa_gate_2400 and cfg_dir in PA_GATE_EXEMPT
    if _pak == "cubic" and _cg < pa_gain_min and not _exempt:
        _bre = _bim = 0.0
    # ── phase-noise level: per radio only where resolvable ──────────────────
    # (PN_LEVEL_GAIN_MIN); otherwise every radio gets the fleet mean. In
    # practice: per radio at 915 / 2400, fleet mean at 433.
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

    _curve = _pn_curve(radio, cfg_dir)
    if continuous_pn and _curve is None:
        # refuse rather than silently fall back to another model
        raise SystemExit(f"no phase-noise curve for {radio}/{cfg_dir} in "
                         f"{PN_CURVES_FILE.name} (fit_pn_curve.py); "
                         "--per-burst-pn uses the older per-burst mask")
    if _curve is not None and _pn_shared:
        # same rule for the curve: in-burst part at the fleet mean
        _curve = shift_pn_curve(_curve, float(np.mean(
            [_curve_at(c, 1e4) for c in _pn_curve(None, cfg_dir)]))
            - _curve_at(_curve, 1e4))
    _rx_spur = _rx_spur_params(cfg_dir)
    _pa_mod = _pa_mod_params(cfg_dir)
    if rx_spur and _rx_spur is None:
        raise SystemExit(f"--rx-spur requested but {RX_SPUR_FILE.name} has no "
                         f"entry for {cfg_dir}")

    prof = {
        "radio": radio, "config": cfg_dir, "fc_hz": fc, "n_real_runs": len(rows),
        "ref_ppm": float(cfo_ppm.mean()),
        # the clock and CFO estimators sit a fixed distance apart
        "clock_offset_ppm": float(np.median(clk_ppm - cfo_ppm)),
        "cp0_samp": 0.0, "phi0_rad": 0.0,
        "leak_phase_rad": 0.0, "dc_phase_rad": 0.0,
        "iq_amp_db": float(col("iq_amp_db").mean()),
        "iq_phase_deg": float(col("iq_phase_deg").mean()),
        "snr_db": float(col("snr_mean_db").mean()),
        "lo_leak_dbc": float(np.median(lk)),
        # the leakage tone's frequency relative to the CFO (fleet constant per band)
        "lo_leak_offset_hz": _leak["offset_hz"] if _leak is not None else 0.0,
        "leak_source": "measured (lo_leakage.json)" if _leak is not None else "log",
        "rx_dc_frac": float(np.median(col("rx_dc_frac"))),
        # post-PA ISI taps (symbol-spaced), from the route chosen above
        "isi_taps": {"lags": _htp["lags"],
                     "c": [[c[0], c[1]] for c in _htp["c_inject"]]},
        # Burst transient (LO settling): g(t) = 1 + sum c_k t^k, k = 1..3, c
        # complex, fitted on the data portion and folded into the per-run base
        # waveform. The PA is y = x + b x|x|^2 with b complex (real part:
        # compression, imaginary: AM/PM), as separate scalar knobs.
        "settling": [list(v) for v in _htp.get("settling",
                                               tp.get("settling", []))],
        "ripple_fir": (None if _rip is None else
                       [[v.real, v.imag] for v in _rip]),
        # measured preamble part of the transient (None: --cubic-preamble, the
        # measured-shape route, or no file)
        "preamble_transient": (None if (cubic_preamble or _mshape is not None)
                               else _preamble_transient(cfg_dir)),
        "settling_shape": (None if _mshape is None else
                           {"t": _mshape[0].tolist(),
                            "phi_deg": _mshape[1].tolist()}),
        "pa_kind": _pak, "pa_cubic_gain": _cg,
        "pa_b_re": _bre, "pa_b_im": _bim,
        "pa_A": float(tp.get("pa_A", 1e6)),
        # Phase noise: pn_mask is the older per-burst model (--per-burst-pn);
        # pn_tangential_deg and pn_radial_pct_fitted are diagnostics of what
        # the ISI taps leave, not injected (pn_radial_pct only with
        # --radial-filler). pn_level_db shifts whichever model is in use.
        "pn_mask": _pn_mask,
        "pn_level_shared": _pn_shared,
        "pn_level_db": 0.0,
        "pn_slope_db_dec": float(tp.get("pn_slope_db_dec", 0.0)),
        "pn_tangential_deg": float(tp.get("extra_phase_deg", 0.0)),
        "pn_radial_pct": (float(tp.get("extra_radial_pct", 0.0))
                          if radial_filler else 0.0),
        "pn_radial_pct_fitted": float(tp.get("extra_radial_pct", 0.0)),
        # the default model: one continuous curve per run (PN_CURVES_FILE)
        "pn_curve": _curve,
        "pn_continuous": bool(continuous_pn),
        # receiver spur: loaded so --dry-run shows it, applied only if enabled
        "rx_spur": _rx_spur,
        "rx_spur_enabled": bool(rx_spur),
        # gain-89 PA modulation; pa_clean records which PA coefficients are in use
        "pa_mod": _pa_mod,
        "pa_mod_enabled": bool(pa_mod) and _pa_mod is not None,
        # the modulation also swings the ISI taps, and the cubic swing has its
        # measured direction (--old-pa-mod-swing: cubic only, in phase)
        "pa_mod_swing": (bool(pa_mod) and _pa_mod is not None and "tap_swing" in _pa_mod
                         and not old_pa_mod_swing),
        "pa_clean": bool(pa_clean),
        # centre for the case where the clock is not derived from ref_ppm
        "clock_mismatch_ppm": float(clk_ppm.mean()),
    }

    # Burst transient as fleet shape x level (cubic route only, _transient_shape);
    # the radio's own coefficients are kept as settling_fitted.
    _ts = (None if (full_transient or _mshape is not None or not prof["settling"])
           else _transient_shape(cfg_dir, "ripple_fit" if _rip is not None else ""))
    if _ts is not None:
        prof["settling_fitted"] = prof["settling"]
        prof["transient_shape"] = [[c.real, c.imag] for c in _ts]
        prof["transient_level_deg"] = _transient_level(prof["settling"], _ts)
        prof["settling"] = [[prof["transient_level_deg"] * c.real,
                             prof["transient_level_deg"] * c.imag] for c in _ts]

    # Post-PA response as fleet response x shift (_isi_base); the radio's own
    # taps are kept as isi_taps_fitted.
    _src_taps = ("pa_clean" if pa_clean else "ripple_fit" if _rip is not None
                 else "hammerstein" if _hamm else None)
    _ib = (None if raw_taps or list(prof["isi_taps"]["lags"]) != ISI_LAGS
           else _isi_base(cfg_dir, _src_taps))
    if _ib is not None:
        _own = np.array([complex(a, b) for a, b in prof["isi_taps"]["c"]])
        _A = np.asarray(_ib[0])
        # this radio's shift: least squares against its own taps
        _g = np.linspace(-500.0, 500.0, 20001)
        _T = _A[None, :] * np.exp(1j * np.outer(_g, ISI_LAGS) * 2 * np.pi * 1e3 / (FS / SPS))
        _sh = float(_g[np.argmin(np.sum(np.abs(_T - _own[None, :]) ** 2, axis=1))])
        prof["isi_taps_fitted"] = prof["isi_taps"]
        prof["isi_base"] = [[c.real, c.imag] for c in _A]
        prof["isi_shift_khz"] = _sh
        prof["isi_shift_fleet_sd_khz"] = _ib[1]
        prof["isi_taps"] = {"lags": list(ISI_LAGS),
                            "c": [[c.real, c.imag] for c in _isi_taps_from_shift(_A, _sh)]}

    # RECEIVE FILTER: the measured BB60 response where the fits saw captures
    # with its passband divided out (the ripple file records it), else the
    # flat-passband version so the passband the fits absorbed is not applied
    # twice (_flat_passband). --bb60-passband = the measured filter regardless.
    _rd = _load_versioned(RIPPLE_FILE, RIPPLE_FORMAT, key="_format_version") or {}
    prof["rx_fir"] = ("measured" if (bb60_passband or _rd.get("_bb60_equalised"))
                      else "flat-passband")

    var = {k: dict(v) for k, v in DEFAULT_VARIATION.items()}
    if _ts is not None:
        var["transient_level_deg"] = {"kind": "fixed"}
    if _ib is not None:
        var["isi_shift_khz"] = {"kind": "fixed"}
    # gain-89 PA modulation: start phase at the measured fleet value, drawn per run
    if prof["pa_mod_enabled"] and "phase_rad" in _pa_mod and not random_pa_mod_phase:
        prof["pa_mod_phase_rad"] = _pa_mod["phase_rad"]
        var["pa_mod_phase_rad"] = {"kind": "gauss", "sd": _pa_mod["phase_sd_rad"]}
    # oscillator spread from the clock mismatch, the less noisy of the two estimators
    var["ref_ppm"].update(sd=float(clk_ppm.std()), phi=phi)
    var["iq_amp_db"].update(sd=float(col("iq_amp_db").std()))
    var["iq_phase_deg"].update(sd=float(col("iq_phase_deg").std()))
    var["snr_db"].update(sd=float(col("snr_mean_db").std()))
    # leakage level: a mixture where BIC clearly prefers one (gain-89 sessions
    # with two states), else a robust Gaussian (median, IQR sd), within the
    # measured range
    _mix = fit_mixture_bic(lk)
    if _mix is not None:
        var["lo_leak_dbc"] = {"kind": "mixture", **_mix,
                              "lo": float(lk.min()), "hi": float(lk.max())}
    else:
        var["lo_leak_dbc"].update(sd=float((np.percentile(lk, 75)
                                            - np.percentile(lk, 25)) / 1.349),
                                  lo=float(lk.min()), hi=float(lk.max()))
    # sd ready in case clock_mismatch_ppm is switched to its own spec
    var["clock_mismatch_ppm"].setdefault("sd", float(clk_ppm.std()))
    if fingerprint_only:
        _src = ("pa_clean" if pa_clean else "ripple_fit" if _rip is not None
                else "hammerstein" if _hamm else None)
        _fleet_nuisance(prof, var, log, cfg_dir, _src,
                        FP_SNR_DROP_DB if snr_drop_db is None else snr_drop_db)
    return prof, var


@_ft.lru_cache(maxsize=4)
def _cfo_drift_ppm_cached(log_path, leak_path):
    log = json.loads(Path(log_path).read_text())
    fc = {}                                   # radio/token/cfg -> that session's carrier
    for k, v in log.items():
        p = k.split("/")
        _r = re.match(r"TX(.+?)_RX", p[0]) if len(p) == 3 else None
        if _r and v.get("fc_hz"):
            # radio from the session key, as profile_from_log selects rows
            # (some sessions' tx_serial field names the wrong radio)
            fc[f"{_r.group(1)}/{p[0].rsplit('_', 1)[-1]}/{p[1].lstrip('g')}"] = v["fc_hz"]
    d = _load_versioned(LEAK_FILE, LEAK_FORMAT)
    if d is None:
        return None
    by = {}
    for k, runs in d["sessions"].items():
        cf = [x["cfo_hz"] for x in runs.values() if x.get("cfo_hz") is not None]
        if k in fc and len(cf) >= 10:
            by.setdefault(k.rsplit("/", 1)[0], []).append(np.mean(cf) / fc[k] * 1e6)
    sd = [np.std(v, ddof=1) for v in by.values() if len(v) >= 4]
    return float(np.percentile(sd, FP_CFO_DRIFT_PCT)) if len(sd) >= 5 else None


def _cfo_drift_ppm(log):
    """The reference's drift over hours (ppm): the spread of a radio's
    config-mean CFO within one session (the six configs are captured in turn),
    at the fleet's FP_CFO_DRIFT_PCT percentile. From the data folder in use."""
    return _cfo_drift_ppm_cached(str(LOG), str(LEAK_FILE))


def _fleet_nuisance(prof, var, log, cfg_dir, taps_src, snr_drop_db):
    """Fingerprint-only variation (default; --per-radio-nuisance turns it off).

    Device-fixed impairments stay per radio: CFO / clock, TX LO leakage, PA
    cubic, phase-noise level, transient level. The rest are drawn from the
    whole fleet each run, so they cannot identify a radio:
      isi_shift_khz   Gaussian at the fleet spread (it follows the capture
                      day); with --raw-taps, one of the fleet's tap sets
      iq_amp/phase    the fleet's per-run values (mostly run noise)
      rx_dc_frac      the fleet's per-run values (receiver)
      snr_db          uniform from snr_drop_db below the fleet's p10 to its p90
      ref_ppm         spread widened by the hour-scale drift (_cfo_drift_ppm),
                      runs independent
    The pools come from the data folder in use."""
    rows = [v for k, v in log.items()
            if len(k.split("/")) == 3 and k.split("/")[1] == f"g{cfg_dir}"]
    q = np.linspace(0, 100, 101)
    for k_var, k_log in (("iq_amp_db", "iq_amp_db"), ("iq_phase_deg", "iq_phase_deg"),
                         ("rx_dc_frac", "rx_dc_frac")):
        x = np.array([v[k_log] for v in rows if v.get(k_log) is not None], float)
        var[k_var] = {"kind": "quantile", "q": [float(z) for z in np.percentile(x, q)],
                      "source": f"fleet, {len(x)} runs"}
    s = np.array([v["snr_mean_db"] for v in rows if v.get("snr_mean_db") is not None], float)
    var["snr_db"] = {"kind": "uniform", "lo": float(np.percentile(s, 10) - snr_drop_db),
                     "hi": float(np.percentile(s, 90))}
    _drift = _cfo_drift_ppm(log)
    if _drift is not None:
        var["ref_ppm"]["sd"] = float(np.hypot(var["ref_ppm"]["sd"], _drift))
        var["ref_ppm"]["drift_ppm"] = _drift
        # independent runs (at the measured phi a dataset would barely move)
        var["ref_ppm"]["phi"] = 0.0
    if prof.get("isi_base") is not None:
        # shifts are centred on 0 by construction
        var["isi_shift_khz"] = {"kind": "gauss", "mu": 0.0,
                                "sd": prof["isi_shift_fleet_sd_khz"]}
        return
    lags = prof["isi_taps"]["lags"]
    pool, names = [], []
    for k, tp in sorted(json.loads(TAPS.read_text()).items()):
        if not k.endswith(f"/{cfg_dir}"):
            continue
        b = tp.get(taps_src) if taps_src else tp
        if isinstance(b, dict) and b.get("c_inject") is not None and \
                list(b.get("lags", tp.get("lags", []))) == list(lags):
            pool.append([[c[0], c[1]] for c in b["c_inject"]])
            names.append(k.split("/")[0])
    if len(pool) >= 2:
        var["isi_taps"] = {"kind": "pool", "pool": pool, "radios": names}


# ════════════════════════════════════════════════════════════════════════════
# synthesis
# ════════════════════════════════════════════════════════════════════════════
def _pn_amplitude(mask, n, fs):
    """The loop-invariant half of add_phase_noise: the shaped amplitude
    spectrum, computed once per run. Together with _pn_draw it reproduces
    add_phase_noise exactly for the same seed."""
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


def _child_rng(rng):
    """An independent, deterministic Generator derived from rng's current state
    without advancing it, so an optional feature cannot shift the run's other
    draws."""
    import hashlib
    h = hashlib.sha256(repr(rng.bit_generator.state).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:16], "little"))


def _shift_lin(v, k):
    """v[n+k] with zero fill (a linear shift; np.roll would wrap)."""
    out = np.zeros_like(v)
    if k > 0:
        out[:-k] = v[k:]
    elif k < 0:
        out[-k:] = v[:k]
    else:
        out[:] = v
    return out


def _isi_wave(w, taps, lags):
    """Symbol-spaced ISI on the waveform, after the PA:
    out[n] = w[n] + sum_d c_d w[n - d*SPS] (delay convention, as in the fits).
    Zero-filled, not circular."""
    out = w.copy()
    for c, d in zip(taps, lags):
        out = out + c * _shift_lin(w, -int(d) * SPS)
    return out


def _pa(x, pa):
    """Memoryless PA (add_pa_nonlinearity 'cubic' / 'rapp'), normalised on the
    data portion as in the fit."""
    if pa["kind"] not in ("rapp", "cubic"):
        return x
    return add_pa_nonlinearity(x, model=pa["kind"],
                               b=complex(pa.get("b_re", 0.0), pa.get("b_im", 0.0)),
                               A=pa.get("A", 1e6), p=pa.get("p", 3.0),
                               norm_slice=slice(DATA_START, DATA_END))


def synth_run(prof, var, rng, state, n_bursts, data_symbols=None, first_burst=0):
    """One run: draw its parameters, then build the burst train.

    first_burst (optional): generate bursts first_burst ... first_burst+n_bursts-1
    of a longer run, so the time-dependent parts (the gain-89 PA modulation, the
    CFO and leakage phase) are where they would be that far into the run; the
    sampling grid stays relative to the first generated burst. 0 = run start.

    data_symbols (optional): the run's N_DATA QPSK symbols, e.g. from a real
    TX file. The random draw still happens, so every other draw is unchanged.

    Impairment order:

      once per run   SRRC(ZC preamble + random QPSK)  [x settling, cubic route]
      once per run   one phase-noise process across all bursts [not with --per-burst-pn]
      per burst      phase noise (own draw, or the run's slice) -> PA -> ISI
                     -> [settling, measured route]
                     -> [ripple FIR] -> IQ -> TX leakage -> CFO -> RX DC
                     -> timing -> AWGN
      once per run   BB60 receive filter -> additive floor

    Everything after the SRRC is at sample rate. PA before ISI: the PA is in
    the transmitter and the ISI is everything after it (the other order fits
    12x worse). Each settling route sits where its fit put it: the fitted cubic
    before the PA, the measured shape after the ISI. Leakage is added before
    the CFO so it lands at the CFO offset; RX DC after it, at 0 Hz.
    """
    p = {}
    for k, spec in var.items():
        if k in ("isi_taps", "pa_mod_phase_rad"):     # drawn below, from their own streams
            continue
        p[k], state[k] = draw(spec, prof.get(k, 0.0), rng, state.get(k))
    p["cp0_samp"] = (p["cp0_samp"] + 0.5) % 1.0 - 0.5      # cyclic

    cfo_hz = p["ref_ppm"] * 1e-6 * prof["fc_hz"]
    # 'derived' (default): the sampling clock shares the LO's reference;
    # any other kind draws it independently
    if var.get("clock_mismatch_ppm", {}).get("kind", "derived") == "derived":
        mismatch = p["ref_ppm"] + prof["clock_offset_ppm"]
    else:
        mismatch = p["clock_mismatch_ppm"]
    p["clock_mismatch_ppm"] = mismatch     # record what was used
    advance = -mismatch * 1e-6 * BURST_SPACING     # +ppm => falling ramp
    lags = prof["isi_taps"]["lags"]
    taps = [complex(a, b) for a, b in prof["isi_taps"]["c"]]
    _ts = var.get("isi_taps", {})
    if _ts.get("kind") == "gauss" and _ts.get("rel_sd", 0) > 0:
        r = _ts["rel_sd"]
        taps = [c + complex(rng.normal(0, r * abs(c)), rng.normal(0, r * abs(c)))
                for c in taps]
    elif prof.get("isi_base") is not None:
        # fleet response slid by this run's shift (a knob; 0 = the fleet centre)
        taps = list(_isi_taps_from_shift(
            [complex(a, b) for a, b in prof["isi_base"]],
            p.get("isi_shift_khz", prof.get("isi_shift_khz", 0.0))))
    elif _ts.get("kind") == "pool":
        # one fitted tap set per run; the index goes into the ground truth
        p["isi_taps_pool_index"] = int(rng.integers(len(_ts["pool"])))
        taps = [complex(a, b) for a, b in _ts["pool"][p["isi_taps_pool_index"]]]
    pn = {"tangential_deg": p["pn_tangential_deg"],
          "radial_pct": p["pn_radial_pct"]}
    # mask levels are dB, so the knob is an offset
    pn_mask = ([(f, lv + p["pn_level_db"]) for f, lv in prof["pn_mask"]]
               if prof.get("pn_mask") else None)
    dc = p["rx_dc_frac"]

    # the same preamble every burst, as on the real TX; data drawn per run
    pre = PREAMBLE
    data = IDEAL[rng.integers(0, 4, N_DATA)]
    if data_symbols is not None:            # a given sequence; the draw above keeps the stream
        data = np.asarray(data_symbols, complex)
        assert data.shape == (N_DATA,), f"data_symbols must have {N_DATA} symbols"
    # The whole burst grid (preamble, data, null tail), so the ISI later sees
    # each symbol's true neighbours and also filters the preamble. The TX
    # reference stays clean.
    grid = np.concatenate([pre, data, np.zeros(N_TAIL, complex)])
    tx = scipy.signal.upfirdn(SRRC, grid, up=SPS)[MF_DELAY:MF_DELAY + BURST_LEN]
    tx = (tx / np.sqrt(np.mean(np.abs(tx[DATA_START:DATA_END]) ** 2))
          ).astype(np.complex64)

    # Pulse shaping once per run (identical for every burst). Phase noise is
    # applied per burst to the waveform, as an oscillator does.
    base = scipy.signal.upfirdn(SRRC, grid, up=SPS)[MF_DELAY:MF_DELAY + BURST_LEN]
    # Burst transient, applied once per run: the measured phase-only shape
    # where one exists (applied after the ISI, per burst, as fit_hammerstein
    # removed it), else the cubic (folded into `base` before the PA, as fitted,
    # with the measured preamble part spliced in).
    _t = np.arange(BURST_LEN, dtype=float) / BURST_LEN
    _shape = prof.get("settling_shape")
    _set = [complex(a, b_) for a, b_ in prof.get("settling", [])]
    if prof.get("transient_shape") is not None:
        # fleet shape x this run's level (0 = no transient)
        _lv = p.get("transient_level_deg", prof.get("transient_level_deg", 0.0))
        _set = [_lv * complex(a, b_) for a, b_ in prof["transient_shape"]]
    # band ripple, applied per burst after the ISI by linear convolution
    # (np.convolve "same"), exactly the operator its taps were fitted with
    _rip_fir = (np.array([complex(a, b_) for a, b_ in prof["ripple_fir"]],
                         dtype=np.complex128)
                if prof.get("ripple_fir") else None)
    _set_phase = (np.exp(1j * np.deg2rad(np.interp(
        _t, np.asarray(_shape["t"]), np.asarray(_shape["phi_deg"]))))
        if _shape else None)
    if _set_phase is None and _set:
        _g = 1.0 + sum(c * _t ** (m + 1) for m, c in enumerate(_set))
        if prof.get("preamble_transient"):
            _g = _splice_preamble(_g, _t, prof["preamble_transient"])
        base = base * _g
    base = base / np.sqrt(np.mean(np.abs(base[DATA_START:DATA_END]) ** 2))
    _n_sym = N_PRE + N_DATA + N_TAIL
    _x_sym = np.arange(_n_sym) * SPS           # symbol instants in sample index
    _x_samp = np.arange(BURST_LEN)
    _pn_amp = (_pn_amplitude(pn_mask, _n_sym, FS / SPS)
               if pn_mask is not None else None)
    # Continuous phase noise (default): one draw for the whole run at symbol
    # rate, from a child generator so no other draw moves.
    _pn_run = None
    # PA modulation phase at the run's first burst, one per run from a child
    # generator: the measured start phase (variation spec 'pa_mod_phase_rad'),
    # or uniform without it (--random-pa-mod-phase, or no measurement)
    _pa_mod_phase = None
    if prof.get("pa_mod_enabled"):
        _crng = _child_rng(np.random.default_rng(_child_rng(rng).integers(2**63)))
        if var.get("pa_mod_phase_rad"):
            _pa_mod_phase = draw(var["pa_mod_phase_rad"], prof["pa_mod_phase_rad"], _crng)[0]
        else:
            _pa_mod_phase = float(_crng.uniform(0, 2 * np.pi))
        p["pa_mod_phase_rad"] = _pa_mod_phase
    if prof.get("pn_continuous"):
        assert BURST_SPACING % SPS == 0, "burst slots must sit on the symbol grid"
        _slot = BURST_SPACING // SPS
        _n_run = (n_bursts - 1) * _slot + _n_sym
        _full = [(f, lv + p["pn_level_db"]) for f, lv in prof["pn_curve"]]
        _pn_run = _pn_draw(_pn_amplitude(_full, _n_run, FS / SPS), _n_run,
                           _child_rng(rng))
    # White noise through the receive filter keeps sum(h^2) of its power; the
    # AWGN is scaled by it so the in-band SNR lands on target after filtering.
    _RX_RETAIN = 0.8734                      # sum(h^2) of RX_FIR
    _rx_fir = RX_FIR
    if prof.get("rx_fir") == "flat-passband":
        _rx_fir = RX_FIR_FLAT
        _RX_RETAIN = float(np.sum(RX_FIR_FLAT ** 2))

    rx = np.zeros(n_bursts * BURST_SPACING, dtype=np.complex64)
    # transmit-envelope gate for the post-filter floor, from the waveform
    # before AWGN so it reaches ~0 where the TX is idle
    gate = np.zeros(n_bursts * BURST_SPACING, dtype=np.float32)
    for j in range(n_bursts):
        b = base
        # phase noise at symbol rate, interpolated up to sample rate (the curve
        # is defined up to the symbol-rate Nyquist)
        if _pn_run is not None:
            # this burst's slice of the run-long process, by its nominal slot
            ph = _pn_run[j * _slot:j * _slot + _n_sym]
            b = b * np.exp(1j * np.interp(_x_samp, _x_sym, ph))
            # advance the main stream as the per-burst draw would (and discard
            # it), so the two phase-noise models differ only in phase noise
            if _pn_amp is not None:
                _pn_draw(_pn_amp, _n_sym, rng)
        elif pn_mask is not None:
            ph = _pn_draw(_pn_amp, _n_sym, rng)
            b = b * np.exp(1j * np.interp(_x_samp, _x_sym, ph))
        elif pn["tangential_deg"] > 0:          # fallback: no mask fitted yet
            ph = np.deg2rad(rng.normal(0, pn["tangential_deg"], _n_sym))
            b = b * np.exp(1j * np.interp(_x_samp, _x_sym, ph))
        # radial (AM) filler: off unless --radial-filler (0 in every profile)
        if pn["radial_pct"] > 0:
            am = rng.normal(0, pn["radial_pct"] / 100.0, _n_sym)
            b = b * (1 + np.interp(_x_samp, _x_sym, am))
        b = b / np.sqrt(np.mean(np.abs(b[DATA_START:DATA_END]) ** 2))
        # PA, then ISI. This burst's PA-modulation phasor, shared by both parts:
        _rot = (np.exp(1j * (2 * np.pi * prof["pa_mod"]["freq_hz"] * (j + first_burst) * BURST_SPACING / FS
                             + _pa_mod_phase)) if _pa_mod_phase is not None else None)
        _pa_par = {"kind": prof["pa_kind"], "b_re": p["pa_b_re"], "b_im": p["pa_b_im"],
                   "A": prof.get("pa_A", 1e6), "p": 3.0}
        _pa_in = b
        b = _pa(_pa_in, _pa_par)
        if _rot is not None and "b" in _PA_MOD_MODE:
            # the cubic modulated per burst, on the data portion only (the real
            # preamble barely carries it), contributing only its shape: its mean
            # complex gain is projected out, since the gain step below already
            # carries the whole measured line
            _db = (complex(*prof["pa_mod"]["b_swing"]) if prof.get("pa_mod_swing")
                   else prof["pa_mod"]["b_amp"]) * _rot
            _bm = _pa(_pa_in, dict(_pa_par, b_re=p["pa_b_re"] + _db.real,
                                   b_im=p["pa_b_im"] + _db.imag))[DATA_START:]
            b = b.copy()
            b[DATA_START:] = _bm * (np.vdot(_bm, b[DATA_START:]) / np.vdot(_bm, _bm))
        if _rot is not None and prof.get("pa_mod_swing") and "taps" in _PA_MOD_MODE:
            # the taps swing with the modulation (data portion, like the cubic):
            # taps_j = taps + c * r_j, c measured per lag
            _sw = dict(zip(prof["pa_mod"]["tap_swing_lags"], prof["pa_mod"]["tap_swing"]))
            _tj = [c + complex(*_sw.get(int(d), (0.0, 0.0))) * _rot for c, d in zip(taps, lags)]
            _bj = _isi_wave(b, _tj, lags)
            b = _isi_wave(b, taps, lags)
            b[DATA_START:] = _bj[DATA_START:]
        else:
            b = _isi_wave(b, taps, lags)
        # measured settling: after the ISI, as fit_hammerstein removed it
        if _set_phase is not None:
            b = b * _set_phase
        if _rip_fir is not None:
            b = np.convolve(b, _rip_fir, mode="same")
        b = b / np.sqrt(np.mean(np.abs(b[DATA_START:DATA_END]) ** 2))
        # PA gain modulation, after the normalisation (which would remove its
        # magnitude): a step in the data portion's complex gain at DATA_START
        if _rot is not None and "data" in _PA_MOD_MODE:
            b = b.copy()
            b[DATA_START:] = b[DATA_START:] * (1.0 + prof["pa_mod"]["amp"] * _rot)
        b = add_iq_imbalance(b, IQ_SIGN * p["iq_amp_db"], p["iq_phase_deg"])
        # TX LO leakage: a tone lo_leak_offset_hz from the CFO (0 with
        # --old-leakage), phase continuous across bursts
        b = b + 10 ** (p["lo_leak_dbc"] / 20.0) * np.exp(1j * (
            p["leak_phase_rad"] + 2 * np.pi * prof.get("lo_leak_offset_hz", 0.0) / FS
            * (np.arange(len(b)) + (j + first_burst) * BURST_SPACING)))
        # one CFO per run, phase continuous across bursts; no per-burst dither
        # (the measured per-burst scatter is mostly the estimator's floor, and
        # the slow LO wander comes from the continuous phase noise)
        b = add_cfo(b, cfo_hz / FS,
                    phase_offset=p["phi0_rad"] + 2 * np.pi * cfo_hz / FS
                    * ((j + first_burst) * BURST_SPACING))
        b = b + dc * np.exp(1j * p["dc_phase_rad"])
        # Accumulated clock offset: the integer part moves the burst in the
        # capture (burst positions drift with the clock, as on real captures),
        # the fractional part is the symbol clock phase.
        tot = p["cp0_samp"] + advance * j
        frac = (tot + 0.5) % 1.0 - 0.5
        shift = int(round(tot - frac))
        b = add_symbol_clock_phase(b, frac, method="fir", N_taps=25)
        # AWGN last. Its budget is what remains after the leakage and DC (which
        # also sit in the null tail, where SNR is estimated), referenced to the
        # data-portion power, and raised by 1/sum(h^2) for the receive filter.
        tot = 10 ** (-p["snr_db"] / 10.0) / _RX_RETAIN
        eff = tot - 10 ** (p["lo_leak_dbc"] / 10.0) - dc ** 2
        ratio = np.mean(np.abs(b) ** 2) / np.mean(
            np.abs(b[DATA_START:DATA_END]) ** 2)
        # gate from the clean envelope, smoothed over ~1 symbol
        _e = np.convolve(np.abs(b) ** 2, np.ones(SPS) / SPS, mode="same")
        b = add_awgn_snr(b, -10 * np.log10(max(eff, tot * 1e-3))
                         + 10 * np.log10(ratio), db=True, rng=rng)
        pos = j * BURST_SPACING + shift
        rx[pos:pos + BURST_LEN] = b.astype(np.complex64)
        gate[pos:pos + BURST_LEN] = np.sqrt(
            _e / max(float(np.mean(_e[DATA_START:DATA_END])), 1e-30))

    # Receiver: the spur (opt-in) acts on everything reaching the receiver,
    # indexed from the capture start; then the receive filter, once over the
    # whole stream as the BB60 filters continuously.
    if prof.get("rx_spur_enabled"):
        rx = add_rx_spur(rx, prof["rx_spur"]["amps_rad"], prof["rx_spur"]["phases_rad"])
    rx = add_decimation_filter(rx, FS, fir=_rx_fir).astype(np.complex64)
    # Post-filter floor, flat in frequency, relative to the in-band noise, and
    # gated by the transmit envelope: real captures show it only while
    # transmitting, and it is incoherent (it averages down across bursts).
    # Mechanism not established (receiver nonlinearity or ADC products fit).
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
    # as the real BB60 writer: one annotation of its chunk size
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
          # hand-built profiles have no n_real_runs
        f"({str(prof['n_real_runs']) + ' real runs' if 'n_real_runs' in prof else 'hand-built'}"
        f", fc {prof['fc_hz']/1e6:.2f} MHz)")
    print(f"  {'parameter':<18}{'value':>14}   variation")
    for k in ["ref_ppm", "clock_mismatch_ppm", "iq_amp_db", "iq_phase_deg",
              "snr_db", "lo_leak_dbc", "cp0_samp", "phi0_rad", "rx_dc_frac",
              "pa_b_re", "pa_b_im", "pn_level_db", "pn_radial_pct"]:
        s = var.get(k, {"kind": "fixed"})
        # if/elif, not a dict of f-strings: those would all be evaluated (and
        # fail on a mixture's list-valued sd)
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
        elif _k == "quantile":
            d = (f"empirical, p5..p95 [{s['q'][5]:.4g}, {s['q'][95]:.4g}]"
                 f"  ({s.get('source', '')})")
        else:
            d = _k
        if k == "lo_leak_dbc":
            d += (f"   [{prof.get('leak_source', 'log')}, tone at CFO "
                  f"{prof.get('lo_leak_offset_hz', 0.0):+.2f} Hz]")
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
    _tv = var.get("isi_taps", {})
    if prof.get("isi_base") is not None:
        _sv = var.get("isi_shift_khz", {"kind": "fixed"})
        print(f"  {'isi_shift_khz':<18}{prof['isi_shift_khz']:>14.4g}   "
              + (f"Gaussian sd={_sv['sd']:.4g} about {_sv.get('mu', prof['isi_shift_khz']):.4g} (fleet)"
                 if _sv["kind"] == "gauss" else "held at the profile value")
              + "   post-PA response: fleet ripple slid by this; at 0 kHz:")
    elif _tv.get("kind") == "pool":
        print(f"  {'isi_taps':<18}{'':>14}   POOLED: one of {len(_tv['pool'])} radios' "
              f"fitted sets per run (fingerprint-only); this radio's own:")
    else:
        print(f"  {'isi_taps':<18}{'':>14}   fixed, {len(t['lags'])} complex taps")
    for k, c in zip(t["lags"], prof["isi_base"] if prof.get("isi_base") is not None else t["c"]):
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
    elif prof.get("transient_shape") is not None:
        print(f"  {'settling':<18}{prof['transient_level_deg']:>14.4g}   deg rms (transient_level_deg) "
              f"x fleet cubic shape on the data portion; preamble "
              + ("MEASURED profile (srrc_preamble_transient.json)"
                 if prof.get("preamble_transient")
                 else "EXTRAPOLATED cubic (--cubic-preamble)"))
    elif prof.get("settling"):
        _s = [complex(a, b_) for a, b_ in prof["settling"]]
        print(f"  {'settling':<18}{'':>14}   fitted complex cubic, |profile| "
              f"span {abs(sum(_s)):.4f}  on the data portion; preamble "
              + ("MEASURED profile (srrc_preamble_transient.json)"
                 if prof.get("preamble_transient")
                 else "EXTRAPOLATED cubic (--cubic-preamble)"))
    if prof.get("pa_mod"):
        print(f"  {'pa gain mod':<18}{'':>14}   {100 * prof['pa_mod']['amp']:.2f} % at "
              f"{prof['pa_mod']['freq_hz']:+.2f} Hz (burst rate), fleet   "
              + ("INJECTED" if prof.get("pa_mod_enabled") else "off (--no-pa-mod)"))
        if prof.get("pa_mod_enabled"):
            _ps = var.get("pa_mod_phase_rad")
            print(f"  {'pa mod phase':<18}{'':>14}   at the first burst: "
                  + (f"{np.degrees(prof['pa_mod_phase_rad']):+.1f} deg, Gaussian sd "
                     f"{np.degrees(_ps['sd']):.1f} deg (measured)" if _ps
                     else "uniform (--random-pa-mod-phase, or not measured)"))
            print(f"  {'pa mod swing':<18}{'':>14}   "
                  + ("cubic and ISI taps (measured, data portion)" if prof.get("pa_mod_swing")
                     else "cubic only, in phase (--old-pa-mod-swing, or not measured)"))
    if prof.get("pa_clean"):
        print(f"  {'pa coefficients':<18}{'':>14}   CLEAN refit (pa_clean block), "
              f"b {prof['pa_b_re']:+.4f}{prof['pa_b_im']:+.4f}j")
    else:
        print(f"  {'pa coefficients':<18}{'':>14}   ripple_fit (joint least squares)"
              f" -- --no-pa-clean, or no pa_clean block")
    print(f"  {'rx filter':<18}{'':>14}   BB60 anti-alias, "
          + ("measured response" if prof.get("rx_fir", "measured") == "measured"
             else "passband flattened (the fits already contain the measured passband)"))
    if prof.get("rx_spur"):
        _a = np.degrees(prof["rx_spur"]["amps_rad"])
        print(f"  {'rx spur':<18}{'':>14}   receiver PM, 1168-sample period "
              f"(4280.8 Hz): {_a[0]:.3f} / {_a[1]:.3f} deg peak (fund. / 2nd), "
              f"fleet   " + ("INJECTED" if prof.get("rx_spur_enabled")
                             else "off (enable with --rx-spur)"))
    if prof.get("pn_mask") and not prof.get("pn_continuous"):
        print(f"  {'pn':<18}{'':>14}   1/f^a mask, per burst, slope "
              f"{prof.get('pn_slope_db_dec', 0):+.2f} dB/dec, level "
              f"{prof['pn_mask'][2][1]:.1f} dB @10 kHz"
              + ("   [FLEET MEAN -- per-radio level not resolvable]"
                 if prof.get("pn_level_shared") else "   [per-radio]"))
    if prof.get("pn_curve"):
        print(f"  {'pn curve':<18}{'':>14}   S_phi dB(rad^2/Hz), slow part fleet, "
              f"in-burst part "
              + ("FLEET MEAN" if prof.get("pn_level_shared") else "this radio")
              + ":   " + ("USED, one continuous draw per run"
                          if prof.get("pn_continuous")
                          else "not used (--per-burst-pn)"))
        _c = prof["pn_curve"]
        for i in range(0, len(_c), 7):
            print(f"  {'':<18}{'':>14}   " + "  ".join(
                f"{f:g}Hz {lv:.1f}" for f, lv in _c[i:i + 7]))
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
    ap.add_argument("--per-burst-pn", action="store_true",
                    help="the PREVIOUS phase-noise model: the in-burst 1/f^a mask "
                         "drawn independently per burst (no burst-to-burst LO "
                         "wander). Reproduces datasets made before 2026-09-27 bit "
                         "for bit. Default: one curve, one continuous draw per run")
    ap.add_argument("--cubic-preamble", action="store_true",
                    help="the PREVIOUS burst transient in the preamble: the data-"
                         "portion cubic extrapolated backwards (1.4-1.6x the real "
                         "preamble offset). Default: the measured preamble profile")
    # accepted for old command lines; continuous is the default now
    ap.add_argument("--continuous-pn", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-pa-clean", dest="pa_clean", action="store_false", default=True,
                    help="use ripple_fit's PA cubic + taps (the joint least squares) "
                         "instead of the clean refit (pa_clean block, fitted through the "
                         "generator's own chain; ON by default since 2026-09-30). "
                         "Reproduces earlier datasets")
    # accepted for old command lines; the clean refit is the default now
    ap.add_argument("--pa-clean", dest="pa_clean", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-pa-mod", dest="pa_mod", action="store_false", default=True,
                    help="leave out the gain-89 PA modulation (fleet-common data-portion "
                         "gain + cubic modulation at -2.15 / -2.38 Hz, 433 / 915 MHz; ON by "
                         "default since 2026-09-30). Reproduces earlier datasets")
    # accepted for old command lines; the modulation is on by default now
    ap.add_argument("--pa-mod", dest="pa_mod", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--old-leakage", action="store_true",
                    help="the PREVIOUS TX LO leakage and CFO: the characterisation log's "
                         "values (leakage read +-5 Hz around the biased x^4 CFO, as a "
                         "fitted mixture; x^4 CFO). Default since 2026-10-01: the level's "
                         "distribution fitted to the re-measured runs, the tone at the CFO "
                         "plus the band offset, and the precise CFO (data/lo_leakage.json)")
    ap.add_argument("--rx-spur", action="store_true",
                    help="add the BB60 receiver's phase spur (1168-sample period, "
                         "phase fixed to the capture start) as real captures carry "
                         "it (off by default; the fitted parameters exclude it)")
    ap.add_argument("--per-radio-nuisance", action="store_true",
                    help="the PREVIOUS variation spec: ISI taps, IQ imbalance, RX DC and "
                         "SNR per radio at the measured spread, CFO at its measured spread. "
                         "Default since 2026-10-03: fingerprint-only -- those drawn from "
                         "the fleet, SNR extended FP_SNR_DROP_DB lower, CFO spread "
                         "widened by the drift measured over a session")
    ap.add_argument("--snr-drop-db", type=float, default=None,
                    help=f"fingerprint-only: how far below the fleet's p10 SNR the "
                         f"per-run draw reaches (default {FP_SNR_DROP_DB:g} dB)")
    ap.add_argument("--gate-pa-2400", action="store_true",
                    help="the PREVIOUS 89_2400 PA: cubic gated off. Default since "
                         "2026-10-03: the clean refit's cubic is injected there")
    ap.add_argument("--old-pa-mod-swing", action="store_true",
                    help="the PREVIOUS gain-89 PA modulation: only the cubic swings, in phase with "
                         "the gain step. Default since 2026-10-04: the ISI taps swing too, and the "
                         "cubic at its measured direction (pa_gain_mod.json tap_swing)")
    ap.add_argument("--random-pa-mod-phase", action="store_true",
                    help="the PREVIOUS gain-89 PA modulation phase: uniform per run. Default "
                         "since 2026-10-04: the measured start phase (pa_gain_mod.json)")
    ap.add_argument("--bb60-passband", action="store_true",
                    help="the PREVIOUS receive filter: the measured BB60 response including its "
                         "passband rise, on fits that already contain it. Default since "
                         "2026-10-04: its flat-passband version on such fits, the measured one "
                         "on fits made with the passband divided out")
    ap.add_argument("--raw-taps", action="store_true",
                    help="the PREVIOUS post-PA linear block: each radio's own two fitted "
                         "taps (and, fingerprint-only, one of the fleet's tap sets per run). "
                         "Default since 2026-10-04: one fleet response per config slid in "
                         "frequency by isi_shift_khz")
    ap.add_argument("--full-transient", action="store_true",
                    help="the PREVIOUS burst transient: each radio's own complex cubic "
                         "(6 numbers). Default since 2026-10-04: one fleet shape per "
                         "config x the radio's level in degrees (transient_level_deg)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the profile and variation spec, write nothing")
    a = ap.parse_args(argv)

    prof, var = profile_from_log(a.radio, a.config, a.radial_filler,
                                 continuous_pn=not a.per_burst_pn, rx_spur=a.rx_spur,
                                 cubic_preamble=a.cubic_preamble,
                                 pa_clean=a.pa_clean, pa_mod=a.pa_mod,
                                 old_leakage=a.old_leakage,
                                 fingerprint_only=not a.per_radio_nuisance,
                                 pa_gate_2400=a.gate_pa_2400, snr_drop_db=a.snr_drop_db,
                                 full_transient=a.full_transient, raw_taps=a.raw_taps,
                                 bb60_passband=a.bb60_passband,
                                 random_pa_mod_phase=a.random_pa_mod_phase,
                                 old_pa_mod_swing=a.old_pa_mod_swing)
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
