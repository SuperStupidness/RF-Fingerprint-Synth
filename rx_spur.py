"""Receiver phase spur: estimate, remove, add.

Every SRRC capture in this campaign carries a small phase modulation that
repeats every 1168 samples at 5 MS/s (4280.82 Hz), plus its 2nd harmonic. It
belongs to the RECEIVER, not the radios:

  - the same frequency on every radio, config and run (4280.8 +- 0.1 Hz);
  - its phase is FIXED relative to the capture start (+165..+176 deg over 107
    runs, 6 radios, several days) while the transmitted bursts land at random
    positions -- referenced to the bursts instead, the phase is random;
  - pure phase modulation, scaling with the carrier (0.25 deg peak at 433 MHz,
    0.5 at 915, 1.0-1.5 at 2400), so it rides on the BB60's oscillator.

It is not a device property, so the generator's measured parameters are fitted
on captures with it REMOVED (remove_rx_spur), and the generator does not
produce it unless asked (--rx-spur, parameters in data/rx_spur.json).

The phase added at capture sample n (counted from the start of the capture) is

    phi(n) = sum_h  a_h * sin(2 pi h n / RX_SPUR_PERIOD + theta_h)

with a_h in radians. Needs numpy and scipy only.
"""
from __future__ import annotations

import numpy as np
import scipy.signal

RX_SPUR_PERIOD = 1168          # samples at 5 MS/s
RX_SPUR_HARMONICS = (1, 2)


def rx_spur_phase(n, amps, phases, period=RX_SPUR_PERIOD, harmonics=RX_SPUR_HARMONICS):
    """phi(n) in radians for capture sample indices n."""
    n = np.asarray(n, dtype=np.float64)
    return sum(a * np.sin(2 * np.pi * h * n / period + th)
               for h, a, th in zip(harmonics, amps, phases))


def add_rx_spur(x, amps, phases, start=0):
    """Apply the spur to a capture whose first sample is capture sample `start`."""
    n = start + np.arange(len(x))
    return x * np.exp(1j * rx_spur_phase(n, amps, phases))


def remove_rx_spur(x, amps, phases, start=0):
    """Undo add_rx_spur."""
    n = start + np.arange(len(x))
    return x * np.exp(-1j * rx_spur_phase(n, amps, phases))


def _x4_cfo(seg, fs, zero_pad=16):
    s = seg.astype(np.complex128) ** 4
    N = 2 ** int(np.ceil(np.log2(len(s)))) * zero_pad
    mag = np.abs(np.fft.fft(s, n=N))
    k = int(np.argmax(mag))
    a, b, c = (np.log(mag[(k + j) % N] + 1e-300) for j in (-1, 0, 1))
    den = a - 2 * b + c
    d = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den != 0 else 0.0
    return (np.fft.fftfreq(N, 1 / fs)[k] + d * fs / N) / 4


def _starts(rx, template, burst_len):
    c = np.abs(scipy.signal.correlate(rx, template, mode='full'))[len(template) - 1:]
    st, _ = scipy.signal.find_peaks(c, height=c.max() * 0.5, distance=int(0.8 * burst_len))
    return st[st + burst_len <= len(rx)]


def _deramp_rows(A):
    n = A.shape[-1]
    X = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
    c, *_ = np.linalg.lstsq(X, A.reshape(-1, n).T, rcond=None)
    return (A.reshape(-1, n) - (X @ c).T).reshape(A.shape)


def estimate_rx_spur(rx, tx_burst, fs=5e6, pre_len=500, data=slice(500, 5500),
                     n_bursts=200):
    """Estimate (amps, phases) of the receiver spur from one real capture.

    Data-aided and self-referencing: one constant CFO, each burst's preamble
    phase removed, then each burst's phase RELATIVE TO THE BURST AVERAGE on the
    data portion. Everything deterministic (ISI, PA, settling) is common to all
    bursts and cancels; the spur does not, because consecutive bursts sit at
    different positions within its 1168-sample period. A least-squares fit of
    sin/cos at each harmonic against CAPTURE sample index, with a per-burst
    mean and slope as nuisance terms, gives the amplitudes and phases.

    Returns amps (radians, peak), phases (radians), and a dict of diagnostics.
    """
    L = len(tx_burst)
    rx = np.asarray(rx, dtype=np.complex128)
    st = _starts(rx, tx_burst[:pre_len], L)
    f0 = np.median([_x4_cfo(rx[s:s + L][data], fs) for s in st[::5]])
    n_all = np.arange(len(rx))
    xc = rx * np.exp(-2j * np.pi * f0 * n_all / fs)
    st = _starts(xc, tx_burst[:pre_len], L)[:n_bursts]
    B = np.array([xc[s:s + L] for s in st])
    B *= np.exp(-1j * np.angle(np.sum(B[:, :pre_len] * np.conj(tx_burst[:pre_len]), axis=1)))[:, None]
    avg = B.mean(axis=0)
    w = np.abs(avg[data])
    keep = w > 0.3 * w.mean()                   # skip near-zero samples of the average
    psi = np.angle(B[:, data] * np.conj(avg[data]))[:, keep]
    idx = (st[:, None] + np.arange(L)[None, data][:, keep])      # capture sample index
    cols = []
    for h in RX_SPUR_HARMONICS:
        arg = 2 * np.pi * h * idx / RX_SPUR_PERIOD
        cols += [np.sin(arg), np.cos(arg)]
    X = np.stack([_deramp_rows(c) for c in cols], axis=-1).reshape(-1, len(cols))
    y = _deramp_rows(psi).ravel()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    amps, phases = [], []
    for i in range(len(RX_SPUR_HARMONICS)):
        s_, c_ = coef[2 * i], coef[2 * i + 1]          # a sin(x+th) = a cos th sin x + a sin th cos x
        amps.append(float(np.hypot(s_, c_)))
        phases.append(float(np.arctan2(c_, s_)))
    resid = y - X @ coef
    return amps, phases, dict(n_bursts=len(st), cfo_hz=float(f0),
                              resid_rms_deg=float(np.degrees(resid.std())))
