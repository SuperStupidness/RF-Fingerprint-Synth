#!/usr/bin/env python3
"""Round trip for the phase-noise curves (data/pn_curves.json).

Generates runs with the default model (one curve, one continuous draw per run)
and, for comparison, the previous per-burst model (--per-burst-pn), measures
both views the curves were fitted to, and compares with the real targets
stored in the data file -- so no raw captures are needed.

    python analysis_scripts/continuous_pn_roundtrip.py              # all configs
    python analysis_scripts/continuous_pn_roundtrip.py 89_433 --seeds 2

PATH view (fleet): per-burst preamble phase angle(sum rx*conj(tx)) after ONE
constant (median x^4) CFO, unwrapped, quadratic removed, Welch nperseg=256 at
the burst rate; half-octave band means as L(f) = S_phi/2 in dBc/Hz, and the
wander ratio. USE ~20 RUNS PER CONFIG: each run is ONE draw of a ~1 s process
whose slowest components barely complete a cycle, so with 5-6 runs the low
bands scatter by +-4 dB and the wander ratio by +-0.3.

IN-BURST view (per radio): the tangential symbol deviation as fit_isi_taps.py
builds it (per-burst x^4 CFO + preamble phase, matched filter, clock phase,
cluster centroid, across-burst mean, per-burst mean + ramp removed), Welch with
one burst per segment, minus the radial spectrum; L(f) at the stored
frequencies and the ratio of the total PN rms to real.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.signal

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import synth_dataset as sd                                     # noqa: E402

PRE = sd.N_PRE * sd.SPS
DS = slice(sd.DATA_START, sd.DATA_END)
L = sd.BURST_LEN
NB_IN = 40                            # bursts per in-burst measurement (as fit_isi_taps)
IDEAL = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)


def _cfo_x4(x, fs, zero_pad=16):
    s = x.astype(np.complex128) ** 4
    N = 2 ** int(np.ceil(np.log2(len(s)))) * zero_pad
    mag = np.abs(np.fft.fft(s, n=N))
    k = int(np.argmax(mag))
    a, b, c = (np.log(mag[(k + j) % N] + 1e-300) for j in (-1, 0, 1))
    den = a - 2 * b + c
    d = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den != 0 else 0.0
    return (np.fft.fftfreq(N, 1 / fs)[k] + d * fs / N) / 4


def _starts(rx, tmpl):
    c = np.abs(scipy.signal.correlate(rx, tmpl, mode='full'))[len(tmpl) - 1:]
    st, _ = scipy.signal.find_peaks(c, height=c.max() * 0.5, distance=int(0.8 * L))
    return st[st + L <= len(rx)]


def _coarse(tx, rx):
    """One constant CFO removed, bursts re-detected (ZC CFO-timing coupling)."""
    st = _starts(rx, tx[:PRE])
    f0 = np.median([_cfo_x4(rx[s:s + L][DS], sd.FS) for s in st[::10]])
    xc = rx * np.exp(-2j * np.pi * f0 * np.arange(len(rx)) / sd.FS)
    return _starts(xc, tx[:PRE]), xc


def phase_path(tx, rx):
    st, xc = _coarse(tx, rx)
    t = (st + PRE / 2) / sd.FS
    ph = np.unwrap([np.angle(np.vdot(tx[:PRE], xc[s:s + PRE])) for s in st])
    tt = t - t[0]
    w = ph - np.polyval(np.polyfit(tt, ph, 2), tt)
    f, S = scipy.signal.welch(w, fs=1 / np.mean(np.diff(t)), nperseg=256, detrend='linear')
    return f, S, w


def _deramp(A):
    x = np.arange(A.shape[1], dtype=float)
    X = np.column_stack([x, np.ones_like(x)])
    c, *_ = np.linalg.lstsq(X, A.T, rcond=None)
    return A - (X @ c).T


def inburst(tx, rx):
    """Mean tangential and radial Welch spectra, and their variance gap."""
    st, xc = _coarse(tx, rx)
    n = np.arange(L)
    nv = np.arange(sd.N_DATA * sd.SPS, dtype=float)
    w = np.exp(-2j * np.pi * nv / sd.SPS)
    base = np.arange(sd.N_DATA, dtype=float) * sd.SPS
    d = (len(sd.SRRC) - 1) // 2
    ch = []
    for s in st[:NB_IN]:
        f = xc[s:s + L]
        f = f * np.exp(-2j * np.pi * _cfo_x4(f[DS], sd.FS) * n / sd.FS)
        f = f * np.exp(-1j * np.angle(np.vdot(tx[:PRE], f[:PRE])))
        mfd = np.convolve(f, sd.SRRC)[d:d + L][DS]
        cp = -np.angle(np.dot(np.abs(mfd) ** 2, w)) * sd.SPS / (2 * np.pi)
        sy = np.interp(base + cp, nv, mfd.real) + 1j * np.interp(base + cp, nv, mfd.imag)
        ch.append(sy / np.sqrt(np.mean(np.abs(sy) ** 2)))
    sn = np.concatenate(ch)
    lab = np.argmin(np.abs(sn[:, None] - IDEAL[None, :]), axis=1)
    ctr = np.array([sn[lab == k].mean() for k in range(4)])[lab]
    dv = ((sn - ctr) * np.conj(ctr / np.abs(ctr))).reshape(len(ch), -1)
    dv = dv - dv.mean(axis=0)
    am = np.maximum(np.abs(sn).reshape(len(ch), -1), 1e-12)
    t, r = _deramp(dv.imag / am), _deramp(dv.real / am)
    fq, Pt = scipy.signal.welch(t, fs=sd.FS / sd.SPS, nperseg=sd.N_DATA, axis=1)
    _, Pr = scipy.signal.welch(r, fs=sd.FS / sd.SPS, nperseg=sd.N_DATA, axis=1)
    return fq, Pt.mean(0) - Pr.mean(0), float(t.var() - r.var())


def band_db(f, S, q):
    q = max(q, 3.0)
    m = (f >= q / 2 ** 0.5) & (f <= q * 2 ** 0.5) & (f > 0)
    return 10 * np.log10(S[m].mean() / 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('configs', nargs='*')
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--bursts', type=int, default=762)
    a = ap.parse_args(argv)

    doc = json.loads(sd.PN_CURVES_FILE.read_text())
    configs = a.configs or list(doc['path_targets'])
    sessions = doc['path_sources']['sessions']
    for c in configs:
        tgt = doc['path_targets'][c]
        bands = [float(k) for k in tgt['L_dBc_Hz'] if float(k) >= 3.0]
        print(f'\n{c}')
        print('  PATH (measured - real) dB at ' + ' '.join(f'{q:g}' for q in bands)
              + ' Hz | wander ratio')
        inb = {}
        for cont in (False, True):
            res = []
            for radio, sess in sessions.items():
                os.environ['SG_SESSION'] = sess.split('_')[-1]
                try:
                    prof, var = sd.profile_from_log(radio, c, continuous_pn=cont)
                except SystemExit:               # generator guard, or no curve
                    continue
                for seed in range(a.seeds):
                    tx, rx, _ = sd.synth_run(prof, var, np.random.default_rng(1000 + seed),
                                             {}, a.bursts)
                    tx = np.asarray(tx, complex)[:L]
                    rx = np.asarray(rx, complex)
                    res.append(phase_path(tx, rx))
                    inb.setdefault((radio, cont), []).append(inburst(tx, rx))
            if not res:
                print('    no radio could be generated')
                break
            f = res[0][0]
            S = np.mean([r[1] for r in res], axis=0)
            d = [band_db(f, S, q) - tgt['L_dBc_Hz'][f'{q:g}'] for q in bands]
            wr = np.degrees(np.mean([r[2].std() for r in res])) / tgt['wander_rms_deg']
            print(f'    {"curve (default)" if cont else "per-burst (old)":16} '
                  + ' '.join(f'{x:+6.1f}' for x in d) + f' | {wr:5.2f}   ({len(res)} runs)',
                  flush=True)
        radios = sorted({r for r, _ in inb})
        if not radios:
            continue
        fs_ = doc['curves'][f'{radios[0]}/{c}'].get('inburst_target', {}).get('f_hz', [])
        print('  IN-BURST (measured - real) dB at ' + ' '.join(f'{q/1e3:g}k' for q in fs_)
              + ' | PN rms ratio')
        for radio in radios:
            e = doc['curves'].get(f'{radio}/{c}', {})
            it = e.get('inburst_target')
            if not it:
                print(f'    {radio}: no stored in-burst target ({e.get("source", "no curve")})')
                continue
            for cont in (False, True):
                v = inb.get((radio, cont))
                if not v:
                    continue
                fq = v[0][0]
                ex = np.mean([x[1] for x in v], axis=0)
                d = [10 * np.log10(max(ex[int(np.argmin(np.abs(fq - q)))], 1e-30) / 2) - t
                     for q, t in zip(it['f_hz'], it['L_dBc_Hz'])]
                rr = np.sqrt(max(np.mean([x[2] for x in v]), 0) / it['var_excess_rad2'])
                print(f'    {radio} {"curve" if cont else "old  "} '
                      + ' '.join(f'{x:+5.1f}' for x in d) + f' | {rr:5.2f}', flush=True)


if __name__ == '__main__':
    main()
