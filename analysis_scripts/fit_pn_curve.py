#!/usr/bin/env python3
"""Fit ONE phase-noise curve per radio/config (data/pn_curves.json) to BOTH
real measurements at once, exactly.

The curve is S_phi(f) in dB(rad^2/Hz) at fixed log-spaced knots from 1.5 Hz to
500 kHz, interpolated in log f (the generator's convention), and the generator
draws it as ONE continuous process per run at symbol rate. Two views of the
real captures constrain it:

  in-burst   the tangential symbol deviation fit_isi_taps.py builds (per-burst
             CFO + preamble phase removed, matched filter, clock phase, cluster
             centroid, across-burst mean removed, per-burst mean + ramp
             removed), Welch with one burst per segment, minus the radial
             (AWGN) spectrum. Per radio. Sees ~1 kHz to 500 kHz.
  path       the preamble phase of every burst after ONE constant CFO,
             quadratic removed, Welch at the burst rate (766 Hz), plus its
             variance (the wander). Fleet-common per config. Sees 1.5-383 Hz,
             and the in-burst part through aliasing.

Neither view alone covers 200 Hz-3 kHz, which is why fitting them separately
and joining the two curves left a knee (the flat 6-50 kHz plateau and the fall
into it) that no straight line followed. Fitting them together puts it where
both views agree.

Every step of both measurements is LINEAR in the phase, so the expected
measurement is a fixed linear map of the injected spectrum -- no random draws:

    E[measurement] = M @ S          S = the curve on the generator's FFT grid

M is built once from the generator's own grid (one run, 762 bursts). Its parts:
  * path: the preamble average (Dirichlet kernel |H|^2) and the burst-rate
    sampling, then quadratic removal and Welch (same operator as before);
  * in-burst: the SMOOTHING the phase gets on its way to a symbol decision --
    the generator interpolates the symbol-rate phase to sample rate and the
    SRRC pulse and matched filter weight it by srrc(n)^2 around each symbol
    instant (|V(f)|^2 below). Treating the deviation as the phase itself
    double-counted that smoothing and left the total in-burst PN 5-8% low.
    Cross-symbol terms land equally on the tangential and radial axes, so the
    excess cancels them; then deramp, across-burst mean and Welch, exactly.

The solve is least squares in dB on: up to 50 in-burst frequencies (1-300 kHz,
only where the excess is >= 0.3x the thermal yardstick), the total in-burst
excess variance, 7 path bands (3-200 Hz) and the wander variance, plus a light
second-difference penalty on the knots (binding only where neither view looks)
and a penalty on any rise above 100 kHz.

INPUT: deviation spectra from fit_isi_taps.py run with SG_PN_SPECTRUM_OUT (and
SG_DESPUR=1, so the receiver spur is not fitted as the radio's). Profiles
without spectra (captures not on disk) get their config's median curve,
shifted by their own in-burst level offset (pn_level_db_10k in isi_taps.json).

    python analysis_scripts/fit_pn_curve.py SPECTRA_DIR            # report only
    python analysis_scripts/fit_pn_curve.py SPECTRA_DIR --write    # data/pn_curves.json

Afterwards, analysis_scripts/continuous_pn_roundtrip.py checks the result on
real generator output.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.optimize
import scipy.signal

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import synth_dataset as sd                                     # noqa: E402

FSYM = sd.FS / sd.SPS                 # synthesis rate of the phase process
SLOT = sd.BURST_SPACING // sd.SPS     # symbols between burst starts
NPRE = sd.N_PRE                       # preamble symbols averaged per burst
K = 762                               # bursts per run
NSYM = sd.N_PRE + sd.N_DATA + sd.N_TAIL
N_RUN = (K - 1) * SLOT + NSYM         # length of one run's phase process
NPERSEG_PATH = 256
KNOTS = [1.5, 3, 6, 12, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400,
         12800, 25600, 51200, 100000, 200000, 500000]
PATH_BANDS = [3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 200.0]
F_IN = np.unique(np.round(np.logspace(3, np.log10(3e5), 50) / 1e3) * 1e3)
W_PATH, W_VAR, W_SMOOTH = 2.0, 2.0, 0.25        # residual weights (dB units)
# An in-burst bin is used only where the phase-noise excess stands clear of the
# thermal yardstick: excess >= REL_MIN x radial. At 2400 MHz and on some 89_433
# radios the phase noise is below the per-bin noise above ~150-250 kHz, where
# the excess is estimation noise (and sometimes negative); fitting it in dB
# drove those fits 20-37 dB off. Above the last usable bin the total variance
# and the smoothness penalty hold the curve.
REL_MIN = 0.3
# No RISE above 100 kHz: past the last usable bin only the total variance sees
# the curve, and on 12 of 122 profiles it pulled the 500 kHz knot up by 1-9 dB
# to absorb wideband tangential noise that is not LO phase noise (on
# 30BF7C1/89_433 the whole in-burst excess is flat to 300 kHz). Such profiles
# now keep a small variance shortfall instead of an unphysical tail.
F_NO_RISE, W_NO_RISE = 1e5, 3.0
# in-burst target stored with each curve, so the round trip needs no captures
F_STORE = [1e3, 2e3, 4e3, 7e3, 1e4, 2e4, 5e4, 1e5, 1.5e5, 2e5, 3e5]
N_CELLS = 3000                        # log-frequency cells for M


def band_db(f, S, q):
    """Half-octave band mean as L(f) dBc/Hz."""
    q = max(q, 3.0)
    m = (f >= q / 2 ** 0.5) & (f <= q * 2 ** 0.5) & (f > 0)
    return 10 * np.log10(S[m].mean() / 2)


# ── the two measurements as operators on a lag covariance ───────────────────

def _path_operator():
    """G (bins x lags), g (lags): E[Welch PSD] = G @ c, E[var] = g @ c, for a
    burst-rate path with Cov(x_k, x_l) = c[|k-l|]."""
    t = np.arange(K) * SLOT / FSYM
    Q = np.column_stack([np.ones(K), t, t ** 2])
    P = np.eye(K) - Q @ np.linalg.pinv(Q)                # quadratic removal
    n = np.arange(NPERSEG_PATH)
    Ql = np.column_stack([np.ones(NPERSEG_PATH), n])
    D = np.eye(NPERSEG_PATH) - Ql @ np.linalg.pinv(Ql)   # Welch's detrend='linear'
    h = scipy.signal.get_window('hann', NPERSEG_PATH)
    nb = NPERSEG_PATH // 2 + 1
    B = np.exp(-2j * np.pi * np.outer(np.arange(nb), n) / NPERSEG_PATH) @ (h[:, None] * D)
    fs_path = FSYM / SLOT
    scale = np.full(nb, 2.0 / (fs_path * np.sum(h ** 2)))
    scale[[0, -1]] /= 2
    starts = range(0, K - NPERSEG_PATH + 1, NPERSEG_PATH // 2)
    G = np.zeros((nb, K))
    for s0 in starts:
        V = np.zeros((K, nb), complex)
        V[s0:s0 + NPERSEG_PATH] = B.T
        Uf = np.fft.fft(P @ V, 2 * K, axis=0)
        ac = np.fft.ifft(np.abs(Uf) ** 2, axis=0).real[:K].T
        ac[:, 1:] *= 2
        G += ac
    G *= (scale / len(starts))[:, None]
    g = np.array([np.trace(P, offset=d) for d in range(K)]) / K
    g[1:] *= 2
    return np.fft.rfftfreq(NPERSEG_PATH, d=1 / fs_path), G, g


def _inburst_operator(n_bursts):
    """Same for the in-burst deviation: one N_DATA-symbol burst, mean + ramp
    removed, Hann Welch with one segment, across-burst mean removed."""
    N = sd.N_DATA
    n = np.arange(N)
    X = np.column_stack([np.ones(N), n])
    D = np.eye(N) - X @ np.linalg.pinv(X)
    h = scipy.signal.get_window('hann', N)
    nb = N // 2 + 1
    B = np.exp(-2j * np.pi * np.outer(np.arange(nb), n) / N) * h[None, :]
    Uf = np.fft.fft(D @ B.T, 2 * N, axis=0)
    ac = np.fft.ifft(np.abs(Uf) ** 2, axis=0).real[:N].T
    ac[:, 1:] *= 2
    scale = np.full(nb, 2.0 / (FSYM * np.sum(h ** 2)))
    scale[[0, -1]] /= 2
    keep = 1 - 1 / n_bursts                              # across-burst mean removal
    g = np.array([np.trace(D, offset=d) for d in range(N)]) / N
    g[1:] *= 2
    return np.fft.rfftfreq(N, d=1 / FSYM), ac * scale[:, None] * keep, g * keep


def symbol_kernel():
    """|V(f)|^2 on the symbol-rate grid: how the phase is smoothed before a
    symbol decision. The phase is drawn at symbol rate, interpolated linearly
    to sample rate, and the TX pulse times the matched filter weights it by
    w(n) = srrc(n)^2 around each symbol instant (the coherent term of the
    decision; the cross-symbol terms are equal on both axes and cancel in the
    excess). v(i) = sum_n w(n) tri(i - n/SPS)."""
    w = sd.SRRC ** 2
    w = w / w.sum()
    n = np.arange(len(w)) - (len(w) - 1) / 2
    i = np.arange(-int(np.ceil(n.max() / sd.SPS)) - 1, int(np.ceil(n.max() / sd.SPS)) + 2)
    v = np.array([np.sum(w * np.maximum(0, 1 - np.abs(k - n / sd.SPS))) for k in i])
    return i, v / v.sum()


def build_operator(n_bursts):
    """M (outputs x cells), the cell frequencies, and the output layout, with

        E[path Welch bins, path var, in-burst Welch bins, in-burst var] = M @ S

    where S is the curve (linear rad^2/Hz) at each cell's frequency. Cells
    group the generator's FFT bins (1.006 Hz apart, one run) on a log grid,
    one bin per cell at the bottom; the curve moves < 0.02 dB inside a cell,
    while every oscillating kernel stays exact per bin."""
    fp, Gp, gp = _path_operator()
    fi, Gi, gi = _inburst_operator(n_bursts)
    f = np.fft.rfftfreq(N_RUN, d=1 / FSYM)
    wt = np.full(len(f), 2.0)                            # irfft: X_b and its mirror
    wt[0] = 0.0
    if N_RUN % 2 == 0:
        wt[-1] = 1.0
    # c[m] = irfft(amp^2 H)[m] / N_RUN with amp^2 = S N_RUN FSYM / 2
    #      = sum_b S_b (FSYM / 2 / N_RUN) wt_b H_b cos(2 pi b m / N_RUN)
    base = FSYM / 2 / N_RUN * wt
    H2 = np.ones_like(f)
    x = np.pi * f[1:] / FSYM
    H2[1:] = (np.sin(NPRE * x) / (NPRE * np.sin(x))) ** 2
    ii, v = symbol_kernel()
    V2 = np.abs(np.exp(-2j * np.pi * np.outer(f / FSYM, ii)) @ v) ** 2

    edges = np.unique(np.searchsorted(f, np.logspace(0, np.log10(f[-1]), N_CELLS)))
    edges = edges[edges >= 1]
    fc = np.sqrt(f[edges] * f[np.r_[edges[1:], len(f)] - 1])
    lag_p = np.arange(K) * SLOT
    lag_i = np.arange(sd.N_DATA)
    Mp = np.zeros((len(fp) + 1, len(edges)))
    Mi = np.zeros((len(fi) + 1, len(edges)))
    Ap, Ai = np.vstack([Gp, gp]), np.vstack([Gi, gi])
    step = 20000
    for b0 in range(1, len(f), step):
        b = np.arange(b0, min(b0 + step, len(f)))
        ph = 2 * np.pi * b / N_RUN
        cell = np.searchsorted(edges, b, side='right') - 1
        Kp = np.cos(np.outer(lag_p, ph)) * (base[b] * H2[b])
        Ki = np.cos(np.outer(lag_i, ph)) * (base[b] * V2[b])
        np.add.at(Mp.T, cell, (Ap @ Kp).T)
        np.add.at(Mi.T, cell, (Ai @ Ki).T)
    return dict(M=np.vstack([Mp, Mi]), fc=fc, fp=fp, fi=fi, n_p=len(fp), n_i=len(fi))


def curve_on(fc, levels):
    return 10.0 ** (np.interp(np.log10(fc), np.log10(KNOTS), levels) / 10.0)


def predict(op, levels):
    y = op['M'] @ curve_on(op['fc'], levels)
    n_p = op['n_p']
    return dict(Sp=y[:n_p], vp=y[n_p], Si=y[n_p + 1:-1], vi=y[-1])


# ── targets and the solve ───────────────────────────────────────────────────

def load_spectra(path):
    """MEDIAN excess (tangential - radial) spectrum over the runs of one file.

    Median, not mean (2026-10-04): one broken run measurement moves a mean of
    six. 30BF779/89_2400 run 100 read 30 deg^2 against 1.5-1.7 for the other
    five (18x), which put that curve ~3x too high in variance; 30ECB71/89_433
    run 60 read 3.3x. The direct extractor reads both runs normally, and no
    other profile has a run above 2x its median."""
    d = json.loads(Path(path).read_text())
    f = np.array(d['f'])
    ex = np.median([np.array(r['tang']) - np.array(r['rad']) for r in d['runs']], axis=0)
    rad = np.median([np.array(r['rad']) for r in d['runs']], axis=0)
    return dict(f=f, excess=ex, rad=rad, var=float(np.median([r['var_excess'] for r in d['runs']])),
                n_runs=len(d['runs']), n_bursts=int(d['runs'][0]['n_bursts']),
                despur=bool(d.get('despur')))


class Fitter:
    def __init__(self, op, path_target):
        self.op = op
        self.t_p = np.array([path_target['L_dBc_Hz'][f'{q:g}'] for q in PATH_BANDS])
        self.t_w = np.radians(np.sqrt(path_target['wander_ms_deg2'])) ** 2
        self.idx = [int(np.argmin(np.abs(op['fi'] - q))) for q in F_IN]

    def usable(self, spec):
        return [i for i in self.idx if spec['excess'][i] >= REL_MIN * spec['rad'][i]]

    def residuals(self, levels, spec):
        pr = predict(self.op, levels)
        idx = self.usable(spec)
        t_in = 10 * np.log10(spec['excess'][idx] / 2)
        r_in = 10 * np.log10(pr['Si'][idx] / 2) - t_in
        r_vi = 10 * np.log10(pr['vi'] / spec['var'])
        r_p = np.array([band_db(self.op['fp'], pr['Sp'], q) for q in PATH_BANDS]) - self.t_p
        r_w = 10 * np.log10(pr['vp'] / self.t_w)
        return r_in, r_vi, r_p, r_w

    def start(self, spec):
        """Path bands below 200 Hz, in-burst excess above 1 kHz, joined in log f."""
        lo = [(q, t + 3.01) for q, t in zip(PATH_BANDS, self.t_p)]
        ex = spec['excess']
        ok = set(self.usable(spec))
        hi = [(q, 10 * np.log10(ex[i])) for q in (1e3, 3e3, 1e4, 3e4, 1e5, 3e5)
              for i in [int(np.argmin(np.abs(self.op['fi'] - q)))] if i in ok]
        pts = lo + hi
        return np.interp(np.log10(KNOTS), np.log10([p[0] for p in pts]), [p[1] for p in pts])

    def fit(self, spec):
        def res(lv):
            r_in, r_vi, r_p, r_w = self.residuals(lv, spec)
            rise = np.maximum(0.0, np.diff(lv[np.array(KNOTS) >= F_NO_RISE]))
            return np.concatenate([r_in, [W_VAR * r_vi], W_PATH * r_p, [W_PATH * r_w],
                                   W_SMOOTH * np.diff(lv, 2), W_NO_RISE * rise])
        sol = scipy.optimize.least_squares(res, self.start(spec), diff_step=1e-4)
        return sol.x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('spectra_dir', help='directory of fit_isi_taps SG_PN_SPECTRUM_OUT files')
    ap.add_argument('--targets', default=str(sd.PN_CURVES_FILE),
                    help="file holding the per-config path targets, and where --write stores the "
                         "curves (default: the generator's pn_curves.json; SG_DATA_DIR moves it)")
    ap.add_argument('--write', action='store_true', help='store the curves in the --targets file')
    a = ap.parse_args(argv)

    doc = json.loads(Path(a.targets).read_text())
    taps = json.loads(sd.TAPS.read_text())          # the generator's profile list (SG_DATA_DIR / SG_TAPS_FILE)
    specs = {}
    for p in sorted(Path(a.spectra_dir).glob('*.json')):
        d = json.loads(p.read_text())
        if 'runs' in d and 'f' in d:
            specs[f"{d['radio']}/{d['config']}"] = load_spectra(p)
    nbs = {s['n_bursts'] for s in specs.values()}
    assert len(nbs) == 1, f'mixed bursts per run: {nbs}'
    print(f'{len(specs)} spectra; building the measurement operator ...', flush=True)
    op = build_operator(nbs.pop())

    curves, rows = {}, []
    for key in sorted(taps):
        radio, cfg = key.split('/')
        if key not in specs:
            continue
        s = specs[key]
        fitter = Fitter(op, doc['path_targets'][cfg])
        lv = fitter.fit(s)
        r_in, r_vi, r_p, r_w = fitter.residuals(lv, s)
        ok = set(fitter.usable(s))
        store = [(q, i) for q in F_STORE for i in [int(np.argmin(np.abs(op['fi'] - q)))] if i in ok]
        curves[key] = dict(curve=[[float(f), round(float(x), 2)] for f, x in zip(KNOTS, lv)],
                           source='fit', n_runs=s['n_runs'], despurred=s['despur'],
                           resid_inburst_rms_db=round(float(np.sqrt(np.mean(r_in ** 2))), 3),
                           resid_inburst_var_db=round(float(r_vi), 3),
                           resid_path_db=[round(float(x), 3) for x in r_p],
                           resid_wander_db=round(float(r_w), 3),
                           inburst_target=dict(
                               f_hz=[q for q, i in store],
                               L_dBc_Hz=[round(float(10 * np.log10(s['excess'][i] / 2)), 2)
                                         for q, i in store],
                               usable_to_hz=float(op['fi'][max(fitter.usable(s))]),
                               var_excess_rad2=s['var'], n_bursts=s['n_bursts']))
        rows.append((key, r_in, r_vi, r_p, r_w))
        print(f'{key:16} in-burst rms {np.sqrt(np.mean(r_in ** 2)):.2f} dB, var {r_vi:+.2f} dB, '
              f'path rms {np.sqrt(np.mean(r_p ** 2)):.2f} dB, wander {r_w:+.2f} dB', flush=True)

    # profiles with no captures on disk: config median, shifted by own level
    for key in sorted(set(taps) - set(curves)):
        radio, cfg = key.split('/')
        peers = [k for k in curves if k.endswith('/' + cfg)]
        med = np.median([[p[1] for p in curves[k]['curve']] for k in peers], axis=0)
        offs = np.median([taps[k]['pn_level_db_10k'] for k in peers])
        d = taps[key]['pn_level_db_10k'] - offs
        # the per-radio (in-burst) part only; the slow part is fleet-common
        lv = [x[1] for x in sd.shift_pn_curve([[f, m] for f, m in zip(KNOTS, med)], d)]
        curves[key] = dict(curve=[[float(f), round(float(x), 2)] for f, x in zip(KNOTS, lv)],
                           source=f'config median of {len(peers)} fits, in-burst part shifted '
                                  f'{d:+.2f} dB by pn_level_db_10k (no captures on disk)')
        print(f'{key:16} config median, shifted {d:+.2f} dB')

    r_in = np.concatenate([r[1] for r in rows])
    print(f'\n{len(rows)} fitted, {len(curves) - len(rows)} from the config median. '
          f'In-burst residual rms {np.sqrt(np.mean(r_in ** 2)):.2f} dB; '
          f'total in-burst variance {np.min([r[2] for r in rows]):+.2f}..{np.max([r[2] for r in rows]):+.2f} dB; '
          f'path bands {np.min([r[3] for r in rows]):+.2f}..{np.max([r[3] for r in rows]):+.2f} dB; '
          f'wander {np.min([r[4] for r in rows]):+.2f}..{np.max([r[4] for r in rows]):+.2f} dB')
    if a.write:
        doc['curves'] = {k: curves[k] for k in sorted(curves)}
        Path(a.targets).write_text(json.dumps(doc, indent=1))
        print(f'wrote {a.targets}')


if __name__ == '__main__':
    main()
