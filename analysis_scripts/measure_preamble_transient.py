#!/usr/bin/env python3
"""Measure the burst-transient phase over the PREAMBLE, per config
(data/srrc_preamble_transient.json).

The fitted settling cubic (isi_taps.json) is estimated on the data portion
only, and the generator applied it over the whole burst, so the preamble was
EXTRAPOLATION: 1.4-1.6x the real preamble offset and 3-4x at the burst start
(12 deg against 3 at 89_2400). This measures the preamble part directly. The
transient is fleet-common (shape correlation 0.97-0.999 across radios and
configs, radio sd of the preamble offset 0.1-0.5 deg), so one profile per
config, averaged over radios and runs.

Estimator, identical for real and synthetic captures:
  * one constant x^4 CFO, bursts re-detected (ZC CFO-timing coupling);
  * each burst aligned on its DATA portion -- data-aided CFO and phase from
    rx * conj(tx) -- so the preamble keeps its offset relative to the data
    (aligning on the preamble, as the fits do, would hide exactly this);
  * burst average, then a short sample-level channel (17 taps) fitted
    tx -> average on the data portion: time-invariant ISI is referenced out,
    the time-varying transient is not;
  * phase of the average against that model in 50-sample blocks, the
    data-portion mean removed. The receiver spur is removed first.

    python analysis_scripts/measure_preamble_transient.py            # report
    python analysis_scripts/measure_preamble_transient.py --write    # store
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import synth_dataset as sd                                     # noqa: E402
from rx_spur import estimate_rx_spur, remove_rx_spur          # noqa: E402
from continuous_pn_roundtrip import _cfo_x4, _starts           # noqa: E402

OUT = sd.PREAMBLE_FILE              # the generator's file (SG_DATA_DIR / SG_PREAMBLE_FILE move it)
L, PRE = sd.BURST_LEN, sd.N_PRE * sd.SPS
DS, DE = sd.DATA_START, sd.DATA_END
BLK, NB = 50, 300
TAPS = np.arange(-8, 9)
T = (np.arange(DE // BLK) + 0.5) * BLK / L          # block centres, burst position
CONFIGS = ["77_433", "89_433", "77_915", "89_915", "77_2400", "89_2400"]
RUNS = (25, 45, 75, 95)


def _cfo_da(x, ref, fs):
    """Data-aided CFO: the tone left in x * conj(ref)."""
    s = x * np.conj(ref)
    N = 2 ** int(np.ceil(np.log2(len(s)))) * 16
    mag = np.abs(np.fft.fft(s, n=N))
    k = int(np.argmax(mag))
    a, b, c = (np.log(mag[(k + j) % N] + 1e-300) for j in (-1, 0, 1))
    den = a - 2 * b + c
    d = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den != 0 else 0.0
    return np.fft.fftfreq(N, 1 / fs)[k] + d * fs / N


def transient_profile(tx, rx, fs=sd.FS):
    """Block phase (deg) over preamble + data, data-portion mean removed."""
    tx = np.asarray(tx, complex)[:L]
    rx = np.asarray(rx, complex)
    st = _starts(rx, tx[:PRE])
    c0 = np.median([_cfo_x4(rx[s:s + L][DS:DE], fs) for s in st[::10]])
    xc = rx * np.exp(-2j * np.pi * c0 * np.arange(len(rx)) / fs)
    st = _starts(xc, tx[:PRE])
    n = np.arange(L)
    acc = np.zeros(L, complex)
    for s in st[:NB]:
        f = xc[s:s + L]
        f = f * np.exp(-2j * np.pi * _cfo_da(f[DS:DE], tx[DS:DE], fs) * n / fs)
        acc += f * np.exp(-1j * np.angle(np.vdot(tx[DS:DE], f[DS:DE])))
    y = acc / min(NB, len(st))
    X = np.stack([np.roll(tx, k) for k in TAPS], 1)
    sl = slice(DS + 50, DE - 50)
    h, *_ = np.linalg.lstsq(X[sl], y[sl], rcond=None)
    m = X @ h
    ph = np.degrees([np.angle(np.vdot(m[i:i + BLK], y[i:i + BLK]))
                     for i in range(0, DE, BLK)])
    return ph - ph[T >= DS / L].mean()


def _real(a):
    sess, cfg, run = a
    rd = Path(sess) / cfg / f"run_{run:03d}" / "valid"
    try:
        tx = np.fromfile(next(rd.glob("*tx_*.sigmf-data")), np.complex64)
        rx = np.fromfile(next(rd.glob("capture_*_combined.sigmf-data")), np.complex64)
    except StopIteration:
        return None
    rx = rx.astype(complex)
    amps, phs, _ = estimate_rx_spur(rx, tx[:L].astype(complex))
    return cfg, Path(sess).name, transient_profile(tx, remove_rx_spur(rx, amps, phs))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--captures", default=str(BASE.parent / "New_captures"))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--session-map", default=None,
                    help="JSON {radio: session dir} to measure instead of pn_curves.json's "
                         "path_sources (e.g. generalisation_work/july_sessions.json)")
    ap.add_argument("--out", default=str(OUT), help="where --write stores the profile")
    a = ap.parse_args(argv)
    if a.session_map:
        dirs = {k: Path(v) for k, v in json.loads(Path(a.session_map).read_text()).items()
                if not k.startswith("_")}
    else:
        dirs = {k: Path(a.captures) / s for k, s in
                json.loads(sd.PN_CURVES_FILE.read_text())["path_sources"]["sessions"].items()}
    sessions = {k: d.name for k, d in dirs.items()}
    jobs = [(str(d), c, r) for d in dirs.values() for c in CONFIGS for r in RUNS]
    with ProcessPoolExecutor(6) as ex:
        res = [x for x in ex.map(_real, jobs) if x]
    pre = T < DS / L
    doc = {"format_version": 1,
           "what_it_is": ("Burst-transient phase over the preamble, per config, fleet mean. "
                          "The generator keeps its fitted cubic on the data portion and uses "
                          "this profile for t < data start, joined continuously there and "
                          "scaled by the radio's data-portion transient rms relative to this "
                          "profile's."),
           "estimator": __doc__.split("Estimator, identical")[1].split("python analysis")[0].strip(),
           "units": {"t": "burst position, sample_index / BURST_LEN (block centres)",
                     "phi_deg": "degrees, data-portion mean removed; multiply the signal by exp(1j*deg2rad(phi))"},
           "t": [round(float(x), 5) for x in T],
           "configs": {}}
    print("config   n   preamble offset deg   first block   data rms   radio sd of offset")
    for c in CONFIGS:
        P = [p for cc, _, p in res if cc == c]
        by_sess = {}
        for cc, s, p in res:
            if cc == c:
                by_sess.setdefault(s, []).append(np.mean(p[pre]))
        m = np.mean(P, axis=0)
        sd_r = float(np.std([np.mean(v) for v in by_sess.values()]))
        print(f"{c:8} {len(P):2d}   {m[pre].mean():6.2f}              {m[0]:6.2f}      {m[~pre].std():5.2f}      {sd_r:.2f}")
        doc["configs"][c] = {"phi_deg": [round(float(x), 3) for x in m],
                             "n_captures": len(P), "n_radios": len(by_sess),
                             "preamble_offset_deg": round(float(m[pre].mean()), 3),
                             "radio_sd_of_offset_deg": round(sd_r, 3),
                             "data_rms_deg": round(float(m[~pre].std()), 3)}
    doc["source"] = {"sessions": sessions, "runs": list(RUNS), "bursts_per_capture": NB}
    if a.write:
        Path(a.out).write_text(json.dumps(doc, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
