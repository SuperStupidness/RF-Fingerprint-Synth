"""
wifi_ofdm — 802.11a/g (non-HT, CBW20) OFDM receiver + OFDM-aware capture
validation, for the Wi-Fi fingerprinting captures.

The single-carrier constellation/cluster gate in post_process_utils rejects
OFDM (no symbol constellation). This module provides the OFDM analog:

    demod_wifi(x20)              — receiver on a 20 Msps burst (sync→CFO→
                                  channel-est→equalize→pilot CPE→EVM)
    wifi_capture_quality(...)   — load a SigMF chunk, decimate to 20 Msps,
                                  find a burst, return {found, evm_pct, cfo_hz, ...}
    validate_wifi_and_catalogue — drop-in replacement for validate_and_catalogue
                                  that gates on EVM instead of cluster score

Detection uses the L-LTF delay-64 autocorrelation (two identical long-training
symbols), which is CFO-robust (a frequency offset only rotates the correlation,
not its magnitude). Timing is then refined with an LTS matched filter after CFO
correction. The pure-DSP functions depend only on numpy/scipy so they can be
unit-tested off the capture host; validate_wifi_and_catalogue lazily imports the
SigMF helpers from post_process_utils.

Cross-validated against the MATLAB WLAN Toolbox reference RX (see
verify_wifi_demod.{m,py}): identical equalized symbols to ~1e-8 on clean bursts.
"""
import os
import numpy as np

FS_NATIVE = 20e6
NFFT = 64
NCP  = 16
NSYM_LEN = NFFT + NCP                 # 80 samples per OFDM symbol
NDATA_SYM_DEFAULT = 270               # our CBW20 QPSK-1/2 burst (22000-sample packet)

# --- standard L-LTF frequency-domain sequence, subcarriers k = -26..26 (DC=0) ---
_L = np.array([
    1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1,
    0,
    1,-1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1],
    dtype=float)
_KK      = np.arange(-26, 27)
_PILOTS  = {-21, -7, 7, 21}
_DATA_K  = np.array([k for k in range(-26, 27) if k != 0 and k not in _PILOTS])  # 48
_USED_K  = _KK[_L != 0]

def _bins(kset):
    return np.asarray(kset) % NFFT

_XLTF = np.zeros(NFFT, complex); _XLTF[_bins(_KK)] = _L
_LTS_TIME = np.fft.ifft(_XLTF)        # 64-sample time-domain long training symbol


def resample_to_20(iq, fs):
    """Resample a capture at `fs` down to the 20 Msps native OFDM rate."""
    fs = float(fs)
    if abs(fs - FS_NATIVE) < 1.0:
        return np.asarray(iq, dtype=np.complex64)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(round(fs)), int(FS_NATIVE))
    up, down = int(FS_NATIVE) // g, int(round(fs)) // g
    return resample_poly(iq, up, down).astype(np.complex64)


def _detect_lltf(x):
    """CFO-robust preamble detection via the delay-64 autocorrelation.

    The metric is normalized to ~1 over the repeated training symbols; it is
    energy-masked so the near-zero-power null tail can't produce a spurious
    peak. Returns (d, metric, P): d indexes into the preamble plateau (STF or
    LTF — both repeat at 64), metric∈[0,1], and P is the complex correlation
    used for the coarse CFO estimate. Exact LTS timing is refined by matched
    filter (after CFO removal) in demod_wifi.
    """
    if len(x) < 2 * NFFT + 8:
        return 0, 0.0, 0j
    c  = np.conj(x[:-NFFT]) * x[NFFT:]              # conj(x[n]) x[n+64]
    ones = np.ones(NFFT)
    P  = np.convolve(c, ones, 'valid')             # windowed sum over 64
    e  = np.convolve(np.abs(x[NFFT:])**2, ones, 'valid')
    M  = np.abs(P) / (e + 1e-12)
    M[e < 0.5 * e.max()] = 0.0                      # ignore low-energy regions
    d  = int(np.argmax(M))
    return d, float(M[d]), P[d]


def _apply_cfo(x, cfo_hz, fs=FS_NATIVE):
    n = np.arange(len(x))
    return x * np.exp(-1j * 2 * np.pi * cfo_hz / fs * n)


def demod_wifi(x20, nsym=NDATA_SYM_DEFAULT, detect_thresh=0.5):
    """
    Demod one 802.11a/g CBW20 burst contained in `x20` (20 Msps).

    Chain: L-LTF autocorr detect → coarse CFO → LTS-MF fine timing →
    channel estimate → zero-forcing equalize → per-symbol pilot common-phase
    correction → data-subcarrier EVM vs nearest QPSK.

    Returns dict: found, detect_metric, cfo_hz, n_symbols, evm_pct, eq (array).
    """
    x20 = np.asarray(x20, dtype=complex)
    out = dict(found=False, detect_metric=0.0, cfo_hz=0.0, n_symbols=0,
               evm_pct=np.nan, eq=np.array([], complex))

    d, metric, P = _detect_lltf(x20)
    out['detect_metric'] = metric
    if metric < detect_thresh:
        return out

    # coarse CFO from the delay-64 autocorrelation phase
    cfo = np.angle(P) * FS_NATIVE / (2 * np.pi * NFFT)
    x = _apply_cfo(x20, cfo)

    # locate the LTS by matched filter (CFO removed → coherent, sharp peaks;
    # also disambiguates the LTF from the STF, which the autocorr cannot).
    # The two identical LTS give peaks 64 apart; take the earlier as LTS1.
    mf = np.abs(np.correlate(x, _LTS_TIME, mode="valid"))
    p = int(np.argmax(mf))
    s1 = p - NFFT if (p - NFFT >= 0 and mf[p - NFFT] > 0.7 * mf[p]) else p
    if s1 + 2 * NFFT + NSYM_LEN >= len(x):
        return out

    # fine CFO from the two refined LTS, then re-correct
    a, b = x[s1:s1 + NFFT], x[s1 + NFFT:s1 + 2 * NFFT]
    cfo_fine = np.angle(np.vdot(a, b)) * FS_NATIVE / (2 * np.pi * NFFT)
    x = _apply_cfo(x, cfo_fine)
    cfo += cfo_fine

    # channel estimate from the two LTS
    Y = 0.5 * (np.fft.fft(x[s1:s1 + NFFT]) + np.fft.fft(x[s1 + NFFT:s1 + 2 * NFFT]))
    H = np.ones(NFFT, complex)
    H[_bins(_USED_K)] = Y[_bins(_USED_K)] / _XLTF[_bins(_USED_K)]

    data_start = s1 + 2 * NFFT + NSYM_LEN            # + 2 LTS + SIGNAL field
    avail = (len(x) - data_start) // NSYM_LEN
    nsym = avail if nsym is None else min(nsym, avail)
    if nsym < 1:
        return out

    dbin, pbin = _bins(_DATA_K), _bins(sorted(_PILOTS))
    syms = []
    for i in range(nsym):
        seg = x[data_start + i * NSYM_LEN: data_start + (i + 1) * NSYM_LEN]
        eqs = np.fft.fft(seg[NCP:]) / H
        # common-phase error from pilots (BPSK): (±1·e^{jφ})² = e^{j2φ},
        # so φ = ½·angle(Σ pilot²) — sign/π-ambiguity is harmless for QPSK EVM.
        phi = 0.5 * np.angle(np.sum(eqs[pbin] ** 2))
        syms.append(eqs[dbin] * np.exp(-1j * phi))

    eq = np.concatenate(syms)
    ref = (np.sign(eq.real) + 1j * np.sign(eq.imag)) / np.sqrt(2)
    evm = 100 * np.sqrt(np.mean(np.abs(eq - ref) ** 2) / np.mean(np.abs(ref) ** 2))

    out.update(found=True, cfo_hz=float(cfo), n_symbols=int(nsym),
               evm_pct=float(evm), eq=eq)
    return out


def wifi_capture_quality(data_path, fs, nsym=NDATA_SYM_DEFAULT,
                         detect_thresh=0.5, max_load=4_000_000):
    """Load a SigMF-data chunk, decimate to 20 Msps, demod one burst.

    Returns the demod_wifi dict (adds nothing hardware-specific). Only the first
    `max_load` samples are read — one burst is enough to gauge quality.
    """
    iq = np.fromfile(data_path, dtype=np.complex64, count=max_load)
    if iq.size == 0:
        return dict(found=False, detect_metric=0.0, cfo_hz=0.0,
                    n_symbols=0, evm_pct=np.nan, eq=np.array([], complex))
    x20 = resample_to_20(iq, fs)
    return demod_wifi(x20, nsym=nsym, detect_thresh=detect_thresh)


# ══════════════════════════════════════════════════════════════
# OFDM-aware validation — mirrors post_process_utils.validate_and_catalogue
# ══════════════════════════════════════════════════════════════
_WIFI_CATALOGUE_COLUMNS = [
    'timestamp', 'experiment_name', 'modulation', 'tx_file', 'rx_combined_file',
    'num_rx_chunks', 'total_rx_samples', 'duration_s', 'sample_rate', 'center_freq',
    'power_dbm_min', 'power_dbm_max', 'power_dbm_mean',
    'wifi_found', 'wifi_evm_pct', 'wifi_cfo_hz', 'wifi_detect_metric',
    'device_type', 'serial_number', 'source_dir',
]


def validate_wifi_and_catalogue(source_dir, target_dir, experiment_yaml=None,
                                power_threshold_dbm=-50.0, evm_threshold_pct=25.0,
                                detect_thresh=0.5, nsym=NDATA_SYM_DEFAULT,
                                catalogue_name='wifi_catalogue.csv'):
    """
    Validate Wi-Fi/OFDM captures, combine valid ones, and catalogue them.

    Same structure as post_process_utils.validate_and_catalogue, but Stage 2 is
    an OFDM demod: a session passes if a valid 802.11 packet is detected on the
    representative chunk with EVM below `evm_threshold_pct`.

    Returns list of valid combined RX SigMF data files.
    """
    import csv
    from datetime import datetime
    from .post_process_utils import (
        identify_capture_files, check_continuous_signal, combine_rx_chunks,
        load_sigmf_meta, extract_catalogue_fields)
    import shutil

    print(f"\n{'='*60}\n  Wi-Fi/OFDM Capture Validator\n"
          f"  Source: {source_dir}\n  Target: {target_dir}\n{'='*60}\n")

    experiment_name = modulation = 'unknown'
    if experiment_yaml is not None:
        import yaml as _yaml
        exp = _yaml.safe_load(open(experiment_yaml))
        experiment_name = exp.get('experiment', {}).get('name', 'unknown')
        modulation = str(exp.get('signal', {}).get('type', 'WIFI')).upper()

    info = identify_capture_files(source_dir)
    tx_files, rx_groups = info['tx_files'], info['rx_groups']
    print(f"  Found {len(tx_files)} TX file(s), {len(rx_groups)} RX session(s)\n")
    if not rx_groups:
        print("  No RX captures found!")
        return []

    os.makedirs(target_dir, exist_ok=True)
    csv_path = os.path.join(target_dir, catalogue_name)
    tx_basename = tx_files[0][0] if tx_files else 'unknown'
    valid_outputs = []

    for session_key, rx_group in rx_groups.items():
        print(f"  -- {session_key} ({len(rx_group)} chunks) --")

        # Stage 1: continuous power
        all_passed, powers, failed_idx = check_continuous_signal(
            rx_group, power_threshold_dbm)
        pmin, pmax, pmean = min(powers), max(powers), float(np.mean(powers))
        print(f"    Power: min={pmin:.1f} max={pmax:.1f} mean={pmean:.1f} dBm")
        if not all_passed:
            print(f"    → REJECT: {len(failed_idx)} chunk(s) below threshold")
            continue

        # Stage 2: OFDM demod on the representative (middle) chunk
        mid = len(rx_group) // 2
        _, mid_data, mid_meta = rx_group[mid]
        fs = load_sigmf_meta(mid_meta).get('global', {}).get('core:sample_rate', FS_NATIVE)
        q = wifi_capture_quality(mid_data, fs, nsym=nsym, detect_thresh=detect_thresh)
        print(f"    OFDM: found={q['found']} detect={q['detect_metric']:.2f} "
              f"EVM={q['evm_pct']:.2f}% CFO={q['cfo_hz']:.0f}Hz", end='')
        if not (q['found'] and q['evm_pct'] <= evm_threshold_pct):
            print(f"  → REJECT (EVM>{evm_threshold_pct:.0f}% or no packet)")
            continue
        print("  → PASS")

        # Stage 3: combine + catalogue
        out_base = os.path.join(target_dir, session_key + '_combined')
        comb_data, comb_meta, total = combine_rx_chunks(rx_group, out_base)
        mf = extract_catalogue_fields(load_sigmf_meta(comb_meta))
        sr = mf['sample_rate']; dur = total / sr if sr else 0
        print(f"    Combined: {total:,} samples ({dur:.2f}s)")

        for tx_base, tx_data, tx_meta in tx_files:
            dd = os.path.join(target_dir, os.path.basename(tx_data))
            if not os.path.exists(dd):
                shutil.copy2(tx_data, dd)
                shutil.copy2(tx_meta, os.path.join(target_dir, os.path.basename(tx_meta)))

        entry = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': experiment_name, 'modulation': modulation,
            'tx_file': tx_basename, 'rx_combined_file': os.path.basename(comb_data),
            'num_rx_chunks': len(rx_group), 'total_rx_samples': total,
            'duration_s': f"{dur:.4f}", 'sample_rate': sr, 'center_freq': mf['center_freq'],
            'power_dbm_min': f"{pmin:.1f}", 'power_dbm_max': f"{pmax:.1f}",
            'power_dbm_mean': f"{pmean:.1f}",
            'wifi_found': q['found'], 'wifi_evm_pct': f"{q['evm_pct']:.3f}",
            'wifi_cfo_hz': f"{q['cfo_hz']:.1f}", 'wifi_detect_metric': f"{q['detect_metric']:.3f}",
            'device_type': mf['device_type'], 'serial_number': mf['serial_number'],
            'source_dir': os.path.abspath(source_dir),
        }
        exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=_WIFI_CATALOGUE_COLUMNS)
            if not exists:
                w.writeheader()
            w.writerow(entry)
        valid_outputs.append(comb_data)

    print(f"\n{'='*60}\n  Results: {len(valid_outputs)}/{len(rx_groups)} sessions valid")
    if valid_outputs:
        print(f"  Catalogue: {csv_path}")
    print(f"{'='*60}\n")
    return valid_outputs
