# Changelog

Newest first. Where the default output changes, the listed flag reproduces the previous output bit for bit.

## 2026-10-04

**Default output changes: the burst transient, the ISI and the receive filter.** Previous output: `--full-transient --raw-taps --bb60-passband`.

### Changed
- The burst transient is one fleet shape per config scaled by a per-radio level, `transient_level_deg` (its rms phase swing over the data). It replaces each radio's six complex-cubic coefficients, which are kept in the profile as `settling_fitted`.
- The ISI (the post-PA linear response) is one fleet response per config, slid in frequency by a per-radio `isi_shift_khz`. It replaces each radio's two complex taps, which are kept as `isi_taps_fitted`. In fingerprint mode the shift is drawn per run from a Gaussian at the fleet spread, instead of picking one of the fleet's tap sets.
- The BB60 receive filter's passband is flattened when the data folder's fits already contain it (the fitted taps absorbed its +0.4 dB rise at 500 kHz, so it was counted twice). The measured filter is used when the fits were made with the passband divided out (`_bb60_equalised` in the ripple file).
- **Shipped data refitted.** `data/` now holds all 24 radios, each from its first capture session only (fits and per-run measurements from the same session), fitted with the BB60 passband divided out of the captures, so the generator uses the measured receive filter. Not reproducible by a flag: the previous `data/` is in the repository history.
- `data/fitted_blocks_30BF779_89_433.npz` and the README result figures rebuilt from the new data.

### Added
- `--full-transient`, `--raw-taps`, `--bb60-passband`.
- `transient_level_deg` and `isi_shift_khz` as variation-spec knobs.
- `gauss` variations take an optional absolute centre `mu`.

### Fixed
- Per-symbol ISI was about 20% too small, and the gain-89, 2400 MHz amplitude distribution had lost its two peaks: the receive-filter double counting above.
- `analysis_scripts/fit_pn_curve.py` combines runs by median. One broken run measurement (symbol timing near its wrap) had set 30BF779/89_2400's curve about 3x too high in variance.
- The CFO drift margin counted only 22 of 24 radios (two sessions carry the wrong `tx_serial`); it now reads the radio from the session key (0.050 -> 0.047 ppm).
- Bundled library: `_sinc_interp` works in chunks, about 1.8x faster, bit-identical output.
- `requirements.txt` now lists `pyfftw`, which `cel_signal_gen_lib/impairments/hardware.py` imports. A clean environment could not import the generator without it.

## 2026-10-03

**Default output changes.** Previous output: `--per-radio-nuisance --gate-pa-2400`.

### Changed
- Variation is now fingerprint-only. Carrier/clock offset, TX LO leakage, PA cubic and phase-noise level stay per radio. ISI taps, IQ imbalance and receiver DC are drawn from the whole fleet each run.
- SNR is drawn per run over a wider range, reaching 10 dB below the fleet's measured values.
- The carrier offset's run-to-run spread includes the reference drift measured over a capture session (fleet 90th percentile), and runs are drawn independently.
- The PA cubic is now applied at 89_2400.

### Added
- `--per-radio-nuisance`, `--gate-pa-2400`, `--snr-drop-db`.
- Variation kinds `quantile` (empirical distribution) and `pool` (one of a list of fitted vectors).
- `isi_taps_pool_index` in the ground truth.

## 2026-10-02

**No change to the default output.**

### Added
- `SG_DATA_DIR`: point the generator at an alternative data folder.

### Changed
- The precise CFO is used only when it covers every run of a session.
- Generator code tidied: one cached loader for the data files; the settling route is resolved once.
- Notebooks updated for the 2026-10-01 leakage change; `paper_figures` uses 30BF779 throughout.
- `fig_input_distributions.py` takes its values from the generator.

## 2026-10-01

**Default output changes: LO leakage and CFO centre.** Previous output: `--old-leakage`.

### Changed
- TX LO leakage level is fitted per session to a re-measurement (`data/lo_leakage.json`). The old values came from a search window that often missed the tone.
- The leakage tone sits at the CFO plus a small per-band offset.
- The CFO distribution uses the precise per-run CFO.

### Added
- `--old-leakage`.

## 2026-09-30

**Default output changes: PA cubic, ISI taps, and the gain-89 PA modulation.** Previous output: `--no-pa-clean --no-pa-mod`.

### Changed
- Clean PA refit (`pa_clean` block in `data/isi_taps.json`) is on by default for 124 profiles.
- Gain-89 PA modulation is on by default.

### Fixed
- The PA modulation's cubic part now acts on the data portion only and no longer double-counts the gain change.
- The Hammerstein guard no longer blocks profiles that use `ripple_fit` (e.g. 30BF795/89_2400).

### Added
- `--no-pa-clean`, `--no-pa-mod`.

## 2026-09-29

**No change to the default output.**

### Added
- `--pa-clean`: PA cubic and ISI taps refitted through the generator's own chain.
- `--pa-mod`: the gain-89 PA modulation (`data/pa_gain_mod.json`).

## 2026-09-28

**Default output changes: the preamble only.** Previous output: `--cubic-preamble`.

### Changed
- The burst transient over the preamble is now measured (`data/srrc_preamble_transient.json`) instead of extrapolated from the data-portion fit.

### Added
- `--cubic-preamble`.
- `analysis_scripts/measure_preamble_transient.py`.
- Bundled library: `add_pa_nonlinearity` gains `model='cubic'` and `model='rapp'`; `add_decimation_filter` gains `fir=`.

### Changed (bundled library)
- `add_sampling_clock_drift` uses a windowed-sinc resampler by default; `method='linear'` gives the old output. The generator does not call it.

## 2026-09-27

**Default output changes: phase noise.** Previous output: `--per-burst-pn`.

### Changed
- Phase noise is one jointly fitted curve per radio and config (`data/pn_curves.json`), drawn as one continuous process per run.

### Added
- `analysis_scripts/fit_pn_curve.py`.
- `--per-burst-pn`.

### Removed
- `data/pn_close_in_masks.json` and `analysis_scripts/calibrate_close_in.py`.

## 2026-09-26

**Default output changes: phase noise.**

### Changed
- In-burst phase noise refitted on captures with the receiver spur removed.

### Added
- `--rx-spur`: the BB60 receiver's phase spur (`data/rx_spur.json`), off by default.
- `--continuous-pn` (became the default on 2026-09-27).
- `rx_spur.py`: estimate, remove or add the receiver spur.
- `analysis_scripts/continuous_pn_roundtrip.py`.
