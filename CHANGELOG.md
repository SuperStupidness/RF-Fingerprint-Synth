# Changelog

Newest first. Where the default output changes, the listed flag reproduces the previous output bit for bit.

## 2026-10-03

**Default output changes.** Previous output: `--per-radio-nuisance --gate-pa-2400`.

### Changed
- Variation is now fingerprint-only. Carrier/clock offset, TX LO leakage, PA cubic and phase-noise level stay per radio. ISI taps, IQ imbalance and receiver DC are drawn from the whole fleet each run.
- SNR is drawn per run over a wider range, reaching 10 dB below the fleet's measured values.
- The carrier offset's run-to-run spread is 3× wider.
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
