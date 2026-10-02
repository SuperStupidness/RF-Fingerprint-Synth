# Synthetic RF Fingerprint Generator

**Note: Claude (Anthropic) assisted with coding, analysis, and documentation formatting. Measurements, modeling, and validation are original author work.**

Generates synthetic SRRC-QPSK captures in SigMF format based on measured profiles from 23 real USRP B210 transmitters. 

We are trying to find a solution to store the full dataset. For now here are some samples. Sample dataset link: https://drive.google.com/drive/folders/1A3LalT3Ojt_YU07PwynkjhlF8aStcJZV?usp=sharing.

The generator applies hardware impairments (fitted from real captures) and redraws run-specific variations while keeping device-specific traits fixed. Outputs match real capture directory layouts, allowing your characterization code to read real and synthetic data interchangeably.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ (tested on 3.13.5). The core generator uses only `numpy` and `scipy`. Tutorials require `matplotlib` and `notebook`.

## Tutorials

Five Jupyter notebooks guide you through the system:

1. **`getting_started.ipynb`** — Run the generator using real hardware profiles. Covers loading profiles, splitting fixed vs. varying parameters, plotting constellations, and saving datasets. (Runs in < 1 minute).
2. **`build_your_own_radio.ipynb`** — Create custom transmitters from scratch. Ideal for parameter sweeps, controlled fleets, testing extreme impairments, and verifying estimators.
3. **`paper_figures.ipynb`** — Reproduce the paper's measurement figures: the radio population across all 23 transmitters, the July-vs-August cross-session replication, the per-run input distributions for a representative device, and the fitted-blocks fidelity check. Runs in under a minute.
4. **`estimators.ipynb`** — The reverse process. Runs estimators on synthetic data to recover impairment parameters, scoring them against the injected ground truth to verify accuracy.
5. **`wifi_reconstruction.ipynb`** — Rebuilds a *real* 802.11g capture from its transmitted waveform plus fitted impairments, block by block, and scores the rebuild against the measured receiver output. The deterministic part reproduces the stored model bit for bit. This is the measurement side rather than the generator: parameters here were fitted from one capture, not sampled from a distribution.

### Standalone figure scripts

`analysis_scripts/` holds the figure scripts the notebooks call, each runnable on its own. One is not covered by the notebooks:

```bash
python analysis_scripts/fig_wifi_reconstruction.py     # -> figures/27_wifi_recon.png / .pdf
```

This is the 802.11g OFDM reconstruction against a real capture — PSD, time domain and constellation, each beside its error. It reads `data/wifi_recon_30BF795_89_2400.npz`, packaged here because the captures it came from are not shipped, and demodulates through the validated receiver in `analysis_scripts/wifi_ofdm.py` (L-LTF detect, LTS channel estimate, zero-forcing equalise, pilot common-phase tracking; cross-checked against the MATLAB WLAN Toolbox to ~1e-8).

Note this figure is about the OFDM **reconstruction chain**, not the SRRC generator that the rest of the repository builds — the two share the impairment model but not the waveform.

## Quickstart

```bash
# Preview the profile and variation spec without writing to disk
python synth_dataset.py --radio 30BF7B6 --config 77_433 --dry-run

# Generate 2 runs of 6 bursts
python synth_dataset.py --radio 30BF7B6 --config 77_433 \
       --runs 2 --bursts 6 --out ./out
```

*Note: `--config` uses the format `<gain>_<band MHz>` (e.g., gain 77 or 89, band 433, 915, or 2400).*

Add `--cubic-preamble` to use the previous burst transient in the preamble (extrapolated, see below); it reproduces pre-2026-09-28 datasets together with `--per-burst-pn`.

Add `--per-burst-pn` to use the previous phase-noise model (one independent draw per burst); it reproduces datasets made before 2026-09-27 exactly. The default is one curve per radio, drawn as a single process across the run (see below).

Add `--rx-spur` to include the BB60 receiver's phase spur, which every real capture carries (see below). It is off by default.

Add `--no-pa-clean`, `--no-pa-mod` or `--old-leakage` to restore the PA fit, the gain-89 PA modulation or the TX LO leakage of earlier versions; each reproduces the previous output bit for bit (see [`CHANGELOG.md`](CHANGELOG.md)).

### Output Layout

```text
out/77_433/
  profile.json              # The device profile and variation spec used
  ground_truth.json         # Per-run exact values of every varying parameter
  run_001/valid/
    *_tx_*.sigmf-data/meta  # Transmitted waveform
    capture_*_combined.*    # Synthetic receive capture
```

`ground_truth.json` allows you to score your estimators against exact injected values rather than other estimators.

## Signal Chain & Parameters

**Signal flow:** SRRC QPSK burst → phase noise → PA → ISI taps → burst transient → IQ imbalance → TX LO leakage → CFO → RX DC → sampling clock → AWGN → [RX spur, opt-in] → RX anti-alias filter.

**Phase noise: one curve per radio and config.** The phase noise is one spectrum from 1.5 Hz to 500 kHz (`data/pn_curves.json`), drawn once per run, so consecutive bursts share the slow LO wander that real captures have (1–8° rms over a run, measured once per burst on the preamble). Each curve is fitted to two real measurements at once: the symbol-by-symbol phase deviation inside the bursts (per radio, 1 kHz up) and the burst-to-burst preamble phase (the same for every radio in a config, 1.5–383 Hz). The fit computes exactly what each measurement would read for a given curve, including the matched filter's smoothing, so no Monte Carlo is involved. The fitted shape falls to about 5 kHz, stays flat from about 6 to 50 kHz and rolls off above, like a PLL loop response. The slow part scales as 20·log10 of the carrier frequency and does not identify the radio; the in-burst level does, where it is resolvable. `--dry-run` prints the curve, `analysis_scripts/fit_pn_curve.py` refits it, and `analysis_scripts/continuous_pn_roundtrip.py` checks generator output against the stored real measurements. `--per-burst-pn` restores the previous model: a straight-line in-burst mask drawn independently for each burst, with no burst-to-burst memory.

**Burst transient in the preamble.** The burst-repetitive phase transient (a bowl over the burst, about 0.8° rms per GHz of carrier) is fitted as a cubic on the data portion only. Applying that cubic over the preamble too extrapolated it: the preamble sat 1.4–1.6× too far from the data in phase, and the burst start 3–4× (12° against 3° real at 89_2400). The preamble part is now a measured profile, one per config (`data/srrc_preamble_transient.json`, from `analysis_scripts/measure_preamble_transient.py`; it barely varies between radios, sd 0.1–0.5°). It is referenced to the data portion exactly as measured and blends into the cubic over the last 100 preamble samples. The data portion is unchanged. `--cubic-preamble` restores the extrapolation.

**Receiver spur (`--rx-spur`, off by default).** Every real capture carries a small phase modulation that repeats every 1168 samples at 5 MS/s (4280.82 Hz), plus its 2nd harmonic: 0.26° peak at 433 MHz, 0.56° at 915, 1.4° at 2400. It comes from the BB60 receiver, not the radios: its phase is fixed to the capture start (identical on captures made days apart), while the transmitted bursts land at random positions; its level changes between capture sessions but does not follow the radio. The generator's fitted parameters were measured on captures with it removed, so by default the output has none. `--rx-spur` adds it back as real captures carry it, with one level and phase per config (`data/rx_spur.json`). `rx_spur.py` has the estimator and the add/remove functions, so you can also measure or remove it on your own captures.

Parameters are sourced from specific files:

| Source File | What It Controls |
| :--- | :--- |
| `data/isi_taps.json` | ISI taps, PA cubic, burst transient (fitted per radio/config via least squares), and the straight-line phase-noise mask used only by `--per-burst-pn`. |
| `data/radio_characterisation.json` | Per-run measurements (CFO, clock, IQ, LO leakage, SNR) defining centers and spreads. |
| `data/srrc_ripple_per_config.json` | Common-mode SRRC band ripple (one FIR per config). |
| `data/pn_curves.json` | Phase noise: one curve per radio/config (1.5 Hz–500 kHz), with the real measurements it was fitted to and its fit residuals. |
| `data/srrc_preamble_transient.json` | Burst-transient phase over the preamble, one measured profile per config. |
| `data/rx_spur.json` | Receiver phase spur (1168-sample period): level and phase per config, used only with `--rx-spur`. |
| `data/lo_leakage.json` | TX LO leakage level and the precise carrier offset, per run (re-measured; replaces the log's values). |
| `data/pa_gain_mod.json` | Gain-89 PA modulation: frequency and size per config, fleet-wide. |
| `data/bb60_rx_fir_5msps_n10.npy` | Measured BB60C anti-alias response. |
| `data/repeat_log.json` | August re-capture of the same fleet, used by the cross-session figure. |
| `data/fitted_blocks_30BF779_89_433.npz` | Extracted per-symbol deviations, real and synthetic, for the fidelity figure (the raw captures it came from are not shipped). |

## Variation Kinds

Each parameter in the variation spec (`profile.json`) uses a `kind` rule to determine how it redraws per run:

| Kind | Behavior |
| :--- | --- |
| `fixed` | Held constant at the profile value. |
| `derived` | Tied to another parameter (e.g., sampling clock follows reference oscillator). |
| `uniform` | Drawn uniformly (used for phases and sampling-grid offsets). |
| `gauss` | Gaussian draw at the measured standard deviation (optionally clamped to bounds). |
| `ar1` | Gaussian draw that wanders across runs (run-to-run memory set by `phi`). |
| `mixture` | Finite Gaussian mixture (e.g., TX LO leakage in the gain-89 sessions that show two states). |
| `scipy` | Any `scipy.stats` distribution by name. |

## Known Limitations

These are measured constraints of the current physical model:

*   **Phase-noise level:** Modeled as a fleet mean. Per-radio levels aren't resolvable with the current gate statistic.
*   **Burst transients:** Held constant per configuration (scaling with carrier frequency) because within-radio scatter exceeds between-radio spread.
*   **Constellation spread at gain 89, 433/915 MHz:** Measured with `radio_characterise.py` (with the 2026-09-26 merged phase-noise curve), the phase spread is 0.95–1.00 of real at 77_433, 77_915 and 89_2400 and 0.87–0.92 at 89_915 and 77_2400, but only 0.61 at 89_433. The amplitude spread is 0.91–0.99 of real, except 0.68 at 89_433 and 0.75 at 89_915. The shortfall is not phase noise. At 89_433 the cubic PA gives about 76% of the real AM/PM pattern, and at gain 89 (433/915) real captures carry a noise-like distortion (about −35 dBc, only while transmitting) that the generator does not model. That distortion is likely a receiver-overload effect of the capture setup.
*   **ISI lattice:** Uses three fixed taps, resulting in slightly sharper amplitude states than real, smoother hardware.
*   **LO-leakage states at gain 89:** Some gain-89 sessions show two leakage levels 10–17 dB apart, which runs switch between at random. They are fitted as a mixture; the mechanism is unresolved. (The apparent bimodality at 915 and 2400 MHz in earlier versions was a measurement error, see [`CHANGELOG.md`](CHANGELOG.md), 2026-10-01.)
*   **Clock phase mean:** The `clock_phase_mean_samp` in the log is a wrapping artifact, not a true radio property.
*   **Phase noise above ~200 kHz:** At 2400 MHz and on some 89_433 radios the in-burst phase noise falls below the thermal noise above about 105–250 kHz, so there the curve is set only by the total in-burst variance and a no-rise constraint.
*   **Wideband tangential noise on a few profiles:** 30BF7C1/89_433 carries a tangential excess that stays flat to 300 kHz, about 10 dB above its fleet peers. It is not LO phase noise (likely the gain-89 distortion above), and one curve cannot reproduce it: its in-burst phase-noise rms comes out 0.82× real. 30ECB71/89_915 is similar but milder (0.90×).

## Bundled Library & Provenance

The repository includes a trimmed version of `PA_modelling_with_GMP/cel_signal_gen_lib` containing the impairment primitives and filter designs. 
*   **Licensing Note:** `core/filter_design.py` includes SRRC design code by Matt @ WaveWalkerDSP.com (Copyright 2021) released under the MIT license. This notice must be retained in any redistribution.

**Data Provenance:**
*   `data/radio_characterisation.json` contains the measured per-run log for 23 radios × 6 configurations × 100 runs. 
*   `data/isi_taps.json` contains the fitted blocks. The generator requires paired ripple curves to run, preventing double-counting of band shapes. Its phase-noise fields are fitted on captures with the receiver spur removed (see [`CHANGELOG.md`](CHANGELOG.md)).
*   `data/pn_curves.json` and `data/rx_spur.json` record the real measurements they were fitted to, and how.

## Update Log

See [`CHANGELOG.md`](CHANGELOG.md). Each entry states whether the default output changed, and which flag restores the previous behaviour.
