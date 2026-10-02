# Synthetic RF Fingerprint Generator

**Note: Claude (Anthropic) assisted with coding, analysis, and documentation formatting. Measurements, modeling, and validation are original author work.**

Generates synthetic SRRC-QPSK captures in SigMF format from measured profiles of 23 real USRP B210 transmitters. Hardware impairments are fitted from real captures; device-specific traits stay fixed while run-specific variations are redrawn. Output matches the real capture layout, so the same analysis code reads real and synthetic data.

Sample dataset (full set pending a storage solution): https://drive.google.com/drive/folders/1A3LalT3Ojt_YU07PwynkjhlF8aStcJZV?usp=sharing

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+ (tested on 3.13.5). The generator needs only `numpy` and `scipy`; the notebooks also need `matplotlib` and `notebook`.

## Tutorials

1. **`getting_started.ipynb`** — generate from a real device profile: fixed vs. varying parameters, constellations, writing a dataset. Under a minute.
2. **`build_your_own_radio.ipynb`** — build a transmitter by hand, for sweeps, controlled fleets and stress cases.
3. **`estimators.ipynb`** — the reverse: recover each impairment from a capture and score it against the injected truth.
4. **`paper_figures.ipynb`** — reproduce the paper's measurement figures. Under a minute.
5. **`wifi_reconstruction.ipynb`** — rebuild a real 802.11g capture block by block from fitted impairments (the measurement side, not the generator).

The figure scripts in `analysis_scripts/` also run on their own; `fig_wifi_reconstruction.py` is the one no notebook calls.

## Quickstart

```bash
# preview the profile and variation spec
python synth_dataset.py --radio 30BF7B6 --config 77_433 --dry-run

# generate 2 runs of 6 bursts
python synth_dataset.py --radio 30BF7B6 --config 77_433 --runs 2 --bursts 6 --out ./out
```

`--config` is `<gain>_<band MHz>`: gain 77 or 89, band 433, 915 or 2400.

| Flag | Effect |
| :--- | :--- |
| `--rx-spur` | Add the BB60 receiver's phase spur, which real captures carry (off by default). |
| `--per-burst-pn` | Previous phase-noise model: an independent draw per burst. |
| `--cubic-preamble` | Previous burst transient: the fitted cubic extrapolated over the preamble. |
| `--no-pa-clean` | Previous PA fit (joint least squares). |
| `--no-pa-mod` | Leave out the gain-89 PA modulation. |
| `--old-leakage` | Previous TX LO leakage and carrier offset. |

The opt-out flags reproduce earlier output bit for bit; [`CHANGELOG.md`](CHANGELOG.md) says which flag goes with which version.

### Output layout

```text
out/77_433/
  profile.json              # device profile and variation spec
  ground_truth.json         # exact per-run values of every varying parameter
  run_001/valid/
    *_tx_*.sigmf-data/meta  # transmitted waveform
    capture_*_combined.*    # synthetic receive capture
```

`ground_truth.json` lets you score an estimator against the injected value rather than against another estimator.

## Signal chain

<p align="center"><img src="figures/signal_chain.png" width="520" alt="Signal chain: transmit path (phase noise, PA, ISI, burst transient, IQ imbalance, TX leakage) then receive path (CFO, RX DC, sampling clock, AWGN, BB60 filter, ADC floor)"></p>

Fitted blocks are held fixed per radio; measured blocks are redrawn each run from that radio's measured spread; constant blocks belong to the receiver. Not drawn: the gain-89 PA modulation (inside the PA), the measured preamble transient (inside the burst transient), the leakage tone's small offset from the CFO, and the receiver spur (`--rx-spur`).

| Data file | What it sets |
| :--- | :--- |
| `data/radio_characterisation.json` | Per-run measurements (clock, IQ, SNR, ...) that set each parameter's centre and spread. |
| `data/isi_taps.json` | Per radio/config: ISI taps, PA cubic and burst transient. |
| `data/srrc_ripple_per_config.json` | Common-mode band ripple, one FIR per config. |
| `data/pn_curves.json` | Phase noise: one curve per radio/config (1.5 Hz–500 kHz), drawn as one process per run. |
| `data/srrc_preamble_transient.json` | Measured burst transient over the preamble, per config. |
| `data/pa_gain_mod.json` | Gain-89 PA modulation, fleet-wide. |
| `data/lo_leakage.json` | TX LO leakage level and precise carrier offset, per run. |
| `data/rx_spur.json` | Receiver phase spur, used only with `--rx-spur`. |
| `data/bb60_rx_fir_5msps_n10.npy` | Measured BB60C anti-alias response. |
| `data/repeat_log.json`, `data/fitted_blocks_30BF779_89_433.npz` | Data for the cross-session and fidelity figures. |

Each fitted file records the measurements it came from and how. `rx_spur.py` estimates, removes or adds the receiver spur on any capture.

## Results

One radio, 30BF7B6, in all six configurations: a real capture against the generator's default output, demodulated the same way. The real capture has the receiver spur removed, since the generator leaves it out by default.

**Spectrum.** The in-band shape, the noise floor and, at gain 89 (433 and 915 MHz), the PA's spectral regrowth all line up.

<p align="center"><img src="figures/results_30BF7B6_psd.png" width="760" alt="30BF7B6 power spectrum, real against synthetic, six configurations"></p>

**Constellation.** The synthetic clusters reproduce the real shapes, including the elongated gain-89 clusters.

<p align="center"><img src="figures/results_30BF7B6_constellation.png" width="760" alt="30BF7B6 (+,+) constellation cluster, real against synthetic, six configurations"></p>

**Per-symbol deviation.** The tangential and radial spreads are within about 15 % of real in every configuration. The synthetic constellation is slightly tighter in most of them, most at 89 / 433 MHz (2.9° against 3.5° tangential), which is the gain-89 shortfall under Known limitations.

<p align="center"><img src="figures/results_30BF7B6_deviation.png" width="760" alt="30BF7B6 per-symbol tangential deviation, real against synthetic, six configurations"></p>

The figure comes from `analysis_scripts/fig_readme_results.py` in the measurement repository; the real captures are not shipped.

## Variation kinds

Each parameter in the variation spec has a `kind`:

| Kind | Redrawn each run as |
| :--- | :--- |
| `fixed` | Not redrawn: held at the profile value. |
| `derived` | Tied to another parameter (the sampling clock follows the reference oscillator). |
| `uniform` | Uniform over a range (phases, sampling-grid offset). |
| `gauss` | Gaussian at the measured spread, optionally bounded. |
| `ar1` | Gaussian that wanders across runs (memory set by `phi`). |
| `mixture` | Gaussian mixture (e.g. TX LO leakage in gain-89 sessions with two states). |
| `scipy` | Any `scipy.stats` distribution, by name. |

## Known limitations

- **Gain 89 is slightly too clean.** The burst-to-burst deviation is about 2 dB below real, and the AM/AM droop is off by 0.1–0.35 dB.
- **In-burst frequency pull is not modelled.** While transmitting data, real radios sit about 7 ppb below their long-term frequency.
- **Phase noise above ~200 kHz** is below the thermal noise at 2400 MHz and on some 89_433 radios, so there the curve is set by the total in-burst variance and a no-rise constraint.
- **A few profiles carry wideband tangential noise that is not LO phase noise** (30BF7C1/89_433, and milder 30ECB71/89_915); one curve reproduces 0.82–0.90× of it.
- **ISI is three fixed taps**, so amplitude states are slightly sharper than on real hardware.
- **Two-state leakage at gain 89** is fitted as a mixture; its mechanism is unresolved.
- **`clock_phase_mean_samp`** in the log is a wrapping artifact, not a radio property.

## Bundled library

`PA_modelling_with_GMP/cel_signal_gen_lib` is a trimmed copy of the impairment and filter-design library.

**Licensing:** `core/filter_design.py` includes SRRC design code by Matt @ WaveWalkerDSP.com (Copyright 2021), released under the MIT license. This notice must be retained in any redistribution.

## Update log

See [`CHANGELOG.md`](CHANGELOG.md).
