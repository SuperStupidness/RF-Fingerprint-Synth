# Synthetic RF Fingerprint Generator

Generates synthetic SRRC-QPSK captures in SigMF format based on measured profiles from 23 real USRP B210 transmitters. 

We are trying to find a solution to store the full dataset. For now here are some samples. Sample dataset link: https://drive.google.com/drive/folders/1A3LalT3Ojt_YU07PwynkjhlF8aStcJZV?usp=sharing.

The generator applies hardware impairments (fitted from real captures) and redraws run-specific variations while keeping device-specific traits fixed. Outputs match real capture directory layouts, allowing your characterization code to read real and synthetic data interchangeably.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ (tested on 3.13.5). The core generator uses only `numpy` and `scipy`. Tutorials require `matplotlib` and `notebook`.

## Tutorials

Three Jupyter notebooks guide you through the system:

1. **`getting_started.ipynb`** — Run the generator using real hardware profiles. Covers loading profiles, splitting fixed vs. varying parameters, plotting constellations, and saving datasets. (Runs in < 1 minute).
2. **`build_your_own_radio.ipynb`** — Create custom transmitters from scratch. Ideal for parameter sweeps, controlled fleets, testing extreme impairments, and verifying estimators.
3. **`estimators.ipynb`** — The reverse process. Runs estimators on synthetic data to recover impairment parameters, scoring them against the injected ground truth to verify accuracy.

## Quickstart

```bash
# Preview the profile and variation spec without writing to disk
python synth_dataset.py --radio 30BF7B6 --config 77_433 --dry-run

# Generate 2 runs of 6 bursts
python synth_dataset.py --radio 30BF7B6 --config 77_433 \
       --runs 2 --bursts 6 --out ./out
```

*Note: `--config` uses the format `<gain>_<band MHz>` (e.g., gain 77 or 89, band 433, 915, or 2400).*

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

**Signal flow:** SRRC QPSK burst → phase noise → PA → ISI taps → burst transient → IQ imbalance → TX LO leakage → CFO → RX DC → sampling clock → AWGN → RX anti-alias filter.

Parameters are sourced from specific files:

| Source File | What It Controls |
| :--- | :--- |
| `isi_taps.json` | ISI taps, PA cubic, burst transient, phase-noise mask (fitted per radio/config via least squares). |
| `radio_characterisation.json` | Per-run measurements (CFO, clock, IQ, LO leakage, SNR) defining centers and spreads. |
| `srrc_ripple_per_config.json` | Common-mode SRRC band ripple (one FIR per config). |
| `bb60_rx_fir_5msps_n10.npy` | Measured BB60C anti-alias response. |

## Variation Kinds

Each parameter in the variation spec (`profile.json`) uses a `kind` rule to determine how it redraws per run:

| Kind | Behavior |
| :--- | --- |
| `fixed` | Held constant at the profile value. |
| `derived` | Tied to another parameter (e.g., sampling clock follows reference oscillator). |
| `uniform` | Drawn uniformly (used for phases and sampling-grid offsets). |
| `gauss` | Gaussian draw at the measured standard deviation (optionally clamped to bounds). |
| `ar1` | Gaussian draw that wanders across runs (run-to-run memory set by `phi`). |
| `mixture` | Finite Gaussian mixture (e.g., used for bimodal TX LO leakage). |
| `scipy` | Any `scipy.stats` distribution by name. |

## Known Limitations

These are measured constraints of the current physical model:

*   **Phase-noise level:** Modeled as a fleet mean. Per-radio levels aren't resolvable with the current gate statistic.
*   **Burst transients:** Held constant per configuration (scaling with carrier frequency) because within-radio scatter exceeds between-radio spread.
*   **Phase fidelity:** The model captures ~96% of measured amplitude spread but only ~76% of phase spread.
*   **ISI lattice:** Uses three fixed taps, resulting in slightly sharper amplitude states than real, smoother hardware.
*   **LO-leakage bimodality:** The two lobes are empirically fitted; the physical mechanism is unresolved and per-run draws are treated independently.
*   **Clock phase mean:** The `clock_phase_mean_samp` in the log is a wrapping artifact, not a true radio property.

## Bundled Library & Provenance

The repository includes a trimmed version of `PA_modelling_with_GMP/cel_signal_gen_lib` containing the impairment primitives and filter designs. 
*   **Licensing Note:** `core/filter_design.py` includes SRRC design code by Matt @ WaveWalkerDSP.com (Copyright 2021) released under the MIT license. This notice must be retained in any redistribution.

**Data Provenance:**
*   `radio_characterisation.json` contains the measured per-run log for 23 radios × 6 configurations × 100 runs. 
*   `isi_taps.json` contains the fitted blocks. The generator requires paired ripple curves to run, preventing double-counting of band shapes.

*Note: Claude (Anthropic) assisted with coding, analysis, and documentation formatting. Measurements, modeling, and validation are original author work.*