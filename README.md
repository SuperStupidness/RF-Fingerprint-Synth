# Synthetic RF-fingerprinting dataset generator

Generates synthetic SRRC-QPSK captures in the SigMF layout, from device
profiles measured on real hardware. Every impairment in the chain is fitted or
measured from captures of 23 USRP B210 transmitters recorded on a Signal Hound
BB60C — none are invented — and each synthetic run redraws the quantities that
genuinely vary run to run, leaving the device-fixed ones alone.

The output is written in the same directory layout the real capture pipeline
produces, so the same characterisation code reads real and synthetic data
unchanged.

## Install

```bash
pip install -r requirements.txt     # numpy + scipy only
```

Python 3.10+. Verified on 3.13.5 with numpy 2.1.3 and scipy 1.15.3.
The generator itself needs only numpy and scipy; matplotlib and
notebook are for `getting_started.ipynb`.

## Tutorials

Three notebooks, in order.

**`getting_started.ipynb`** — the generator as shipped, driven by profiles
measured from real hardware. Six steps:
loading a measured device profile, the split between what is fixed per
device and what varies per run, generating a run and plotting its
constellation, reading the ground truth, changing how a parameter varies
(including swapping in any `scipy.stats` distribution), and writing a
dataset to disk. It runs end to end in under a minute.

**`build_your_own_radio.ipynb`** — building a transmitter that does not exist,
from a dictionary you write yourself. Covers the full profile schema, a
round-trip check (inject a known carrier offset, measure it back with an
independent estimator, confirm sign and size), switching one impairment at a
time, generating a fleet of devices that differ only in what you choose, and
sweeping a single impairment to make a controlled dataset. Use this when you
need a controlled fleet, a parameter sweep, or an impairment larger than
anything in the measured set.

**`estimators.ipynb`** — the other direction: recovering the impairment
parameters from a capture. These are the estimators the measured profiles
themselves were built with — burst detection and alignment, carrier offset,
sampling clock, the joint ISI/PA least squares, IQ imbalance, the thermal and
phase-noise split, and LO leakage. Each is run on generated data and scored
against the values `synth_run` injected, which is the only check that catches a
convention error: two of the estimators here report the opposite sign to the
injector, and one returns a number that is purely its own noise floor.

## Quickstart

```bash
# show the profile and variation spec without writing anything
python synth_dataset.py --radio 30BF7B6 --config 77_433 --dry-run

# write 2 runs of 6 bursts
python synth_dataset.py --radio 30BF7B6 --config 77_433 \
       --runs 2 --bursts 6 --out ./out
```

`--config` is `<gain>_<band MHz>`: gain 77 or 89, band 433, 915 or 2400.
Run `--dry-run` first — it prints every parameter and how it varies.

### Output layout

```
out/77_433/
  profile.json              the device profile + variation spec actually used
  ground_truth.json         per-run draw of every varying parameter
  run_001/valid/
    *_tx_*.sigmf-data/meta  the transmitted waveform
    capture_*_combined.*    the synthetic receive capture
```

`ground_truth.json` is the point of the generator: it records what was injected
on every run, so an estimator can be scored against truth rather than against
another estimator.

## The chain

Signal order matters and is documented in the module docstring of
`synth_dataset.py`. Briefly: SRRC QPSK burst → phase noise → PA → ISI taps →
burst transient → IQ imbalance → TX LO leakage → CFO → RX DC → sampling clock →
AWGN → receiver anti-alias filter.

Parameters come from two places:

| source | what it fixes |
| --- | --- |
| `isi_taps.json` | ISI taps, PA cubic, burst transient, phase-noise mask — fitted per radio/config from burst-averaged least squares |
| `radio_characterisation.json` | per-run measurements (CFO, clock, IQ, LO leakage, SNR) that set the centres and the run-to-run spreads |
| `srrc_ripple_per_config.json` | common-mode SRRC band ripple, one FIR per config |
| `bb60_rx_fir_5msps_n10.npy` | measured BB60C anti-alias response |

## Variation kinds

Each parameter's `kind` in the variation spec decides how it is redrawn per run:

| kind | meaning |
| --- | --- |
| `fixed` | held at the profile value |
| `derived` | tied to another parameter (the sampling clock follows the reference oscillator) |
| `uniform` | drawn uniformly — used for phases and the sampling-grid offset |
| `gauss` | Gaussian at the measured sd, optionally clamped to measured physical bounds |
| `ar1` | Gaussian that wanders across runs, `phi` setting the run-to-run memory |
| `mixture` | finite Gaussian mixture — TX LO leakage is bimodal on most radio/configs |
| `scipy` | any `scipy.stats` distribution by name plus params |

Specs are plain JSON and are written into every run's `profile.json`, so a
dataset always carries the exact sampling rules that produced it. Adding a new
distribution means adding a branch in `draw()`; the `scipy` kind covers the
~110 continuous distributions without any code change.

## Known limitations

These are measured facts about the current model, not TODOs hidden in a corner.

- **Phase-noise level is a fleet mean.** Per-radio level is not resolvable with
  the current gate statistic, so every radio gets the same level. Slope and
  shape are fitted.
- **Burst transient is not device-specific.** Within-radio run-to-run scatter
  exceeds the between-radio spread, so it is held per configuration and scales
  with carrier frequency.
- **Amplitude fidelity is better than phase.** Against a real capture the full
  chain reproduces about 96 % of the measured amplitude spread but only ~76 %
  of the phase spread; the missing term is a tangential Gaussian of roughly
  1.8 degrees at the compressed gain.
- **The ISI lattice is slightly too sharp.** Three fixed taps give a finite set
  of per-symbol amplitude states, where real hardware is smoother — the taps do
  not vary within a burst.
- **The LO-leakage bimodality is empirical.** The two lobes are reproduced by a
  fitted mixture; the physical mechanism is unresolved, and the per-run draws
  are independent, which may not be how the hardware behaves.
- **`clock_phase_mean_samp` in the characterisation log is a wrapping
  artifact**, not a radio property. The well-defined quantity is the burst-0
  intercept, which is not present in this log.

## Bundled library

`PA_modelling_with_GMP/cel_signal_gen_lib` is trimmed to what this project
uses plus the modulation generators, which are useful if you want something
other than SRRC QPSK:

- `core/filter_design` — SRRC and related pulse shapes (used by the generator)
- `impairments/{channel,hardware}` — the impairment primitives the chain calls
- `modulations/` — PSK, QAM, ASK, FSK and APSK signal generators (needs matplotlib)
- `visualization/{time_domain,constellation}` — required by `modulations`

`core/filter_design.py` contains SRRC design code by Matt @ WaveWalkerDSP.com,
Copyright 2021, released under the MIT licence. That notice is retained in the
file and must stay there in any redistribution.

## Acknowledgement

Claude (Anthropic) was used throughout this project to assist with coding,
analysis and documentation. The measurements, the modelling decisions and the
validation are the authors'.

## Provenance

`radio_characterisation.json` is the measured per-run log for 23 radios × 6
configurations × 100 runs. `isi_taps.json` carries the fitted blocks, each
entry paired with the ripple FIR it was fitted against — the generator refuses
to run if a ripple curve is in scope but the paired fit is missing, because
injecting one without the other double-counts the band shape.
