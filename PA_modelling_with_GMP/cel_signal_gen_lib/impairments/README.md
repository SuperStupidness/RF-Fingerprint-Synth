# Hardware & Channel Impairments

> **AI-generated code notice**
> The functions in this folder were written by **Claude** (Anthropic) in a prior session,
> in collaboration with the repository owner. The docstrings were also AI-assisted.
> All functions were reviewed and validated by the owner before use.

---

## What it does

Applies physical hardware imperfections and channel effects to a complex baseband signal
(`np.ndarray` of complex64/128). Each function takes a signal in, returns an impaired signal out.
They are designed to be composable — you can chain them in any order.

For the physically correct ordering when building a transmitter simulation,
see `cel_signal_gen_lib/generators/README.md`.

---

## `hardware.py` — Transmitter impairments

| Function | What it models |
|---|---|
| `add_cfo(signal, cfo)` | **Carrier frequency offset** — static LO crystal error. `cfo` is a fraction of sample rate (e.g. `50e-6` for 50 ppm). |
| `add_iq_imbalance(signal, amp_imbalance_db, phase_imbalance_deg)` | **IQ mismatch** — amplitude and phase difference between the I and Q mixer paths. Creates a ghost "image" signal. |
| `add_phase_noise(signal, phase_noise_std_dev, rng)` | **LO phase noise** — random phase drift modelled as a Wiener process. `phase_noise_std_dev` controls the oscillator quality (larger = noisier). |
| `add_carrier_frequency_drift(signal, cfo, max_drift, drift_type, rng)` | **LO thermal drift** — slow frequency wander within a burst using an Ornstein-Uhlenbeck process. |
| `add_sampling_clock_drift(signal, drift_ppm, max_drift_ppm, drift_type, rng)` | **ADC clock offset/drift** — resamples the signal to simulate a crystal frequency error. Supports constant offset or OU random walk. |
| `add_sampling_clock_jitter(signal, jitter_std, rng)` | **ADC timing jitter** — per-sample random timing error around each sample point. |
| `add_pa_nonlinearity(signal, model, input_backoff_db)` | **Power amplifier distortion** — AM/AM and AM/PM compression. Supports `'saleh'` and `'rapp'` models. `input_backoff_db` sets how far below saturation the PA operates. |
| `add_quantization(signal, n_bits)` | **ADC quantization** — uniform quantization to `n_bits`. |
| `add_spectral_inversion(signal)` | **Spectral inversion** — flips the spectrum (conjugates the signal). |
| `add_symbol_clock_phase(signal, sps, tau)` | **Symbol clock phase offset** — shifts the sampling phase relative to the symbol grid. |

---

## `channel.py` — Channel impairments

| Function | What it models |
|---|---|
| `add_awgn(signal, noise_power)` | **AWGN** — additive white Gaussian noise with a fixed noise power. |
| `add_awgn_snr(signal, desired_snr, rng)` | **AWGN at a target SNR** — automatically scales noise power to achieve the desired SNR in dB. Preferred over `add_awgn` for most use cases. |

---

## `nonlinear.py` — PA model fitting & application

Lower-level functions for fitting and applying PA behavioural models from measured data.
Useful if you have real PA measurements and want to use them instead of the parametric models in `hardware.py`.

| Function | What it does |
|---|---|
| `apply_saleh(signal, params)` | Apply a Saleh PA model with given parameters |
| `apply_rapp(signal, params)` | Apply a Rapp PA model |
| `apply_polynomial(signal, params)` | Apply a polynomial PA model |
| `apply_gmp(signal, params)` | Apply a Generalised Memory Polynomial (GMP) model |
| `apply_volterra(signal, params)` | Apply a Volterra series model |
| `fit_pa_saleh(input_amp, output_amp, output_phase)` | Fit Saleh model parameters from measured I/O data |
| `fit_pa_taylor(input_signal, output_signal)` | Fit Taylor series coefficients from measured I/O data |

---

## Basic usage example

```python
import numpy as np
from cel_signal_gen_lib.impairments.hardware import (
    add_cfo, add_iq_imbalance, add_phase_noise, add_pa_nonlinearity,
)
from cel_signal_gen_lib.impairments.channel import add_awgn_snr

rng = np.random.default_rng(seed=0)

# Start with your clean baseband signal
signal = my_modulated_signal   # np.ndarray, complex

# Apply in physical order (DAC → IQ → LO → PA → channel)
signal = add_iq_imbalance(signal, amp_imbalance_db=0.3, phase_imbalance_deg=1.5)
signal = add_cfo(signal, cfo=50e-6)                     # 50 ppm
signal = add_phase_noise(signal, phase_noise_std_dev=5e-4, rng=rng)
signal = add_pa_nonlinearity(signal, model='saleh', input_backoff_db=6.0)
signal = add_awgn_snr(signal, desired_snr=30, rng=rng)  # 30 dB SNR
```

> For dataset generation at scale, use `cel_signal_gen_lib/generators/` instead —
> it handles all of this automatically with the correct physical ordering.
