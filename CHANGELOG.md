# Changelog

Newest first. Each entry says whether the generator's **default output** changes. Where it does, a flag reproduces the previous output bit for bit, so older datasets can always be regenerated.

### 2026-10-02: tidier generator code, updated notebooks, and a separate data folder

**No change to the default output.** Same seed, same captures: checked bit for bit on 16 cases (four configs, with the default and each opt-out flag).

*   **Generator code.**
    *   The seven versioned data files load through one cached loader instead of seven copies of the same checks.
    *   The settling route (ripple, Hammerstein or cubic) is resolved once, in that order, before its requirements are checked.
    *   The PA-modulation phasor is computed once per burst for both of its parts.
*   **The precise CFO is used only where it covers every run of a session.** A session re-measured on every 4th run only (the `D:/repeat` sessions) previously mixed precise and x⁴ values; it now keeps the x⁴ CFO throughout, and says so. No shipped profile is affected.
*   **`SG_DATA_DIR`** points the generator at a whole alternative data folder, for example one refitted on a subset of capture sessions so that another can serve as a held-out test set. Each file's own `SG_*_FILE` override still applies on top.
*   **Notebooks.**
    *   `getting_started` and `paper_figures` no longer describe the leakage as bimodal (see 2026-10-01).
    *   `estimators` §10 gains a subsection showing why the null-tail leakage estimator misses the tone on real captures, and how the long-term carrier offset fixes it.
    *   `build_your_own_radio` lists the optional profile blocks (continuous phase noise, preamble transient, PA modulation, leakage tone offset, receiver spur, ripple) and compares the per-burst and continuous phase-noise models.
    *   `paper_figures` now uses 30BF779 for every figure.
*   **`analysis_scripts/fig_input_distributions.py`** takes every curve and sample from the generator itself: the profile, the variation spec, the re-measured leakage and the precise CFO. With the precise CFO the clock-CFO offset is +0.0001 ppm, not the +0.0057 ppm the x⁴ estimate implied.

### 2026-10-01: measured TX LO leakage and precise CFO

**Changes to the default output: LO leakage level and frequency, and the CFO centre.** `--old-leakage` reproduces the previous output bit for bit.

*   **What was wrong.** The log's leakage was read within ±5 Hz of the x⁴ CFO. That CFO is biased by about −7 ppb (−3 / −6.5 / −15 to −18 Hz at 433 / 915 / 2400), so at 915 and 2400 the search missed the tone. At 2400 it read 25–30 dB low; at 915 it flipped between two levels, which the generator modelled as a mixture.
*   **The re-measurement** covers 17,000 runs (`analysis_scripts/remeasure_lo_leakage.py`; table built by `build_lo_leakage_table.py`; data in `data/lo_leakage.json`):
    *   The tone sits at the precise CFO plus a fleet constant per band: −0.54 / −0.60 / −1.19 Hz, with p5–p95 within ±0.05 Hz.
    *   Runs are independent and unrelated to the CFO.
    *   Gain 77 has one level per radio: within-session sd 0.6 dB, sessions agree to 0.3 dB, radios span −40.7 to −51.5 dBc.
    *   Gain 89 sits about 14 dB lower. It is uncorrelated with the gain-77 level, and some sessions show two states 10–17 dB apart (10/46 at 433, 3/45 at 915, none at 2400).
*   **Now:**
    *   the level's distribution is fitted per session to the re-measured runs, like every other measured block. It is a robust Gaussian, or the two-component mixture where BIC clearly prefers one. That covers the gain-89 sessions with two states; at gain 77 a mixture, where picked, only models skew or tails (components 0.2–2 dB apart);
    *   the tone is placed at the CFO plus the band offset, with its phase continuous across bursts;
    *   the CFO distribution uses the precise per-run CFO in place of the log's x⁴ values.
    *   The clock mismatch is unchanged: it is derived, and its offset to the CFO absorbs the bias.
*   Sessions without a table entry keep the log's values, and the generator says so.

### 2026-09-30: two fixes from the full-chain pilot (`analysis_scripts/pilot_full_chain.py`)

*   **The PA modulation's cubic change now acts on the data portion only, and contributes only its shape.**
    *   Previously it went through the PA for the whole burst, preamble included, which doubled the preamble phase wander at 89_433 (2.6–3.0° against real 1.1–1.8°).
    *   On the data portion alone it also shifted the data's mean complex gain, which the gain step already carries in full. That counted the line twice (8.3 % against real 4.9 %).
    *   Its mean gain relative to the unmodulated output is now projected out, as the lock was measured. Result: line −2.14 Hz at 4.8 % (real 4.7–5.1 %) at 89_433, and 1.5 % (real 1.3–1.7 %) at 89_915.
    *   Changes the default output at 89_433 / 89_915 only; everything else is bit-identical.
*   **The Hammerstein-fit guard no longer blocks profiles that use `ripple_fit`.** It raised before the ripple route, which discards the Hammerstein fit anyway, so profiles such as 30BF795/89_2400 could not be generated at all. No output changes for any profile that ran before.

### 2026-09-30: clean PA refit on by default, for the whole fleet

**Changes to the default output: PA cubic and ISI taps.** `--pa-clean` (below) is now on by default. `--no-pa-clean` reproduces the previous output bit for bit (checked at 89_433, 89_915 and 77_2400; no other drawn parameter changes).

*   `data/isi_taps.json` now has a `pa_clean` block for 124 profiles. 122 were refitted by `analysis_scripts/refit_pa_clean_fleet.py`: every radio with two sessions on disk, except 77_915 where a session lacks it. Each is fitted on the same session and runs as `ripple_fit`, with its settling held.
*   **Same session:** 2–6 dB closer than `ripple_fit`, with the full gain-89 AM/PM span (61 % / 79 % → 105 % / 100 % at 89_433 / 89_915).
*   **The radio's other session:** +1.3 dB at 433 MHz, unchanged at 915 MHz, +0.4–0.5 dB at 2400 MHz, and never worse by more than 0.34 dB. Every block records its held-out scores.
*   Profiles without a block (one-session radios: 30ECB6B's gain-77 and 2400 configs) keep `ripple_fit`, and the generator says so.

### 2026-09-30: PA modulation on by default

**Changes to the default output at gain 89, 433 / 915 MHz only.** `--pa-mod` (below) is now on by default: every real capture at those configs carries it. `--no-pa-mod` reproduces the previous output bit for bit (checked at 77_433, 89_2400, 89_433 and 89_915; no other drawn parameter changes).

### 2026-09-29: PA at gain 89 (both new blocks opt-in; default output unchanged)

*   **`--pa-clean`**: PA cubic + ISI taps refitted through the generator's own chain (settling → PA → ISI → ripple FIR) instead of the joint least squares, which does not pass the cubic through the taps. On held-out runs this gets 90–107 % of the real AM/PM span against 46–91 % and is up to 2.6 dB closer. It uses a `pa_clean` block in `data/isi_taps.json`, written by `analysis_scripts/refit_pa_clean.py`; only 30ECB6B at 89_433 / 89_915 has one so far.
*   **`--pa-mod`**: the gain-89 modulation that every real capture carries. The data portion's complex gain relative to the preamble rotates at −2.15 Hz (433) / −2.38 Hz (915) at the burst rate, ±5 % / ±1.6 %, with a locked modulation of the cubic. It is fleet-common and transmit-side (`data/pa_gain_mod.json`), and in the round trip the line is reproduced to within 0.5 % (433). This is what looked like a random TX distortion at gain 89.
*   Also: the gain-89 compression is the B210 PA, not the BB60 front end. It was unchanged when ~20 dB less reached the receiver (`analysis_scripts/gain_experiment.py`).

### 2026-09-28: measured burst transient in the preamble

**Changes to the default output: the preamble only.** The data portion, every drawn parameter and every noise draw are unchanged; `--cubic-preamble` reproduces the previous output bit for bit (checked on three profiles).

*   The fitted transient cubic was extrapolated over the preamble, where it was never fitted. Measured with one estimator on real and synthetic captures (bursts aligned on their data portion, so the preamble's offset is kept), the preamble offset relative to the data was 1.46 / 3.16 / 8.13° synthetic against 1.01 / 1.94 / 5.22° real at 77_433 / 77_915 / 77_2400, and the first 10 µs of the burst 1.9 / 4.1 / 10.5° against 0.5 / 0.6 / 2.3°. Now 0.91 / 1.86 / 4.94° and 0.33 / 0.55 / 2.06°. Across all six configs the offset is within 0.1–0.3° of real, validated on different runs from the ones the profile was measured on.
*   This matters most for anything that aligns on, or learns from, the preamble.
*   New: `data/srrc_preamble_transient.json` and `analysis_scripts/measure_preamble_transient.py`.
*   Bundled library: `add_pa_nonlinearity` gains the fitted models, `model='cubic'` (complex `b`) and `model='rapp'`, normalised on a chosen slice. `add_decimation_filter` gains `fir=` for a measured receive response; its docstring's "0.75 x rate" BB60 note is corrected to the measured 0.675 x fs. `synth_dataset.py` now calls both instead of private copies; the output is bit-identical (checked on three profiles and every PA branch).
*   Bundled library: `add_sampling_clock_drift` now resamples with a 33-tap Kaiser-windowed sinc by default instead of linear interpolation (error vs an exact resample −83 to −97 dB instead of −23 to −40 dB). `method='linear'` reproduces the old output bit for bit. `synth_dataset.py` does not call it, so the generator output is unchanged.

### 2026-09-27: one phase-noise curve, now the default

**Changes to the default output.** The phase noise is now one curve per radio and config (`data/pn_curves.json`), drawn as a single process across each run. Generating with the same seed gives different phase noise than before; every other drawn parameter and every noise draw is unchanged. `--per-burst-pn` reproduces the previous output bit for bit (checked on three profiles).

*   **One curve, fitted to both measurements at once.** The 2026-09-26 option joined two separately fitted pieces: fleet close-in points below 200 Hz and each radio's straight-line in-burst mask above 3 kHz. The gap between them was interpolated. The real in-burst spectrum turned out not to be a straight line. It falls to about 5 kHz, is flat from about 6 to 50 kHz and rolls off above, so the line ran 3–5 dB high at 4–10 kHz and 3 dB low at 50 kHz. Now one curve (19 points, 1.5 Hz–500 kHz) is fitted to the in-burst deviation and the burst-to-burst phase path together. The expected value of each measurement is computed exactly, including the smoothing by the pulse shape and matched filter, which the old in-burst fit left out (worth 0.5 dB at 100 kHz and 1.9 dB at 200 kHz).
*   **Fit quality (122 profiles fitted from captures, 9 without captures take their config's median):** in-burst residual 0.4–0.7 dB rms (median per config), total in-burst variance within ±0.1 dB for the median profile, path bands within 0.4 dB except on 30BF7C1/89_433 (see Known Limitations).
*   **Round trip through the generator** (`continuous_pn_roundtrip.py`, 4 runs per radio, 6 radios per config): burst-to-burst phase within ±1 dB from 3 to 200 Hz except the 6 Hz band (−0.7 to −2.9 dB); wander 0.89–1.11× real (the old per-burst model: 28–37 dB low at 3 Hz, wander 0.06–0.19×). In-burst spectrum from 1 to 10 kHz within ±1 dB on 22 of 29 radio/configs and within ±1.6 dB on 27 (old model: +2 to +5 dB at 4–10 kHz, and −3 to −7 dB at 1–2 kHz at 2400). In-burst phase-noise rms 0.94–1.04× real, except 0.82–0.90× on the three profiles named in Known Limitations and on 30BF795/77_2400, where the fleet-mean level rule sets the curve 0.7 dB below that radio.
*   **Removed:** `data/pn_close_in_masks.json` and `analysis_scripts/calibrate_close_in.py` (both added 2026-09-26, never released). The fleet path targets now live in `data/pn_curves.json`.

**New tools:**

*   `analysis_scripts/fit_pn_curve.py`: fits all curves in about a minute from deviation spectra written by `fit_isi_taps.py` (set `SG_PN_SPECTRUM_OUT`, with `SG_DESPUR=1`).
*   `analysis_scripts/continuous_pn_roundtrip.py`: now checks both views, the burst-to-burst path and the in-burst spectrum, against the targets stored with each curve.

### 2026-09-26: phase noise and the receiver spur

**Changes to the default output.** Generating with the same seed as before now gives different phase noise. Nothing else changes.

*   **In-burst phase noise refitted without the receiver spur.** Every real capture carries a small phase modulation from the BB60 receiver (below). It fell in the lowest bins of the in-burst phase-noise fit, making the fitted line too steep and too high. All 131 profiles in `data/isi_taps.json` were refitted on captures with the spur removed. 122 were refitted directly; 9 whose captures were not available got the median correction for their config, which is noted in each profile's `pn_rx_spur` field. Typical change: slope shallower by 1.5 / 2.9 / 4.6 dB per decade and level at 10 kHz lower by 1.4 / 2.6 / 4.2 dB, at 433 / 915 / 2400 MHz. The burst transient and the other fitted blocks are unaffected.

**New, optional (both off by default):**

*   **`--continuous-pn`: one phase-noise curve, one draw per run.** *(Superseded 2026-09-27: now the default, with a jointly fitted curve.)* It joins close-in points (1.5–200 Hz, same for every radio, `data/pn_close_in_masks.json`) to each radio's in-burst curve (3 kHz up) and draws the result as one continuous process across the run, so consecutive bursts share the slow phase wander that real captures have. Without it, the synthetic between-burst phase is 25–37 dB too low at 3 Hz and its wander is 0.06–0.25× real. With it, it is within ±1.5 dB from 3 to 200 Hz and 0.84–1.21× real. The in-burst phase-noise spread is also closer to real: within −8% to +13%. `--dry-run` prints the combined curve.
*   **`--rx-spur`: the BB60 receiver's phase spur.** It is a phase modulation repeating every 1168 samples at 5 MS/s (4280.82 Hz), plus its 2nd harmonic: 0.26° peak at 433 MHz, 0.56° at 915 and 1.4° at 2400. It comes from the receiver, not the radios: its phase is fixed to the capture start (identical on captures made days apart) while the transmitted bursts land at random positions, and its level changes between capture sessions without following the radio. `--rx-spur` adds it as real captures carry it, with one level and phase per config (`data/rx_spur.json`).

**New tools:**

*   `rx_spur.py`: estimate, remove or add the receiver spur on any capture.
*   `analysis_scripts/calibrate_close_in.py` *(removed 2026-09-27; `fit_pn_curve.py` replaces it)*.
*   `analysis_scripts/continuous_pn_roundtrip.py`: checks generator output against the stored real measurement. Use about 20 runs per config.

**Reproducibility:** with both new flags off, the output depends only on the refitted `data/isi_taps.json`. Turning either flag on changes nothing but the phase: every other drawn parameter and every noise draw stays identical.