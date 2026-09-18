from typing import Optional
from functools import lru_cache
import numpy as np
import pyfftw
import scipy.signal


def apply_analog_filter(signal: np.ndarray, ripple_db: float, cutoff: float) -> np.ndarray:
    """Simulate a 2nd-order Chebyshev Type-1 analog baseband low-pass filter.

    Models a simple active RC-based DAC output smoother / anti-alias filter
    found in real RF modem chipsets. Order and family are fixed; ripple_db and
    cutoff vary per device due to component manufacturing tolerances
    (typically 1–5% for capacitors/resistors).

    N=2 is deliberate — real baseband analog filters are 1st–2nd order.
    Higher orders (e.g. N=5) produce multi-sample group delay that swamps
    the fingerprint and smears constellation clusters.

    Parameters
    ----------
    signal :
        Complex baseband signal.
    ripple_db :
        Peak-to-peak passband ripple (dB). Typical range 0.1–1.0.
    cutoff :
        Normalised cutoff frequency (0.0–1.0, where 1.0 = Nyquist = fs/2).
        Typical value ~0.8 leaves a 20% guard band.
    """
    sos = scipy.signal.cheby1(N=2, rp=ripple_db, Wn=cutoff, btype='lowpass', output='sos')
    # sosfilt: causal one-pass filter — matches real hardware where the RF front-end
    # filter runs before the signal is ever observed. Adds non-linear phase (group delay).
    # To bypass for visualization, set tx_analog_filter_ripple_db=None in the profile.
    return scipy.signal.sosfilt(sos, signal)


def add_dc_offset(signal: np.ndarray, frac: float, rng: np.random.Generator,
                  reference_rms: Optional[float] = None) -> np.ndarray:
    """Add a DC offset to simulate LO self-mixing in a zero-IF receiver.

    The offset magnitude is fixed (a property of the specific receiver hardware),
    but the phase is drawn fresh per burst because the LO leakage path length
    varies with temperature and load.

    Parameters
    ----------
    signal :
        Complex baseband signal.
    frac :
        DC offset magnitude as a fraction of the reference RMS (e.g. 0.02 = 2%).
    rng :
        Per-burst RNG — used only for the random phase draw.
    reference_rms :
        RMS value to scale against. If provided, this is used instead of
        computing from the signal — pass the pre-AWGN RMS so DC offset
        magnitude is independent of SNR.
    """
    rms = reference_rms if reference_rms is not None else float(np.sqrt(np.mean(np.abs(signal) ** 2)))
    phase = rng.uniform(0.0, 2.0 * np.pi)
    return signal + frac * rms * np.exp(1j * phase)


def add_cfo(signal, cfo, phase_offset = 0):
    signal_cfo = signal * np.exp(1j * (2*np.pi*cfo*np.arange(0,len(signal)) + phase_offset))
    return signal_cfo

def add_symbol_clock_phase(
    signal, 
    fractional_offset, 
    method='fft', 
    N_taps=21
):
    """
    Applies a fractional clock offset to a signal using one of two methods.

    Parameters
    ----------
    signal : numpy.ndarray
        The input complex baseband signal.
    fractional_offset : float
        The desired time shift, in fractional samples (e.g., 0.4).
    method : str, optional
        'fft': Uses a frequency-domain phase ramp (default).
               This is a "perfect" shift but has circular (wrap-around)
               edge effects.
        'fir': Uses a time-domain windowed-sinc FIR filter.
               This is an approximation but has more standard 'zero-padded'
               edge effects.
    N_taps : int, optional
        The number of taps for the FIR filter, *only* used if method='fir'.
        Must be an odd number.
    
    Returns
    -------
    numpy.ndarray
        The time-shifted signal.
    """
    
    # No shift, no work
    if fractional_offset == 0:
        return signal

    if method == 'fft':
        # 1. Get length and go to frequency domain
        N = len(signal)
        signal_fft = pyfftw.interfaces.numpy_fft.fft(signal)
        
        # 2. Create the frequency ramp for the shift
        freq_k = np.fft.fftfreq(N)
        phase_ramp = np.exp(-1j * 2 * np.pi * freq_k * fractional_offset)
        
        # 3. Apply shift and return to time domain
        shifted_fft = signal_fft * phase_ramp
        return pyfftw.interfaces.numpy_fft.ifft(shifted_fft)

    elif method == 'fir':
        # 1. Check that N_taps is odd
        if N_taps % 2 == 0:
            raise ValueError("N_taps for 'fir' method must be odd.")
            
        # 2. Create the filter tap indices
        # (e.g., for N=21, n goes from -10, -9, ..., 9, 10)
        n = np.arange(-(N_taps - 1) // 2, N_taps // 2 + 1)
        
        # 3. Create the windowed-sinc filter
        # Note: the 'delay' is the fractional_offset
        h = np.sinc(n - fractional_offset)
        h *= np.hamming(N_taps)
        h /= np.sum(h) # Normalize for unity gain
        
        # 4. Apply the filter using convolution
        # We use 'mode=same' to keep the signal the same length
        return scipy.signal.oaconvolve(signal, h, mode='same')
    
    else:
        raise ValueError(f"Unknown method '{method}'. Must be 'fft' or 'fir'.")

def add_phase_noise(signal, phase_noise_std_dev=None, phase_noise_mask=None,
                    sample_rate=None, rng=np.random.default_rng()):
    """
    Applies phase noise to a complex baseband signal.

    Supports two modes:
    - Wiener process: Simple random walk giving a 1/f² profile.
    - Shaped mask: Arbitrary phase noise profile from an S_phi dB(rad^2/Hz)
                   specification. NOTE this is not dBc/Hz - see below.

    Parameters
    ----------
    signal : numpy.ndarray
        The input complex baseband signal.
    phase_noise_std_dev : float, optional
        Standard deviation (in radians) of the Gaussian step for the
        Wiener process mode. Produces a pure 1/f² phase noise profile.
    phase_noise_mask : list of tuples, optional
        [(offset_freq_hz, level_dB), ...], interpolated linearly in dB against
        log frequency. Requires sample_rate.

        THE LEVEL IS S_phi, dB(rad^2/Hz) - NOT L(f) IN dBc/Hz.

        A datasheet or spectrum analyser quotes L(f), the single-sideband
        dBc/Hz, and the two differ by a factor of two:

            S_phi(f) = 2 L(f)      i.e.   S_phi_dB = L_dB + 3.0102

        Pasting a datasheet mask in directly therefore injects 3.01 dB TOO
        LITTLE phase noise, silently - nothing here can detect the mistake.
        Convert first. fit_phase_noise_mask and fit_phase_noise_pole already
        emit S_phi (they add the 3.0102), so masks from those go straight in.

        Verified end to end: the variance of the generated process equals the
        integral of the mask, sum S_phi(f) df over 0..sample_rate/2, to within
        0.5% on flat and sloped masks, and the shape is reproduced to better
        than 0.7 dB across four decades.

        np.interp CLAMPS outside the mask, so the level is held FLAT below the
        first point and above the last - it does not continue the slope. Put a
        point at sample_rate/2 unless you want the top of the band flat.
    sample_rate : float, optional
        Sample rate in Hz. Required when using phase_noise_mask.
    rng : numpy.random.Generator, optional
        A NumPy random number generator instance.

    Returns
    -------
    numpy.ndarray
        The signal with phase noise applied.
    """
    N = len(signal)

    if phase_noise_std_dev is not None:
        # Wiener process mode (original behavior)
        if phase_noise_std_dev <= 0:
            return signal
        phase_steps = rng.normal(0, phase_noise_std_dev, N)
        phase_noise_process = np.cumsum(phase_steps)

    elif phase_noise_mask is not None:
        if sample_rate is None:
            raise ValueError("sample_rate is required for shaped phase noise.")

        # Sort mask by offset frequency
        mask = sorted(phase_noise_mask, key=lambda x: x[0])
        mask_freqs = np.array([m[0] for m in mask])
        mask_levels_dbc = np.array([m[1] for m in mask])

        # One-sided frequency bins
        freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)
        n_freqs = len(freqs)

        # Interpolate mask in log-frequency space (linear dBc/Hz)
        log_mask_freqs = np.log10(mask_freqs)
        psd_dbc = np.zeros(n_freqs)
        psd_dbc[1:] = np.interp(
            np.log10(freqs[1:]), log_mask_freqs, mask_levels_dbc
        )

        # Convert dBc/Hz to linear PSD
        S_phi = np.zeros(n_freqs)
        S_phi[1:] = 10.0 ** (psd_dbc[1:] / 10.0)

        # Amplitude spectrum for irfft
        # For numpy irfft convention: PSD(f_k) = 2|X[k]|² / (N * fs)
        # So |X[k]| = sqrt(S_phi * N * fs / 2)
        amplitude = np.sqrt(S_phi * N * sample_rate / 2.0)
        amplitude[0] = 0.0  # no DC offset

        # Unit complex Gaussian noise
        noise = (rng.standard_normal(n_freqs) +
                 1j * rng.standard_normal(n_freqs)) / np.sqrt(2.0)
        noise[0] = 0.0
        if N % 2 == 0:
            noise[-1] = noise[-1].real  # Nyquist bin must be real

        # Shape spectrum and transform to time domain
        phase_noise_process = np.fft.irfft(amplitude * noise, n=N)

    else:
        raise ValueError(
            "Must provide either phase_noise_std_dev or phase_noise_mask."
        )

    return signal * np.exp(1j * phase_noise_process)

def add_iq_imbalance(signal, amp_imbalance_db, phase_imbalance_deg):
    """
    Applies IQ imbalance to a complex baseband signal with power preservation.
    
    Parameters:
    - signal: The input complex signal (NumPy array).
    - amp_imbalance_db: Amplitude imbalance in dB. 
                        (e.g., 1 dB means Q branch is 1 dB stronger than I).
    - phase_imbalance_deg: Phase imbalance (quadrature skew) in degrees.
                           (e.g., 5 degrees means Q branch leads I by 5 degrees,
                           resulting in 95° separation instead of 90°).
    
    Returns:
    - imbalanced_signal: Complex signal with IQ imbalance applied, 
                         normalized to preserve average power.
    """
    
    # 1. Convert inputs to linear gain and radians
    # Use symmetric amplitude imbalance
    g_i = 10**(-amp_imbalance_db / 40.0)  # I branch gain
    g_q = 10**(amp_imbalance_db / 40.0)   # Q branch gain
    
    # Phase skew (radians)
    phi = np.deg2rad(phase_imbalance_deg)
    
    # 2. Separate I and Q components
    I = np.real(signal)
    Q = np.imag(signal)
    
    # 3. Apply imbalance with symmetric amplitude scaling
    I_imb = I * g_i
    Q_imb = (I * np.sin(phi) + Q * np.cos(phi)) * g_q
    
    # 4. Calculate power normalization factor
    # The average output power is scaled by (g_i^2 + g_q^2) / 2
    # We must divide the amplitude by sqrt of this to preserve power.
    
    power_scale = np.sqrt((g_i**2 + g_q**2) / 2.0)
    
    # 5. Normalize to preserve power
    I_imb = I_imb / power_scale
    Q_imb = Q_imb / power_scale
    
    # 6. Recombine into a new complex signal
    imbalanced_signal = I_imb + 1j * Q_imb
    
    return imbalanced_signal

def add_amplitude_variance(signal, std_pct, sps=1, rng=np.random.default_rng()):
    """Per-symbol multiplicative amplitude jitter: s *= (1 + a), a ~ N(0, std_pct/100).

    Reproduces the measured constellation "amplitude variance" feature — the
    radial spread of the symbol clusters about their centroid, in percent. The
    jitter is drawn once per symbol and held constant across that symbol's `sps`
    samples (centred on the symbol instant), so it maps closely onto the measured
    amp_var_pct. NOTE: after an RX matched filter the recovered std is ~0.9x the
    injected value (the block-constant hold captures ~88% of the pulse energy for
    the campaign's SRRC β=0.35/span=10) — set the parameter ~10-15% higher to hit
    an exact measured target, or verify against your own MF. `sps=1` makes it a
    per-sample perturbation.

    Distinct from PA nonlinearity (deterministic AM/AM) and AWGN (additive):
    this is a multiplicative, memoryless, per-symbol gain ripple.

    Parameters
    ----------
    signal : np.ndarray
        Complex baseband signal (oversampled at `sps` samples/symbol).
    std_pct : float
        Target radial std as a percentage (e.g. 3.0 = 3%). <=0 or None disables.
    sps : int
        Samples per symbol; the jitter is constant within each symbol block.
    rng : np.random.Generator
        Per-burst RNG.
    """
    if not std_pct or std_pct <= 0:
        return signal
    # center-aligned symbol blocks: sample n belongs to symbol round(n/sps), so the
    # constant hold is centred on each symbol instant (k*sps) — assumes the signal's
    # symbol centres sit at integer multiples of sps (as MultiBurstSource produces).
    idx = (np.arange(len(signal)) + sps // 2) // sps
    a = rng.normal(0.0, std_pct / 100.0, int(idx[-1]) + 1)
    return signal * (1.0 + a[idx])


def add_phase_variance(signal, std_deg, sps=1, rng=np.random.default_rng()):
    """Per-symbol multiplicative phase jitter: s *= exp(1j*theta), theta ~ N(0, std_deg).

    Reproduces the measured "phase variance" feature — the tangential spread of
    the symbol clusters about their centroid, in degrees. Like
    add_amplitude_variance, the jitter is per-symbol (held over `sps` samples,
    centred on the symbol instant); the recovered std after an RX matched filter
    is ~0.9x the injected value (see add_amplitude_variance), so set ~10-15%
    higher to hit an exact measured target.

    Distinct from add_phase_noise: that is a CUMULATIVE Wiener random walk
    (1/f^2 oscillator phase noise); this is I.I.D. per-symbol jitter — a flat,
    memoryless tangential spread (matches the measured platykurtic blob).

    Parameters
    ----------
    signal : np.ndarray
        Complex baseband signal (oversampled at `sps` samples/symbol).
    std_deg : float
        Target tangential std in degrees (e.g. 3.0 = 3 deg). <=0 or None disables.
    sps : int
        Samples per symbol; the jitter is constant within each symbol block.
    rng : np.random.Generator
        Per-burst RNG.
    """
    if not std_deg or std_deg <= 0:
        return signal
    idx = (np.arange(len(signal)) + sps // 2) // sps
    theta = rng.normal(0.0, np.deg2rad(std_deg), int(idx[-1]) + 1)
    return signal * np.exp(1j * theta[idx])


def add_spectral_inversion(signal):
    """
    Applies spectral inversion to a complex baseband signal.

    This flips the spectrum around DC by conjugating the signal,
    swapping upper and lower sidebands.

    Parameters
    ----------
    signal : numpy.ndarray
        The input complex baseband signal.

    Returns
    -------
    numpy.ndarray
        The spectrally inverted signal.
    """
    return np.conj(signal)


@lru_cache(maxsize=32)
def _decimation_taps(fs, bandwidth, stop_db, transition_frac):
    """Kaiser anti-alias taps. Cached: the filter is a device/session constant,
    so designing it per call repeats identical work on every burst."""
    from scipy.signal import kaiserord, firwin
    fcut = bandwidth / 2.0
    width = (transition_frac * fcut) / (fs / 2.0)        # normalized to Nyquist
    numtaps, beta = kaiserord(stop_db, width)
    numtaps |= 1                                          # force odd (symmetric)
    return firwin(numtaps, fcut / (fs / 2.0), window=('kaiser', beta))


def add_decimation_filter(signal, fs, bandwidth, stop_db=100.0, transition_frac=0.12):
    """Model an SDR decimation / anti-alias brick-wall (e.g. the Signal Hound
    BB60 IQ filter).

    Flat passband to +/- bandwidth/2, sharp transition, deep stopband —
    reproduces the "flat noise floor then cliff" the BB60 imprints when
    decimating to the target IQ rate (the BB60 auto bandwidth is 0.75 x the
    decimated rate). Without this, a synthetic capture has no band edge and a
    raw-IQ classifier trivially separates it from real BB60 captures.

    Linear-phase Kaiser FIR; group delay compensated so len(out) == len(in).

    Parameters
    ----------
    signal : np.ndarray      complex baseband sampled at fs
    fs : float               sample rate (Hz)
    bandwidth : float        flat IQ bandwidth (Hz); passband edge = bandwidth/2
    stop_db : float          stopband attenuation (dB)
    transition_frac : float  transition width as a fraction of the passband edge
    """
    from scipy.signal import oaconvolve
    taps = _decimation_taps(fs, bandwidth, stop_db, transition_frac)
    gd = (len(taps) - 1) // 2
    return oaconvolve(signal, taps)[gd:gd + len(signal)].astype(signal.dtype)


def add_quantization(signal, n_bits, full_scale=None):
    """
    Applies uniform mid-tread quantization to a complex baseband signal,
    simulating an ADC with a given bit depth.

    Parameters
    ----------
    signal : numpy.ndarray
        The input complex baseband signal.
    n_bits : int
        Number of quantization bits (e.g., 8, 10, 12, 16).
    full_scale : float, optional
        The full-scale amplitude of the quantizer. Values outside
        [-full_scale, full_scale] are clipped. If None, uses the
        peak amplitude of the signal (no clipping occurs).

    Returns
    -------
    numpy.ndarray
        The quantized complex signal.
    """
    n_levels = 2 ** n_bits

    # Default full-scale to peak amplitude
    if full_scale is None:
        full_scale = max(np.max(np.abs(np.real(signal))),
                         np.max(np.abs(np.imag(signal))))

    # Step size
    step = (2 * full_scale) / n_levels

    # Separate, clip, and quantize I and Q independently
    I = np.clip(np.real(signal), -full_scale, full_scale)
    Q = np.clip(np.imag(signal), -full_scale, full_scale)

    I_q = step * np.floor(I / step + 0.5)
    Q_q = step * np.floor(Q / step + 0.5)

    # Clamp to valid range (prevents boundary rounding overshoot)
    max_val = full_scale - step / 2
    I_q = np.clip(I_q, -max_val, max_val)
    Q_q = np.clip(Q_q, -max_val, max_val)

    return I_q + 1j * Q_q

def add_carrier_frequency_drift(signal, cfo=0.0, max_drift=0.001,
                                mean_reversion=0.001, drift_type='random_walk',
                                rng=np.random.default_rng()):
    """
    Applies a time-varying carrier frequency offset to simulate LO drift.

    The random walk mode uses an Ornstein-Uhlenbeck (mean-reverting)
    process centered around a nominal CFO, ensuring the instantaneous
    frequency stays within realistic limits.

    Parameters
    ----------
    signal : numpy.ndarray
        The input complex baseband signal.
    cfo : float, optional
        Nominal carrier frequency offset in normalized frequency
        (cycles/sample). The drift wanders around this value.
        Default is 0.0.
    max_drift : float, optional
        Maximum frequency deviation from cfo in normalized frequency.
        For 'random_walk', used as a 3-sigma bound.
        For 'linear', frequency ramps from cfo to cfo + max_drift.
        Default is 0.001.
    mean_reversion : float, optional
        Strength of the mean-reverting pull (0 to 1). Smaller values
        give slower, more correlated drift. Larger values snap back
        more aggressively. Default is 0.001. Only used for
        'random_walk'.
    drift_type : str, optional
        'random_walk' (default): Ornstein-Uhlenbeck bounded random walk.
        'linear': Frequency ramps linearly from cfo to cfo ± max_drift over
                  the signal duration. The sign is drawn from rng each call
                  so bursts drift both up and down.
    rng : numpy.random.Generator, optional
        Random number generator. Used by 'random_walk' for the OU noise and
        by 'linear' to draw the drift direction (± sign).

    Returns
    -------
    numpy.ndarray
        The signal with carrier frequency drift applied.
    """
    N = len(signal)
    n = np.arange(N)

    if drift_type == 'linear':
        # Frequency ramps from cfo to cfo ± max_drift; sign drawn per burst
        sign = 1 if rng.random() < 0.5 else -1
        inst_freq = cfo + sign * max_drift * n / (N - 1)
        phase = 2 * np.pi * np.cumsum(inst_freq)

    elif drift_type == 'random_walk':
        # Ornstein-Uhlenbeck process centered around cfo
        sigma_target = max_drift / 3.0
        noise_std = sigma_target * np.sqrt(2.0 * mean_reversion)

        white_noise = noise_std * rng.standard_normal(N)
        drift = scipy.signal.lfilter(
            [1.0], [1.0, -(1.0 - mean_reversion)], white_noise
        )

        inst_freq = cfo + drift
        phase = 2 * np.pi * np.cumsum(inst_freq)

    else:
        raise ValueError(
            f"Unknown drift_type '{drift_type}'. Must be 'linear' or 'random_walk'."
        )

    return signal * np.exp(1j * phase)


def add_sampling_clock_drift(signal, drift_ppm=0.0, max_drift_ppm=0.0,
                              mean_reversion=0.001, drift_type='constant',
                              rng=np.random.default_rng()):
    """
    Applies sampling clock drift to simulate an ADC clock rate offset,
    optionally with a wandering clock rate.

    Parameters
    ----------
    signal : numpy.ndarray
        The input complex baseband signal.
    drift_ppm : float, optional
        Constant clock rate offset in parts per million.
        Default is 0.0.
    max_drift_ppm : float, optional
        Maximum additional clock rate deviation in ppm.
        For 'random_walk', used as a 3-sigma bound.
        Default is 0.0.
    mean_reversion : float, optional
        Strength of the mean-reverting pull (0 to 1).
        Default is 0.001. Only used for 'random_walk'.
    drift_type : str, optional
        'constant' (default): steady rate offset set by drift_ppm.
        'random_walk': OU wandering rate bounded by max_drift_ppm.
    rng : numpy.random.Generator, optional
        Random number generator (only used for 'random_walk').

    Returns
    -------
    numpy.ndarray
        The resampled signal with clock drift applied.
    """
    N = len(signal)
    n = np.arange(N, dtype=np.float64)

    nominal_rate = drift_ppm * 1e-6

    if drift_type == 'constant':
        t_drift = n * (1.0 + nominal_rate)

    elif drift_type == 'random_walk':
        max_drift = max_drift_ppm * 1e-6
        sigma_target = max_drift / 3.0
        noise_std = sigma_target * np.sqrt(2.0 * mean_reversion)

        white_noise = noise_std * rng.standard_normal(N)
        rate_drift = scipy.signal.lfilter(
            [1.0], [1.0, -(1.0 - mean_reversion)], white_noise
        )

        inst_rate = 1.0 + nominal_rate + rate_drift
        t_drift = np.concatenate([[0.0], np.cumsum(inst_rate[:-1])])

    else:
        raise ValueError(
            f"Unknown drift_type '{drift_type}'. Must be 'constant' or 'random_walk'."
        )

    # Linear interpolation — fast and vectorized
    result_I = np.interp(t_drift, n, np.real(signal), left=0.0, right=0.0)
    result_Q = np.interp(t_drift, n, np.imag(signal), left=0.0, right=0.0)

    return result_I + 1j * result_Q


def add_sampling_clock_jitter(signal, jitter_std, rng=np.random.default_rng()):
    """
    Applies random per-sample timing jitter to simulate ADC clock
    instability.

    Parameters
    ----------
    signal : numpy.ndarray
        The input complex baseband signal.
    jitter_std : float
        Standard deviation of the timing jitter in samples.
    rng : numpy.random.Generator, optional
        A NumPy random number generator instance.

    Returns
    -------
    numpy.ndarray
        The signal with sampling jitter applied.
    """
    N = len(signal)
    n = np.arange(N, dtype=np.float64)

    t_jittered = n + rng.normal(0, jitter_std, N)
    t_jittered = np.clip(t_jittered, 0, N - 1)

    result_I = np.interp(t_jittered, n, np.real(signal))
    result_Q = np.interp(t_jittered, n, np.imag(signal))

    return result_I + 1j * result_Q

def add_pa_nonlinearity(signal, model='saleh',
                        alpha_a=2.1587, beta_a=1.1517,
                        alpha_p=4.0033, beta_p=9.1040,
                        coeffs=None, input_backoff_db=0.0):
    """
    Applies memoryless power amplifier nonlinearity to a complex
    baseband signal.

    Supports two models:
    - Saleh: Empirical AM/AM and AM/PM model, commonly used for
             traveling wave tube amplifiers (TWTA). Exhibits soft
             saturation.
    - Taylor: Odd-order polynomial model. General purpose, can
              approximate any memoryless nonlinearity. Complex
              coefficients model both AM/AM and AM/PM.

    Parameters
    ----------
    signal : numpy.ndarray
        The input complex baseband signal.
    model : str, optional
        'saleh' (default) or 'taylor'.
    alpha_a : float, optional
        Saleh AM/AM numerator. Default 2.1587 (Saleh 1981).
    beta_a : float, optional
        Saleh AM/AM denominator. Default 1.1517.
    alpha_p : float, optional
        Saleh AM/PM numerator (radians). Default 4.0033.
    beta_p : float, optional
        Saleh AM/PM denominator. Default 9.1040.
    coeffs : list of complex, optional
        Taylor polynomial coefficients [a1, a3, a5, ...] for odd
        orders 1, 3, 5, ... Only odd orders are used because
        even-order products fall out of band at baseband.
        Complex coefficients produce AM/PM distortion.
        Default is [1.0, -0.1, 0.01].
    input_backoff_db : float, optional
        Input back-off in dB. Positive values reduce the input
        amplitude, pushing the PA toward its linear region.
        Default is 0.0.

    Returns
    -------
    numpy.ndarray
        The signal after PA nonlinearity.
    """
    # Apply input back-off
    backoff_linear = 10.0 ** (-input_backoff_db / 20.0)
    x = signal * backoff_linear

    r = np.abs(x)
    theta = np.angle(x)

    if model == 'saleh':
        # AM/AM: output amplitude as a function of input amplitude
        r_out = alpha_a * r / (1.0 + beta_a * r ** 2)

        # AM/PM: additional phase rotation as a function of input amplitude
        phi = alpha_p * r ** 2 / (1.0 + beta_p * r ** 2)

        return r_out * np.exp(1j * (theta + phi))

    elif model == 'taylor':
        if coeffs is None:
            coeffs = [1.0, -0.1, 0.01]

        # y = sum_k a_{2k+1} * |x|^{2k} * x
        # Only odd-order terms: order 1, 3, 5, ...
        y = np.zeros_like(x)
        for k, a_k in enumerate(coeffs):
            y += a_k * (r ** (2 * k)) * x

        return y

    else:
        raise ValueError(
            f"Unknown model '{model}'. Must be 'saleh' or 'taylor'."
        )

def fit_pa_saleh(input_amplitude, output_amplitude, output_phase_shift):
    """
    Fits the Saleh PA model parameters to measured AM/AM and AM/PM data.

    Parameters
    ----------
    input_amplitude : numpy.ndarray
        Measured input envelope amplitudes.
    output_amplitude : numpy.ndarray
        Measured output envelope amplitudes corresponding to each
        input amplitude.
    output_phase_shift : numpy.ndarray
        Measured phase shift (in radians) at each input amplitude.

    Returns
    -------
    dict
        Fitted parameters: alpha_a, beta_a, alpha_p, beta_p.
    """
    from scipy.optimize import curve_fit

    def am_am(r, alpha_a, beta_a):
        return alpha_a * r / (1.0 + beta_a * r ** 2)

    def am_pm(r, alpha_p, beta_p):
        return alpha_p * r ** 2 / (1.0 + beta_p * r ** 2)

    # Fit AM/AM
    popt_am, _ = curve_fit(
        am_am, input_amplitude, output_amplitude,
        p0=[2.0, 1.0], bounds=(0, np.inf)
    )

    # Fit AM/PM
    popt_pm, _ = curve_fit(
        am_pm, input_amplitude, output_phase_shift,
        p0=[4.0, 9.0], bounds=(0, np.inf)
    )

    params = {
        'alpha_a': popt_am[0],
        'beta_a': popt_am[1],
        'alpha_p': popt_pm[0],
        'beta_p': popt_pm[1],
    }

    return params


def fit_pa_taylor(input_signal, output_signal, max_order=5):
    """
    Fits the Taylor polynomial PA model to measured input/output
    complex baseband data.

    Uses least-squares to find the odd-order coefficients that
    best map input to output. Coefficients can be complex to
    capture both AM/AM and AM/PM effects.

    Parameters
    ----------
    input_signal : numpy.ndarray
        Measured complex input signal.
    output_signal : numpy.ndarray
        Measured complex output signal.
    max_order : int, optional
        Maximum polynomial order (must be odd). Default is 5,
        giving coefficients for orders 1, 3, 5.

    Returns
    -------
    dict
        'coeffs': list of complex coefficients [a1, a3, a5, ...].
        'max_order': the max order used.
    """
    if max_order % 2 == 0:
        raise ValueError("max_order must be odd.")

    r = np.abs(input_signal)
    num_coeffs = (max_order + 1) // 2  # orders 1, 3, 5, ...

    # Build basis matrix: column k = |x|^{2k} * x
    # y = a1*x + a3*|x|^2*x + a5*|x|^4*x + ...
    A = np.zeros((len(input_signal), num_coeffs), dtype=complex)
    for k in range(num_coeffs):
        A[:, k] = (r ** (2 * k)) * input_signal

    # Solve via least squares directly in the complex domain
    coeffs, _, _, _ = np.linalg.lstsq(A, output_signal, rcond=None)

    # Clean up near-zero imaginary parts
    coeffs = [c if np.abs(c.imag) > 1e-10 * np.abs(c.real)
              else complex(c.real) for c in coeffs]

    return {
        'coeffs': coeffs,
        'max_order': max_order,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Impairments characterised bottom-up from B210/BB60C captures (30ECB81, 2026-08).
# add_* applies an impairment; fit_* estimates one from a capture.
# Evidence for each choice is in analysis_scripts/bottom_up/.
# ══════════════════════════════════════════════════════════════════════════════


def add_isi_taps(signal, taps, ks=None, sps=1):
    """Apply symbol-spaced channel taps to a sample-rate signal.

    The taps are zero-stuffed to symbol spacing and the output re-centred on the
    main tap, so no group delay is introduced.

    Three taps is the right order for this hardware: stable run to run
    (|mean|/sd 5-33 over 20 runs) and longer sets buy almost nothing, because
    what they would fit is not a function of the symbol pattern at all. See
    add_burst_phase_transient.

    Parameters
    ----------
    signal : numpy.ndarray
        Complex baseband signal at sample rate.
    taps : array_like
        Complex tap weights, one per lag in ks. The main tap carries the
        channel's static rotation; do NOT normalise it away.
    ks : array_like of int, optional
        Tap lags in SYMBOLS. Defaults to centred, -L..+L.
    sps : int
        Samples per symbol.
    """
    taps = np.asarray(taps, dtype=complex)
    if ks is None:
        L = len(taps) // 2
        ks = np.arange(-L, L + 1)
    ks = np.asarray(ks, dtype=int)
    h = np.zeros((ks.max() - ks.min()) * sps + 1, dtype=complex)
    h[(ks - ks.min()) * sps] = taps
    i0 = (0 - ks.min()) * sps
    return scipy.signal.oaconvolve(signal, h, mode='full')[i0:i0 + len(signal)]


def _shift_pad(v, k):
    """Shift by k samples with zero fill - a LINEAR shift, not a circular one.

    np.roll would wrap the tail of the array round to the front, asserting that
    the sample before the first one was the last one. That is false: in a burst
    the data portion is preceded by the preamble, not by its own end. The guard
    in the fits below drops the affected rows either way, so the two give an
    identical answer - but the design matrix should not state something untrue
    about the signal just because nothing currently reads it.
    """
    out = np.zeros_like(v)
    if k > 0:
        out[k:] = v[:-k]
    elif k < 0:
        out[:k] = v[-k:]
    else:
        out[:] = v
    return out


def fit_isi_taps(rx_symbols, tx_symbols, n_taps=3, guard=8):
    """FIT. Symbol-spaced channel taps from burst-averaged symbols.

    Every burst carries the same symbol sequence, so the burst mean keeps the
    data-dependent (ISI) error while averaging the random error down by sqrt(N).

    Parameters
    ----------
    rx_symbols : numpy.ndarray
        Received symbols, either (n_bursts, n_symbols) - averaged internally -
        or an already-averaged (n_symbols,).
    tx_symbols : numpy.ndarray
        Transmitted symbols, same length.
    n_taps : int
        Number of taps (odd).
    guard : int
        Symbols skipped at each end, where the neighbours are unknown.

    Returns
    -------
    dict
        taps, ks, residual_rms.
    """
    y = np.asarray(rx_symbols, dtype=complex)
    if y.ndim == 2:
        y = y.mean(axis=0)
    y = y / np.sqrt(np.mean(np.abs(y) ** 2))
    a = np.asarray(tx_symbols, dtype=complex)
    a = a / np.sqrt(np.mean(np.abs(a) ** 2))
    L = n_taps // 2
    if guard < L:
        raise ValueError(f'guard ({guard}) must be >= n_taps // 2 ({L}); '
                         'otherwise the zero-filled edge rows enter the fit')
    ks = np.arange(-L, L + 1)
    A = np.stack([_shift_pad(a, int(k)) for k in ks], axis=1)
    sl = slice(guard, -guard) if guard else slice(None)
    taps, *_ = np.linalg.lstsq(A[sl], y[sl], rcond=None)
    resid = y[sl] - A[sl] @ taps
    return {'taps': taps, 'ks': ks,
            'residual_rms': float(np.sqrt(np.mean(np.abs(resid) ** 2)))}


def add_burst_frequency_settling(signal, freq_poly_hz, burst_len, data_start,
                                 data_len, fs, ramp_len=0):
    """Apply a burst-repetitive carrier-frequency settling transient.

    Same physics as add_carrier_frequency_drift - the LO frequency is not
    constant - but a different regime, and the two are complementary:

        add_carrier_frequency_drift : SLOW drift across the whole signal,
                                      linear ramp or OU random walk, RANDOM.
        this function              : FAST settling inside each burst,
                                      identical every burst, DETERMINISTIC.

    The transmitter keys on and off at every burst edge, so the LO and its
    supply see the same step each time and relax the same way. Being identical
    burst to burst, it survives burst averaging intact and shows up as a
    deterministic error - which is why it is invisible to anything that treats
    burst-to-burst variation as noise.

    freq_poly_hz gives the instantaneous frequency offset in Hz as a polynomial
    in position within the data portion (0 at its start, 1 at its end), highest
    order first. Phase is obtained by integration, so a CONSTANT frequency term
    integrates to a linear phase ramp, which any per-burst CFO estimator removes
    along with the true CFO. Only the VARIATION about the mean is observable,
    and the profile is mean-removed here to match.

    Measured on a B210 at 2.4 GHz: a swing of order 100 Hz across the burst,
    settling within the first third. Apply AFTER the channel filter and before
    the carrier terms.

    Parameters
    ----------
    signal : numpy.ndarray
        Complex baseband signal, an integer number of bursts long.
    freq_poly_hz : array_like
        Frequency offset in Hz versus normalised position, highest order first
        (numpy.polyval convention). Quadratic is what a settling LO produces.
    burst_len : int
        Samples per burst.
    data_start, data_len : int
        Start sample and length of the data portion within a burst.
    fs : float
        Sample rate in Hz.
    ramp_len : int
        Raised-cosine entry width in samples, avoiding a step at the preamble
        boundary. One filter span is a sensible choice.
    """
    k = (np.arange(burst_len) - data_start) / float(data_len)
    t_span = data_len / float(fs)                  # seconds spanned by k = 0..1
    ph_poly = np.polyint(np.asarray(freq_poly_hz, dtype=float)) * 2 * np.pi * t_span
    prof = np.polyval(ph_poly, k)
    prof = prof - prof.mean()                      # mean absorbed by CFO removal
    w = np.zeros(burst_len)
    w[data_start:] = 1.0
    if ramp_len:
        # Window spans exactly ramp_len samples. Slicing symmetrically
        # about data_start gives 2*(ramp_len//2) slots and fails to
        # broadcast for any odd ramp_len - one filter span of an
        # odd-tap SRRC, for instance.
        lo = max(0, data_start - ramp_len // 2)
        hi = min(burst_len, lo + ramp_len)
        w[lo:hi] = 0.5 - 0.5 * np.cos(np.pi * np.arange(hi - lo) / ramp_len)
    n_bursts = int(np.ceil(len(signal) / burst_len))
    return signal * np.exp(1j * np.tile(prof * w, n_bursts)[:len(signal)])


def fit_burst_frequency_settling(rx_symbols, model_symbols, constellation,
                                 symbol_rate, degree=3, guard=8):
    """FIT. Carrier-frequency settling profile from the channel-model residual.

    Fit the channel taps FIRST and pass their prediction as model_symbols, so
    this measures only what they leave behind. That residual is a function of
    POSITION IN THE BURST, not of the symbol pattern, which is why no linear
    filter reproduces it at any tap spacing or length.

    The phase residual is fitted with a polynomial of the given degree and then
    DIFFERENTIATED to give frequency, which is the physically meaningful form: a
    cubic phase profile is a quadratic frequency profile, i.e. an LO that starts
    the burst offset and settles. The linear phase term is usually near zero
    because a per-burst CFO estimator has already removed it.

    Parameters
    ----------
    rx_symbols : numpy.ndarray
        Received symbols, (n_bursts, n_symbols) or burst-averaged (n_symbols,).
    model_symbols : numpy.ndarray
        Channel-model prediction for the same symbols.
    constellation : numpy.ndarray
        Ideal constellation points, unit mean power.
    symbol_rate : float
        Symbols per second, needed to convert phase slope into Hz.
    degree : int
        Phase polynomial degree; 3 is where the explained variance saturates,
        giving a quadratic frequency profile.
    guard : int
        Symbols skipped at each end.

    Returns
    -------
    dict
        freq_poly_hz (for add_burst_frequency_settling), phase_poly_rad,
        variance_explained, freq_excursion_hz (peak-to-peak about the mean),
        profile_rms_deg.
    """
    y = np.asarray(rx_symbols, dtype=complex)
    if y.ndim == 2:
        y = y.mean(axis=0)
    y = y / np.sqrt(np.mean(np.abs(y) ** 2))
    m = np.asarray(model_symbols, dtype=complex)
    c = np.asarray(constellation, dtype=complex)
    lab = np.argmin(np.abs(y[:, None] - c[None, :]), axis=1)
    z = (y - m) * np.conj(c[lab])

    n = len(y)
    k = np.arange(n) / n
    P = np.vander(k, degree + 1)
    sl = slice(guard, -guard) if guard else slice(None)
    ph_poly, *_ = np.linalg.lstsq(P[sl], z.imag[sl], rcond=None)
    fit = P @ ph_poly
    ve = 1.0 - np.var(z.imag[sl] - fit[sl]) / np.var(z.imag[sl])

    # phase (rad) vs normalised position -> frequency (Hz)
    t_span = n / float(symbol_rate)
    freq_poly = np.polyder(ph_poly) / (2 * np.pi * t_span)
    f = np.polyval(freq_poly, k)
    return {'freq_poly_hz': freq_poly,
            'phase_poly_rad': ph_poly,
            'variance_explained': float(ve),
            'freq_excursion_hz': float(f.max() - f.min()),
            'profile_rms_deg': float(np.degrees(fit.std()))}


def add_coherent_leakage(signal, eps, delta_f_hz, burst_len, fs):
    """Apply a coherent leakage / feedback artefact as a per-burst complex gain.

    WARNING - on the hardware this was characterised on, this is a property of
    the MEASUREMENT SETUP and not of the radio: eps and delta_f were identical
    across four different devices and repeated on capture days weeks apart.
    Reproducing it in a fingerprinting dataset teaches a classifier to recognise
    the rig. Provided so the effect can be modelled deliberately, for instance to
    study its impact on estimators, not because it belongs in training data.

    Signature: the per-burst gain traces a CIRCLE in the complex plane, giving
    equal AM and PM modulation depth 90 degrees apart. A depth ratio of exactly
    1.0 is what distinguishes it from an ordinary gain wander.

    delta_f is ALIASED - the gain is observable only once per burst, so the true
    rate is delta_f, or burst_rate * n +/- delta_f for any integer n.

    Parameters
    ----------
    signal : numpy.ndarray
        Complex baseband signal.
    eps : float
        Leakage amplitude relative to the wanted signal, e.g. 0.053 for 5.3%.
    delta_f_hz : float
        Offset of the leaking term from the carrier, in Hz.
    burst_len : int
        Samples per burst.
    fs : float
        Sample rate in Hz.
    """
    n_bursts = int(np.ceil(len(signal) / burst_len))
    t = np.arange(n_bursts) * burst_len / fs
    g = 1.0 + eps * np.exp(2j * np.pi * delta_f_hz * t)
    return signal * np.repeat(g, burst_len)[:len(signal)]


def fit_coherent_leakage(per_burst_gain, burst_period_s):
    """FIT. Coherent leakage amplitude and frequency offset from per-burst gain.

    per_burst_gain is each burst's complex gain relative to the burst mean, for
    instance the projection of each burst onto the mean burst. Several cycles of
    the modulation are needed to identify delta_f; over a quarter cycle the fit
    returns nonsense and makes matters worse.

    Returns
    -------
    dict
        eps, delta_f_hz, am_pm_ratio (near 1.0 implies additive coherent
        leakage), variance_explained.
    """
    g = np.asarray(per_burst_gain, dtype=complex)
    g = g / np.abs(g).mean()
    u = g - g.mean()
    n = len(u)
    S = np.fft.fft(u * np.hanning(n), 1 << 16)
    df = float(np.fft.fftfreq(1 << 16, burst_period_s)[np.argmax(np.abs(S))])
    ph = np.exp(2j * np.pi * df * np.arange(n) * burst_period_s)
    eps = complex(np.vdot(ph, u) / np.vdot(ph, ph))
    res = u - eps * ph
    am = np.abs(g).std() * np.sqrt(2.0)
    pm = np.angle(g).std() * np.sqrt(2.0)
    return {'eps': float(abs(eps)), 'delta_f_hz': df,
            'am_pm_ratio': float(am / pm) if pm else float('inf'),
            'variance_explained': float(1.0 - np.mean(np.abs(res) ** 2)
                                        / np.mean(np.abs(u) ** 2))}


def fit_iq_imbalance(rx_symbols, constellation):
    """FIT. IQ imbalance by a data-aided 2x2 real fit over decided symbols.

    Fits [I_s; Q_s] = M @ [I_d; Q_d]. The columns of M are the images of the unit
    I and Q axes: their norm ratio is the gain imbalance and their departure from
    orthogonality is the phase imbalance. Rotation-invariant, so residual carrier
    phase cannot leak in.

    Prefer this to the complementary variance E[s^2]/E[|s|^2]. That estimator
    measures IMPROPERNESS, and a repeated finite symbol burst is improper on its
    own: for QPSK s^2 is +/-j, so E[s^2] = j*(n_13 - n_24)/N, which never
    averages away because the same symbols recur every burst. On a 1000-symbol
    burst that bias is about 1/sqrt(1000), and it has been observed to masquerade
    as 3.3 degrees of imbalance where the true value was 0.1.

    Returns
    -------
    dict
        amp_imbalance_db, phase_imbalance_deg - both signed to match
        add_iq_imbalance, so fit -> add round-trips.
    """
    s = np.asarray(rx_symbols, dtype=complex).ravel()
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    c = np.asarray(constellation, dtype=complex)
    d = c[np.argmin(np.abs(s[:, None] - c[None, :]), axis=1)]
    # The constant column is NOT optional. LO leakage is a tone at the CFO, so
    # after per-burst CFO correction it lands as a DC offset on the
    # constellation. With an infinite balanced symbol stream that offset is
    # orthogonal to the regressors and harmless -- but a capture repeats ONE
    # fixed 1000-symbol burst whose quadrant counts do not balance (mean ~0.03,
    # not 0), so without an intercept the offset couples straight into M as
    # apparent gain/phase imbalance, sinusoidally in the leakage phase.
    # Measured: 2% leakage fakes 0.007 dB / 0.048 deg, which is 24% / 44% of a
    # typical true imbalance; 5% leakage exceeds the true phase imbalance
    # outright. With the intercept the estimate is exact at any leakage.
    D = np.column_stack([d.real, d.imag, np.ones(len(d))])
    Y = np.column_stack([s.real, s.imag])
    M = np.linalg.lstsq(D, Y, rcond=None)[0].T[:, :2]
    cI, cQ = M[:, 0], M[:, 1]
    nI, nQ = np.linalg.norm(cI), np.linalg.norm(cQ)
    # nQ/nI, not nI/nQ: add_iq_imbalance uses g_i = 10^(-a/40), so a POSITIVE
    # amplitude imbalance means Q is the larger axis. This ordering makes
    # fit -> add round-trip. NOTE radio_characterise.py reports the opposite
    # sign for the amplitude term (its phase term agrees).
    return {'amp_imbalance_db': float(20.0 * np.log10(nQ / nI)),
            'phase_imbalance_deg': float(
                90.0 - np.degrees(np.arccos(
                    np.clip(cI @ cQ / (nI * nQ), -1.0, 1.0))))}


def fit_phase_noise_mask(phase_dev, radial_dev, symbol_rate,
                         fit_band=(3e3, 1.5e5), mask_freqs=None, nperseg=256):
    """FIT. A 1/f^alpha phase-noise mask ready to pass to add_phase_noise.

    phase_dev and radial_dev are the per-symbol tangential and radial deviations
    about the ideal points, shape (n_bursts, n_symbols), with the burst mean and
    a per-burst linear ramp already removed from BOTH. That symmetry matters:
    de-ramping one and not the other understates the phase noise. The radial
    channel serves as the AWGN yardstick.

    Only fit_band is fitted. Below it a short burst has no resolution; above it
    the AWGN floor dominates. Everything outside is extrapolation.

    The returned mask is in S_phi dB(rad^2/Hz), which is what add_phase_noise
    expects - 3.01 dB ABOVE the same curve written as SSB dBc/Hz. Generate the
    process at symbol rate and resample up: generating straight at sample rate
    extrapolates the mask far beyond the measured band, and because the slope is
    shallow the variance integral is dominated by that extrapolation.

    Do not use the Wiener mode of add_phase_noise for this hardware. It is 1/f^2,
    i.e. -20 dB/decade, against a measured -8.7 to -10.4.

    THIS IS THE DEFAULT MODEL. fit_phase_noise_pole fits the physically
    motivated PLL one-pole form instead; it was measured against this one and
    fits worse on real data (see its docstring for the numbers), so it is a
    diagnostic, not a replacement.

    Returns
    -------
    dict
        slope_db_per_decade, level_db_at_10k, mask (list of (Hz, S_phi dB)),
        fit_rms_db.
    """
    ph = np.asarray(phase_dev, dtype=float)
    am = np.asarray(radial_dev, dtype=float)
    f, P = scipy.signal.welch(ph, fs=symbol_rate, nperseg=nperseg,
                              noverlap=nperseg // 2, axis=-1)
    _, Pa = scipy.signal.welch(am, fs=symbol_rate, nperseg=nperseg,
                               noverlap=nperseg // 2, axis=-1)
    P = P.mean(axis=0) if P.ndim > 1 else P
    Pa = Pa.mean(axis=0) if Pa.ndim > 1 else Pa
    floor = Pa[f > 0.35 * symbol_rate / 2].mean()
    ex = np.maximum(P - floor, 1e-30)
    b = (f >= fit_band[0]) & (f <= fit_band[1])
    slope, icept = np.polyfit(np.log10(f[b]), 10.0 * np.log10(ex[b] / 2.0), 1)
    resid = 10.0 * np.log10(ex[b] / 2.0) - (slope * np.log10(f[b]) + icept)
    if mask_freqs is None:
        mask_freqs = (1e2, 1e3, 1e4, 1e5, symbol_rate / 2)
    mask = [(float(q), float(slope * np.log10(q) + icept + 3.0102))
            for q in mask_freqs]
    return {'slope_db_per_decade': float(slope),
            'level_db_at_10k': float(slope * 4.0 + icept),
            'mask': mask,
            'fit_rms_db': float(np.sqrt(np.mean(resid ** 2)))}


def add_bulk_delay(signal, tau_samples):
    """Delay a signal by tau samples, fractional amounts allowed.

    A delay multiplies the spectrum by a phase that tilts linearly with
    frequency, so the shift is applied that way rather than by rolling the
    array. Unlike numpy.roll this handles non-integer tau, which matters
    because a delay that is a whole number of SAMPLES is still a fraction of a
    SYMBOL, and it is the symbol-fraction that no symbol-spaced channel model
    can absorb.

    The transform is circular, so the first and last few samples wrap. Guard
    them (the fit below uses 64) rather than trusting the edges.

    Parameters
    ----------
    signal : numpy.ndarray
        Complex baseband signal at sample rate.
    tau_samples : float
        Delay in samples; positive delays, negative advances.
    """
    n = len(signal)
    f = np.fft.fftfreq(n)
    return np.fft.ifft(np.fft.fft(np.asarray(signal, dtype=complex))
                       * np.exp(-2j * np.pi * f * tau_samples))


def fit_residual_delay(rx_signal, tx_signal, sps=5, beta=0.35, max_lag=14,
                       guard=64, n_fft=4096):
    """FIT. Bulk delay left between rx and tx after burst alignment, in SAMPLES.

    Burst detection correlates against the preamble and picks a sample index,
    so it can land a few samples off. Measured on four B210s: exactly 0 on
    healthy captures, but +6.000 samples on 30BF795 and -1.000 on 30BF779 at
    2400 (both gains, reproducible to three decimals). A whole number of
    samples is the signature of a detector picking the wrong index; a real
    analog path delay would be some arbitrary fraction.

    Such an offset is invisible to fit_isi_taps and cannot be repaired by
    adding taps to it: symbol-spaced taps shift in steps of sps samples, and 6
    is not a multiple of 5. With excess bandwidth (beta > 0) no symbol-spaced
    filter synthesises a fractional-symbol delay, which is why the residual
    plateaus no matter how long the tap set gets. Everything downstream then
    tries to explain the misalignment as hardware: on 30BF795 at 2400 it turned
    up as a spurious -1.04 dB of PA compression that vanished once the delay
    was removed.

    Fit this FIRST, before fit_isi_taps, and pass the corrected signal on.

    The delay is read off as a group delay - the slope of the phase of the
    fitted sample-spaced response - measured only across the band the signal
    occupies, |f| < (1 + beta) / 2 / sps. Restricting to that band is what
    makes it reliable; a tap-energy centroid over all lags is dragged by
    out-of-band noise taps and mis-estimates by up to a full sample on clean
    captures, which then damages them.

    Parameters
    ----------
    rx_signal, tx_signal : numpy.ndarray
        Burst-averaged received and transmitted signal at SAMPLE rate, equal
        length. Burst-average the rx first so random error is averaged down.
    sps : int
        Samples per symbol.
    beta : float
        SRRC roll-off, setting the occupied bandwidth.
    max_lag : int
        Half-width in samples of the tap set used to see the delay. Must
        comfortably exceed the delay being looked for.
    guard : int
        Samples skipped at each end.
    n_fft : int
        FFT length for the phase slope.

    Returns
    -------
    dict
        tau_samples (feed to add_bulk_delay with the sign negated to undo it),
        tau_symbols, peak_lag, energy_frac (tap energy within +-2 of the peak;
        near 1 means a clean translation rather than a dispersive channel),
        residual_before / residual_after (the same 3-tap symbol-spaced channel
        fitted either side of the correction), and residual_floor (the
        sample-spaced fit, which absorbs the delay on its own - residual_after
        should land on it).
    """
    y = np.asarray(rx_signal, dtype=complex)
    x = np.asarray(tx_signal, dtype=complex)
    y = y / np.sqrt(np.mean(np.abs(y) ** 2))
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))

    if guard < max_lag:
        raise ValueError(f'guard ({guard}) must be >= max_lag ({max_lag}); '
                         'otherwise the zero-filled edge rows enter the fit')
    ks = np.arange(-max_lag, max_lag + 1)
    sl = slice(guard, -guard) if guard else slice(None)
    A = np.stack([_shift_pad(x, int(k)) for k in ks], axis=1)[sl]
    c, *_ = np.linalg.lstsq(A, y[sl], rcond=None)
    # the sample-spaced fit absorbs the delay itself, so its residual is the
    # FLOOR a correct delay removal should reach - not the "before" figure
    floor = float(np.sqrt(np.mean(np.abs(y[sl] - A @ c) ** 2)))

    w = np.abs(c) ** 2
    j0 = int(np.argmax(w))
    near = np.abs(ks - ks[j0]) <= 2
    energy_frac = float(np.sum(w[near]) / np.sum(w))

    # group delay: phase slope of the tap response over the occupied band only
    h = np.zeros(n_fft, dtype=complex)
    h[:len(c)] = c
    H = np.fft.fft(np.roll(h, -j0))
    f = np.fft.fftfreq(n_fft)
    keep = np.abs(f) < (1.0 + beta) / 2.0 / sps
    order = np.argsort(f[keep])
    slope = np.polyfit(f[keep][order],
                       np.unwrap(np.angle(H[keep][order])), 1)[0]
    tau = float(ks[j0] - slope / (2.0 * np.pi))

    # before/after are like for like: the SAME 3-tap symbol-spaced channel the
    # pipeline actually uses, fitted on the uncorrected and corrected signal.
    ks3 = np.arange(-1, 2) * sps
    A3 = np.stack([_shift_pad(x, int(k)) for k in ks3], axis=1)[sl]

    def _r3(v):
        cc, *_ = np.linalg.lstsq(A3, v[sl], rcond=None)
        return float(np.sqrt(np.mean(np.abs(v[sl] - A3 @ cc) ** 2)))

    return {'tau_samples': tau, 'tau_symbols': tau / float(sps),
            'peak_lag': int(ks[j0]), 'energy_frac': energy_frac,
            'residual_before': _r3(y),
            'residual_after': _r3(add_bulk_delay(y, -tau)),
            'residual_floor': floor}


def fit_phase_noise_pole(phase_dev, radial_dev, symbol_rate,
                         fit_band=(3e3, 1.5e5), mask_freqs=None, nperseg=256,
                         pole_grid=(1e2, 1e7, 600)):
    """FIT. A ONE-POLE phase-noise spectrum. DIAGNOSTIC ONLY - NOT THE DEFAULT.

    fit_phase_noise_mask, with its free 1/f^alpha slope, remains the model this
    library uses. This function exists to test whether the physically motivated
    PLL form describes the hardware better. It was measured against the free
    slope and it does NOT, so do not swap it in.

    Fit residual, dB rms, lower is better. Both models have two free
    parameters, so this is a fair comparison:

        MODULATED captures, 3-150 kHz, four radios
            band     one-pole      free slope
            433      0.82-1.39     1.43-1.89     one-pole better
            915      1.21-1.36     1.01-1.39     comparable
            2400     2.26-2.46     1.40-1.85     free better

        CW captures, ~4 decades, floor subtracted, one radio
            band     one-pole      free slope    powerlaw3
            433        5.87          4.36          24.2
            915        6.57          5.17          22.3
            2400       6.04          3.81          15.9     free wins everywhere

    The CW result is the decisive one: it was taken over the four to five
    decades of span needed to tell the shapes apart, which the 1.7-decade
    modulated band cannot do. Note also that ALL residuals are poor - 4+ dB
    against 0.6 dB on synthetic one-pole data - so the honest reading is that
    none of these models describes this oscillator well, not that the free
    slope is right.

    What the function fits, and why it seemed like a good idea
    ---------------------------------------------------------
    The PLL-oscillator model of Mohammadian & Tellambura (IEEE Access, 2021,
    eq. 10), reduced to the part observable here:

        S(f) = K0 * (1 + (f/fz)^2) / (1 + (f/fp)^2)  ->  K0 / (1 + (f/fp)^2)

    The zero fz sits far above the measured band (100 MHz in the 802.11ad
    parameterisation, against a 150 kHz fit ceiling here), so its numerator is 1
    to within a fraction of a dB and it is dropped rather than fitted to noise.

    The argument for it was that physical slopes are -30 (flicker), -20 (white
    FM) or 0 (thermal floor) dB/decade, while measured slopes here are -5.7 to
    -13.0 - none of them. A straight line across a KNEE reports the corner's
    position as a fake slope. That reasoning is sound and the effect is real
    (on synthetic one-pole data the free fit reports slopes from -1.5 to -19.5
    purely as the corner moves), but it evidently is not what limits accuracy
    on this hardware.

    Its autocorrelation (ibid., eq. 11) is a delta plus one exponential - in
    discrete time, white noise plus AR(1), with ar[1] = exp(-2*pi*fp/fs). One
    pole would therefore be the whole model, which is why AR(8) was
    over-parameterised.

    CAUTION, an earlier claim here was wrong: it stated that the old AR fit's
    ar[1] ~ 1.0 was explained by a corner far below the symbol rate. Measured
    corners are 22.6-74.4 kHz on modulated data and 1.3-10.1 kHz on CW, giving
    ar[1] = 0.63-0.87 at a 1 MHz symbol rate - nowhere near 1.0. The AR
    ill-conditioning is NOT explained by this model and remains unresolved.

    A one-pole spectrum can only produce slopes between 0 and -20 dB/decade.
    Every measurement on this hardware falls inside that range; railing against
    a pole_grid endpoint means it does not, and the extra components of eq. (7)
    would be needed.

    Parameters are as fit_phase_noise_mask, which shares the PSD pipeline and
    the same AWGN correction, plus:

    pole_grid : (lo, hi, n)
        Log-spaced search range for the corner, in Hz. K0 is solved in closed
        form at each trial pole, so this is a 1-D search and always converges.

    Returns
    -------
    dict
        pole_hz, plateau_db (K0 as S_phi dB at DC), ar1_at_symbol_rate,
        level_db_at_10k (comparable to fit_phase_noise_mask's), mask,
        fit_rms_db, railed (True if the corner hit a search endpoint).
    """
    ph = np.asarray(phase_dev, dtype=float)
    am = np.asarray(radial_dev, dtype=float)
    f, P = scipy.signal.welch(ph, fs=symbol_rate, nperseg=nperseg,
                              noverlap=nperseg // 2, axis=-1)
    _, Pa = scipy.signal.welch(am, fs=symbol_rate, nperseg=nperseg,
                               noverlap=nperseg // 2, axis=-1)
    P = P.mean(axis=0) if P.ndim > 1 else P
    Pa = Pa.mean(axis=0) if Pa.ndim > 1 else Pa
    floor = Pa[f > 0.35 * symbol_rate / 2].mean()
    ex = np.maximum(P - floor, 1e-30)
    b = (f >= fit_band[0]) & (f <= fit_band[1])
    fb = f[b]
    y = 10.0 * np.log10(ex[b] / 2.0)

    lo, hi, n = pole_grid
    poles = np.logspace(np.log10(lo), np.log10(hi), int(n))
    best = None
    for fp in poles:
        shape = -10.0 * np.log10(1.0 + (fb / fp) ** 2)
        k0 = np.mean(y - shape)                      # closed-form plateau
        r = float(np.sqrt(np.mean((y - shape - k0) ** 2)))
        if best is None or r < best[0]:
            best = (r, float(fp), float(k0))
    rms, fp, k0 = best

    if mask_freqs is None:
        mask_freqs = (1e2, 1e3, 1e4, 1e5, symbol_rate / 2)
    curve = lambda q: k0 - 10.0 * np.log10(1.0 + (np.asarray(q, float) / fp) ** 2)
    mask = [(float(q), float(curve(q) + 3.0102)) for q in mask_freqs]
    return {'pole_hz': fp,
            'plateau_db': k0,
            'ar1_at_symbol_rate': float(np.exp(-2.0 * np.pi * fp / symbol_rate)),
            'level_db_at_10k': float(curve(1e4)),
            'mask': mask,
            'fit_rms_db': rms,
            'railed': bool(fp <= poles[1] or fp >= poles[-2])}


PN_CARRIER_EXPONENT = 1.34   # MEASURED on one B210; see scale_phase_noise_mask


def scale_phase_noise_mask(mask, carrier_from_hz, carrier_to_hz,
                           exponent=PN_CARRIER_EXPONENT):
    """Project a phase-noise mask from one carrier frequency to another.

    Oscillator phase noise rises with carrier frequency. Textbook theory says
    QUADRATICALLY (exponent 2.0): multiply a reference by N and its phase noise
    goes up by 20*log10(N). This hardware does not do that.

    MEASURED, one B210 (30ECB6B), CW at 433 / 915 / 2400 MHz, 10 s captures,
    only the offsets where the AM/PM ratio confirmed the measurement sat well
    above the noise floor:

        band pair      n at 100 Hz    n at 1 kHz
        433  -> 915       0.87           1.35
        915  -> 2400      1.51           1.58
        433  -> 2400      1.23           1.48

        mean 1.34, range 0.87 - 1.58,  against a theoretical 2.0

    TREAT THE DEFAULT AS PROVISIONAL. It rests on ONE radio and two usable
    offset frequencies, and it is not even constant across those - the exponent
    drifts with offset, so a single number applied uniformly across the mask is
    already an approximation within this one unit. Whether it holds for other
    B210s, other silicon, or any wider population is UNTESTED. Pass exponent
    explicitly whenever you have a better number, or 2.0 to get the textbook
    behaviour and see how much it matters.

    Why it is not 2.0 is unresolved. A pure reference-multiplication chain
    gives exactly 2; the AD9361 selects different VCO and divider settings per
    band, so its LO chain is not a simple multiply. That is interpretation, not
    measurement.

    Parameters
    ----------
    mask : list of (Hz, S_phi dB)
        As returned by fit_phase_noise_mask or fit_phase_noise_pole.
    carrier_from_hz, carrier_to_hz : float
        Carrier the mask was measured at, and the one wanted.
    exponent : float
        Power law in carrier frequency. Default is the measured 1.34.

    Returns
    -------
    list of (Hz, S_phi dB)
        Same offsets, shifted by 10 * exponent * log10(to / from) dB.
    """
    if carrier_from_hz <= 0 or carrier_to_hz <= 0:
        raise ValueError('carrier frequencies must be positive')
    shift = 10.0 * float(exponent) * np.log10(carrier_to_hz / carrier_from_hz)
    return [(float(f), float(s + shift)) for f, s in mask]
