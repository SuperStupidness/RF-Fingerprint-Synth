import numpy as np

def add_awgn(signal, noise_power=0.1, db=False, rng=np.random.default_rng()):
    """
    Add Additive White Gaussian Noise (AWGN) to a signal.
    
    Parameters
    ----------
    signal : array_like
        Input signal to which noise will be added
    noise_power : float, optional
        Power of the noise to be added, default=0.1
    db : bool, optional
        If True, noise_power is interpreted in decibels, default=False
        
    Returns
    -------
    noisy_signal : ndarray
        Signal with added complex AWGN
    """
    # Generate complex noise with variance/power = 1 and mean = 0
    mean = 0
    # Power are divided to real and imaginary components. If only real, variance = 1
    variance = np.sqrt(2)/2

    N = len(signal)
    noise_real = rng.normal(mean, variance, N)
    noise_imag = rng.normal(mean, variance, N)

    noise = noise_real + 1j*noise_imag

    if db:
        noise_scaled = np.sqrt(10**(noise_power/10))*noise
    else:
        noise_scaled = np.sqrt(noise_power)*noise

    return signal + noise_scaled

def add_awgn_snr(signal, desired_snr=10, db=True, rng=np.random.default_rng()):
    """
    Add Additive White Gaussian Noise (AWGN) to a signal with specified SNR.
    
    Parameters
    ----------
    signal : array_like
        Input signal to which noise will be added
    desired_snr : float, optional
        Desired Signal-to-Noise Ratio, default=10
    db : bool, optional
        If True, desired_snr is interpreted in decibels, default=True
        
    Returns
    -------
    noisy_signal : ndarray
        Signal with added complex AWGN at the specified SNR
    """
    # Generate complex noise with variance/power = 1 and mean = 0
    mean = 0
    # Power are divided to real and imaginary components. If only real, variance = 1
    variance = np.sqrt(2)/2

    N = len(signal)
    noise_real = rng.normal(mean, variance, N)
    noise_imag = rng.normal(mean, variance, N)

    noise = noise_real + 1j*noise_imag

    # Calculate signal power
    signal_power = np.mean(np.abs(signal)**2)

    # Scale noise power to achieve desired snr
    if db:
        noise_power = signal_power/(10**(desired_snr/10))
    else:
        noise_power = signal_power/desired_snr

    noise_scaled = np.sqrt(noise_power)*noise

    return signal + noise_scaled