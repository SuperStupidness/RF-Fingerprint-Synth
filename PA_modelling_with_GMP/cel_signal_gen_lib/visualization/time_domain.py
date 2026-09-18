import numpy as np
import matplotlib.pyplot as plt
from .constellation import signal_constellation_generator

def plot_signal(signal, title, max_samples=500):
    """Plot the I/Q components of a signal"""
    
    # Get the first 500 samples or the entire signal if shorter
    plot_length = min(max_samples, len(signal))
    
    # Extract I/Q components
    i_component = np.real(signal[:plot_length])
    q_component = np.imag(signal[:plot_length])
    
    # Create plot
    plt.figure(figsize=(4.8, 3))
    plt.plot(i_component, 'b-', label='In-Phase (I)')
    plt.plot(q_component, 'r-', label='Quadrature (Q)')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return

def plot_fsk_signal(signal, symbols, phi, delta_f, f_carrier, title, max_samples=200):
    """Plot the real component and instantanteous frequency of a FSK signal"""
    t = np.linspace(0, len(phi), len(phi))

    inst_freq = np.zeros(len(t))
    inst_freq[1:] = np.diff(phi) / (2 * np.pi * (t[1] - t[0]))
    inst_freq[0] = inst_freq[1]  # Avoid undefined first point

    ylim_upper = f_carrier + delta_f * 1.1 
    ylim_lower = f_carrier - delta_f * 1.1  

    fig, ((ax1, ax2)) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t[:max_samples], np.real(signal[:max_samples]))
    ax1.plot(symbols[:max_samples])
    ax1.set_title(title)
    ax1.set_ylabel("Magnitude")
    ax1.legend(["Real", "Bit symbol"])
    
    ax2.plot(inst_freq[:max_samples], color="red") # To check whether it is coherent or not
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Frequency")
    ax2.set_ylim([ylim_lower, ylim_upper])
    ax2.legend(["Frequency"])
    
    plt.show()
    return

def plot_constellation(signal, samples_per_symbol, filter_length=None, modulation_type=None, ax=None):
    # Type and Value Checks for samples_per_symbol
    if not isinstance(samples_per_symbol, int) or samples_per_symbol <= 0:
        raise TypeError(f"samples_per_symbol must be a positive integer, got {samples_per_symbol}")
    elif filter_length is None:
        filter_length = 1
    elif not isinstance(filter_length, int) or filter_length <= 0:
        raise TypeError(f"filter_length must be a positive integer, got {filter_length}")
        
    group_delay = (filter_length - 1) / 2
    
    # We must sample at an integer, so we round.
    # We add the *intentional* clock_offset to this ideal center.
    start_sample = int(np.round(group_delay))
    
    # Slice the array starting from the correct sample, striding by sps
    samples = signal[start_sample::samples_per_symbol]

    # If no Axes object is provided, create a standalone one
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Plot actual received samples
    ax.scatter(np.real(samples), np.imag(samples), marker='x', alpha=0.4, label='Received')

    if modulation_type is not None:
        try:
            ideal_symbols = signal_constellation_generator(modulation_type)
        except NameError:
            pass
        else:
            ax.scatter(np.real(ideal_symbols), np.imag(ideal_symbols), marker='o', facecolors='none', edgecolors='r', s=100, label='Ideal')

    # Aesthetics
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"Constellation Diagram {f'({modulation_type})' if modulation_type else ''}")
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.axis('equal') # Crucial for QAM/PSK shapes
    ax.legend()

    return ax

def return_sampled_values(signal, samples_per_symbol, filter_length=None):
    # Type and Value Checks for samples_per_symbol
    if not isinstance(samples_per_symbol, int) or samples_per_symbol <= 0:
        raise TypeError(f"samples_per_symbol must be a positive integer, got {samples_per_symbol}")
    elif filter_length is None:
        filter_length = 1
    elif not isinstance(filter_length, int) or filter_length <= 0:
        raise TypeError(f"filter_length must be a positive integer, got {filter_length}")
        
    group_delay = (filter_length - 1) / 2
    
    # We must sample at an integer, so we round.
    # We add the *intentional* clock_offset to this ideal center.
    start_sample = int(np.round(group_delay))
    
    # Slice the array starting from the correct sample, striding by sps
    samples = signal[start_sample::samples_per_symbol]

    return samples

    