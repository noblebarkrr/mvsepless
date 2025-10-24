
import numpy as np
from scipy import signal

def remove_center(input_array, samplerate, rdf=0.99999, window_size=2048, overlap=2, window_type="blackman", stereo_mode="stereo"):
    # Validate input
    # if input_array.ndim != 2 or input_array.shape[1] != 2:
        # raise ValueError("Input must be a stereo array with shape (samples, 2)")
    
    left = input_array[0]
    right = input_array[1]
    # mono = np.mean(input_array, axis=1)

    # Adjust window size if input is too short
    nperseg = min(window_size, len(left))
    if nperseg < 16:  # Minimum reasonable window size
        nperseg = 16
        if len(left) < 16:
            # For very short inputs, just return the original with warning
            import warnings
            warnings.warn(f"Input too short ({len(left)} samples), returning original audio")
            return left, right, left, right
    
    noverlap = nperseg // overlap  # Ensure noverlap < nperseg
    if noverlap >= nperseg:
        noverlap = nperseg - 1  # Ensure at least 1 sample difference

    # Compute STFT
    f, t, Z_left = signal.stft(left, fs=samplerate, nperseg=nperseg, noverlap=noverlap, window=window_type)
    f, t, Z_right = signal.stft(right, fs=samplerate, nperseg=nperseg, noverlap=noverlap, window=window_type)
    # f, t, Z_mono = signal.stft(mono, fs=samplerate, nperseg=nperseg, noverlap=noverlap, window=window_type)

    if stereo_mode == "mono":
        Z_common_left = np.minimum(np.abs(Z_left), np.abs(Z_right)) * np.exp(1j*np.angle(Z_mono))
        Z_common_right = np.minimum(np.abs(Z_left), np.abs(Z_right)) * np.exp(1j*np.angle(Z_mono))
    else:
        Z_common_left = np.minimum(np.abs(Z_left), np.abs(Z_right)) * np.exp(1j*np.angle(Z_right))
        Z_common_right = np.minimum(np.abs(Z_left), np.abs(Z_right)) * np.exp(1j*np.angle(Z_left))

    reduction_factor = rdf

    Z_new_left = Z_left - Z_common_left * reduction_factor
    Z_new_right = Z_right - Z_common_right * reduction_factor

    # Compute ISTFT
    _, new_left = signal.istft(Z_new_left, fs=samplerate, nperseg=nperseg, noverlap=noverlap, window=window_type)
    _, new_right = signal.istft(Z_new_right, fs=samplerate, nperseg=nperseg, noverlap=noverlap, window=window_type)
    _, common_signal_left = signal.istft(Z_common_left, fs=samplerate, nperseg=nperseg, noverlap=noverlap, window=window_type)
    _, common_signal_right = signal.istft(Z_common_right, fs=samplerate, nperseg=nperseg, noverlap=noverlap, window=window_type)

    # Trim to original length
    new_left = new_left[:len(left)]
    new_right = new_right[:len(right)]
    common_signal_left = common_signal_left[:len(left)]
    common_signal_right = common_signal_right[:len(left)]

    # Normalize
    peak = np.max([np.abs(new_left).max(), np.abs(new_right).max()])
    if peak > 1.0:
        new_left = new_left / peak
        new_right = new_right / peak

    inverted_center_left = -common_signal_left
    inverted_center_right = -common_signal_right

    mixed_left = left + inverted_center_left
    mixed_right = right + inverted_center_right

    peak_mixed = np.max([np.abs(mixed_left).max(), np.abs(mixed_right).max()])
    if peak_mixed > 1.0:
        mixed_left = mixed_left / peak_mixed
        mixed_right = mixed_right / peak_mixed

    return common_signal_left, common_signal_right, new_left, new_right
