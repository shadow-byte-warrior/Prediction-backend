import pyedflib
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

class EEGPreprocessor:
    """
    Clean and prepare EEG signals for deep learning.
    """
    
    def __init__(self, sampling_rate=256, low_freq=0.5, high_freq=100, notch_freq=60):
        self.fs = sampling_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.notch_freq = notch_freq
        
    def bandpass_filter(self, data, order=4):
        """Remove frequencies outside 0.5-100 Hz range."""
        nyquist = self.fs / 2
        low = self.low_freq / nyquist
        high = min(self.high_freq / nyquist, 0.99)
        
        b, a = butter(order, [low, high], btype='band')
        
        if len(data.shape) == 1:
            return filtfilt(b, a, data)
        else:
            return np.array([filtfilt(b, a, ch) for ch in data])
    
    def notch_filter(self, data, Q=30):
        """Remove 60 Hz power line interference."""
        nyquist = self.fs / 2
        freq = self.notch_freq / nyquist
        
        b, a = iirnotch(freq, Q)
        
        if len(data.shape) == 1:
            return filtfilt(b, a, data)
        else:
            return np.array([filtfilt(b, a, ch) for ch in data])
    
    def normalize(self, data):
        """Z-score normalization per channel."""
        if len(data.shape) == 1:
            return (data - np.mean(data)) / (np.std(data) + 1e-8)
        else:
            return np.array([(ch - np.mean(ch)) / (np.std(ch) + 1e-8) for ch in data])
    
    def remove_artifacts(self, data, threshold=5):
        """Clip extreme values (likely artifacts)."""
        return np.clip(data, -threshold, threshold)
    
    def preprocess(self, data):
        """Apply full preprocessing pipeline."""
        # Step 1: Bandpass filter
        data = self.bandpass_filter(data)
        # Step 2: Notch filter  
        data = self.notch_filter(data)
        # Step 3: Normalize
        data = self.normalize(data)
        # Step 4: Remove artifacts
        data = self.remove_artifacts(data)
        
        return data

def preprocess_edf(file_path, window_size=4, overlap=0.0):
    """
    Load an EDF file, preprocess it, and return windowed segments for inference.
    
    Args:
        file_path: Path to the .edf file
        window_size: Length of window in seconds (default 4 for model input)
        overlap: Overlap fraction (0.0 for distinct non-overlapping windows)
                 For inference, we typically want to check every segment.
    
    Returns:
        segments: Numpy array of shape (n_windows, 1024, 18) suitable for model input.
        If file cannot be loaded or validation fails, returns None.
    """
    try:
        f = pyedflib.EdfReader(file_path)
        
        # Determine number of channels but limit to 18 as per model requirement
        n_channels = min(f.signals_in_file, 18)
        
        # Read signal (only first 18 channels)
        signals = np.zeros((n_channels, f.getNSamples()[0]))
        for i in range(n_channels):
            signals[i, :] = f.readSignal(i)
        
        fs = int(f.getSampleFrequency(0))
        f.close()
        
        # Handle case where file has < 18 channels if necessary (zeropad?)
        # For now assume input >= 18 channels as trained on CHB-MIT
        if n_channels < 18:
            # Pad with zeros if fewer channels
            padded = np.zeros((18, signals.shape[1]))
            padded[:n_channels, :] = signals
            signals = padded
        
        # Preprocessing
        preprocessor = EEGPreprocessor(sampling_rate=fs)
        processed_signals = preprocessor.preprocess(signals)
        
        # Windowing
        window_samples = window_size * fs
        step_samples = int(window_samples * (1 - overlap))
        
        windows = []
        n_samples = processed_signals.shape[1]
        
        for start in range(0, n_samples - window_samples + 1, step_samples):
            end = start + window_samples
            window = processed_signals[:, start:end]
            
            # Model expects shape (1024, 18) per sample, but our signal is (18, 1024)
            # Notebook logic: X_reshaped = np.transpose(X, (0, 2, 1)) -> (samples, time, channels)
            # So we need to transpose each window to (1024, 18)
            window_T = window.transpose()
            
            windows.append(window_T)
            
        return np.array(windows)

    except Exception as e:
        print(f"Error processing EDF file: {e}")
        return None
