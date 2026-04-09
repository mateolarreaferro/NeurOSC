# simulator_service.py
"""
Synthetic EEG Data Simulator for BioMus
Generates realistic brain wave patterns for debugging without hardware
"""
import threading
import time
from typing import List, Tuple, Dict, Optional
import numpy as np
from collections import deque

from osc_sender import OSCSender


class SmoothingBuffer:
    """Exponential moving average buffer for smoothing data streams"""

    def __init__(self, alpha: float = 0.3, num_values: int = 5):
        self.alpha = alpha
        self.smoothed_values: List[Optional[float]] = [None] * num_values
        self.enabled = True

    def update(self, values: List[float]) -> List[float]:
        if not self.enabled:
            return values
        result = []
        for i, val in enumerate(values):
            if i >= len(self.smoothed_values):
                self.smoothed_values.append(None)
            if self.smoothed_values[i] is None:
                self.smoothed_values[i] = val
            else:
                self.smoothed_values[i] = self.alpha * val + (1 - self.alpha) * self.smoothed_values[i]
            result.append(self.smoothed_values[i])
        return result

    def reset(self):
        self.smoothed_values = [None] * len(self.smoothed_values)


class SimulatorService:
    """
    Simulates OpenBCI Ganglion EEG data with realistic brain wave patterns.
    Useful for testing and debugging without actual hardware.
    """

    def __init__(self, sampling_rate: int = 200, num_channels: int = 4):
        """
        Initialize simulator

        Args:
            sampling_rate: Samples per second (Ganglion default is 200 Hz)
            num_channels: Number of EEG channels (Ganglion has 4)
        """
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.connected = False
        self.streaming = False

        # Data buffer
        self.buffer = deque(maxlen=50000)  # Store up to 250 seconds at 200 Hz
        self.lock = threading.Lock()

        # Generation thread
        self.thread = None
        self.running = False

        # Simulation state
        self.time_elapsed = 0.0
        self.mode = "normal"  # normal, meditation, focused, drowsy

        # Brain wave frequencies (Hz)
        self.DELTA = (0.5, 4.0)    # Deep sleep
        self.THETA = (4.0, 8.0)     # Drowsiness, meditation
        self.ALPHA = (8.0, 13.0)    # Relaxed, calm
        self.BETA = (13.0, 30.0)    # Active thinking, focus
        self.GAMMA = (30.0, 50.0)   # High-level cognition

        # OSC sender
        self.osc = OSCSender()

        # Smoothing configuration
        self.smoothing_alpha = 0.3
        self.smoothing_enabled = True
        self.band_smoothers: Dict[str, SmoothingBuffer] = {}
        self.engagement_smoother = SmoothingBuffer(alpha=0.2, num_values=4)

    def configure_smoothing(self, enabled: bool, alpha: float = 0.3):
        """Configure smoothing parameters"""
        self.smoothing_enabled = enabled
        self.smoothing_alpha = max(0.05, min(1.0, alpha))
        for smoother in self.band_smoothers.values():
            smoother.enabled = enabled
            smoother.alpha = self.smoothing_alpha
        self.engagement_smoother.enabled = enabled
        self.engagement_smoother.alpha = self.smoothing_alpha

    def configure_artifact_detection(self, enabled: bool, **kwargs):
        """Stub for API compatibility - simulator doesn't generate artifacts"""
        pass
        
    def connect(self):
        """Simulate connection"""
        self.connected = True
        print("[SIMULATOR] Connected to synthetic EEG source")
        
    def disconnect(self):
        """Simulate disconnection"""
        self.stop_stream()
        self.connected = False
        print("[SIMULATOR] Disconnected")
        
    def start_stream(self, buffer_size: int = 45000):
        """Start generating synthetic data"""
        if not self.connected:
            raise RuntimeError("Simulator not connected")
        if self.streaming:
            return
            
        self.streaming = True
        self.running = True
        self.time_elapsed = 0.0
        
        # Start generation thread
        self.thread = threading.Thread(target=self._generate_loop, daemon=True)
        self.thread.start()
        print("[SIMULATOR] Started streaming synthetic EEG data")
        
    def stop_stream(self):
        """Stop generating data"""
        if not self.streaming:
            return
            
        self.running = False
        self.streaming = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
            
        print("[SIMULATOR] Stopped streaming")
        
    def set_mode(self, mode: str):
        """
        Change simulation mode
        
        Args:
            mode: 'normal', 'meditation', 'focused', 'drowsy'
        """
        valid_modes = ['normal', 'meditation', 'focused', 'drowsy']
        if mode in valid_modes:
            self.mode = mode
            print(f"[SIMULATOR] Mode changed to: {mode}")
        else:
            print(f"[SIMULATOR] Invalid mode: {mode}. Valid: {valid_modes}")
            
    def _generate_sample(self, t: float, channel: int) -> float:
        """
        Generate a single sample for a channel at time t
        
        Simulates realistic EEG with multiple frequency components
        """
        # Base amplitudes for different bands (microvolts)
        # Adjust based on simulation mode
        if self.mode == "meditation":
            # High alpha, high theta, low beta
            alpha_amp = 40.0
            theta_amp = 30.0
            beta_amp = 5.0
            delta_amp = 10.0
            gamma_amp = 2.0
        elif self.mode == "focused":
            # High beta, low alpha
            alpha_amp = 10.0
            theta_amp = 5.0
            beta_amp = 35.0
            delta_amp = 3.0
            gamma_amp = 8.0
        elif self.mode == "drowsy":
            # High delta, high theta, low beta
            alpha_amp = 15.0
            theta_amp = 25.0
            beta_amp = 5.0
            delta_amp = 40.0
            gamma_amp = 2.0
        else:  # normal
            alpha_amp = 20.0
            theta_amp = 10.0
            beta_amp = 15.0
            delta_amp = 8.0
            gamma_amp = 5.0
            
        # Add channel-specific variation
        channel_factor = 1.0 + 0.1 * channel
        
        # Generate signal as sum of sine waves at different frequencies
        signal = 0.0
        
        # Delta band (0.5-4 Hz)
        delta_freq = np.random.uniform(*self.DELTA)
        signal += delta_amp * channel_factor * np.sin(2 * np.pi * delta_freq * t)
        
        # Theta band (4-8 Hz)
        theta_freq = np.random.uniform(*self.THETA)
        signal += theta_amp * channel_factor * np.sin(2 * np.pi * theta_freq * t)
        
        # Alpha band (8-13 Hz) - often dominant when relaxed
        alpha_freq = np.random.uniform(*self.ALPHA)
        signal += alpha_amp * channel_factor * np.sin(2 * np.pi * alpha_freq * t)
        
        # Beta band (13-30 Hz)
        beta_freq = np.random.uniform(*self.BETA)
        signal += beta_amp * channel_factor * np.sin(2 * np.pi * beta_freq * t)
        
        # Gamma band (30-50 Hz) - lower amplitude
        gamma_freq = np.random.uniform(*self.GAMMA)
        signal += gamma_amp * channel_factor * np.sin(2 * np.pi * gamma_freq * t)
        
        # Add some noise for realism
        noise = np.random.normal(0, 2.0)
        signal += noise
        
        # Add slow drift
        drift = 5.0 * np.sin(2 * np.pi * 0.1 * t)
        signal += drift
        
        return signal
        
    def _generate_loop(self):
        """Main loop for generating synthetic data"""
        dt = 1.0 / self.sampling_rate  # Time between samples
        
        while self.running:
            # Generate one sample for each channel
            samples = []
            for channel in range(self.num_channels):
                sample = self._generate_sample(self.time_elapsed, channel)
                samples.append(sample)
                
            # Add to buffer (format: [timestamp, ch1, ch2, ch3, ch4])
            data_point = [self.time_elapsed] + samples
            
            with self.lock:
                self.buffer.append(data_point)
                
            # Advance time
            self.time_elapsed += dt
            
            # Sleep to maintain sampling rate
            time.sleep(dt)
            
    def get_board_data(self, num_samples: int) -> np.ndarray:
        """
        Get recent data from buffer (mimics BrainFlow interface)
        
        Returns:
            numpy array of shape (num_channels + 1, num_samples)
            First row is timestamps, rest are channel data
        """
        with self.lock:
            if len(self.buffer) == 0:
                return np.array([]).reshape(self.num_channels + 1, 0)
                
            # Get last num_samples
            n = min(num_samples, len(self.buffer))
            recent_data = list(self.buffer)[-n:]
            
            # Convert to numpy array and transpose
            # Shape: (n_samples, n_channels+1) -> (n_channels+1, n_samples)
            data_array = np.array(recent_data).T
            
            return data_array
            
    def get_current_board_data(self, max_samples: int = 250) -> np.ndarray:
        """
        Get all available data (mimics BrainFlow interface)
        
        Returns:
            numpy array of shape (num_channels + 1, actual_samples)
        """
        return self.get_board_data(max_samples)
    
    # ---------- Data access helpers (mimicking EEGService) ----------
    
    def get_timeseries_window(
        self,
        window_sec: float = 4.0,
        max_points: int = 512,
    ):
        """
        Returns (channel_names, data[channels][samples])
        """
        if not (self.connected and self.streaming):
            return [], []
        
        n_samples = int(window_sec * self.sampling_rate)
        data = self.get_board_data(n_samples)
        
        if data.shape[1] == 0:
            return [], []
        
        # Skip timestamp row, get channel data
        ts_data = []
        for ch_idx in range(self.num_channels):
            channel_series = data[ch_idx + 1, :]  # +1 to skip timestamp
            
            # Downsample if too many points
            if channel_series.size > max_points:
                step = int(np.floor(channel_series.size / max_points))
                channel_series = channel_series[::step]
            
            ts_data.append(channel_series.tolist())
        
        channel_names = [f"CH{idx+1}" for idx in range(self.num_channels)]
        return channel_names, ts_data
    
    def get_fft_spectrum(
        self,
        window_sec: float = 4.0,
        min_freq: float = 0.5,
        max_freq: float = 40.0,
    ):
        """
        Returns (channel_names, freqs, psd[channels][freqs])
        """
        if not (self.connected and self.streaming):
            return [], [], []
        
        n_samples = int(window_sec * self.sampling_rate)
        data = self.get_board_data(n_samples)
        
        if data.shape[1] == 0:
            return [], [], []
        
        channel_names = [f"CH{idx+1}" for idx in range(self.num_channels)]
        all_psd = []
        freq_list = []
        
        for ch_idx in range(self.num_channels):
            sig = data[ch_idx + 1, :]  # +1 to skip timestamp
            
            if sig.size < 32:
                all_psd.append([])
                continue
            
            # Simple FFT-based PSD
            fft_result = np.fft.rfft(sig)
            psd = np.abs(fft_result) ** 2 / len(sig)
            freqs = np.fft.rfftfreq(len(sig), 1.0 / self.sampling_rate)
            
            # Filter to frequency range
            mask = (freqs >= min_freq) & (freqs <= max_freq)
            filtered_freqs = freqs[mask]
            filtered_psd = psd[mask]
            
            # Convert to dB scale
            psd_db = 10 * np.log10(filtered_psd + 1e-10)
            
            if ch_idx == 0:
                freq_list = filtered_freqs.tolist()
            
            all_psd.append(psd_db.tolist())
        
        return channel_names, freq_list, all_psd
    
    def get_band_powers(
        self,
        window_sec: float = 4.0,
        bands=None,
        use_relative: bool = True,
        apply_smoothing: bool = True,
    ):
        """
        Returns (channel_names, band_names, band_values[channels][bands])
        """
        if bands is None:
            bands = [
                ("delta", 0.5, 4.0),
                ("theta", 4.0, 8.0),
                ("alpha", 8.0, 13.0),
                ("beta", 13.0, 30.0),
                ("gamma", 30.0, 40.0),
            ]

        if not (self.connected and self.streaming):
            return [], [b[0] for b in bands], []

        n_samples = int(window_sec * self.sampling_rate)
        data = self.get_board_data(n_samples)

        if data.shape[1] == 0:
            return [], [b[0] for b in bands], []

        band_names = [b[0] for b in bands]
        channel_names = [f"CH{idx+1}" for idx in range(self.num_channels)]
        all_band_vals = []

        for ch_idx in range(self.num_channels):
            ch_name = f"CH{ch_idx+1}"
            sig = data[ch_idx + 1, :]  # +1 to skip timestamp

            if sig.size < 32:
                all_band_vals.append([0.0] * len(bands))
                continue

            # Calculate FFT
            fft_result = np.fft.rfft(sig)
            psd = np.abs(fft_result) ** 2 / len(sig)
            freqs = np.fft.rfftfreq(len(sig), 1.0 / self.sampling_rate)

            # Calculate band powers
            ch_band_vals = []
            for _, fmin, fmax in bands:
                band_mask = (freqs >= fmin) & (freqs < fmax)
                band_power = np.sum(psd[band_mask])
                ch_band_vals.append(float(band_power))

            # Convert to relative power (percentage) if requested
            if use_relative:
                total_power = sum(ch_band_vals)
                if total_power > 0:
                    ch_band_vals = [(bp / total_power) * 100 for bp in ch_band_vals]
                else:
                    ch_band_vals = [0.0] * len(bands)

            # Apply smoothing if enabled
            if apply_smoothing and self.smoothing_enabled:
                if ch_name not in self.band_smoothers:
                    self.band_smoothers[ch_name] = SmoothingBuffer(
                        alpha=self.smoothing_alpha, num_values=len(bands)
                    )
                ch_band_vals = self.band_smoothers[ch_name].update(ch_band_vals)

            all_band_vals.append(ch_band_vals)

        return channel_names, band_names, all_band_vals

    def get_band_powers_with_artifacts(
        self,
        window_sec: float = 4.0,
        bands=None,
        use_relative: bool = True,
    ):
        """
        Returns band powers with artifact info (simulator always returns clean).
        Returns: (channel_names, band_names, band_values, artifact_flags, signal_quality)
        """
        channels, band_names, values = self.get_band_powers(
            window_sec=window_sec, bands=bands, use_relative=use_relative
        )
        # Simulator generates clean data - no artifacts
        artifact_flags = [False] * len(channels)
        signal_quality = [1.0] * len(channels)
        return channels, band_names, values, artifact_flags, signal_quality

    def get_engagement_index(
        self,
        window_sec: float = 4.0,
    ):
        """
        Calculate engagement index: Beta / (Alpha + Theta)
        Returns: (channel_names, per_channel_values, cross_channel_average)
        """
        if not (self.connected and self.streaming):
            return [], [], 0.0

        channels, band_names, values = self.get_band_powers(
            window_sec=window_sec,
            use_relative=False,
            apply_smoothing=False
        )

        if not channels or not values:
            return [], [], 0.0

        engagement_values = []
        for ch_bands in values:
            if len(ch_bands) < 5:
                engagement_values.append(0.0)
                continue

            theta = ch_bands[1]
            alpha = ch_bands[2]
            beta = ch_bands[3]

            denominator = alpha + theta + 0.001
            engagement = beta / denominator
            engagement = max(0.0, min(5.0, engagement))
            engagement_values.append(engagement)

        if self.smoothing_enabled:
            engagement_values = self.engagement_smoother.update(engagement_values)

        avg_engagement = float(np.mean(engagement_values)) if engagement_values else 0.0
        return channels, engagement_values, avg_engagement

    # ---------- OSC glue ----------

    def configure_osc(self, ip: str, port: int, enabled: bool,
                      send_raw: bool, send_bands: bool):
        self.osc.configure(ip, port, enabled, send_raw, send_bands)

    def osc_push_timeseries(self, channel_names: List[str], data: List[List[float]]):
        self.osc.send_timeseries(channel_names, data)

    def osc_push_bands(self, channel_names: List[str],
                       band_names: List[str],
                       values: List[List[float]]):
        self.osc.send_bands(channel_names, band_names, values)
