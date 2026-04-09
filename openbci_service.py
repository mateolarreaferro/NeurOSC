# openbci_service.py
import threading
from typing import Optional, List, Tuple, Dict
from collections import deque

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations

from osc_sender import OSCSender


# ─── DSP Pipeline ────────────────────────────────────────────────────────────

class DSPPipeline:
    """
    Windowed EEG signal processing pipeline.

    All filters are applied statelessly (sosfiltfilt) on each window.
    This is correct because get_current_board_data returns overlapping
    sliding windows, not sequential non-overlapping chunks.

    Chain: Detrend → CAR → Bandpass (1-45 Hz) → Notch (60 Hz)
    """

    def __init__(self, sampling_rate: int = 200, num_channels: int = 4):
        self.fs = sampling_rate
        self.n_ch = num_channels
        self.enabled = True

        # Filter design (done once, applied statelessly per window)
        self._design_filters()

        # Feature toggles
        self.car_enabled = True
        self.bandpass_enabled = True
        self.notch_enabled = True

        # Artifact detection
        self.artifact_enabled = True
        self.amplitude_threshold = 100.0   # µV
        self.gradient_threshold = 50.0     # µV per sample
        self.flatline_var_threshold = 0.5  # µV²

        # Band power normalization (for OSC output, not chart)
        self.log_transform = False          # off by default - chart uses log scale already
        self.baseline_normalize = False     # off by default - needs 30s to stabilize
        self.baseline_window_sec = 30.0
        self._baseline_history: Dict[str, Dict[str, deque]] = {}
        self._baseline_max_samples = int(self.baseline_window_sec / 0.1)

    def _design_filters(self):
        """Design filter coefficients. Called once or when fs changes."""
        # Bandpass: 1-45 Hz, 4th order Butterworth
        self.bp_sos = butter(4, [1.0, 45.0], btype='band', fs=self.fs, output='sos')

        # Notch: 60 Hz, Q=30 (~2 Hz bandwidth)
        b_notch, a_notch = iirnotch(60.0, 30.0, self.fs)
        # Pack into SOS format for sosfiltfilt
        self.notch_sos = np.array([[b_notch[0], b_notch[1], b_notch[2],
                                    1.0, a_notch[1], a_notch[2]]])

    def set_sampling_rate(self, fs: int):
        if fs != self.fs:
            self.fs = fs
            self._design_filters()

    def reset(self):
        """Reset baseline history (call on reconnect)."""
        self._baseline_history.clear()

    # ── Spatial filter ──

    def apply_car(self, data: np.ndarray) -> np.ndarray:
        """
        Common Average Reference: subtract mean of all channels per sample.
        data shape: (n_channels, n_samples)
        """
        if not self.car_enabled or data.shape[0] < 2:
            return data
        return data - data.mean(axis=0, keepdims=True)

    # ── Temporal filters (stateless, safe for overlapping windows) ──

    def filter_channel(self, signal: np.ndarray) -> np.ndarray:
        """Apply bandpass + notch to a single channel window.
        Uses sosfiltfilt (zero-phase, no edge artifacts, stateless)."""
        if not self.enabled or signal.size < 32:
            return signal

        out = signal.copy().astype(np.float64)

        # Detrend first (removes DC offset and linear drift)
        DataFilter.detrend(out, DetrendOperations.LINEAR.value)

        if self.bandpass_enabled and out.size >= 32:
            try:
                out = sosfiltfilt(self.bp_sos, out)
            except Exception:
                pass  # If signal too short for filter, skip

        if self.notch_enabled and out.size >= 16:
            try:
                out = sosfiltfilt(self.notch_sos, out)
            except Exception:
                pass

        return out

    # ── Artifact detection ──

    def detect_artifacts(self, signal: np.ndarray) -> Tuple[bool, str]:
        if not self.artifact_enabled or signal.size < 4:
            return False, ""
        if np.max(np.abs(signal)) > self.amplitude_threshold:
            return True, "amplitude"
        gradient = np.abs(np.diff(signal))
        if np.max(gradient) > self.gradient_threshold:
            return True, "gradient"
        if np.var(signal) < self.flatline_var_threshold:
            return True, "flatline"
        return False, ""

    def get_signal_quality(self, signal: np.ndarray) -> float:
        if signal.size < 4:
            return 0.0
        score = 1.0
        max_amp = np.max(np.abs(signal))
        if max_amp > self.amplitude_threshold:
            score -= 0.4
        elif max_amp > self.amplitude_threshold * 0.7:
            score -= 0.2
        max_grad = np.max(np.abs(np.diff(signal)))
        if max_grad > self.gradient_threshold:
            score -= 0.3
        elif max_grad > self.gradient_threshold * 0.7:
            score -= 0.15
        if np.var(signal) < self.flatline_var_threshold:
            score -= 0.3
        return max(0.0, min(1.0, score))

    # ── Band power normalization (optional, for OSC) ──

    def normalize_band_power(self, ch_name: str, band_name: str, raw_power: float) -> float:
        """Log10 → z-score against rolling baseline."""
        value = raw_power

        if self.log_transform:
            value = np.log10(max(value, 1e-10))

        if not self.baseline_normalize:
            return float(value)

        if ch_name not in self._baseline_history:
            self._baseline_history[ch_name] = {}
        if band_name not in self._baseline_history[ch_name]:
            self._baseline_history[ch_name][band_name] = deque(maxlen=self._baseline_max_samples)

        history = self._baseline_history[ch_name][band_name]
        history.append(value)

        if len(history) < 10:
            return float(value)

        mean = np.mean(history)
        std = np.std(history)
        if std < 1e-10:
            return 0.0
        return float((value - mean) / std)

    def configure(self, **kwargs):
        for key, val in kwargs.items():
            if val is not None and hasattr(self, key):
                setattr(self, key, val)


# ─── Smoothing ────────────────────────────────────────────────────────────────

class SmoothingBuffer:
    """Exponential moving average buffer."""

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


# ─── Device Definitions ──────────────────────────────────────────────────────

DEVICE_CONFIGS = {
    "ganglion": {
        "label": "OpenBCI Ganglion",
        "board_id_serial": BoardIds.GANGLION_BOARD,
        "board_id_ble": BoardIds.GANGLION_NATIVE_BOARD,
        "channel_names": ["CH1", "CH2", "CH3", "CH4"],
        "needs_board_config": True,
    },
    "muse_2": {
        "label": "Muse 2",
        "board_id_serial": BoardIds.MUSE_2_BLED_BOARD,
        "board_id_ble": BoardIds.MUSE_2_BOARD,
        "channel_names": ["TP9", "AF7", "AF8", "TP10"],
        "needs_board_config": False,
    },
    "muse_3": {
        "label": "Muse 3",
        "board_id_serial": BoardIds.MUSE_2_BLED_BOARD,
        "board_id_ble": BoardIds.MUSE_2_BOARD,
        "channel_names": ["TP9", "AF7", "AF8", "TP10"],
        "needs_board_config": False,
    },
    "muse_athena": {
        "label": "Muse Athena",
        "board_id_serial": BoardIds.MUSE_S_BLED_BOARD,
        "board_id_ble": BoardIds.MUSE_S_BOARD,
        "channel_names": ["TP9", "AF7", "AF8", "TP10"],
        "needs_board_config": False,
    },
}


# ─── Main Service ─────────────────────────────────────────────────────────────

class EEGService:
    def __init__(self):
        self.board: Optional[BoardShim] = None
        self.board_id = None
        self.device_type: str = "ganglion"
        self.params = BrainFlowInputParams()
        self.connected = False
        self.streaming = False
        self.lock = threading.Lock()
        self.osc = OSCSender()

        # DSP pipeline
        self.dsp = DSPPipeline(sampling_rate=200, num_channels=4)

        # Smoothing (per-channel buffers for bands)
        self.smoothing_alpha = 0.3
        self.smoothing_enabled = True
        self.band_smoothers: Dict[str, SmoothingBuffer] = {}

    def configure_smoothing(self, enabled: bool, alpha: float = 0.3):
        self.smoothing_enabled = enabled
        self.smoothing_alpha = max(0.05, min(1.0, alpha))
        for smoother in self.band_smoothers.values():
            smoother.enabled = enabled
            smoother.alpha = self.smoothing_alpha

    def _get_channel_names(self) -> List[str]:
        config = DEVICE_CONFIGS.get(self.device_type, DEVICE_CONFIGS["ganglion"])
        return config["channel_names"]

    # ---------- Connection ----------

    def connect(self, serial_port: str = "", mac_address: str = "",
                timeout: int = 15, device_type: str = "ganglion"):
        if self.connected:
            return

        if device_type not in DEVICE_CONFIGS:
            raise ValueError(f"Unknown device type: {device_type}. "
                             f"Supported: {list(DEVICE_CONFIGS.keys())}")

        self.device_type = device_type
        config = DEVICE_CONFIGS[device_type]

        if serial_port:
            self.board_id = config["board_id_serial"]
            self.params.serial_port = serial_port
            self.params.mac_address = ""
        else:
            self.board_id = config["board_id_ble"]
            self.params.serial_port = ""
            self.params.mac_address = mac_address

        self.params.timeout = timeout
        BoardShim.enable_dev_board_logger()
        self.board = BoardShim(self.board_id, self.params)
        self.board.prepare_session()
        self.connected = True

        # Update DSP sampling rate and reset state
        self.dsp.set_sampling_rate(BoardShim.get_sampling_rate(self.board_id))
        self.dsp.reset()
        self.band_smoothers.clear()

        if config["needs_board_config"]:
            try:
                import time
                time.sleep(0.5)
                self.board.config_board("]")
                time.sleep(0.1)
                for cmd in ["!", "@", "#", "$"]:
                    self.board.config_board(cmd)
                time.sleep(0.1)
            except Exception as e:
                print(f"[Neuro] Board config warning: {e}")

    def disconnect(self):
        if not self.connected or self.board is None:
            return
        if self.streaming:
            self.stop_stream()
        self.board.release_session()
        self.board = None
        self.connected = False
        self.dsp.reset()

    # ---------- Streaming ----------

    def start_stream(self, buffer_size: int = 45000):
        if not self.connected or self.board is None:
            raise RuntimeError("Board not connected")
        if self.streaming:
            return
        config = DEVICE_CONFIGS.get(self.device_type, DEVICE_CONFIGS["ganglion"])
        if config["needs_board_config"]:
            import time
            try:
                self.board.config_board("]")
                time.sleep(0.2)
            except Exception:
                pass
        self.board.start_stream(buffer_size)
        self.streaming = True

    def stop_stream(self):
        if not self.streaming or self.board is None:
            return
        self.board.stop_stream()
        self.streaming = False

    # ---------- Data access ----------

    def _get_exg_channels(self) -> List[int]:
        if self.board_id is None:
            return []
        return BoardShim.get_exg_channels(self.board_id)

    def _get_processed_window(self, window_sec: float) -> Tuple[np.ndarray, int]:
        """
        Get a window of data from the board and apply the DSP chain.
        Returns (eeg: shape [n_ch, n_samples], sampling_rate).

        The full chain:
          1. Extract raw EEG channels
          2. CAR (spatial)
          3. Per-channel: detrend → bandpass → notch (all stateless/zero-phase)
        """
        sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        n_samples = int(window_sec * sampling_rate)

        with self.lock:
            data = self.board.get_current_board_data(n_samples)

        ch_indices = self._get_exg_channels()
        if data.shape[1] == 0:
            return np.array([]), sampling_rate

        # Extract EEG channels → (n_channels, n_samples)
        eeg = np.array([data[ch, :].astype(np.float64) for ch in ch_indices])

        # 1. Common Average Reference (spatial)
        eeg = self.dsp.apply_car(eeg)

        # 2. Per-channel temporal filtering (stateless, safe for overlapping windows)
        if self.dsp.enabled:
            for i in range(eeg.shape[0]):
                eeg[i] = self.dsp.filter_channel(eeg[i])

        return eeg, sampling_rate

    def get_timeseries_window(
        self,
        window_sec: float = 4.0,
        max_points: int = 512,
    ) -> Tuple[List[str], List[List[float]]]:
        if not (self.connected and self.streaming and self.board):
            return [], []

        eeg, _ = self._get_processed_window(window_sec)
        if eeg.size == 0:
            return [], []

        ts_data = []
        for i in range(eeg.shape[0]):
            ch = eeg[i]
            if ch.size > max_points:
                step = int(np.floor(ch.size / max_points))
                ch = ch[::step]
            ts_data.append(ch.tolist())

        names = self._get_channel_names()
        channel_names = names[:eeg.shape[0]] if len(names) >= eeg.shape[0] else [f"CH{i+1}" for i in range(eeg.shape[0])]
        return channel_names, ts_data

    def get_fft_spectrum(
        self,
        window_sec: float = 4.0,
        min_freq: float = 0.5,
        max_freq: float = 45.0,
    ) -> Tuple[List[str], List[float], List[List[float]]]:
        if not (self.connected and self.streaming and self.board):
            return [], [], []

        eeg, sampling_rate = self._get_processed_window(window_sec)
        if eeg.size == 0:
            return [], [], []

        names = self._get_channel_names()
        channel_names = names[:eeg.shape[0]] if len(names) >= eeg.shape[0] else [f"CH{i+1}" for i in range(eeg.shape[0])]
        all_psd: List[List[float]] = []
        freq_list: List[float] = []

        for ch_idx in range(eeg.shape[0]):
            sig = eeg[ch_idx]
            if sig.size < 32:
                all_psd.append([])
                continue

            fft_len = DataFilter.get_nearest_power_of_two(sig.size)
            while fft_len >= sig.size and fft_len > 2:
                fft_len = fft_len // 2
            overlap = fft_len // 2

            psd, freqs = DataFilter.get_psd_welch(
                sig, fft_len, overlap, sampling_rate,
                WindowOperations.HANNING.value
            )

            mask = (freqs >= min_freq) & (freqs <= max_freq)
            psd_db = 10 * np.log10(psd[mask] + 1e-10)

            if ch_idx == 0:
                freq_list = freqs[mask].tolist()
            all_psd.append(psd_db.tolist())

        return channel_names, freq_list, all_psd

    def get_band_powers(
        self,
        window_sec: float = 4.0,
        bands: Optional[List[Tuple[str, float, float]]] = None,
        use_relative: bool = True,
        apply_smoothing: bool = True,
    ) -> Tuple[List[str], List[str], List[List[float]]]:
        if bands is None:
            bands = [
                ("delta", 1.0, 4.0),
                ("theta", 4.0, 8.0),
                ("alpha", 8.0, 13.0),
                ("beta", 13.0, 30.0),
                ("gamma", 30.0, 45.0),
            ]

        if not (self.connected and self.streaming and self.board):
            return [], [b[0] for b in bands], []

        eeg, sampling_rate = self._get_processed_window(window_sec)
        if eeg.size == 0:
            return [], [b[0] for b in bands], []

        band_names = [b[0] for b in bands]
        names = self._get_channel_names()
        channel_names = names[:eeg.shape[0]] if len(names) >= eeg.shape[0] else [f"CH{i+1}" for i in range(eeg.shape[0])]
        all_band_vals: List[List[float]] = []

        for ch_idx in range(eeg.shape[0]):
            ch_name = channel_names[ch_idx]
            sig = eeg[ch_idx]
            if sig.size < 32:
                all_band_vals.append([0.0] * len(bands))
                continue

            fft_len = DataFilter.get_nearest_power_of_two(sig.size)
            while fft_len >= sig.size and fft_len > 2:
                fft_len = fft_len // 2
            overlap = fft_len // 2

            psd_tuple = DataFilter.get_psd_welch(
                sig, fft_len, overlap, sampling_rate,
                WindowOperations.HANNING.value
            )

            ch_band_vals: List[float] = []
            for _, fmin, fmax in bands:
                bp = DataFilter.get_band_power(psd_tuple, fmin, fmax)
                ch_band_vals.append(float(bp))

            if use_relative:
                total_power = sum(ch_band_vals)
                if total_power > 0:
                    ch_band_vals = [(bp / total_power) * 100 for bp in ch_band_vals]
                else:
                    ch_band_vals = [0.0] * len(bands)

            # Optional normalization (off by default, for OSC use)
            if self.dsp.log_transform or self.dsp.baseline_normalize:
                ch_band_vals = [
                    self.dsp.normalize_band_power(ch_name, band_names[i], v)
                    for i, v in enumerate(ch_band_vals)
                ]

            # EMA smoothing
            if apply_smoothing and self.smoothing_enabled:
                if ch_name not in self.band_smoothers:
                    self.band_smoothers[ch_name] = SmoothingBuffer(
                        alpha=self.smoothing_alpha, num_values=len(bands)
                    )
                ch_band_vals = self.band_smoothers[ch_name].update(ch_band_vals)

            all_band_vals.append(ch_band_vals)

        return channel_names, band_names, all_band_vals

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
