# osc_sender.py
from typing import List, Dict, Optional, Set
from pythonosc.udp_client import SimpleUDPClient
import logging
import numpy as np

logger = logging.getLogger(__name__)


class OSCSender:
    def __init__(self, ip: str = "127.0.0.1", port: int = 9000):
        self.ip = ip
        self.port = port
        self.client: Optional[SimpleUDPClient] = None
        self.enabled: bool = False
        self.send_raw: bool = True
        self.send_bands_enabled: bool = False
        self.max_floats_per_message = 1000

        # Granular OSC controls
        self.enabled_channels: Set[str] = {"CH1", "CH2", "CH3", "CH4"}
        self.enabled_bands: Set[str] = {"delta", "theta", "alpha", "beta", "gamma"}
        self.send_absolute: bool = True
        self.send_relative: bool = True
        self.send_averages: bool = True
        self.osc_prefix: str = "/neuro"

        # Fixed normalization ranges (uV^2)
        self.band_ranges = {
            'delta': (0.5, 100.0),
            'theta': (0.5, 50.0),
            'alpha': (1.0, 100.0),
            'beta': (0.5, 30.0),
            'gamma': (0.5, 20.0)
        }

    def configure(self, ip: str, port: int, enabled: bool,
                  send_raw: bool, send_bands: bool):
        self.ip = ip
        self.port = port
        self.enabled = enabled
        self.send_raw = send_raw
        self.send_bands_enabled = send_bands
        if enabled:
            self.client = SimpleUDPClient(self.ip, self.port)
        else:
            self.client = None

    def configure_granular(self, channels: Optional[List[str]] = None,
                           bands: Optional[List[str]] = None,
                           send_absolute: Optional[bool] = None,
                           send_relative: Optional[bool] = None,
                           send_averages: Optional[bool] = None,
                           osc_prefix: Optional[str] = None):
        if channels is not None:
            self.enabled_channels = set(channels)
        if bands is not None:
            self.enabled_bands = set(bands)
        if send_absolute is not None:
            self.send_absolute = send_absolute
        if send_relative is not None:
            self.send_relative = send_relative
        if send_averages is not None:
            self.send_averages = send_averages
        if osc_prefix is not None:
            self.osc_prefix = osc_prefix.rstrip("/")

    def _ensure_client(self):
        return self.enabled and self.client is not None

    def send_timeseries(self, channel_names: List[str], data: List[List[float]]):
        if not self._ensure_client() or not self.send_raw:
            return

        try:
            for ch_idx, ch_data in enumerate(data):
                ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"CH{ch_idx+1}"
                if ch_name not in self.enabled_channels:
                    continue

                if len(ch_data) <= self.max_floats_per_message:
                    self.client.send_message(f"{self.osc_prefix}/raw/{ch_name}", ch_data)
                else:
                    for chunk_idx, i in enumerate(range(0, len(ch_data), self.max_floats_per_message)):
                        chunk = ch_data[i:i + self.max_floats_per_message]
                        self.client.send_message(f"{self.osc_prefix}/raw/{ch_name}/chunk{chunk_idx}", chunk)

            # Combined message with enabled channels only
            all_data = []
            for ch_idx, ch_data in enumerate(data):
                ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"CH{ch_idx+1}"
                if ch_name in self.enabled_channels:
                    all_data.extend(ch_data)

            if all_data:
                if len(all_data) <= self.max_floats_per_message:
                    self.client.send_message(f"{self.osc_prefix}/raw", all_data)
                else:
                    for chunk_idx, i in enumerate(range(0, len(all_data), self.max_floats_per_message)):
                        chunk = all_data[i:i + self.max_floats_per_message]
                        self.client.send_message(f"{self.osc_prefix}/raw/chunk{chunk_idx}", chunk)

        except Exception as e:
            logger.error(f"Failed to send timeseries OSC: {e}")

    def _normalize_band_value(self, band_name: str, value: float) -> float:
        min_val, max_val = self.band_ranges.get(band_name.lower(), (0.0, 100.0))
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))

    def send_bands(
        self,
        channel_names: List[str],
        bands: List[str],
        values: List[List[float]],
    ):
        if not self._ensure_client() or not self.send_bands_enabled:
            return

        try:
            n_channels = len(values)
            values_array = np.array(values)

            # Per-channel individual band powers
            for ch_idx, ch_vals in enumerate(values):
                ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"CH{ch_idx+1}"
                if ch_name not in self.enabled_channels:
                    continue

                for band_idx, band_name in enumerate(bands):
                    if band_name not in self.enabled_bands:
                        continue

                    abs_value = ch_vals[band_idx]

                    if self.send_absolute:
                        self.client.send_message(f"{self.osc_prefix}/bands/{ch_name}/{band_name}", abs_value)

                    if self.send_relative:
                        rel_value = self._normalize_band_value(band_name, abs_value)
                        self.client.send_message(f"{self.osc_prefix}/bands/{ch_name}/{band_name}-relative", rel_value)

            # Cross-channel aggregates
            if self.send_averages:
                # Filter to enabled channels
                enabled_indices = [i for i, ch in enumerate(channel_names) if ch in self.enabled_channels]

                for band_idx, band_name in enumerate(bands):
                    if band_name not in self.enabled_bands:
                        continue

                    band_values = values_array[enabled_indices, band_idx] if enabled_indices else values_array[:, band_idx]

                    mean_val = float(np.mean(band_values))
                    self.client.send_message(f"{self.osc_prefix}/bands/{band_name}", mean_val)

                    max_val = float(np.max(band_values))
                    min_val = float(np.min(band_values))
                    self.client.send_message(f"{self.osc_prefix}/bands/{band_name}/max", max_val)
                    self.client.send_message(f"{self.osc_prefix}/bands/{band_name}/min", min_val)

                # Muse-compatible combined messages
                for band_idx, band_name in enumerate(bands):
                    if band_name not in self.enabled_bands:
                        continue

                    abs_values = [float(values[ch_idx][band_idx])
                                  for ch_idx in range(n_channels)
                                  if channel_names[ch_idx] in self.enabled_channels]
                    self.client.send_message(f"{self.osc_prefix}/elements/{band_name}_absolute", abs_values)

                    rel_values = [self._normalize_band_value(band_name, values[ch_idx][band_idx])
                                  for ch_idx in range(n_channels)
                                  if channel_names[ch_idx] in self.enabled_channels]
                    self.client.send_message(f"{self.osc_prefix}/elements/{band_name}_relative", rel_values)

        except Exception as e:
            logger.error(f"Failed to send bands OSC: {e}")
