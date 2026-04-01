# NeurOSC

Real-time EEG visualization and OSC streaming for OpenBCI Ganglion.

Connect your Ganglion, see your brain waves, and send band powers to any creative app over OSC.

---

## Quick Start (no coding required)

1. **Unzip** `Neuro.zip`
2. **Right-click** `Neuro.app` and choose **Open** (don't double-click — macOS blocks unsigned apps the first time)
3. Click **Open** in the security dialog that appears
4. The app opens in your browser automatically

That's it. No Python, no terminal, no installs.

---

## Connecting Your Ganglion

1. Turn on the Ganglion board
2. Pair it in **System Settings > Bluetooth** (look for "Ganglion")
3. In the app, leave the serial port field empty and click **Connect**
4. Click **Start** to begin streaming

If you're using a **BLED112 USB dongle** instead of Bluetooth, enter the serial port path (e.g. `/dev/tty.usbmodem1`) before connecting.

---

## What You'll See

Three visualization modes (switch with the tabs at the top):

- **Traces** — raw 4-channel EEG waveforms in real time
- **FFT** — frequency spectrum (power spectral density)
- **Bands** — bar chart of delta, theta, alpha, beta, gamma power per channel

---

## Sending OSC

The right panel controls OSC output:

1. Set the **IP** and **port** of the app you want to receive data (default: `127.0.0.1:9000`)
2. Click **Enable OSC**
3. Choose what to send:
   - **Data types**: Raw timeseries, Band powers, or both
   - **Channels**: Toggle CH1–CH4 individually
   - **Bands**: Toggle delta, theta, alpha, beta, gamma individually
   - **Values**: Absolute, Relative (0–1 normalized), Averages (cross-channel mean/min/max)
4. The **Address Preview** at the bottom shows every OSC message being sent

### OSC addresses

| Address | Data |
|---------|------|
| `/neuro/raw/CH1` | Raw timeseries for channel 1 |
| `/neuro/bands/CH1/alpha` | Alpha power for channel 1 |
| `/neuro/bands/CH1/alpha-relative` | Alpha power normalized 0–1 |
| `/neuro/bands/alpha` | Alpha averaged across all enabled channels |
| `/neuro/bands/alpha/max` | Max alpha across channels |
| `/neuro/bands/alpha/min` | Min alpha across channels |
| `/neuro/elements/alpha_absolute` | All channels in one message [CH1, CH2, CH3, CH4] |
| `/neuro/elements/alpha_relative` | All channels normalized [0–1, 0–1, 0–1, 0–1] |

The address prefix (`/neuro`) can be changed in the **Address Prefix** field.

### Receiving in Max/MSP

```
[udpreceive 9000]
|
[OSC-route /neuro/bands]
|
[OSC-route /alpha]
```

### Receiving in TouchDesigner

Add an **OSC In CHOP**, set port to `9000`, then use a **Select CHOP** to pick channels.

---

## DSP Pipeline

Click **DSP Pipeline** in the sidebar to configure signal processing:

- **CAR** — Common Average Reference (spatial noise reduction)
- **BP 1-45Hz** — Bandpass filter (removes drift and high-frequency noise)
- **Notch 60Hz** — Removes power line interference
- **Artifacts** — Rejects bad data (amplitude spikes, movement, flatline)
- **Log / Z-Score** — Optional normalization for OSC output
- **Smooth** — Adjustable smoothing (slide left = smoother, right = more responsive)

All filters are on by default. Log and Z-Score are off by default.

---

## Simulator Mode

Don't have the Ganglion nearby? Click **Enable** under Simulator, then Connect and Start. You'll get synthetic brain waves with selectable states: Normal, Meditation, Focused, Drowsy.

---

## Running from Source (developers)

**Requirements:** Python 3.9+

```bash
# Clone the repo
git clone <repo-url>
cd NeurOSC

# Create a virtual environment and activate it
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Run
python launcher.py
```

### Rebuilding the app

```bash
pip install pyinstaller
bash build.sh
# Output: dist/Neuro.app
```

---

## Troubleshooting

**App won't open / "damaged" warning**
Right-click > Open. If that doesn't work, run this in Terminal once:
```
xattr -cr /path/to/Neuro.app
```

**Ganglion won't connect**
- Make sure it's paired in System Settings > Bluetooth
- Try power cycling the board
- If using the BLED112 dongle, enter the serial port path

**No data showing**
- Make sure you clicked both **Connect** and **Start**
- Check the status indicator (green dot = connected, "Live" pill = streaming)

**OSC not arriving**
- Make sure OSC is enabled (green "Live" indicator in the OSC panel)
- Check that at least one data type (Raw or Bands) is toggled on
- Verify the port matches your receiving app
