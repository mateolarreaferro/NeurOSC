# main.py
from typing import Optional
from pathlib import Path
import glob
import sys
import asyncio

import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from openbci_service import EEGService, DEVICE_CONFIGS
from simulator_service import SimulatorService

app = FastAPI(title="Neuro - EEG Band Visualization & OSC")

# Resolve paths: use PyInstaller's _MEIPASS when bundled, otherwise this file's dir
if getattr(sys, '_MEIPASS', None):
    _HERE = Path(sys._MEIPASS)
else:
    _HERE = Path(__file__).resolve().parent

# Static files mount (create dir if missing - e.g. empty in bundled app)
_static_dir = _HERE / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

templates = Jinja2Templates(directory=str(_HERE / "templates"))

# Global service instances
service = EEGService()
simulator = SimulatorService()

# Mode flag
USE_SIMULATOR = False


def get_active_service():
    """Always return the currently active service - never cache this."""
    return simulator if USE_SIMULATOR else service


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -------- REST API --------

@app.post("/api/connect")
async def api_connect(payload: dict):
    active = get_active_service()

    if USE_SIMULATOR:
        try:
            active.connect()
            return {"status": "ok", "simulator": True}
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    else:
        serial_port = payload.get("serial_port", "")
        mac_address = payload.get("mac_address", "")
        device_type = payload.get("device_type", "ganglion")
        timeout = int(payload.get("timeout", 15))
        try:
            active.connect(serial_port=serial_port, mac_address=mac_address,
                           timeout=timeout, device_type=device_type)
            return {"status": "ok", "simulator": False, "device_type": device_type}
        except Exception as e:
            msg = str(e)
            # Provide actionable guidance for common BrainFlow errors
            if "17" in msg or "GENERAL_ERROR" in msg:
                msg += (" — This usually means BrainFlow native libraries are missing "
                        "from the app bundle. If you repackaged the .app yourself, "
                        "rebuild with the official build.sh script. "
                        "Check /api/diagnostics for details.")
            elif "BOARD_NOT_READY" in msg:
                msg += (" — Make sure no other app (e.g. OpenBCI GUI) is using "
                        "the serial port, and that the dongle is plugged in.")
            return JSONResponse({"status": "error", "message": msg}, status_code=500)


@app.post("/api/disconnect")
async def api_disconnect():
    active = get_active_service()
    active.disconnect()
    return {"status": "ok"}


@app.get("/api/status")
async def api_status():
    active = get_active_service()
    result = {
        "connected": active.connected,
        "streaming": active.streaming,
        "simulator": USE_SIMULATOR,
    }
    if not USE_SIMULATOR and hasattr(active, 'device_type'):
        result["device_type"] = active.device_type
    return result


@app.get("/api/diagnostics")
async def api_diagnostics():
    """Check BrainFlow native library availability and system info."""
    import os
    import platform

    results = {
        "platform": platform.system(),
        "python": platform.python_version(),
        "bundled": getattr(sys, '_MEIPASS', None) is not None,
    }

    # Check BrainFlow native libs
    try:
        import brainflow
        bf_dir = os.path.dirname(brainflow.__file__)
        lib_dir = os.path.join(bf_dir, "lib")
        results["brainflow_version"] = brainflow.__version__
        results["brainflow_lib_dir"] = lib_dir
        results["brainflow_lib_exists"] = os.path.isdir(lib_dir)

        critical_libs = {
            "BoardController": "Core BrainFlow (required)",
            "GanglionLib": "Ganglion via BLED112 dongle",
            "BrainFlowBluetooth": "Native Bluetooth (Ganglion BLE)",
            "simpleble-c": "SimpleBLE backend (Muse, native BLE)",
            "MuseLib": "Muse device support",
            "DataHandler": "Signal processing (required)",
        }

        lib_status = {}
        if os.path.isdir(lib_dir):
            lib_files = os.listdir(lib_dir)
            for lib_name, description in critical_libs.items():
                found = any(lib_name in f for f in lib_files)
                lib_status[lib_name] = {"found": found, "purpose": description}
        else:
            for lib_name, description in critical_libs.items():
                lib_status[lib_name] = {"found": False, "purpose": description}

        results["native_libs"] = lib_status
        results["all_libs_ok"] = all(v["found"] for v in lib_status.values())
    except ImportError:
        results["brainflow_version"] = None
        results["error"] = "brainflow package not installed"

    # Supported devices
    results["supported_devices"] = [
        {"id": k, "label": v["label"]} for k, v in DEVICE_CONFIGS.items()
    ]

    return results


@app.post("/api/use_simulator")
async def api_use_simulator(payload: dict):
    global USE_SIMULATOR
    use_sim = bool(payload.get("enabled", True))
    USE_SIMULATOR = use_sim

    if service.streaming:
        service.stop_stream()
    if simulator.streaming:
        simulator.stop_stream()

    return {
        "status": "ok",
        "simulator": USE_SIMULATOR,
        "message": "Simulator enabled" if USE_SIMULATOR else "Real hardware enabled"
    }


@app.post("/api/simulator/mode")
async def api_simulator_mode(payload: dict):
    mode = payload.get("mode", "normal")
    simulator.set_mode(mode)
    return {"status": "ok", "mode": mode}


@app.get("/api/devices")
async def api_list_devices():
    """List supported device types."""
    return {
        "devices": [
            {"id": k, "label": v["label"], "channels": v["channel_names"]}
            for k, v in DEVICE_CONFIGS.items()
        ]
    }


@app.get("/api/ports")
async def api_list_ports():
    ports = []
    bluetooth_devices = []

    if sys.platform == "darwin":
        ports.extend(glob.glob("/dev/tty.usbmodem*"))
        ports.extend(glob.glob("/dev/tty.usbserial*"))
        ports.extend(glob.glob("/dev/cu.usbmodem*"))
        ports.extend(glob.glob("/dev/cu.usbserial*"))

        try:
            import subprocess
            result = subprocess.run(
                ["system_profiler", "SPBluetoothDataType"],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.split('\n')
            bt_keywords = ['ganglion', 'muse', 'athena']
            for i, line in enumerate(lines):
                if any(kw in line.lower() for kw in bt_keywords):
                    for j in range(max(0, i-5), min(len(lines), i+10)):
                        if 'Address:' in lines[j]:
                            mac = lines[j].split('Address:')[1].strip()
                            name = line.strip().rstrip(':')
                            # Infer device type from name
                            name_lower = name.lower()
                            if 'athena' in name_lower:
                                dev_type = "muse_athena"
                            elif 'muse-s' in name_lower or 'muse s' in name_lower:
                                dev_type = "muse_athena"
                            elif 'muse' in name_lower and '3' in name_lower:
                                dev_type = "muse_3"
                            elif 'muse' in name_lower:
                                dev_type = "muse_2"
                            else:
                                dev_type = "ganglion"
                            bluetooth_devices.append({
                                "name": name,
                                "mac": mac,
                                "type": "bluetooth",
                                "device_type": dev_type,
                            })
                            break
        except Exception as e:
            print(f"Bluetooth scan error: {e}")

    elif sys.platform.startswith("linux"):
        ports.extend(glob.glob("/dev/ttyUSB*"))
        ports.extend(glob.glob("/dev/ttyACM*"))
    elif sys.platform == "win32":
        import serial.tools.list_ports
        detected = serial.tools.list_ports.comports()
        ports = [port.device for port in detected]

    ports = sorted(list(set(ports)))

    return {
        "ports": ports,
        "bluetooth": bluetooth_devices,
        "count": len(ports) + len(bluetooth_devices),
    }


@app.post("/api/start")
async def api_start(payload: dict):
    active = get_active_service()
    try:
        buffer_size = int(payload.get("buffer_size", 45000))
        active.start_stream(buffer_size=buffer_size)
        return {"status": "ok", "simulator": USE_SIMULATOR}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/stop")
async def api_stop():
    active = get_active_service()
    active.stop_stream()
    return {"status": "ok"}


@app.post("/api/dsp_config")
async def api_dsp_config(payload: dict):
    """Configure DSP pipeline: filters, normalization, artifact detection."""
    service.dsp.configure(**{k: v for k, v in payload.items() if v is not None})
    # Smoothing is separate
    if "smoothing_alpha" in payload:
        service.configure_smoothing(
            enabled=payload.get("smoothing_enabled", service.smoothing_enabled),
            alpha=float(payload["smoothing_alpha"]),
        )
    elif "smoothing_enabled" in payload:
        service.configure_smoothing(
            enabled=payload["smoothing_enabled"],
            alpha=service.smoothing_alpha,
        )
    return {"status": "ok"}


@app.get("/api/dsp_status")
async def api_dsp_status():
    """Get current DSP pipeline configuration."""
    d = service.dsp
    return {
        "car_enabled": d.car_enabled,
        "bandpass_enabled": d.bandpass_enabled,
        "notch_enabled": d.notch_enabled,
        "artifact_enabled": d.artifact_enabled,
        "log_transform": d.log_transform,
        "baseline_normalize": d.baseline_normalize,
        "baseline_window_sec": d.baseline_window_sec,
        "amplitude_threshold": d.amplitude_threshold,
        "gradient_threshold": d.gradient_threshold,
        "smoothing_enabled": service.smoothing_enabled,
        "smoothing_alpha": service.smoothing_alpha,
    }


@app.post("/api/osc_config")
async def api_osc_config(payload: dict):
    ip = payload.get("ip", "127.0.0.1")
    port = int(payload.get("port", 9000))
    enabled = bool(payload.get("enabled", False))
    send_raw = bool(payload.get("send_raw", True))
    send_bands = bool(payload.get("send_bands", False))
    # Configure OSC on both services so it works regardless of mode
    service.configure_osc(ip, port, enabled, send_raw, send_bands)
    simulator.osc.configure(ip, port, enabled, send_raw, send_bands)
    return {"status": "ok"}


@app.post("/api/osc_granular")
async def api_osc_granular(payload: dict):
    """Configure granular OSC output: per-channel, per-band, absolute/relative, prefix"""
    kwargs = dict(
        channels=payload.get("channels"),
        bands=payload.get("bands"),
        send_absolute=payload.get("send_absolute"),
        send_relative=payload.get("send_relative"),
        send_averages=payload.get("send_averages"),
        osc_prefix=payload.get("osc_prefix"),
    )
    # Configure on both services
    service.osc.configure_granular(**kwargs)
    simulator.osc.configure_granular(**kwargs)
    return {"status": "ok"}


# -------- WebSocket for stream --------

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()

    config = {
        "mode": "bands",
        "window_sec": 4.0,
        "send_interval_ms": 100
    }

    stream_task = None

    try:
        init_msg = await ws.receive_json()
        config["mode"] = init_msg.get("mode", "bands")
        config["window_sec"] = float(init_msg.get("window_sec", 4.0))
        config["send_interval_ms"] = int(init_msg.get("interval_ms", 100))

        async def stream_data():
            while True:
                # Re-resolve active service each iteration so simulator toggle works mid-stream
                active = get_active_service()

                if not (active.connected and active.streaming):
                    await asyncio.sleep(0.5)
                    continue

                try:
                    if config["mode"] == "timeseries":
                        channels, data = active.get_timeseries_window(window_sec=config["window_sec"])
                        if channels:
                            active.osc_push_timeseries(channels, data)
                            await ws.send_json({
                                "type": "timeseries",
                                "channels": channels,
                                "data": data,
                            })

                    elif config["mode"] == "fft":
                        channels, freqs, psd = active.get_fft_spectrum(window_sec=config["window_sec"])
                        if channels:
                            await ws.send_json({
                                "type": "fft",
                                "channels": channels,
                                "freqs": freqs,
                                "psd": psd,
                            })

                    elif config["mode"] == "bands":
                        channels, band_names, values = active.get_band_powers(window_sec=config["window_sec"])
                        if channels and len(channels) > 0 and len(values) > 0:
                            active.osc_push_bands(channels, band_names, values)
                            await ws.send_json({
                                "type": "bands",
                                "channels": channels,
                                "bands": band_names,
                                "values": values,
                            })
                        else:
                            await ws.send_json({
                                "type": "bands",
                                "channels": [],
                                "bands": band_names if band_names else [],
                                "values": [],
                            })
                except Exception as e:
                    await ws.send_json({"type": "error", "message": str(e)})

                await asyncio.sleep(config["send_interval_ms"] / 1000.0)

        stream_task = asyncio.create_task(stream_data())

        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.1)
                config["mode"] = msg.get("mode", config["mode"])
                config["window_sec"] = float(msg.get("window_sec", config["window_sec"]))
                config["send_interval_ms"] = int(msg.get("interval_ms", config["send_interval_ms"]))
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # Always cancel the stream task on disconnect
        if stream_task is not None:
            stream_task.cancel()
            try:
                await stream_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await ws.close()
        except Exception:
            pass
