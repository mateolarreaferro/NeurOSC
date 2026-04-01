#!/usr/bin/env python3
"""
Neuro - EEG Signal Interface
Double-click this file or run `python launcher.py` to start.
"""
import sys
import os
import socket
import threading
import time
import webbrowser
import signal

def find_free_port(start=8100, end=8200):
    """Find an available port."""
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    return start

def open_browser(port, retries=10):
    """Wait for server to be ready, then open the browser."""
    for _ in range(retries):
        time.sleep(0.5)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect(("127.0.0.1", port))
                # Server is up
                webbrowser.open(f"http://127.0.0.1:{port}")
                return
        except (ConnectionRefusedError, OSError):
            continue
    # Last resort, open anyway
    webbrowser.open(f"http://127.0.0.1:{port}")

def main():
    # Ensure we're running from the right directory
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)

    # If running from a PyInstaller bundle, adjust path
    if getattr(sys, '_MEIPASS', None):
        os.chdir(sys._MEIPASS)

    port = find_free_port()

    print(f"""
    ╔══════════════════════════════════════╗
    ║          NEURO                       ║
    ║          EEG Signal Interface        ║
    ╠══════════════════════════════════════╣
    ║                                      ║
    ║  Running at:                         ║
    ║  http://127.0.0.1:{port:<5}              ║
    ║                                      ║
    ║  Opening in your browser...          ║
    ║                                      ║
    ║  Press Ctrl+C to quit.               ║
    ╚══════════════════════════════════════╝
    """)

    # Open browser in background thread
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    # Start the server
    import uvicorn
    try:
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
    except KeyboardInterrupt:
        print("\nNeuro stopped.")

if __name__ == "__main__":
    main()
