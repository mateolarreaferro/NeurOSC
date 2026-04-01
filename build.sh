#!/bin/bash
# Build Neuro.app - standalone macOS application
# Run: bash build.sh

set -e

echo ""
echo "  Building Neuro.app..."
echo ""

# Clean previous builds
rm -rf build dist

pyinstaller \
    --noconfirm \
    --windowed \
    --name "Neuro" \
    --add-data "templates:templates" \
    --add-data "static:static" \
    --add-data "main.py:." \
    --add-data "openbci_service.py:." \
    --add-data "simulator_service.py:." \
    --add-data "osc_sender.py:." \
    --collect-all brainflow \
    --collect-submodules fastapi \
    --collect-submodules starlette \
    --collect-submodules uvicorn \
    --collect-submodules anyio \
    --hidden-import scipy.signal \
    --hidden-import pythonosc \
    --hidden-import pythonosc.udp_client \
    --hidden-import jinja2.ext \
    --hidden-import email.mime.multipart \
    --hidden-import email.mime.text \
    --exclude-module tkinter \
    --exclude-module matplotlib \
    --exclude-module PIL \
    --exclude-module cv2 \
    --exclude-module mediapipe \
    --exclude-module IPython \
    --exclude-module pytest \
    --osx-bundle-identifier com.neuro.eeg \
    launcher.py

# macOS: set Bluetooth permission in Info.plist
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLIST="dist/Neuro.app/Contents/Info.plist"
    if [ -f "$PLIST" ]; then
        /usr/libexec/PlistBuddy -c "Add :NSBluetoothAlwaysUsageDescription string 'Neuro needs Bluetooth to connect to OpenBCI EEG headsets.'" "$PLIST" 2>/dev/null || true
        /usr/libexec/PlistBuddy -c "Add :NSBluetoothPeripheralUsageDescription string 'Neuro needs Bluetooth to connect to OpenBCI EEG headsets.'" "$PLIST" 2>/dev/null || true
    fi
fi

SIZE=$(du -sh dist/Neuro.app 2>/dev/null | cut -f1 || du -sh dist/Neuro 2>/dev/null | cut -f1)

echo ""
echo "  Build complete!"
echo "  Output: dist/Neuro.app ($SIZE)"
echo ""
echo "  To run: open dist/Neuro.app"
echo "  To distribute: zip -r Neuro.zip dist/Neuro.app"
echo ""
