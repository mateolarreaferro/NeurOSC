#!/bin/bash
# Build Neuro.app - standalone macOS application
# Run: bash build.sh

set -e

echo ""
echo "  Building Neuro.app..."
echo ""

# Clean previous builds
rm -rf build dist

# Locate BrainFlow native libraries directory
BRAINFLOW_LIB=$(python -c "import brainflow, os; print(os.path.join(os.path.dirname(brainflow.__file__), 'lib'))")
echo "  BrainFlow native libs: $BRAINFLOW_LIB"

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
    --add-binary "$BRAINFLOW_LIB/*:brainflow/lib" \
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
    --exclude-module torch \
    --exclude-module torchaudio \
    --exclude-module torchvision \
    --exclude-module transformers \
    --exclude-module lightning \
    --exclude-module sklearn \
    --exclude-module pandas \
    --exclude-module h5py \
    --exclude-module onnxruntime \
    --exclude-module sqlalchemy \
    --exclude-module grpc \
    --exclude-module psycopg2 \
    --exclude-module numba \
    --exclude-module llvmlite \
    --exclude-module librosa \
    --exclude-module av \
    --exclude-module soundfile \
    --exclude-module pygments \
    --exclude-module lxml \
    --exclude-module opentelemetry \
    --exclude-module pygame \
    --osx-bundle-identifier com.neuro.eeg \
    launcher.py

# macOS: set Bluetooth permission in Info.plist
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLIST="dist/Neuro.app/Contents/Info.plist"
    if [ -f "$PLIST" ]; then
        /usr/libexec/PlistBuddy -c "Add :NSBluetoothAlwaysUsageDescription string 'Neuro needs Bluetooth to connect to EEG headsets.'" "$PLIST" 2>/dev/null || true
        /usr/libexec/PlistBuddy -c "Add :NSBluetoothPeripheralUsageDescription string 'Neuro needs Bluetooth to connect to EEG headsets.'" "$PLIST" 2>/dev/null || true
    fi
fi

# Verify critical native libraries made it into the bundle
echo ""
echo "  Verifying BrainFlow native libraries..."
MISSING=0
for LIB in libBoardController libGanglionLib libBrainFlowBluetooth libsimpleble-c libDataHandler libMLModule libMuseLib; do
    if ls dist/Neuro.app/Contents/Frameworks/${LIB}.* dist/Neuro.app/Contents/Resources/${LIB}.* dist/Neuro.app/Contents/MacOS/brainflow/lib/${LIB}.* 2>/dev/null | head -1 > /dev/null 2>&1; then
        echo "    OK: $LIB"
    elif find dist/Neuro.app -name "${LIB}.*" 2>/dev/null | head -1 | grep -q .; then
        echo "    OK: $LIB (found in bundle)"
    else
        echo "    MISSING: $LIB"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "  WARNING: $MISSING native libraries are missing from the bundle!"
    echo "  The app may fail with GENERAL_ERROR(17) when connecting to hardware."
    echo "  Try: pip install --force-reinstall brainflow && bash build.sh"
    echo ""
fi

SIZE=$(du -sh dist/Neuro.app 2>/dev/null | cut -f1 || du -sh dist/Neuro 2>/dev/null | cut -f1)

echo ""
echo "  Build complete!"
echo "  Output: dist/Neuro.app ($SIZE)"
echo ""
echo "  To run: open dist/Neuro.app"
echo "  To distribute: zip -r Neuro.zip dist/Neuro.app"
echo ""
