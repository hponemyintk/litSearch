#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"
REQUIREMENTS="requirements.txt"

# Check Python is available
if ! command -v "$PYTHON" &>/dev/null; then
    echo "Error: $PYTHON not found. Install Python 3 or set PYTHON env var."
    exit 1
fi

# Remove existing venv if present
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

echo "Creating virtual environment in $VENV_DIR..."
"$PYTHON" -m venv "$VENV_DIR"

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from $REQUIREMENTS..."
pip install -r "$REQUIREMENTS"

echo ""
echo "Done! To activate the environment, run:"
echo "  source $VENV_DIR/bin/activate"
