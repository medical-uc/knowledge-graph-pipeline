#!/usr/bin/env bash
set -uo pipefail

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "Error: Python 3.11 is required but not installed." >&2
    exit 1
fi

# Verify exact Python version matches 3.11.x
python_version=$(python3.11 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$python_version" != "3.11" ]; then
    echo "Error: Python 3.11 is required, but found version $python_version." >&2
    exit 1
fi

# Check if .venv already exists
if [ -d ".venv" ]; then
    echo "Found existing virtual environment (.venv)."
else
    echo "Creating virtual environment with Python 3.11..."
    python3.11 -m venv .venv
fi

# Activate and install requirements
source .venv/bin/activate
pip install -r requirements.txt
