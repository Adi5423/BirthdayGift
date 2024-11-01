#!/usr/bin/env bash
set -eo pipefail

echo "Updating pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Build script completed."