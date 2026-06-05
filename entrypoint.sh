#!/bin/bash
set -e

# Run model weight download check/download
echo "========================================="
echo "Checking MuseTalk model weights..."
echo "========================================="
python3 download_models.py
echo "========================================="
echo "Model weights check complete!"
echo "========================================="

# Execute the CMD
exec "$@"
