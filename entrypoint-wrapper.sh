#!/bin/bash
set -e

echo "=== EntryPoint Wrapper: Setting up volumes BEFORE base image entrypoint ==="

# CRITICAL: Create symlink BEFORE sombi/base entrypoint does rsync
# The base image entrypoint runs rsync of ComfyUI+venv (~20GB) to /workspace
# If /workspace is on ephemeral storage (~5GB), this fails with "No space left on device"
if [ ! -L /workspace ] || [ -d /workspace ]; then
    echo "Setting up /workspace symlink..."
    rm -rf /workspace
    mkdir -p /runpod-volume/workspace
    ln -s /runpod-volume/workspace /workspace
    echo "Created /workspace -> /runpod-volume/workspace"
fi

# Also ensure /comfyui is linked to network volume
mkdir -p /runpod-volume/comfyui
if [ ! -e /comfyui ] || [ -L /comfyui ] && [ "$(readlink -f /comfyui)" != "/runpod-volume/comfyui" ]; then
    rm -rf /comfyui
    ln -s /runpod-volume/comfyui /comfyui
    echo "Created /comfyui -> /runpod-volume/comfyui"
fi

echo "=== Volume setup complete, executing original entrypoint ==="

# Find and execute original entrypoint first
# The original entrypoint may do important initialization (rsync, etc.)
if [ -f /usr/local/bin/docker-entrypoint.sh ]; then
    echo "Found original entrypoint at /usr/local/bin/docker-entrypoint.sh"
    exec /usr/local/bin/docker-entrypoint.sh "$@"
elif [ -f /entrypoint.sh ]; then
    echo "Found original entrypoint at /entrypoint.sh"
    exec /entrypoint.sh "$@"
else
    echo "No original entrypoint found, skipping..."
fi

# After original entrypoint completes, run our start.sh
# This point is reached only if original entrypoint didn't exec
echo "=== Original entrypoint done, starting start.sh ==="
exec /start.sh
