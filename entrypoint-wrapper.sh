#!/bin/bash
set -e

echo "=== EntryPoint Wrapper: Setting up volumes BEFORE base image entrypoint ==="

# =============================================================================
# CRITICAL: Check if /runpod-volume is actually mounted
# In RunPod serverless, the network volume may NOT be automatically mounted.
# We must verify /runpod-volume exists as a mount point before using it.
# =============================================================================

# Check if /runpod-volume is a mount point (not just a local directory)
is_runpod_mounted() {
    if mount | grep -q "on /runpod-volume "; then
        return 0  # true - is mounted
    fi
    # Also check if /runpod-volume is a symlink to an existing directory
    if [ -L "/runpod-volume" ] && [ -d "/runpod-volume" ]; then
        # It's a symlink to an existing directory - check if it has actual content
        # by verifying it's not just an empty directory we created
        local target=$(readlink -f /runpod-volume 2>/dev/null)
        if [ -n "$target" ] && [ "$target" != "/" ]; then
            return 0
        fi
    fi
    return 1  # false - not mounted or doesn't exist
}

# Setup /workspace based on whether network volume is available
echo "Setting up /workspace..."
rm -rf /workspace

if is_runpod_mounted; then
    echo "Network volume detected at /runpod-volume, creating symlinks..."
    mkdir -p /runpod-volume/workspace
    ln -s /runpod-volume/workspace /workspace
    echo "Created /workspace -> /runpod-volume/workspace"
else
    echo "WARNING: /runpod-volume not mounted, using local /workspace"
    echo "This will use ephemeral storage which is limited (~5GB)"
    mkdir -p /workspace
    # Create workspace on network volume if possible for models
    if [ -d "/runpod-volume" ]; then
        mkdir -p /runpod-volume/workspace
    fi
fi

# Setup /comfyui symlink if network volume is available
if is_runpod_mounted; then
    mkdir -p /runpod-volume/comfyui
    if [ ! -e /comfyui ] || [ -L /comfyui ] && [ "$(readlink -f /comfyui)" != "/runpod-volume/comfyui" ]; then
        rm -rf /comfyui
        ln -s /runpod-volume/comfyui /comfyui
        echo "Created /comfyui -> /runpod-volume/comfyui"
    fi
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
