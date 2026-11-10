#!/usr/bin/env bash
# One-time host prep for Isaac Lab / Isaac Sim on this laptop.
# Safe to re-run: every step checks the current state first.
#
#   sudo ./host/setup-host.sh            # swap + docker GPU runtime
#   sudo SWAP_GB=24 ./host/setup-host.sh # different swap size
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo $0" >&2
  exit 1
fi

SWAP_GB="${SWAP_GB:-20}"
SWAPFILE=/swapfile

# ---------------------------------------------------------------- swap
# Isaac Sim wants 32 GB RAM. We have ~21 GB, so swap is the safety net that
# turns an OOM kill into a slowdown while Kit loads its extensions.
want_bytes=$((SWAP_GB * 1024 * 1024 * 1024))
have_bytes=$(stat -c %s "$SWAPFILE" 2>/dev/null || echo 0)

if [[ "$have_bytes" -eq "$want_bytes" ]] && swapon --show=NAME --noheadings | grep -qx "$SWAPFILE"; then
  echo "[swap] $SWAPFILE already ${SWAP_GB}G and active, skipping"
else
  echo "[swap] resizing $SWAPFILE to ${SWAP_GB}G"
  # swapoff moves pages back to RAM, so this needs enough free RAM to hold
  # what is currently swapped out. Close big apps first if it is heavily used.
  swapoff "$SWAPFILE" 2>/dev/null || true
  rm -f "$SWAPFILE"
  # fallocate is instant on ext4; fall back to dd on filesystems that need it
  fallocate -l "${SWAP_GB}G" "$SWAPFILE" || dd if=/dev/zero of="$SWAPFILE" bs=1M count=$((SWAP_GB * 1024)) status=progress
  chmod 600 "$SWAPFILE"
  mkswap "$SWAPFILE" >/dev/null
  swapon "$SWAPFILE"
fi
grep -qE "^\s*$SWAPFILE\s" /etc/fstab || echo "$SWAPFILE none swap defaults 0 0" >> /etc/fstab

# ------------------------------------------------- docker GPU runtime
# nvidia-container-toolkit is installed but Docker was never told about it,
# so `--gpus all` and the compose `devices: gpu` block would fail.
if docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
  echo "[docker] nvidia runtime already registered, skipping"
else
  echo "[docker] registering nvidia runtime"
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

echo
echo "Done. Check with: make doctor"
swapon --show
