#!/usr/bin/env bash
# Old-soname libraries Isaac Sim 6.2 links against but Ubuntu 26.04 no longer ships.
#
#   libxml2.so.2  26.04 only has libxml2.so.16 (ABI break, so a symlink is not safe).
#                 Without it omni.kit.asset_converter fails, and the URDF/MJCF/asset
#                 importers fail with it.
#   libicu*.so.74 libxml2.so.2 from 24.04 is linked against ICU 74; 26.04 has ICU 78.
#
# The Ubuntu 24.04 (noble-updates) packages are pinned by SHA256 and only their .so files
# are unpacked into host/compat-lib. Nothing is installed system wide. `make sim` puts that
# directory on LD_LIBRARY_PATH for the Isaac Sim process only.
#
# If a download 404s, noble-updates has moved on: take the new Filename/SHA256 from
# http://archive.ubuntu.com/ubuntu/dists/noble-updates/main/binary-amd64/Packages.gz
set -euo pipefail

POOL=http://archive.ubuntu.com/ubuntu
DEBS=(
  "pool/main/libx/libxml2/libxml2_2.9.14+dfsg-1.3ubuntu3.10_amd64.deb c5a065f85c1e6f605741e365e40cad71c3ceb0e85e2791400bd35b8896e12ad1"
  "pool/main/i/icu/libicu74_74.2-1ubuntu3.1_amd64.deb c9a70989678660eed9a1e904c74fa043da8bec8e2036856fc16e31ced79b04f8"
)

OUT="$(cd "$(dirname "$0")" && pwd)/compat-lib"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT"

for entry in "${DEBS[@]}"; do
  read -r path sha <<<"$entry"
  deb="$WORK/$(basename "$path")"
  echo "fetch  $(basename "$path")"
  curl -fsSL -o "$deb" "$POOL/$path"
  echo "$sha  $deb" | sha256sum -c --quiet
  dpkg-deb -x "$deb" "$WORK/root"
done

# Shared libraries only: no docs, no config, nothing outside lib dirs.
cp -P "$WORK"/root/usr/lib/x86_64-linux-gnu/*.so.* "$OUT/"
echo "installed into $OUT:"
ls "$OUT"
