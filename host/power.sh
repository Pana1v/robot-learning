#!/usr/bin/env bash
# Put the laptop in training mode, or show what the GPU is allowed to draw.
#
#   ./host/power.sh status
#   ./host/power.sh train     # performance profile (no sudo needed)
#   sudo ./host/power.sh boost  # also flip the HP Omen GPU TGP / Dynamic Boost knobs
#   ./host/power.sh quiet     # back to balanced
set -euo pipefail

HPWMI=/sys/devices/platform/hp-wmi

status() {
  local ac
  ac=$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1 || echo "?")
  echo "AC plugged in : $([[ $ac == 1 ]] && echo yes || echo 'NO (GPU stays capped near 50 W on battery)')"
  echo "Power profile : $(powerprofilesctl get)"
  [[ -r $HPWMI/gpu_tgp ]] && echo "Omen gpu_tgp  : $(cat $HPWMI/gpu_tgp)   gpu_ppab: $(cat $HPWMI/gpu_ppab)"
  nvidia-smi --query-gpu=power.limit,power.max_limit,power.draw,temperature.gpu,clocks.sm \
    --format=csv,noheader | awk -F', ' '{print "GPU limit     : "$1" (max "$2"), drawing "$3", "$4"C, "$5}'
}

case "${1:-status}" in
  status) status ;;
  train)
    powerprofilesctl set performance
    status ;;
  boost)
    [[ $EUID -eq 0 ]] || { echo "boost needs sudo" >&2; exit 1; }
    powerprofilesctl set performance || true
    # Same switches the Omen Gaming Hub flips on Windows: custom TGP and
    # Dynamic Boost (PPAB). Firmware only honours them on AC power, and they
    # reset on reboot, so re-run after a restart.
    echo 1 > $HPWMI/gpu_tgp
    echo 1 > $HPWMI/gpu_ppab
    sleep 2
    status ;;
  quiet)
    powerprofilesctl set balanced
    status ;;
  *) echo "usage: $0 {status|train|boost|quiet}" >&2; exit 1 ;;
esac
