#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Checking tools"
missing=()
for cmd in iw tshark python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    missing+=("$cmd")
  fi
done

if [ "${#missing[@]}" -gt 0 ]; then
  echo "Missing tools: ${missing[*]}"
  echo "Install on Ubuntu with:"
  echo "  sudo apt-get update"
  echo "  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y tshark wireless-tools aircrack-ng"
  exit 2
fi

echo "[2/4] Wireless devices"
iw dev || true

echo "[3/4] USB devices likely related to Wi-Fi"
if command -v lsusb >/dev/null 2>&1; then
  lsusb | grep -Ei "wireless|wlan|wi-fi|802\\.11|realtek|ralink|mediatek|atheros|qualcomm|alfa|tp-link|rtl|mt76|ath9k|ath10k" || true
else
  echo "lsusb not installed; install usbutils if needed."
fi

echo "[4/4] Next steps"
cat <<'TXT'
If no wlan interface appears above:
  1. Plug a USB Wi-Fi adapter that supports monitor mode.
  2. Attach/pass-through the USB device to this Ubuntu VM.
  3. Run this script again.

To create a monitor interface manually:
  sudo ip link set wlan0 down
  sudo iw dev wlan0 set type monitor
  sudo ip link set wlan0 up

If NetworkManager keeps taking over the adapter:
  sudo airmon-ng check kill
  sudo airmon-ng start wlan0

Then run the live Rogue AP sensor:
  sudo python3 sdn_runtime/rogue_ap_live_sensor.py \
    --interface wlan0mon \
    --model models/rogue_ap_runtime/rogue_ap_runtime_rf.joblib \
    --alert-url http://127.0.0.1:8080/rogue-ap-alert \
    --out-dir results/rogue_ap_runtime_live
TXT
