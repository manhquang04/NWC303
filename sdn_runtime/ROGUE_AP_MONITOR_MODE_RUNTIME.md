# Rogue AP Runtime With Wi-Fi Monitor Mode

This runtime path is different from ARP Spoofing.

- ARP Spoofing is visible directly to Ryu/OpenFlow through ARP packet-in events.
- Rogue AP is an 802.11 wireless problem, so the SDN controller needs a Wi-Fi monitor-mode sensor.

## Required Hardware

Use a USB Wi-Fi adapter that supports monitor mode on Linux. Good practical choices are adapters with Atheros `ath9k_htc`, MediaTek `mt76`, or Realtek chipsets with working Linux drivers.

The adapter must be attached to the Ubuntu VM, not only to macOS.

## VM Setup

Check tools and wireless interfaces:

```bash
cd /home/manhquang/Downloads/NWC303_LOCAL_RUNTIME
bash sdn_runtime/setup_wifi_monitor_vm.sh
```

Install tools if missing:

```bash
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y tshark wireless-tools aircrack-ng usbutils
```

Enable monitor mode:

```bash
sudo ip link set wlan0 down
sudo iw dev wlan0 set type monitor
sudo ip link set wlan0 up
```

Or with aircrack-ng:

```bash
sudo airmon-ng check kill
sudo airmon-ng start wlan0
```

## Runtime Architecture

```text
Wi-Fi adapter in monitor mode
  -> tshark live 802.11/radiotap fields
  -> rogue_ap_live_sensor.py
  -> runtime window aggregate features
  -> Random Forest runtime scorer
  -> POST /rogue-ap-alert
  -> Ryu controller logs Rogue AP decision
```

## Start Ryu Controller

Use the same controller that handles ARP, now with a Rogue AP REST endpoint:

```bash
cd /home/manhquang/Downloads/NWC303_LOCAL_RUNTIME
ARP_GUARD_EVENT_LOG=/tmp/arp_guard_events.jsonl \
/home/manhquang/Downloads/NWC303/.venv311/bin/ryu-manager sdn_runtime/arp_guard_controller.py
```

The Rogue AP endpoint is:

```text
POST http://127.0.0.1:8080/rogue-ap-alert
```

## Start Rogue AP Live Sensor

```bash
cd /home/manhquang/Downloads/NWC303_LOCAL_RUNTIME
sudo python3 sdn_runtime/rogue_ap_live_sensor.py \
  --interface wlan0mon \
  --model models/rogue_ap_runtime/rogue_ap_runtime_rf.joblib \
  --window-frames 100 \
  --stride-frames 50 \
  --alert-url http://127.0.0.1:8080/rogue-ap-alert \
  --out-dir results/rogue_ap_runtime_live
```

Use `--dry-run` first if you only want local sensor logs:

```bash
sudo python3 sdn_runtime/rogue_ap_live_sensor.py \
  --interface wlan0mon \
  --model models/rogue_ap_runtime/rogue_ap_runtime_rf.joblib \
  --dry-run \
  --max-windows 20
```

## Important Limitations

This is true live wireless sensing, but Rogue AP mitigation is not the same as ARP mitigation:

- Ryu can directly block ARP spoofing on OpenFlow switch ports.
- Ryu cannot directly disable a Rogue AP over the air.
- For Rogue AP, Ryu can log, alert, quarantine a known client on the wired SDN side, or trigger an external WLAN controller/firewall action.

For thesis wording, call this:

> Hybrid SDN + wireless monitor-mode runtime for Rogue AP detection.

Do not claim it is pure OpenFlow-only Rogue AP detection.
