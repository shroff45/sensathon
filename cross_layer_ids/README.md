# 🛡️ Cross-Layer IDS for Software Defined Vehicles

Real-time intrusion detection using physics-based cross-layer
consistency checking across CAN bus, V2X, and onboard sensors.

## Quick Start

```bash
pip install -r requirements.txt
python setup.py
# Flash ESP32 (Arduino IDE → esp32/cross_layer_ids/)
python laptop/stream_to_esp32.py    # Terminal 1
streamlit run laptop/dashboard.py   # Terminal 2
```

## Hardware

```
ESP32 GPIO2 → 220Ω → Green LED → GND
ESP32 GPIO4 → 220Ω → Yellow LED → GND
ESP32 GPIO5 → 220Ω → Red LED → GND
```

## Architecture

```
Laptop Simulator → USB → ESP32 (RF inference + LEDs)
                          ↓ USB
                   Laptop reads results
                          ↓
                   Streamlit Dashboard
```

## Novel Contribution

6 physics-based cross-layer features detecting coordinated
attacks invisible to any single-layer IDS.
