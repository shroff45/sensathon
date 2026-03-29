# Arduino IDE Setup for ESP32

## Install ESP32 Board Package

1. File → Preferences → Additional Boards Manager URLs
2. Add: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
3. Tools → Board → Boards Manager → Search "esp32" → Install

## Board Settings

| Setting | Value |
|---|---|
| Board | ESP32 Dev Module |
| Upload Speed | 921600 |
| CPU Frequency | 240MHz |
| Flash Frequency | 80MHz |
| Flash Mode | QIO |
| Flash Size | 4MB |
| Partition Scheme | Default 4MB |
| Port | (your USB port) |

## Verify Compilation

1. Open `esp32/cross_layer_ids/cross_layer_ids.ino`
2. Verify `rf_model.h` exists in the same folder
3. Click ✓ (Verify) — should compile with 0 errors
4. Click → (Upload)

## Troubleshooting

| Problem | Solution |
|---|---|
| Port not visible | Install CP2102/CH340 driver |
| Upload fails | Hold BOOT button during upload |
| "rf_model.h not found" | Run `python setup.py` first |
| LEDs don't light | Check wiring polarity |
