# Sensathon: Cross-Layer IDS for Software Defined Vehicles 🛡️🚙

Welcome to the **Sensathon** project repository. This project implements a cutting-edge **Cross-Layer Intrusion Detection System (IDS)** specifically designed for modern Software-Defined Vehicles (SDVs).

By leveraging real-time edge computing on microcontrollers like the ESP32, this system applies physics-based cross-layer consistency checking across three fundamental vehicular networks:
- **CAN Bus** (In-vehicle communication)
- **V2X** (Vehicle-to-Everything communication)
- **Onboard Sensors** (Physical state & environment sensing)

This multi-faceted approach allows detection of sophisticated, coordinated attacks that would otherwise appear legitimate to traditional, single-layer detection systems.

---

## 🏗️ Architecture

The system operates efficiently using a split-architecture model:

```text
Laptop Simulator (Data Generation/Streaming) 
         |
         | (USB Serial)
         v
ESP32 Microcontroller (Real-time RF inference + LED Alerts)
         |
         | (USB Serial - Attack Detections)
         v
Laptop (Streamlit Control Dashboard)
```

## 📂 Repository Structure

The core source code of the project resides in the `cross_layer_ids/` directory.

- **`cross_layer_ids/esp32/`**: Contains the C++ firmware (`.ino` and `.h` files) for the ESP32 microcontroller that runs the onboard inference.
- **`cross_layer_ids/laptop/`**: Python scripts for the simulation environment. Includes:
  - `stream_to_esp32.py`: Streams synthesized log data to the ESP32.
  - `dashboard.py`: A local Streamlit dashboard to visualize inference alerts and network activity in real time.
- **`cross_layer_ids/data/`**: Scripts and Jupyter tools (e.g., `train_model.py`, `validate_model.py`) used to generate vehicle attack datasets and train the RandomForest models.
- **`cross_layer_ids/docs/`**: Documentation, setup guides (`arduino_setup.md`), and performance plots.
- **`cross_layer_ids/test_system.py`**: End-to-end integration testing script.
- **`cross_layer_ids/handover_guide.md`**: Guide for contributors and handoffs, detailing recent bug fixes.

---

## 🚀 Quick Start Guide

**1. Clone and Install Dependencies:**
```bash
git clone https://github.com/shroff45/sensathon.git
cd sensathon/cross_layer_ids
pip install -r requirements.txt
python setup.py
```

**2. Hardware Setup:**
Wire your ESP32 board to three LEDs representing system states. The default configuration uses:
- `GPIO2` → 220Ω Resistor → Green LED (Safe state)
- `GPIO4` → 220Ω Resistor → Yellow LED (Warning state)
- `GPIO5` → 220Ω Resistor → Red LED (Attack Detected state)

**3. Flash the ESP32:**
Open `esp32/cross_layer_ids/cross_layer_ids.ino` in the Arduino IDE and flash it to your connected ESP32 board. (Refer to `docs/arduino_setup.md` for specific library requirements).

**4. Run the Full Simulation:**
Launch two separate terminals:

> **Terminal 1** (Streams data to ESP32):
> ```bash
> cd cross_layer_ids
> python laptop/stream_to_esp32.py
> ```

> **Terminal 2** (Runs User Visual Dashboard):
> ```bash
> cd cross_layer_ids
> streamlit run laptop/dashboard.py
> ```

---

## 💡 Novel Contribution
This project implements **6 newly constructed physics-based cross-layer features**. These features correlate data that theoretically must match in the physical world (e.g., V2X reported speed vs. CAN Bus internal wheel speed values). By analyzing the physical impossibilities and discrepancies between logical layers, the system spots advanced stealth attacks that conventional IT-based IDSs miss.

---
*Developed for the Sensathon Hackathon Challenge.*
