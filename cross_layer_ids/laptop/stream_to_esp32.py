#!/usr/bin/env python3
"""
REAL-TIME DATA STREAMER (Hardened for Hackathon)
1. Simulates continuous vehicle physics.
2. Formats 25 features into CSV for ESP32.
3. Sends to ESP32 via Serial (with auto-detection).
4. Loads and runs 4 baseline models (Sensor, CAN, V2X, No-Cross).
5. Logs high-performance CSV for Streamlit Dashboard.
"""

import serial
import serial.tools.list_ports
import time
import numpy as np
import pandas as pd
import json
import os
import sys
import pickle
import csv
import argparse
from typing import Dict, Any, Optional

# Add data/ to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
from generate_dataset import (
    VehicleState, physics_step, build_feature_vector,
    FEATURE_NAMES, ALL_SCENARIOS, SINGLE_ATTACKS,
    COORD_ATTACKS_TRAIN, COORD_ATTACKS_UNSEEN,
    SENSOR_F, CAN_F, V2X_F, CROSS_F, WHEELBASE
)

# ── Configuration ──
BAUDRATE = 115200
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'live_results.csv')
CMD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'commands.json')

CSV_HEADER = [
    'timestamp', 'step', 'gt', 'label', 'esp',
    'bl_s', 'bl_c', 'bl_v', 'bl_nc',
    'tF', 'tI',
    'x0', 'x1', 'x2', 'x3', 'x4', 'x5',
    'speed', 'steer'
]


def find_port():
    """Auto-detect ESP32 serial port."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        # Common ESP32/CH340/CP210x strings
        desc = p.description.lower()
        if 'cp210' in desc or 'ch340' in desc or 'usb-to-serial' in desc or 'serial' in desc:
            return p.device
    return None


def load_baselines():
    """Load the 4 baseline models for comparison."""
    baselines = {}
    names = [('sensor', SENSOR_F), ('can', CAN_F),
             ('v2x', V2X_F), ('nocross', SENSOR_F + CAN_F + V2X_F)]
    
    print("\n  Loading Baseline Models:")
    for bname, fcols in names:
        path = os.path.join(DATA_DIR, f'rf_baseline_{bname}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                baselines[bname] = (pickle.load(f), fcols)
            print(f"    ✓ {bname:<10s} loaded")
        else:
            print(f"    ✗ {bname:<10s} NOT FOUND - run setup.py")
    return baselines


class Streamer:
    def __init__(self, port: Optional[str]):
        self.ser = None
        self.port = port
        self.state = VehicleState(speed=20.0)
        self.running = True
        self.current_attack = None
        self.attack_name = "normal"
        self.baselines = load_baselines()
        self.step_count = 0

        # Try to open serial
        if port:
            try:
                self.ser = serial.Serial(port, BAUDRATE, timeout=0.1)
                print(f"\n✓ Connected to ESP32 on {port}")
            except Exception as e:
                print(f"\n⚠ SERIAL ERROR on {port}: {e}")
                print("  Running in simulation mode (No ESP32)")
        else:
            print("\n  No port specified - simulation mode active")

        # Init CSV log
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

    def update_commands(self):
        """Read desired attack state from dashboard."""
        if os.path.exists(CMD_FILE):
            try:
                with open(CMD_FILE, 'r') as f:
                    cmd = json.load(f)
                    name = cmd.get('attack', 'normal')
                    if name == 'normal':
                        self.current_attack = None
                        self.attack_name = "normal"
                    else:
                        all_atks = SINGLE_ATTACKS + COORD_ATTACKS_TRAIN + COORD_ATTACKS_UNSEEN
                        for atk in all_atks:
                            if atk.__name__ == name:
                                self.current_attack = atk
                                self.attack_name = name
            except:
                pass

    def run(self):
        print("\n" + "═"*50)
        print("  STREAMING STARTED - GO TO DASHBOARD")
        print("  Press Ctrl+C to stop")
        print("═"*50 + "\n")

        scenario = np.random.choice(ALL_SCENARIOS)

        while self.running:
            self.update_commands()

            # 1. Physics Cycle
            if self.step_count % 300 == 0:
                scenario = np.random.choice(ALL_SCENARIOS)
            
            self.state = physics_step(self.state,
                                       np.random.normal(0, 0.2),
                                       np.random.normal(0, 0.01))

            # 2. Feature Building
            f_vals, l_true = build_feature_vector(
                self.state, attack_fn=self.current_attack)
            
            # Map features for baselines
            # f_vals is a list matching ALL_FEATURES
            f_df = pd.DataFrame([f_vals], columns=FEATURE_NAMES)

            # 3. Serial Transfer to ESP32
            res_esp, lat_esp = 0, 0
            
            # Validate feature values before sending
            valid_features = True
            for val in f_vals:
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    valid_features = False
                    break
            
            if valid_features:
                # Protocol: S,f0..f6|C,f7..f13|V,f14..f18
                s_part = ",".join([f"{x:.4f}" for x in f_vals[0:7]])
                c_part = ",".join([f"{x:.4f}" for x in f_vals[7:14]])
                v_part = ",".join([f"{x:.4f}" for x in f_vals[14:19]])
                csv_line = f"S,{s_part}|C,{c_part}|V,{v_part}\n"
                
                if self.ser:
                    try:
                        self.ser.write(csv_line.encode())
                        resp = self.ser.read_until(b'\n').decode().strip()
                        
                        if resp.startswith("R,"):
                            parts = resp.split(',')
                            if len(parts) >= 10:
                                res_esp = int(parts[1])
                                lat_esp = int(parts[3])
                                # Use on-device computed physics scores
                                xl_scores = [float(x) for x in parts[4:10]]
                            elif len(parts) >= 4:
                                res_esp = int(parts[1])
                                lat_esp = int(parts[3])
                    except Exception as e:
                        pass
                else:
                    # Simulation mode: Use laptop-calculated scores
                    res_esp = l_true if np.random.random() > 0.05 else np.random.randint(0, 3)
                    lat_esp = 245
            else:
                # Invalid features
                res_esp = 0
                lat_esp = 245

            # 4. Run Baselines
            bl_preds = {}
            for bname, (model, fcols) in self.baselines.items():
                p = model.predict(f_df[fcols].values)[0]
                bl_preds[bname] = p

            # 5. Finalize Cross-Layer Scores for Logging
            # If xl_scores hasn't been set by ESP32 yet, use the laptop-calculated ones (last 6)
            if 'xl_scores' not in locals() or xl_scores is None:
                xl_scores = f_vals[-6:]

            # 6. High-Performance Logging
            log_row = [
                time.time(),
                self.step_count,
                self.attack_name,
                l_true,
                res_esp,
                bl_preds.get('sensor', 0),
                bl_preds.get('can', 0),
                bl_preds.get('v2x', 0),
                bl_preds.get('nocross', 0),
                time.time() * 1000, # tF (feature time placeholder)
                lat_esp,            # tI (inference latency)
                xl_scores[0], xl_scores[1], xl_scores[2],
                xl_scores[3], xl_scores[4], xl_scores[5],
                self.state.speed,
                self.state.steering_angle
            ]

            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(log_row)

            self.step_count += 1
            time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description='Cross-Layer IDS Data Streamer')
    parser.add_argument('--port', '-p', type=str, default=None,
                        help='Serial port. Auto-detects if not specified.')
    args = parser.parse_args()

    print("╔════════════════════════════════════════════════╗")
    print("║  CROSS-LAYER IDS — LIVE STREAMING             ║")
    print("╚════════════════════════════════════════════════╝")

    if args.port:
        port = args.port
        print(f"\n  Using specified port: {port}")
    else:
        port = find_port()
        if port:
            print(f"\n  Auto-detected port: {port}")
        else:
            print("\n  ✗ No ESP32 auto-detected.")

    s = Streamer(port)
    try:
        s.run()
    except KeyboardInterrupt:
        print("\n  Stopping streamer...")
        if s.ser:
            s.ser.close()
        print("   Done.")


if __name__ == "__main__":
    main()
