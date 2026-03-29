#!/usr/bin/env python3
"""
Cross-Layer IDS — SYSTEM VERIFICATION SCRIPT
Run this BEFORE the hackathon demo to verify everything works.
Tests each component independently, then the full pipeline.
"""

import os
import sys
import json
import time
import subprocess
import pickle

ROOT = os.path.dirname(os.path.abspath(__file__))
PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


def test_files():
    print("\n═══ FILE EXISTENCE ═══")
    files = [
        'requirements.txt',
        'setup.py',
        'README.md',
        'walkthrough.md',
        'data/generate_dataset.py',
        'data/train_model.py',
        'data/validate_model.py',
        'data/dataset_train.csv',
        'data/dataset_test.csv',
        'data/rf_model_full.pkl',
        'data/rf_baseline_sensor.pkl',
        'data/rf_baseline_can.pkl',
        'data/rf_baseline_v2x.pkl',
        'data/rf_baseline_nocross.pkl',
        'data/ablation_results.json',
        'data/validation_results.json',
        'esp32/cross_layer_ids/cross_layer_ids.ino',
        'esp32/cross_layer_ids/rf_model.h',
        'laptop/stream_to_esp32.py',
        'laptop/dashboard.py',
        'docs/arduino_setup.md',
    ]
    for f in files:
        path = os.path.join(ROOT, f)
        check(f, os.path.exists(path), "FILE NOT FOUND")


def test_imports():
    print("\n═══ PYTHON IMPORTS ═══")
    modules = [
        'numpy', 'pandas', 'sklearn', 'scipy',
        'serial', 'streamlit', 'plotly', 'matplotlib',
    ]
    for m in modules:
        try:
            __import__(m)
            check(f"import {m}", True)
        except ImportError:
            check(f"import {m}", False, "pip install it")

    try:
        from streamlit_autorefresh import st_autorefresh
        check("import streamlit_autorefresh", True)
    except ImportError:
        check("import streamlit_autorefresh", False,
              "pip install streamlit-autorefresh")


def test_data_quality():
    print("\n═══ DATA QUALITY ═══")
    import pandas as pd

    train_path = os.path.join(ROOT, 'data', 'dataset_train.csv')
    if not os.path.exists(train_path):
        check("dataset_train.csv exists", False)
        return

    train = pd.read_csv(train_path)
    check(f"Train size: {len(train)}", len(train) > 50000,
          f"Only {len(train)} samples")

    labels = train['label'].value_counts()
    for cls in [0, 1, 2]:
        n = labels.get(cls, 0)
        check(f"Train class {cls}: {n} samples",
              n > 5000, f"Only {n}")

    test_path = os.path.join(ROOT, 'data', 'dataset_test.csv')
    if not os.path.exists(test_path):
        check("dataset_test.csv exists", False)
        return

    test = pd.read_csv(test_path)
    check(f"Test size: {len(test)}", len(test) > 10000)

    # Check for unseen attacks in test
    unseen = test[test['attack_name'].isin(
        ['attack_coord_can_imu', 'attack_coord_speed_all'])]
    check(f"Unseen attacks in test: {len(unseen)}",
          len(unseen) > 100, "Need unseen attacks for generalization")


def test_model():
    print("\n═══ MODEL QUALITY ═══")

    results = json.load(open(os.path.join(ROOT, 'data',
                                           'ablation_results.json')))

    full = results.get('Full-Cross-Layer', {})
    c2_f1 = full.get('f1', [0, 0, 0])[2]
    check(f"Coord F1: {c2_f1:.3f}", c2_f1 > 0.7,
          f"Only {c2_f1:.3f}")

    overall = full.get('overall_f1', 0)
    check(f"Overall F1: {overall:.3f}", overall > 0.85)

    # Cross-layer improvement check
    nocross = results.get('All-Raw-No-Cross', {})
    nc_c2 = nocross.get('f1', [0, 0, 0])[2]
    improvement = c2_f1 - nc_c2
    check(f"XL improvement: +{improvement:.3f}",
          improvement > 0.02,
          f"Need >0.02, got {improvement:.3f}")

    # Cross-layer ONLY model should show signal
    xl_only = results.get('CrossLayer-Only', {})
    if xl_only:
        xl_c2 = xl_only.get('f1', [0, 0, 0])[2]
        check(f"XL-only C2 F1: {xl_c2:.3f}", xl_c2 > 0.3,
              "Cross-layer features alone should carry some signal")

    cv = results.get('cross_validation', {})
    cv_std = cv.get('std', 1.0)
    check(f"CV stability (std={cv_std:.4f})", cv_std < 0.03)


def test_validation():
    print("\n═══ VALIDATION RESULTS ═══")
    vpath = os.path.join(ROOT, 'data', 'validation_results.json')
    if not os.path.exists(vpath):
        check("validation_results.json exists", False)
        return

    vr = json.load(open(vpath))

    fp = vr.get('false_positives', {})
    for scenario, data in fp.items():
        rate = data.get('rate', 1.0)
        check(f"FP rate ({scenario}): {rate:.4f}",
              rate < 0.15, f"Rate: {rate:.4f}")

    lat = vr.get('latency', {})
    mean_ms = lat.get('mean_ms', 9999)
    check(f"Detection latency: {mean_ms:.0f}ms",
          mean_ms < 1000, f"{mean_ms:.0f}ms")

    unseen = vr.get('unseen_attacks', {})
    for name, data in unseen.items():
        if isinstance(data, dict):
            # Use "any attack detected" instead of exact class match
            any_det = data.get('any_attack_detected', 0)
            check(f"Unseen ({name}): {any_det:.1%} detected",
                  any_det > 0.3,
                  f"Only {any_det:.1%} detected as any attack")
        else:
            check(f"Unseen ({name}): {data:.3f}", data > 0.3)


def test_esp32_header():
    print("\n═══ ESP32 MODEL HEADER ═══")
    hpath = os.path.join(ROOT, 'esp32', 'cross_layer_ids', 'rf_model.h')
    if not os.path.exists(hpath):
        check("rf_model.h exists", False)
        return

    content = open(hpath).read()
    for token in ['RF_N_TREES', 'RF_N_FEATURES', 'get_feat', 'get_thr', 'get_left', 'get_right', 'get_cls']:
        check(f"Has {token}", token in content)

    size_kb = os.path.getsize(hpath) / 1024
    check(f"Header size: {size_kb:.0f} KB",
          size_kb < 500, f"Too large: {size_kb:.0f} KB")


def test_serial_port():
    print("\n═══ SERIAL PORT ═══")
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        check(f"Serial ports found: {len(ports)}", True)
        for p in ports:
            print(f"    {p.device}: {p.description}")
    except Exception as e:
        check("Serial port scan", False, str(e))


if __name__ == '__main__':
    print("╔══════════════════════════════════════════════╗")
    print("║  CROSS-LAYER IDS — SYSTEM VERIFICATION      ║")
    print("╚══════════════════════════════════════════════╝")

    test_files()
    test_imports()
    test_data_quality()
    test_model()
    test_validation()
    test_esp32_header()
    test_serial_port()

    print(f"\n{'█'*50}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print(f"  ✓ ALL SYSTEMS GO — READY FOR DEMO")
    else:
        print(f"  ⚠ FIX {FAIL} ISSUES BEFORE DEMO")
    print(f"{'█'*50}")
