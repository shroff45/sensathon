#!/usr/bin/env python3
"""
One-command project setup.
Run: python setup.py
Generates dataset, trains model, exports to C header, validates.
"""

import subprocess
import sys
import os


def run_step(script_path, description):
    """Run a Python script and check for errors."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")

    abs_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        script_path
    )

    result = subprocess.run(
        [sys.executable, abs_path],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    if result.returncode != 0:
        print(f"\n  ✗ FAILED: {description}")
        print(f"    Script: {script_path}")
        sys.exit(1)

    print(f"\n  ✓ COMPLETED: {description}")


if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   CROSS-LAYER IDS — AUTOMATED PROJECT SETUP            ║")
    print("║   This will generate data, train models, and validate.  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    project_root = os.path.dirname(os.path.abspath(__file__))

    # Create all directories
    dirs = [
        'data',
        os.path.join('esp32', 'cross_layer_ids'),
        'laptop',
        'docs',
    ]
    for d in dirs:
        full_path = os.path.join(project_root, d)
        os.makedirs(full_path, exist_ok=True)
        print(f"  ✓ Directory: {d}/")

    # Run pipeline
    run_step(
        os.path.join('data', 'validate_model.py'),
        "Step 3/4: Validate Model Rigor"
    )
    run_step(
        os.path.join('data', 'generate_plots.py'),
        "Step 4/4: Generate Presentation Plots"
    )

    print("\n" + "█" * 60)
    print("  ✓ ALL SETUP COMPLETE")
    print("")
    print("  Artifacts:")
    print("    data/dataset_train.csv")
    print("    data/dataset_test.csv")
    print("    data/rf_model_full.pkl")
    print("    data/ablation_results.json")
    print("    data/validation_results.json")
    print("    esp32/cross_layer_ids/rf_model.h")
    print("    docs/plot1_features.png")
    print("    docs/plot2_ablation.png")
    print("    docs/plot3_confusion.png")
    print("    docs/plot4_comparison.png  ← WINNING SLIDE")
    print("")
    print("  Next: Flash ESP32 → stream_to_esp32.py → dashboard.py")
    print("█" * 60)
