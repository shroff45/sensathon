#!/usr/bin/env python3
"""
Cross-Layer IDS — PRESENTATION PLOT GENERATOR
Run: python docs/generate_plots.py
Generates presentation-quality charts for the hackathon.
"""

import os
import sys
import pandas as pd
import json
import matplotlib
import numpy as np

# Set non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add data/ to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
from generate_dataset import CROSS_F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
DOCS_DIR = os.path.join(ROOT, 'docs')

def generate_dist_plots():
    print("  Generating feature distribution plots...")
    train_path = os.path.join(DATA_DIR, 'dataset_train.csv')
    if not os.path.exists(train_path):
        print("    ✗ dataset_train.csv NOT FOUND")
        return

    df = pd.read_csv(train_path)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for idx, feat in enumerate(CROSS_F):
        ax = axes[idx // 3][idx % 3]
        for cls, color, label in [(0, '#00cc66', 'Normal'),
                                   (1, '#ffaa00', 'Single-Layer'),
                                   (2, '#ff3366', 'Coordinated')]:
            vals = df[df['label'] == cls][feat]
            ax.hist(vals, bins=50, alpha=0.6, color=color,
                    label=label, density=True)
        ax.set_title(feat.replace('xl_', '').replace('_', ' ').title(),
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlabel('Consistency Score (0=agree, 1=violation)')
        ax.set_ylabel('Density')

    plt.suptitle("Cross-Layer Feature Distributions by Attack Class\n"
                 "(Separation proves physics-based detection works)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out = os.path.join(DOCS_DIR, 'feature_distributions.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {out}")

def generate_ablation_chart():
    print("  Generating ablation study chart...")
    ab_path = os.path.join(DATA_DIR, 'ablation_results.json')
    if not os.path.exists(ab_path):
        print("    ✗ ablation_results.json NOT FOUND")
        return

    with open(ab_path, 'r') as f:
        ab = json.load(f)

    # Filter out CV and other metadata
    models = [k for k in ab if k not in ('cross_validation', 'per_attack', 'feature_importance')]
    c2 = [ab[m]['f1'][2] for m in models]
    ov = [ab[m]['overall_f1'] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    w = 0.35
    bars1 = ax.bar(x - w/2, c2, w, label='Coordinated Attack F1', color='#ff3366')
    bars2 = ax.bar(x + w/2, ov, w, label='Overall F1', color='#4488ff')

    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('F1 Score')
    ax.set_title('Ablation Study: The Impact of Cross-Layer Features')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add labels on top
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(DOCS_DIR, 'ablation_chart.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {out}")

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════╗")
    print("║  CROSS-LAYER IDS — PLOT GENERATOR            ║")
    print("╚══════════════════════════════════════════════╝")
    
    os.makedirs(DOCS_DIR, exist_ok=True)
    generate_dist_plots()
    generate_ablation_chart()
    print("\n  Done. Charts ready for presentation.")
