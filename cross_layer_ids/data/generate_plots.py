#!/usr/bin/env python3
"""
Generates 4 static presentation plots for hackathon backup.
Run: python data/generate_plots.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_dataset import (CROSS_F, ALL_FEATURES, SENSOR_F,
                               CAN_F, V2X_F)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
DOCS = os.path.join(ROOT, 'docs')
os.makedirs(DOCS, exist_ok=True)

# ── Load data ──
df_train = pd.read_csv(os.path.join(DATA, 'dataset_train.csv'))
df_test = pd.read_csv(os.path.join(DATA, 'dataset_test.csv'))
clf = pickle.load(open(os.path.join(DATA, 'rf_model_full.pkl'), 'rb'))
ab = json.load(open(os.path.join(DATA, 'ablation_results.json')))

y_test = df_test['label'].values
y_pred = clf.predict(df_test[ALL_FEATURES].values)


# ══════════════════════════════════════════════════════════
# PLOT 1: Cross-Layer Feature Distributions
# ══════════════════════════════════════════════════════════

print("Plot 1: Feature Distributions...")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for idx, feat in enumerate(CROSS_F):
    ax = axes[idx // 3][idx % 3]
    for cls, color, label in [(0, '#00cc66', 'Normal'),
                               (1, '#ffaa00', 'Single-Layer'),
                               (2, '#ff3366', 'Coordinated')]:
        vals = df_train[df_train['label'] == cls][feat]
        ax.hist(vals, bins=60, alpha=0.6, color=color,
                label=label, density=True)
    name = feat.replace('xl_', '').replace('_', ' ').title()
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlabel('Consistency Score')

plt.suptitle("Cross-Layer Feature Distributions by Attack Class\n"
             "Coordinated attacks (red) are physically inconsistent "
             "across layers",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(DOCS, 'plot1_features.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ plot1_features.png")


# ══════════════════════════════════════════════════════════
# PLOT 2: Ablation Study
# ══════════════════════════════════════════════════════════

print("Plot 2: Ablation Study...")

models = [k for k in ab if k not in
          ('cross_validation', 'per_attack', 'feature_importance')]
c2_f1 = [ab[m]['f1'][2] for m in models]
overall = [ab[m]['overall_f1'] for m in models]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(models))
w = 0.35

colors_c2 = ['#888888'] * (len(models) - 1) + ['#ff3366']
colors_ov = ['#aaaaaa'] * (len(models) - 1) + ['#00cc66']

bars1 = ax.bar(x - w/2, c2_f1, w,
               label='Coordinated Attack F1', color=colors_c2)
bars2 = ax.bar(x + w/2, overall, w,
               label='Overall F1', color=colors_ov)

for bar, val in zip(bars1, c2_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, overall):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=9)

ax.set_xticks(x)
short_names = []
for m in models:
    s = m.replace('All-Raw-No-Cross', 'All Raw\n(no XL)')
    s = s.replace('Full-Cross-Layer', 'Full + XL\n(Ours)')
    s = s.replace('CrossLayer-Only', 'XL Only\n(6 feat)')
    s = s.replace('-', '\n')
    short_names.append(s)
ax.set_xticklabels(short_names, fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Ablation Study: Cross-Layer Features Are Necessary\n'
             'Removing them drops coordinated attack detection',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)

if len(models) >= 2:
    ax.annotate('★ OUR\nCONTRIBUTION',
                xy=(len(models)-1, c2_f1[-1]),
                xytext=(len(models)-1.8, min(c2_f1[-1] + 0.15, 1.05)),
                fontsize=12, color='#ff3366', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#ff3366',
                                lw=2))

plt.tight_layout()
plt.savefig(os.path.join(DOCS, 'plot2_ablation.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ plot2_ablation.png")


# ══════════════════════════════════════════════════════════
# PLOT 3: Confusion Matrix
# ══════════════════════════════════════════════════════════

print("Plot 3: Confusion Matrix...")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues', aspect='auto')

labels = ['Normal', 'Single-Layer', 'Coordinated']
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(labels, fontsize=12)
ax.set_yticklabels(labels, fontsize=12)
ax.set_xlabel('Predicted', fontsize=13)
ax.set_ylabel('Actual', fontsize=13)
ax.set_title('Confusion Matrix — Full Cross-Layer Model',
             fontsize=14, fontweight='bold')

for i in range(3):
    for j in range(3):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=16, fontweight='bold', color=color)

plt.colorbar(im, ax=ax, label='Count')
plt.tight_layout()
plt.savefig(os.path.join(DOCS, 'plot3_confusion.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ plot3_confusion.png")


# ══════════════════════════════════════════════════════════
# PLOT 4: Baseline Comparison — THE WINNING SLIDE
# ══════════════════════════════════════════════════════════

print("Plot 4: Baseline Comparison (Winning Slide)...")

from sklearn.metrics import recall_score

model_results = {}

# Full model
model_results['Cross-Layer\n(Ours)'] = {
    'coord_recall': recall_score(y_test, y_pred, labels=[2],
                                  average='macro',
                                  zero_division=0) * 100,
    'overall': (y_pred == y_test).mean() * 100,
}

# Baselines
baseline_configs = [
    ('Sensor\nOnly', 'sensor', SENSOR_F),
    ('CAN\nOnly', 'can', CAN_F),
    ('V2X\nOnly', 'v2x', V2X_F),
    ('All Raw\n(no XL)', 'nocross', SENSOR_F + CAN_F + V2X_F),
]

for display_name, file_suffix, feature_list in baseline_configs:
    bpath = os.path.join(DATA, f'rf_baseline_{file_suffix}.pkl')
    if not os.path.exists(bpath):
        print(f"  ⚠ Missing baseline: {bpath}")
        continue
    bclf = pickle.load(open(bpath, 'rb'))
    X_b = df_test[feature_list].values
    bp = bclf.predict(X_b)
    model_results[display_name] = {
        'coord_recall': recall_score(y_test, bp, labels=[2],
                                      average='macro',
                                      zero_division=0) * 100,
        'overall': (bp == y_test).mean() * 100,
    }

names = list(model_results.keys())
coord_recalls = [model_results[n]['coord_recall'] for n in names]
overalls = [model_results[n]['overall'] for n in names]

fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(names))
w = 0.35

# Our model gets different colors
colors_cr = ['#ff5252'] * (len(names)-1) + ['#2196F3']
colors_ov = ['#ffab40'] * (len(names)-1) + ['#69f0ae']

bars1 = ax.bar(x_pos - w/2, coord_recalls, w,
               label='Coordinated Attack Recall (%)',
               color=colors_cr)
bars2 = ax.bar(x_pos + w/2, overalls, w,
               label='Overall Accuracy (%)',
               color=colors_ov)

for bar, val in zip(bars1, coord_recalls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, overalls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val:.1f}%', ha='center', fontsize=9)

ax.set_xticks(x_pos)
ax.set_xticklabels(names, fontsize=11)
ax.set_ylim(0, 115)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Why Single-Layer Systems Fail\n'
             'Only our cross-layer model catches coordinated attacks',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(DOCS, 'plot4_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ plot4_comparison.png")


print(f"\n{'█'*50}")
print("  ALL 4 PLOTS SAVED TO docs/")
print(f"{'█'*50}")
