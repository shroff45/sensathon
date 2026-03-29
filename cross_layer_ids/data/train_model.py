#!/usr/bin/env python3
"""
Train Random Forest with FIXED ablation study.

Changes from previous version:
  1. All baselines use IDENTICAL hyperparameters (fair comparison)
  2. Added explicit check: does removing xl_ features hurt Class 2?
  3. Added cross-layer-ONLY model (6 features) to show they carry signal
  4. Increased max_depth slightly for better generalization
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pickle
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_dataset import (FEATURE_NAMES, SENSOR_F, CAN_F,
                               V2X_F, CROSS_F, ALL_FEATURES)

project_root = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
data_dir = os.path.join(project_root, 'data')
esp32_dir = os.path.join(project_root, 'esp32', 'cross_layer_ids')
os.makedirs(esp32_dir, exist_ok=True)

df_train = pd.read_csv(os.path.join(data_dir, 'dataset_train.csv'))
df_test = pd.read_csv(os.path.join(data_dir, 'dataset_test.csv'))

X_train = df_train[ALL_FEATURES].values
y_train = df_train['label'].values
X_test = df_test[ALL_FEATURES].values
y_test = df_test['label'].values

CLASS_NAMES = ['Normal', 'Single-Layer', 'Coordinated']

print(f"Training: {len(X_train)} | Test: {len(X_test)} | "
      f"Features: {len(ALL_FEATURES)}")

# ══════════════════════════════════════════════════════════
# HYPERPARAMETERS — SAME FOR ALL MODELS (fair comparison)
# ══════════════════════════════════════════════════════════

PARAMS = {
    'n_estimators': 20,
    'max_depth': 12,           # increased from 10 for better learning
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
}

results = {}


def train_eval(name, feature_names_list):
    """Train and evaluate using specified feature columns.
    Uses SAME hyperparameters for ALL models."""
    X_tr = df_train[feature_names_list].values
    X_te = df_test[feature_names_list].values

    clf = RandomForestClassifier(**PARAMS)
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)

    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    overall = f1_score(y_test, y_pred, average='weighted')

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    results[name] = {
        'precision': p.tolist(), 'recall': r.tolist(),
        'f1': f1.tolist(), 'overall_f1': float(overall),
        'n_features': len(feature_names_list),
        'confusion_matrix': cm.tolist(),
    }

    print(f"\n{'═'*60}")
    print(f"  {name} ({len(feature_names_list)} features)")
    print(f"{'═'*60}")
    print(classification_report(y_test, y_pred,
                                target_names=CLASS_NAMES,
                                zero_division=0))
    print(f"  Confusion matrix:\n{cm}")
    return clf


# ══════════════════════════════════════════════════════════
# ABLATION STUDY — 6 MODELS FOR COMPLETE COMPARISON
# ══════════════════════════════════════════════════════════

# 1. Single-layer baselines
train_eval("Sensor-Only", SENSOR_F)
train_eval("CAN-Only", CAN_F)
train_eval("V2X-Only", V2X_F)

# 2. Cross-layer features ONLY (proves they carry signal)
train_eval("CrossLayer-Only", CROSS_F)

# 3. All raw features, NO cross-layer (the key comparison)
NO_CROSS = SENSOR_F + CAN_F + V2X_F
train_eval("All-Raw-No-Cross", NO_CROSS)

# 4. FULL MODEL — all 25 features
clf_full = train_eval("Full-Cross-Layer", ALL_FEATURES)


# ══════════════════════════════════════════════════════════
# KEY RESULTS TABLE
# ══════════════════════════════════════════════════════════

print("\n" + "█" * 60)
print("  CLASS 2 (COORDINATED) DETECTION COMPARISON")
print("█" * 60)
print(f"\n  {'Model':<25s}  {'#F':>3s}  {'C0':>6s}  {'C1':>6s}  "
      f"{'C2':>6s}  {'All':>6s}")
print(f"  {'─'*25}  {'─'*3}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
for name, r in results.items():
    m = " ◄" if name == "Full-Cross-Layer" else ""
    print(f"  {name:<25s}  {r['n_features']:>3d}  "
          f"{r['f1'][0]:>6.3f}  {r['f1'][1]:>6.3f}  "
          f"{r['f1'][2]:>6.3f}  {r['overall_f1']:>6.3f}{m}")

# Explicit improvement check
no_cross_c2 = results['All-Raw-No-Cross']['f1'][2]
full_c2 = results['Full-Cross-Layer']['f1'][2]
improvement = full_c2 - no_cross_c2
print(f"\n  Cross-layer improvement on Class 2: "
      f"{no_cross_c2:.3f} → {full_c2:.3f} "
      f"(+{improvement:.3f})")
if improvement > 0.05:
    print(f"  ✓ SIGNIFICANT — cross-layer features help")
elif improvement > 0:
    print(f"  ~ MARGINAL — cross-layer features help slightly")
else:
    print(f"  ⚠ NO IMPROVEMENT — check attack design")


# ══════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ══════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("  5-FOLD CROSS-VALIDATION")
print(f"{'═'*60}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    RandomForestClassifier(**PARAMS),
    X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)

print(f"  Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
results['cross_validation'] = {
    'scores': cv_scores.tolist(),
    'mean': float(cv_scores.mean()),
    'std': float(cv_scores.std()),
}


# ══════════════════════════════════════════════════════════
# PER-ATTACK BREAKDOWN
# ══════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("  PER-ATTACK ACCURACY")
print(f"{'═'*60}")

per_attack = {}
for aname in sorted(df_test['attack_name'].unique()):
    mask = df_test['attack_name'] == aname
    if mask.sum() < 10:
        continue
    sp = clf_full.predict(df_test.loc[mask, ALL_FEATURES].values)
    st_arr = df_test.loc[mask, 'label'].values
    acc = float((sp == st_arr).mean())
    per_attack[aname] = {'n': int(mask.sum()), 'accuracy': acc}
    unseen = (" [UNSEEN]" if aname in ('attack_coord_can_imu',
                                        'attack_coord_speed_all')
              else "")
    print(f"  {aname:35s}  n={mask.sum():5d}  acc={acc:.3f}{unseen}")

results['per_attack'] = per_attack


# ══════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("  FEATURE IMPORTANCE")
print(f"{'═'*60}")

importances = clf_full.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
imp_data = {}

# Track how much importance is on cross-layer features
xl_total = 0.0
for i in sorted_idx:
    name = ALL_FEATURES[i]
    imp = importances[i]
    bar = '█' * int(imp * 120)
    tag = " ★" if name.startswith('xl_') else ""
    if name.startswith('xl_'):
        xl_total += imp
    print(f"  {name:28s}  {imp:.4f}  {bar}{tag}")
    imp_data[name] = float(imp)

print(f"\n  Cross-layer features total importance: {xl_total:.3f}")
print(f"  {'✓ SIGNIFICANT' if xl_total > 0.15 else '⚠ LOW'}")

results['feature_importance'] = imp_data


# ══════════════════════════════════════════════════════════
# EXPORT TO C
# ══════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("  EXPORTING TO C HEADER")
print(f"{'═'*60}")

total_nodes = sum(t.tree_.node_count for t in clf_full.estimators_)
ram_kb = total_nodes * 20 / 1024
print(f"  Trees: {len(clf_full.estimators_)}")
print(f"  Nodes: {total_nodes}")
print(f"  RAM: {ram_kb:.1f} KB / 520 KB")
print(f"  {'✓ FITS' if ram_kb < 400 else '✗ TOO LARGE'}")

header_path = os.path.join(esp32_dir, 'rf_model.h')

with open(header_path, 'w') as f:
    f.write("// Auto-generated Random Forest — DO NOT EDIT\n")
    f.write("// Regenerate: python data/train_model.py\n\n")
    f.write("#ifndef RF_MODEL_H\n#define RF_MODEL_H\n\n")
    f.write("#include <stdint.h>\n\n")

    nt = len(clf_full.estimators_)
    f.write(f"#define RF_N_TREES {nt}\n")
    f.write(f"#define RF_N_FEATURES {clf_full.n_features_in_}\n")
    f.write(f"#define RF_N_CLASSES {len(clf_full.classes_)}\n\n")

    for ti, est in enumerate(clf_full.estimators_):
        tree = est.tree_
        nc = tree.node_count
        f.write(f"// Tree {ti}: {nc} nodes\n")

        f.write(f"static const int16_t t{ti}_f[]={{")
        f.write(",".join(str(int(x)) for x in tree.feature))
        f.write("};\n")

        f.write(f"static const float t{ti}_t[]={{")
        f.write(",".join(f"{x:.6f}" for x in tree.threshold))
        f.write("};\n")

        f.write(f"static const int16_t t{ti}_l[]={{")
        f.write(",".join(str(int(x)) for x in tree.children_left))
        f.write("};\n")

        f.write(f"static const int16_t t{ti}_r[]={{")
        f.write(",".join(str(int(x)) for x in tree.children_right))
        f.write("};\n")

        leaf_cls = []
        for i in range(nc):
            if tree.feature[i] == -2:
                leaf_cls.append(int(np.argmax(tree.value[i][0])))
            else:
                leaf_cls.append(-1)
        f.write(f"static const int8_t t{ti}_c[]={{")
        f.write(",".join(str(x) for x in leaf_cls))
        f.write("};\n\n")

    for func, arr, rtype, default in [
        ('get_feat', 'f', 'int16_t', '-2'),
        ('get_left', 'l', 'int16_t', '-1'),
        ('get_right', 'r', 'int16_t', '-1'),
        ('get_cls', 'c', 'int8_t', '0'),
    ]:
        f.write(f"static inline {rtype} {func}"
                f"(int t, int n) {{\n  switch(t) {{\n")
        for i in range(nt):
            f.write(f"    case {i}: return t{i}_{arr}[n];\n")
        f.write(f"    default: return {default};\n")
        f.write("  }\n}\n\n")

    f.write("static inline float get_thr(int t, int n) {\n")
    f.write("  switch(t) {\n")
    for i in range(nt):
        f.write(f"    case {i}: return t{i}_t[n];\n")
    f.write("    default: return 0.0f;\n  }\n}\n\n")

    f.write("#endif\n")

print(f"  → {header_path} ✓")


# ══════════════════════════════════════════════════════════
# SAVE ALL MODELS
# ══════════════════════════════════════════════════════════

pickle.dump(clf_full, open(os.path.join(data_dir,
            'rf_model_full.pkl'), 'wb'))

for feat_set, bname in [(SENSOR_F, 'sensor'), (CAN_F, 'can'),
                         (V2X_F, 'v2x'), (NO_CROSS, 'nocross')]:
    cb = RandomForestClassifier(**PARAMS)
    cb.fit(df_train[feat_set].values, y_train)
    pickle.dump(cb, open(os.path.join(data_dir,
                f'rf_baseline_{bname}.pkl'), 'wb'))

with open(os.path.join(data_dir, 'ablation_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  All saved ✓")
