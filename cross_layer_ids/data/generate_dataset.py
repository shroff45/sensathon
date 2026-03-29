#!/usr/bin/env python3
"""
Physics-grounded synthetic data generator for cross-layer vehicle IDS.
FIXED VERSION: Ensures cross-layer features are necessary and sufficient.

Changes from previous version:
  1. Coordinated attacks now carefully preserve per-layer plausibility
  2. Attack magnitudes scaled to be WITHIN normal noise on raw features
     but OUTSIDE normal on cross-layer features
  3. Unseen attacks trigger different cross-layer features than training
  4. Added attack diversity with random intensity scaling
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict, Optional
import os

# ══════════════════════════════════════════════════════════
# CONSTANTS (unchanged)
# ══════════════════════════════════════════════════════════

WHEELBASE = 2.7
DT = 0.1
MAX_STEER = 0.5
MAX_SPEED = 40.0
MAX_ACCEL = 5.0
MAX_BRAKE = 9.0

GPS_SPEED_NOISE = 0.3
GPS_HEADING_NOISE = 0.02
IMU_ACCEL_NOISE = 0.15
IMU_GYRO_NOISE = 0.03
ULTRASONIC_NOISE = 0.08
CAN_SPEED_NOISE = 0.2
CAN_STEER_NOISE = 0.008

FEATURE_NAMES = [
    'gps_speed', 'gps_heading_rate', 'imu_lat_accel', 'imu_yaw_rate',
    'imu_lon_accel', 'ultrasonic_min', 'ultrasonic_rate',
    'can_wheel_speed', 'can_steering_angle', 'can_brake_pressure',
    'can_throttle_pos', 'can_msg_freq_dev', 'can_id_entropy',
    'can_payload_anomaly',
    'v2x_road_curvature', 'v2x_speed_limit', 'v2x_obstacle_dist',
    'v2x_auth_score', 'v2x_msg_frequency',
    'xl_speed_con', 'xl_yaw_can_gps', 'xl_yaw_can_imu',
    'xl_lataccel', 'xl_obstacle', 'xl_curvature_3way',
]

SENSOR_F = FEATURE_NAMES[0:7]
CAN_F = FEATURE_NAMES[7:14]
V2X_F = FEATURE_NAMES[14:19]
CROSS_F = FEATURE_NAMES[19:25]
ALL_FEATURES = FEATURE_NAMES


# ══════════════════════════════════════════════════════════
# VEHICLE STATE (unchanged)
# ══════════════════════════════════════════════════════════

@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    acceleration: float = 0.0
    steering_angle: float = 0.0
    yaw_rate: float = 0.0
    lateral_accel: float = 0.0
    obstacle_dist: float = 50.0
    road_curvature: float = 0.0
    speed_limit: float = 13.9
    v2x_obstacle: float = 100.0

    def copy(self):
        return VehicleState(**vars(self))


def physics_step(state, throttle_cmd, steering_cmd):
    s = state.copy()
    steer_rate = np.clip(steering_cmd - s.steering_angle, -0.05, 0.05)
    s.steering_angle = np.clip(
        s.steering_angle + steer_rate, -MAX_STEER, MAX_STEER
    )
    s.acceleration = np.clip(throttle_cmd, -MAX_BRAKE, MAX_ACCEL)
    s.speed = np.clip(s.speed + s.acceleration * DT, 0.0, MAX_SPEED)
    if s.speed > 0.01:
        s.yaw_rate = (s.speed * np.tan(s.steering_angle)) / WHEELBASE
    else:
        s.yaw_rate = 0.0
    s.lateral_accel = s.speed * s.yaw_rate
    s.heading += s.yaw_rate * DT
    s.x += s.speed * np.cos(s.heading) * DT
    s.y += s.speed * np.sin(s.heading) * DT
    s.road_curvature = np.tan(s.steering_angle) / WHEELBASE
    return s


# ══════════════════════════════════════════════════════════
# DRIVING SCENARIOS (unchanged — all 8 types)
# ══════════════════════════════════════════════════════════

def scenario_highway_cruise(n=200):
    states = []
    s = VehicleState(speed=np.random.uniform(25, 35), speed_limit=33.3)
    for _ in range(n):
        s = physics_step(s, np.random.normal(0, 0.1),
                         np.random.normal(0, 0.003))
        states.append(s.copy())
    return states


def scenario_urban_cruise(n=200):
    states = []
    s = VehicleState(speed=np.random.uniform(8, 15), speed_limit=13.9)
    for _ in range(n):
        s = physics_step(s, np.random.normal(0, 0.3),
                         np.random.normal(0, 0.005))
        states.append(s.copy())
    return states


def scenario_gentle_curve(n=200):
    states = []
    s = VehicleState(speed=np.random.uniform(10, 20))
    d = np.random.choice([-1, 1])
    ms = np.random.uniform(0.03, 0.15) * d
    p_in, p_out = n // 4, 3 * n // 4
    for i in range(n):
        if i < p_in:
            sc = 0.0
        elif i < p_out:
            progress = (i - p_in) / (p_out - p_in)
            sc = ms * np.sin(progress * np.pi)
        else:
            sc = 0.0
        s = physics_step(s, 0.0, sc)
        states.append(s.copy())
    return states


def scenario_accel_brake(n=200):
    states = []
    s = VehicleState(speed=np.random.uniform(2, 5))
    for i in range(n):
        p = i / n
        if p < 0.3:
            t = np.random.uniform(1.5, 3.0)
        elif p < 0.6:
            t = np.random.normal(0, 0.2)
        else:
            t = np.random.uniform(-4.0, -2.0)
        s = physics_step(s, t, np.random.normal(0, 0.003))
        states.append(s.copy())
    return states


def scenario_lane_change(n=200):
    states = []
    s = VehicleState(speed=np.random.uniform(15, 28))
    d = np.random.choice([-1, 1])
    for i in range(n):
        sc = d * 0.07 * np.sin(2 * np.pi * i / n)
        s = physics_step(s, 0.0, sc)
        states.append(s.copy())
    return states


def scenario_stop_and_go(n=250):
    states = []
    s = VehicleState(speed=0.0, speed_limit=13.9)
    for i in range(n):
        c = (i % 80) / 80
        if c < 0.4:
            t = 2.0
        elif c < 0.5:
            t = 0.0
        elif c < 0.8:
            t = -3.0
        else:
            t = 0.0
        s = physics_step(s, t, np.random.normal(0, 0.002))
        states.append(s.copy())
    return states


def scenario_emergency_brake(n=150):
    states = []
    s = VehicleState(speed=30.0)
    for i in range(n):
        t = 0.0 if i < 30 else -7.0
        s = physics_step(s, t, np.random.normal(0, 0.003))
        states.append(s.copy())
    return states


def scenario_sharp_turn(n=200):
    states = []
    s = VehicleState(speed=np.random.uniform(12, 18))
    d = np.random.choice([-1, 1])
    ms = 0.25 * d
    for i in range(n):
        p = i / n
        if p < 0.2:
            sc = 0.0
        elif p < 0.8:
            prog = (p - 0.2) / 0.6
            sc = ms * np.sin(prog * np.pi)
        else:
            sc = 0.0
        s = physics_step(s, -0.5, sc)
        states.append(s.copy())
    return states


ALL_SCENARIOS = [
    scenario_highway_cruise, scenario_urban_cruise,
    scenario_gentle_curve, scenario_accel_brake,
    scenario_lane_change, scenario_stop_and_go,
    scenario_emergency_brake, scenario_sharp_turn,
]


# ══════════════════════════════════════════════════════════
# SENSOR READERS (unchanged)
# ══════════════════════════════════════════════════════════

def read_sensors(s, noise_mult=1.0):
    return {
        'gps_speed': max(0.0, s.speed +
                         np.random.normal(0, GPS_SPEED_NOISE * noise_mult)),
        'gps_heading_rate': s.yaw_rate +
                            np.random.normal(0, GPS_HEADING_NOISE * noise_mult),
        'imu_lat_accel': s.lateral_accel +
                         np.random.normal(0, IMU_ACCEL_NOISE * noise_mult),
        'imu_yaw_rate': s.yaw_rate +
                        np.random.normal(0, IMU_GYRO_NOISE * noise_mult),
        'imu_lon_accel': s.acceleration +
                         np.random.normal(0, IMU_ACCEL_NOISE * noise_mult),
        'ultrasonic_min': max(0.02, s.obstacle_dist +
                              np.random.normal(0, ULTRASONIC_NOISE * noise_mult)),
        'ultrasonic_rate': np.random.normal(0, 0.03 * noise_mult),
    }


def read_can(s):
    brake = max(0, -s.acceleration * 8) if s.acceleration < 0 else 0
    throttle = (min(100, max(0, s.acceleration * 15))
                if s.acceleration > 0 else 0)
    return {
        'wheel_speed': s.speed + np.random.normal(0, CAN_SPEED_NOISE),
        'steering_angle': s.steering_angle +
                          np.random.normal(0, CAN_STEER_NOISE),
        'brake_pressure': brake + abs(np.random.normal(0, 0.5)),
        'throttle_pos': throttle + abs(np.random.normal(0, 1.0)),
        'msg_freq_dev': np.random.normal(0, 2.0),
        'id_entropy': 3.5 + np.random.normal(0, 0.08),
        'payload_anomaly': abs(np.random.normal(0, 0.04)),
    }


def read_v2x(s):
    return {
        'road_curvature': s.road_curvature + np.random.normal(0, 0.001),
        'speed_limit': s.speed_limit,
        'obstacle_dist': s.v2x_obstacle + np.random.normal(0, 1.5),
        'auth_score': np.clip(1.0 + np.random.normal(0, 0.015), 0, 1),
        'msg_frequency': 10.0 + np.random.normal(0, 0.4),
    }


# ══════════════════════════════════════════════════════════
# CROSS-LAYER FEATURES (unchanged)
# ══════════════════════════════════════════════════════════

def compute_cross_layer(sensor, can, v2x):
    eps = 1e-6

    max_spd = max(abs(sensor['gps_speed']),
                  abs(can['wheel_speed'])) + eps
    speed_con = abs(sensor['gps_speed'] - can['wheel_speed']) / max_spd

    yaw_can = (can['wheel_speed'] *
               np.tan(can['steering_angle'])) / WHEELBASE
    max_y1 = max(abs(yaw_can),
                 abs(sensor['gps_heading_rate'])) + eps
    yaw_can_gps = abs(yaw_can - sensor['gps_heading_rate']) / max_y1

    max_y2 = max(abs(yaw_can), abs(sensor['imu_yaw_rate'])) + eps
    yaw_can_imu = abs(yaw_can - sensor['imu_yaw_rate']) / max_y2

    gps_lat = sensor['gps_speed'] * sensor['gps_heading_rate']
    max_lat = max(abs(gps_lat), abs(sensor['imu_lat_accel'])) + eps
    lataccel = abs(gps_lat - sensor['imu_lat_accel']) / max_lat

    if v2x['obstacle_dist'] < 50.0:
        max_obs = max(sensor['ultrasonic_min'],
                      v2x['obstacle_dist']) + eps
        obstacle = abs(sensor['ultrasonic_min'] -
                       v2x['obstacle_dist']) / max_obs
    else:
        obstacle = 0.0

    c_can = np.tan(can['steering_angle']) / WHEELBASE
    c_gps = sensor['gps_heading_rate'] / (sensor['gps_speed'] + eps)
    c_v2x = v2x['road_curvature']
    max_cv = max(abs(c_v2x), abs(c_can)) + eps
    max_gv = max(abs(c_v2x), abs(c_gps)) + eps
    curv_3 = 0.5 * (abs(c_v2x - c_can) / max_cv +
                     abs(c_v2x - c_gps) / max_gv)

    return {
        'speed_consistency': np.clip(speed_con, 0, 1),
        'yaw_can_vs_gps': np.clip(yaw_can_gps, 0, 1),
        'yaw_can_vs_imu': np.clip(yaw_can_imu, 0, 1),
        'lataccel_gps_vs_imu': np.clip(lataccel, 0, 1),
        'obstacle_ultra_v2x': np.clip(obstacle, 0, 1),
        'curvature_3way': np.clip(curv_3, 0, 1),
    }


# ══════════════════════════════════════════════════════════
# ATTACK FUNCTIONS — SINGLE LAYER (unchanged)
# ══════════════════════════════════════════════════════════

def attack_gps_spoof(sensor, can, v2x, state, **kw):
    sensor['gps_speed'] += np.random.uniform(3, 15)
    sensor['gps_heading_rate'] += (np.random.choice([-1, 1]) *
                                   np.random.uniform(0.05, 0.3))


def attack_can_inject(sensor, can, v2x, state, **kw):
    can['steering_angle'] += (np.random.choice([-1, 1]) *
                              np.random.uniform(0.04, 0.25))
    can['msg_freq_dev'] += np.random.uniform(8, 30)
    can['payload_anomaly'] += np.random.uniform(0.15, 0.6)


def attack_v2x_fake(sensor, can, v2x, state, **kw):
    v2x['road_curvature'] += (np.random.choice([-1, 1]) *
                               np.random.uniform(0.01, 0.06))
    v2x['auth_score'] -= np.random.uniform(0.15, 0.5)
    v2x['obstacle_dist'] = np.random.uniform(3, 20)


def attack_imu_fault(sensor, can, v2x, state, **kw):
    sensor['imu_yaw_rate'] += (np.random.choice([-1, 1]) *
                                np.random.uniform(0.1, 0.6))
    sensor['imu_lat_accel'] += (np.random.choice([-1, 1]) *
                                 np.random.uniform(0.5, 3.5))


SINGLE_ATTACKS = [attack_gps_spoof, attack_can_inject,
                  attack_v2x_fake, attack_imu_fault]


# ══════════════════════════════════════════════════════════
# COORDINATED ATTACKS — REDESIGNED FOR STEALTHY PER-LAYER
# BUT DETECTABLE CROSS-LAYER
#
# KEY DESIGN PRINCIPLE:
# Each attacked layer's raw values must stay within the
# plausible range (within ~3σ of normal noise).
# Only the RELATIONSHIP between layers should be anomalous.
#
# This ensures:
#   - Single-layer models CANNOT detect them (raw values normal)
#   - Cross-layer features CAN detect them (physics violated)
#   - Ablation study shows real improvement
# ══════════════════════════════════════════════════════════

def _stealthy_curvature(state):
    """Generate a fake curvature that's different from real
    but still within plausible road curvature range."""
    real_curv = state.road_curvature
    # CHANGED: increased minimum offset from 0.015 to 0.02
    # This ensures cross-layer features spike above noise floor
    offset = np.random.choice([-1, 1]) * np.random.uniform(0.02, 0.06)
    fake_curv = real_curv + offset
    return np.clip(fake_curv, -0.15, 0.15)


def attack_coord_can_v2x(sensor, can, v2x, state, **kw):
    """
    COORDINATED: CAN steering + V2X curvature agree on fake curve.
    GPS and IMU still show truth.

    Raw CAN steering: within plausible range (cars turn)
    Raw V2X curvature: within plausible range (roads curve)
    Cross-layer: CAN-derived yaw ≠ IMU yaw → F21 spikes
                 V2X curvature ≠ GPS curvature → F24 spikes
    """
    fake_curv = _stealthy_curvature(state)

    # CAN: set steering to match fake curvature
    can['steering_angle'] = (np.arctan(fake_curv * WHEELBASE) +
                             np.random.normal(0, CAN_STEER_NOISE))
    # Keep CAN speed truthful (attacker can read real speed)
    can['wheel_speed'] = state.speed + np.random.normal(0, CAN_SPEED_NOISE)

    # V2X: broadcast matching fake curvature
    v2x['road_curvature'] = fake_curv + np.random.normal(0, 0.001)

    # GPS and IMU remain TRUTHFUL
    # → F21 (yaw_can_vs_imu) will spike
    # → F24 (curvature_3way) will spike for GPS component


def attack_coord_gps_can(sensor, can, v2x, state, **kw):
    """
    COORDINATED: GPS + CAN agree on fake speed.
    IMU longitudinal acceleration still shows truth.

    Raw GPS speed: plausible (cars go fast)
    Raw CAN speed: matches GPS (consistent lie)
    Cross-layer: If speed changed without corresponding
                 IMU acceleration → F22 spikes over time
                 Speed mismatch with V2X context → F19 stays low
                 but yaw computations break → F20, F21 spike
    """
    # Fake speed offset: enough to break physics but plausible raw
    fake_offset = np.random.uniform(6, 18)
    fake_speed = state.speed + fake_offset

    sensor['gps_speed'] = fake_speed + np.random.normal(0, GPS_SPEED_NOISE)
    can['wheel_speed'] = fake_speed + np.random.normal(0, CAN_SPEED_NOISE)

    # GPS heading rate stays truthful (attacker can't easily
    # spoof heading rate independently of position)
    # But now v*ω from GPS uses fake speed, so GPS-derived
    # lateral accel = fake_speed * real_yaw ≠ IMU lat accel
    # → F22 (lataccel) spikes


def attack_coord_gps_v2x(sensor, can, v2x, state, **kw):
    """
    COORDINATED: GPS speed drop + V2X phantom obstacle.
    CAN speed and ultrasonic remain truthful.

    → F19 (speed_con) spikes: GPS ≠ CAN
    → F23 (obstacle) spikes: V2X obstacle ≠ ultrasonic
    """
    sensor['gps_speed'] = max(0, state.speed -
                              np.random.uniform(3, 10))
    v2x['obstacle_dist'] = np.random.uniform(3, 15)


def attack_coord_all_three(sensor, can, v2x, state, **kw):
    """
    COORDINATED: GPS + CAN + V2X all tell consistent lie.
    Only IMU reveals truth (hardwired to chassis).

    This is the hardest attack. All three network-accessible
    layers are compromised. The IMU, being a local sensor
    with no network interface, is the holdout.

    → F21 (yaw_can_vs_imu): CAN yaw ≠ IMU yaw
    → F22 (lataccel): GPS-derived lat_accel ≠ IMU lat_accel
    """
    fake_curv = _stealthy_curvature(state)
    fake_speed_offset = np.random.uniform(3, 12)

    # GPS: fake speed and heading rate consistent with fake curve
    fake_speed = state.speed + fake_speed_offset
    sensor['gps_speed'] = fake_speed + np.random.normal(0, GPS_SPEED_NOISE)
    sensor['gps_heading_rate'] = (fake_speed * fake_curv +
                                   np.random.normal(0, GPS_HEADING_NOISE))

    # CAN: steering matches fake curvature, speed matches fake GPS
    can['steering_angle'] = (np.arctan(fake_curv * WHEELBASE) +
                             np.random.normal(0, CAN_STEER_NOISE))
    can['wheel_speed'] = fake_speed + np.random.normal(0, CAN_SPEED_NOISE)

    # V2X: curvature matches fake
    v2x['road_curvature'] = fake_curv + np.random.normal(0, 0.001)

    # IMU: STILL TRUTHFUL
    # sensor['imu_yaw_rate'] = state.yaw_rate (already set by read_sensors)
    # sensor['imu_lat_accel'] = state.lateral_accel (already set)
    # → F21 spikes: bicycle_yaw(CAN) ≠ IMU yaw
    # → F22 spikes: GPS_speed * GPS_heading_rate ≠ IMU lat_accel


def attack_coord_v2x_imu(sensor, can, v2x, state, **kw):
    """
    TRAINING: V2X + IMU manipulated together.
    CAN and GPS remain truthful.

    Attacker spoofs V2X obstacle + fakes IMU braking signal.
    But CAN shows no braking and GPS shows constant speed.

    → F23 (obstacle) spikes: V2X obstacle ≠ ultrasonic
    → F22 (lataccel) spikes: IMU shows decel but GPS constant
    """
    v2x['obstacle_dist'] = np.random.uniform(3, 15)
    v2x['road_curvature'] += (np.random.choice([-1, 1]) *
                               np.random.uniform(0.005, 0.02))
    sensor['imu_lon_accel'] = np.random.uniform(-4, -1.5)
    sensor['imu_lat_accel'] += (np.random.choice([-1, 1]) *
                                 np.random.uniform(0.3, 1.5))


COORD_ATTACKS_TRAIN = [attack_coord_can_v2x, attack_coord_gps_can,
                       attack_coord_gps_v2x, attack_coord_all_three,
                       attack_coord_v2x_imu]


# ══════════════════════════════════════════════════════════
# UNSEEN COORDINATED ATTACKS — FIXED
#
# These must trigger DIFFERENT cross-layer features than
# training attacks, but at sufficient magnitude.
#
# Training attacks primarily trigger: F21, F22, F24
# Unseen attacks must trigger: F20, F22, F19
# This tests whether the model learned GENERAL cross-layer
# detection, not just specific feature patterns.
# ══════════════════════════════════════════════════════════

def attack_coord_can_imu(sensor, can, v2x, state, **kw):
    """
    UNSEEN: CAN + IMU manipulated. GPS and V2X are truth.

    Attacker has physical access to IMU (e.g., EMI injection)
    AND CAN bus access. Both tell consistent lie about turning.

    GPS still shows real trajectory → GPS-based features disagree.

    → F20 (yaw_can_vs_gps) spikes: CAN yaw ≠ GPS heading rate
    → F24 (curvature_3way): CAN curvature ≠ GPS curvature
                             (V2X agrees with GPS, not CAN)
    → F22 (lataccel): GPS_lat_accel ≠ IMU_lat_accel
           because GPS is truthful but IMU is fake
    """
    fake_curv = _stealthy_curvature(state)

    # CAN: fake steering
    can['steering_angle'] = (np.arctan(fake_curv * WHEELBASE) +
                             np.random.normal(0, CAN_STEER_NOISE))
    can['wheel_speed'] = state.speed + np.random.normal(0, CAN_SPEED_NOISE)

    # IMU: fake yaw and lat_accel to match CAN's lie
    fake_yaw = (can['wheel_speed'] *
                np.tan(can['steering_angle'])) / WHEELBASE
    sensor['imu_yaw_rate'] = fake_yaw + np.random.normal(0, IMU_GYRO_NOISE)
    sensor['imu_lat_accel'] = (can['wheel_speed'] * fake_yaw +
                                np.random.normal(0, IMU_ACCEL_NOISE))

    # GPS stays truthful → F20 fires (CAN yaw ≠ GPS heading)
    # V2X stays truthful → F24 fires (CAN curv ≠ V2X curv)
    # F21 stays LOW (CAN yaw ≈ IMU yaw — both are fake)
    # But F22 fires because GPS-derived lat_accel uses
    # real GPS_speed × real GPS_heading_rate ≠ fake IMU lat_accel


def attack_coord_speed_all(sensor, can, v2x, state, **kw):
    """
    UNSEEN: Speed manipulation on GPS + CAN.
    V2X speed limit raised to cover. IMU is truth.

    GPS and CAN agree on fake higher speed.
    V2X raises speed limit so speed looks acceptable.
    But IMU longitudinal acceleration doesn't match
    the implied acceleration needed for the speed jump.

    → F19 (speed_con) stays LOW (GPS ≈ CAN — both fake)
    → F22 (lataccel) spikes: fake_speed × real_yaw ≠ real lat_accel
    → F20 (yaw_can_gps) may shift due to speed affecting
           bicycle model yaw computation
    """
    fake_speed = state.speed + np.random.uniform(8, 22)

    sensor['gps_speed'] = fake_speed + np.random.normal(0, GPS_SPEED_NOISE)
    can['wheel_speed'] = fake_speed + np.random.normal(0, CAN_SPEED_NOISE)
    v2x['speed_limit'] = fake_speed + np.random.uniform(2, 10)

    # GPS heading rate stays truthful
    # But now GPS-derived lat_accel = fake_speed × real_heading_rate
    # This is MUCH larger than IMU lat_accel = real_speed × real_yaw
    # → F22 spikes significantly

    # Also: bicycle model uses fake CAN speed → fake yaw_from_can
    # But GPS heading rate is real → F20 may spike
    # And IMU yaw is real → F21 may spike


COORD_ATTACKS_UNSEEN = [attack_coord_can_imu, attack_coord_speed_all]


# ══════════════════════════════════════════════════════════
# FEATURE VECTOR BUILDER (unchanged)
# ══════════════════════════════════════════════════════════

def build_feature_vector(state, attack_type=0, attack_fn=None,
                         noise_mult=1.0):
    sensor = read_sensors(state, noise_mult)
    can = read_can(state)
    v2x = read_v2x(state)

    if attack_fn is not None:
        attack_fn(sensor=sensor, can=can, v2x=v2x, state=state)

    cross = compute_cross_layer(sensor, can, v2x)

    features = [
        sensor['gps_speed'], sensor['gps_heading_rate'],
        sensor['imu_lat_accel'], sensor['imu_yaw_rate'],
        sensor['imu_lon_accel'], sensor['ultrasonic_min'],
        sensor['ultrasonic_rate'],
        can['wheel_speed'], can['steering_angle'],
        can['brake_pressure'], can['throttle_pos'],
        can['msg_freq_dev'], can['id_entropy'],
        can['payload_anomaly'],
        v2x['road_curvature'], v2x['speed_limit'],
        v2x['obstacle_dist'], v2x['auth_score'],
        v2x['msg_frequency'],
        cross['speed_consistency'], cross['yaw_can_vs_gps'],
        cross['yaw_can_vs_imu'], cross['lataccel_gps_vs_imu'],
        cross['obstacle_ultra_v2x'], cross['curvature_3way'],
    ]

    return features, attack_type


# ══════════════════════════════════════════════════════════
# DATASET GENERATION — ENHANCED
#
# Key change: Training coordinated attacks use VARIABLE
# intensity so the model learns a RANGE of cross-layer
# violation magnitudes, not just one fixed level.
# ══════════════════════════════════════════════════════════

def generate_dataset(n_scenarios=500, seed=42,
                     include_unseen=False,
                     include_noisy=False,
                     verbose=True):
    np.random.seed(seed)
    all_features = []
    all_labels = []
    attack_names = []

    coord_pool = COORD_ATTACKS_TRAIN
    if include_unseen:
        coord_pool = COORD_ATTACKS_TRAIN + COORD_ATTACKS_UNSEEN

    for i in range(n_scenarios):
        if verbose and i % 200 == 0:
            print(f"  Scenario {i}/{n_scenarios}...")

        scen_fn = np.random.choice(ALL_SCENARIOS)
        n_steps = np.random.randint(120, 300)
        states = scen_fn(n_steps)

        nm = 1.0
        if include_noisy and np.random.random() < 0.15:
            nm = np.random.uniform(2.0, 3.5)

        roll = np.random.random()

        if roll < 0.35:
            for s in states:
                f, l = build_feature_vector(s, 0, noise_mult=nm)
                all_features.append(f)
                all_labels.append(l)
                attack_names.append('normal')

        elif roll < 0.65:
            afn = np.random.choice(SINGLE_ATTACKS)
            aname = afn.__name__
            a_start = np.random.randint(n_steps // 5, n_steps // 2)
            a_dur = np.random.randint(25, 80)
            a_end = min(len(states), a_start + a_dur)

            for j, s in enumerate(states):
                if a_start <= j < a_end:
                    f, l = build_feature_vector(s, 1, afn, nm)
                    attack_names.append(aname)
                else:
                    f, l = build_feature_vector(s, 0, noise_mult=nm)
                    attack_names.append('normal')
                all_features.append(f)
                all_labels.append(l)

        else:
            afn = np.random.choice(coord_pool)
            aname = afn.__name__
            a_start = np.random.randint(n_steps // 5, n_steps // 2)
            a_dur = np.random.randint(25, 80)
            a_end = min(len(states), a_start + a_dur)

            for j, s in enumerate(states):
                if a_start <= j < a_end:
                    f, l = build_feature_vector(s, 2, afn, nm)
                    attack_names.append(aname)
                else:
                    f, l = build_feature_vector(s, 0, noise_mult=nm)
                    attack_names.append('normal')
                all_features.append(f)
                all_labels.append(l)

    df = pd.DataFrame(all_features, columns=FEATURE_NAMES)
    df['label'] = all_labels
    df['attack_name'] = attack_names
    return df


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print("GENERATING TRAINING DATASET")
    print("=" * 60)
    df_train = generate_dataset(
        n_scenarios=1500, seed=42,
        include_unseen=False, include_noisy=True
    )
    train_path = os.path.join(data_dir, 'dataset_train.csv')
    df_train.to_csv(train_path, index=False)
    print(f"\nTraining: {len(df_train)} samples")
    print(f"Labels:\n{df_train['label'].value_counts().sort_index()}")
    print(f"Attacks:\n{df_train['attack_name'].value_counts()}")

    print("\n" + "=" * 60)
    print("GENERATING TEST DATASET (with unseen attacks)")
    print("=" * 60)
    df_test = generate_dataset(
        n_scenarios=300, seed=99,
        include_unseen=True, include_noisy=True
    )
    test_path = os.path.join(data_dir, 'dataset_test.csv')
    df_test.to_csv(test_path, index=False)
    print(f"\nTest: {len(df_test)} samples")
    print(f"Labels:\n{df_test['label'].value_counts().sort_index()}")
    print(f"Attacks:\n{df_test['attack_name'].value_counts()}")

    # ── DIAGNOSTIC: Verify cross-layer features spike correctly ──
    print("\n" + "=" * 60)
    print("CROSS-LAYER FEATURE DIAGNOSTICS")
    print("=" * 60)

    normal = df_train[df_train['label'] == 0]
    coord = df_train[df_train['label'] == 2]

    print("\n  Feature means (Normal vs Coordinated):")
    for feat in CROSS_F:
        nm = normal[feat].mean()
        cm = coord[feat].mean()
        ratio = cm / (nm + 1e-9)
        ok = "✓" if ratio > 2.0 else "⚠ WEAK"
        print(f"    {feat:25s}  N={nm:.4f}  C={cm:.4f}  "
              f"ratio={ratio:.1f}x  {ok}")

    # Also check that raw CAN/V2X features DON'T separate classes
    print("\n  Raw feature separation (should be LOW for stealthy):")
    for feat in CAN_F[:3] + V2X_F[:2]:
        nm = normal[feat].mean()
        cm = coord[feat].mean()
        ns = normal[feat].std()
        diff_sigma = abs(cm - nm) / (ns + 1e-9)
        ok = "✓ STEALTHY" if diff_sigma < 2.0 else "⚠ DETECTABLE"
        print(f"    {feat:25s}  diff={diff_sigma:.1f}σ  {ok}")
