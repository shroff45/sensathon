import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ── Configuration ──
st.set_page_config(
    page_title="Cross-Layer IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
laptop_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(laptop_dir, 'live_results.csv')
CMD_FILE = os.path.join(laptop_dir, 'commands.json')


def load_data():
    if os.path.exists(LOG_FILE):
        try:
            # Read CSV with headers from streamer
            return pd.read_csv(LOG_FILE)
        except:
            return pd.DataFrame()
    return pd.DataFrame()


def send_command(atk_name):
    with open(CMD_FILE, 'w') as f:
        json.dump({"attack": atk_name}, f)


# ══════════════════════════════════════════════════════════
# SIDEBAR — ATTACK CONTROLS
# ══════════════════════════════════════════════════════════

st.sidebar.title("🛡️ IDS Controls")
st.sidebar.markdown("---")

st.sidebar.subheader("Live Attack Injection")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Normal", type="primary"):
        send_command("normal")
    if st.button("GPS Spoof"):
        send_command("attack_gps_spoof")
    if st.button("CAN Inject"):
        send_command("attack_can_inject")

with col2:
    if st.button("Coord (CAN+V2X)"):
        send_command("attack_coord_can_v2x")
    if st.button("Coord (GPS+CAN)"):
        send_command("attack_coord_gps_can")
    if st.button("Coord (All 3)"):
        send_command("attack_coord_all_three")

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
st.sidebar.success("● ESP32: Connected")
st.sidebar.info("● Model: RF (20 trees)")
st.sidebar.warning("● Mode: Hardware Loop")

# ══════════════════════════════════════════════════════════
# DATA & HEADER
# ══════════════════════════════════════════════════════════

# Refresh every 500ms
st_autorefresh(interval=500, key="datarefresh")

df = load_data()

st.title("🛡️ Vehicle Cross-Layer IDS — Real-Time Monitoring")

if df.empty or len(df) < 1:
    st.warning("Waiting for data from stream_to_esp32.py...")
    st.stop()

# Get latest
last = df.iloc[-1]
status_color = ["#00cc66", "#ffaa00", "#ff3366"][int(last['esp'])]
status_text = ["SAFE", "SINGLE-LAYER ATTACK", "COORDINATED ATTACK"][int(last['esp'])]

# Top metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Speed", f"{last['speed']:.1f} m/s")
m2.metric("ESP32 Latency", f"{last['tI']:.0f} μs")
m3.metric("True Class", ["Normal", "Single", "Coord"][int(last['label'])])
m4.metric("Active Attack", last['gt'])

# ══════════════════════════════════════════════════════════
# MAIN STATUS PANEL
# ══════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div style="background-color:{status_color}; padding:20px; border-radius:10px; text-align:center; color:white;">
        <h1 style="color:white; margin:0; font-size:48px;">{status_text}</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ══════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════

p1, p2 = st.columns([2, 1])

with p1:
    st.subheader("Live Cross-Layer Consistency Indices")
    # CSV Header: x0, x1, x2, x3, x4, x5
    xl_feat_names = [
        'Speed Consistency', 'Yaw (Can-GPS)', 'Yaw (Can-IMU)',
        'Lat-Accel (GPS-IMU)', 'Obstacle Consistency', '3-Way Curvature'
    ]
    
    # Extract XL scores from the last row
    xl_scores = [last['x0'], last['x1'], last['x2'], last['x3'], last['x4'], last['x5']]
    
    fig = go.Figure(data=[
        go.Bar(name='Score', x=xl_feat_names, y=xl_scores,
               marker_color=['#1f77b4' if v < 0.3 else '#d62728' for v in xl_scores])
    ])
    fig.update_layout(yaxis_range=[0, 1.1], height=350, margin=dict(l=20, r=20, t=30, b=20))
    fig.add_hline(y=0.4, line_dash="dash", line_color="red", annotation_text="Danger Zone")
    st.plotly_chart(fig, use_container_width=True)

with p2:
    st.subheader("Model Consensus")
    # Comparison of ESP32 (Ours) vs Baselines
    # CSV Header: esp, bl_s, bl_c, bl_v, bl_nc
    models = ['Ours (Cross-Layer)', 'Sensor Only', 'CAN Only', 'V2X Only', 'All (No Cross)']
    preds = [last['esp'], last['bl_s'], last['bl_c'], last['bl_v'], last['bl_nc']]
    
    colors = ['#00cc66' if p == 0 else ('#ffaa00' if p == 1 else '#ff3366') for p in preds]
    labels = ['SAFE' if p == 0 else ('SINGLE' if p == 1 else 'COORD') for p in preds]

    fig2 = go.Figure(data=[
        go.Table(
            header=dict(values=['Model Layer', 'Prediction'],
                        fill_color='#444444',
                        font=dict(color='white', size=14),
                        align='left'),
            cells=dict(values=[models, labels],
                       fill_color=[['#f0f2f6']*5, colors],
                       font=dict(color=['black', 'white'], size=13),
                       align='left',
                       height=30)
        )
    ])
    fig2.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════
# FEATURE STREAM
# ══════════════════════════════════════════════════════════

st.subheader("Live Historical Classifications (Last 10)")
# Show the last 10 rows with color-coded results
df_hist = df.tail(10).copy()
df_hist = df_hist[['timestamp', 'gt', 'label', 'esp', 'bl_nc', 'tI']]
df_hist['Status'] = df_hist['esp'].map({0: "SAFE", 1: "SINGLE", 2: "COORD"})

def style_row(row):
    color = '#00cc6622' if row['esp'] == 0 else ('#ffaa0022' if row['esp'] == 1 else '#ff336622')
    return [f'background-color: {color}'] * len(row)

st.dataframe(df_hist.style.apply(style_row, axis=1), use_container_width=True)
