"""
generate_datasets.py
====================
Generates synthetic spacecraft telemetry and telecommand datasets
representing NORMAL operational behaviour only.
Run this script once before opening the notebook.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────
N_TELEMETRY    = 5000          # number of telemetry rows
N_TELECOMMAND  = 1000          # number of telecommand rows
START_TIME     = datetime(2025, 1, 1, 0, 0, 0)
SAMPLE_RATE_S  = 10            # one telemetry sample every 10 seconds

# ── Telemetry Parameters & their normal operating ranges ───────
TELEMETRY_PARAMS = {
    "BATT_VOLTAGE":      (27.0,  29.0),   # Battery voltage (V)
    "BATT_CURRENT":      (1.5,   3.5),    # Battery current (A)
    "SOLAR_POWER":       (80.0,  120.0),  # Solar panel power (W)
    "CPU_TEMP":          (35.0,  55.0),   # On-board CPU temperature (°C)
    "GYRO_X":            (-0.05, 0.05),   # Gyroscope X-axis (deg/s)
    "GYRO_Y":            (-0.05, 0.05),   # Gyroscope Y-axis (deg/s)
    "GYRO_Z":            (-0.05, 0.05),   # Gyroscope Z-axis (deg/s)
    "ATTITUDE_ROLL":     (-2.0,  2.0),    # Attitude roll angle (deg)
    "ATTITUDE_PITCH":    (-2.0,  2.0),    # Attitude pitch angle (deg)
    "ATTITUDE_YAW":      (-5.0,  5.0),    # Attitude yaw angle (deg)
    "RF_SIGNAL_STRENGTH":(-80.0,-50.0),  # RF link signal strength (dBm)
    "THRUSTER_TEMP":     (18.0,  28.0),   # Thruster temperature (°C)
    "MEMORY_USAGE":      (40.0,  70.0),   # On-board memory usage (%)
    "DATA_RATE":         (50.0,  150.0),  # Data downlink rate (kbps)
    "PAYLOAD_TEMP":      (20.0,  30.0),   # Payload instrument temperature (°C)
}

# ── Telecommand types ──────────────────────────────────────────
COMMANDS = [
    "CMD_SAFE_MODE",
    "CMD_ATTITUDE_ADJUST",
    "CMD_PAYLOAD_ON",
    "CMD_PAYLOAD_OFF",
    "CMD_DOWNLINK_START",
    "CMD_DOWNLINK_STOP",
    "CMD_ORBIT_CORRECTION",
    "CMD_BATTERY_CHARGE",
    "CMD_GYRO_CALIBRATE",
    "CMD_REBOOT_OBC",
]

# ── Generate Telemetry ─────────────────────────────────────────
print("Generating telemetry dataset...")
rows = []
for i in range(N_TELEMETRY):
    ts = START_TIME + timedelta(seconds=i * SAMPLE_RATE_S)
    param = np.random.choice(list(TELEMETRY_PARAMS.keys()))
    lo, hi = TELEMETRY_PARAMS[param]
    # Add mild sinusoidal drift + Gaussian noise for realism
    mid   = (lo + hi) / 2.0
    amp   = (hi - lo) / 4.0
    phase = 2 * np.pi * (i / N_TELEMETRY)
    val   = mid + amp * np.sin(phase) + np.random.normal(0, (hi - lo) * 0.05)
    val   = round(float(np.clip(val, lo, hi)), 4)
    rows.append({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                 "parameter": param,
                 "value": val})

telemetry_df = pd.DataFrame(rows)
telemetry_df.to_csv("telemetry_train.csv", index=False)
print(f"  [OK] telemetry_train.csv  -> {len(telemetry_df)} rows, {telemetry_df['parameter'].nunique()} parameters")

# ── Generate Telecommands ──────────────────────────────────────
print("Generating telecommand dataset...")
cmd_rows = []
for i in range(N_TELECOMMAND):
    ts  = START_TIME + timedelta(seconds=np.random.randint(0, N_TELEMETRY * SAMPLE_RATE_S))
    cmd = np.random.choice(COMMANDS, p=[0.05, 0.20, 0.12, 0.10,
                                        0.12, 0.10, 0.08, 0.08,
                                        0.08, 0.07])
    # Command value = execution flag (1=success, 0=pending)
    val = int(np.random.choice([0, 1], p=[0.05, 0.95]))
    cmd_rows.append({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                     "command": cmd,
                     "value": val})

telecommand_df = pd.DataFrame(cmd_rows).sort_values("timestamp").reset_index(drop=True)
telecommand_df.to_csv("telecommand_train.csv", index=False)
print(f"  [OK] telecommand_train.csv -> {len(telecommand_df)} rows, {telecommand_df['command'].nunique()} commands")
print("\nDataset generation complete. You may now open the notebook.")
