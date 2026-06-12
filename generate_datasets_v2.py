"""
generate_datasets_v2.py
=======================
Generates v2 synthetic spacecraft telemetry & telecommand datasets.
  - 50 telemetry parameters  (10 000 rows)
  - 175 telecommand types    (175 rows, one per command)
Run once before opening the v2 notebook.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

START_TIME     = datetime(2025, 1, 1, 0, 0, 0)
N_TELEMETRY    = 10_000
SAMPLE_RATE_S  = 10

# ── 50 Telemetry Parameters ────────────────────────────────────
TELEMETRY_PARAMS = {
    # Power Subsystem
    "BATT_VOLTAGE_1":        (26.5,  29.5),
    "BATT_VOLTAGE_2":        (26.5,  29.5),
    "BATT_CURRENT_1":        (1.0,   4.0),
    "BATT_CURRENT_2":        (1.0,   4.0),
    "BATT_TEMP_1":           (15.0,  35.0),
    "BATT_TEMP_2":           (15.0,  35.0),
    "BATT_SOC":              (60.0,  100.0),
    "SOLAR_PANEL_VOLT_A":    (28.0,  34.0),
    "SOLAR_PANEL_VOLT_B":    (28.0,  34.0),
    "SOLAR_PANEL_CURR_A":    (3.0,   8.0),
    "SOLAR_PANEL_CURR_B":    (3.0,   8.0),
    "SOLAR_POWER_TOTAL":     (80.0,  130.0),
    "BUS_VOLTAGE":           (27.5,  29.5),
    "BUS_CURRENT":           (5.0,   12.0),
    # Thermal Subsystem
    "THRUSTER_TEMP_1":       (15.0,  30.0),
    "THRUSTER_TEMP_2":       (15.0,  30.0),
    "PAYLOAD_TEMP_CAM":      (18.0,  28.0),
    "PAYLOAD_TEMP_SENSOR":   (18.0,  28.0),
    "OBC_TEMP":              (30.0,  55.0),
    "TRANS_TEMP":            (25.0,  50.0),
    "STRUCT_TEMP_PANEL_X":   (-10.0, 50.0),
    "STRUCT_TEMP_PANEL_Y":   (-10.0, 50.0),
    "STRUCT_TEMP_PANEL_Z":   (-10.0, 50.0),
    "RADIATOR_TEMP":         (-30.0, 10.0),
    # ADCS (Attitude Determination & Control)
    "GYRO_X":                (-0.08, 0.08),
    "GYRO_Y":                (-0.08, 0.08),
    "GYRO_Z":                (-0.08, 0.08),
    "MAGNETOMETER_X":        (-60.0, 60.0),
    "MAGNETOMETER_Y":        (-60.0, 60.0),
    "MAGNETOMETER_Z":        (-60.0, 60.0),
    "REACTION_WHEEL_SPD_X":  (-3000, 3000),
    "REACTION_WHEEL_SPD_Y":  (-3000, 3000),
    "REACTION_WHEEL_SPD_Z":  (-3000, 3000),
    "ATTITUDE_ROLL":         (-3.0,  3.0),
    "ATTITUDE_PITCH":        (-3.0,  3.0),
    "ATTITUDE_YAW":          (-5.0,  5.0),
    "STAR_TRACKER_VALID":    (0.0,   1.0),
    "SUN_SENSOR_ANGLE":      (0.0,   180.0),
    # Communication Subsystem
    "RF_SIGNAL_STRENGTH":    (-85.0, -45.0),
    "RF_NOISE_FLOOR":        (-110.0,-80.0),
    "TX_POWER":              (1.0,   5.0),
    "LINK_MARGIN":           (5.0,   25.0),
    "DATA_RATE_KBPS":        (50.0,  200.0),
    "PACKET_LOSS_RATE":      (0.0,   2.0),
    # OBC & Memory
    "CPU_USAGE":             (10.0,  60.0),
    "MEMORY_USAGE":          (30.0,  75.0),
    "FLASH_WRITE_RATE":      (0.0,   5.0),
    "WATCHDOG_COUNTER":      (0.0,   100.0),
    # Propulsion
    "TANK_PRESSURE":         (280.0, 320.0),
    "THRUSTER_VALVE_TEMP":   (10.0,  25.0),
}

# ── 175 Telecommand Types ──────────────────────────────────────
COMMANDS = [
    # Power commands
    "CMD_BATT_CHARGE_ON","CMD_BATT_CHARGE_OFF","CMD_BATT_DISCHARGE_MODE",
    "CMD_SOLAR_PANEL_DEPLOY","CMD_SOLAR_PANEL_RETRACT","CMD_SOLAR_TRACK_ON",
    "CMD_SOLAR_TRACK_OFF","CMD_BUS_RESET","CMD_POWER_MODE_ECO","CMD_POWER_MODE_NORMAL",
    "CMD_POWER_MODE_HIGH","CMD_EPS_RESET","CMD_LOAD_SHED_ON","CMD_LOAD_SHED_OFF",
    "CMD_BATT_HEATER_ON","CMD_BATT_HEATER_OFF",
    # Thermal commands
    "CMD_HEATER_THRUSTER1_ON","CMD_HEATER_THRUSTER1_OFF",
    "CMD_HEATER_THRUSTER2_ON","CMD_HEATER_THRUSTER2_OFF",
    "CMD_HEATER_PAYLOAD_ON","CMD_HEATER_PAYLOAD_OFF",
    "CMD_HEATER_OBC_ON","CMD_HEATER_OBC_OFF",
    "CMD_RADIATOR_LOUVRE_OPEN","CMD_RADIATOR_LOUVRE_CLOSE",
    "CMD_THERMAL_CTRL_AUTO","CMD_THERMAL_CTRL_MANUAL",
    # ADCS commands
    "CMD_ATTITUDE_HOLD","CMD_ATTITUDE_ADJUST","CMD_NADIR_POINT","CMD_SUN_POINT",
    "CMD_SAFE_ATTITUDE","CMD_GYRO_CALIBRATE","CMD_GYRO_RESET","CMD_GYRO_BIAS_UPDATE",
    "CMD_MAGNETORQUER_X_ON","CMD_MAGNETORQUER_X_OFF",
    "CMD_MAGNETORQUER_Y_ON","CMD_MAGNETORQUER_Y_OFF",
    "CMD_MAGNETORQUER_Z_ON","CMD_MAGNETORQUER_Z_OFF",
    "CMD_RW_X_DESATURATE","CMD_RW_Y_DESATURATE","CMD_RW_Z_DESATURATE",
    "CMD_RW_X_SPEED_SET","CMD_RW_Y_SPEED_SET","CMD_RW_Z_SPEED_SET",
    "CMD_STAR_TRACKER_ON","CMD_STAR_TRACKER_OFF","CMD_STAR_TRACKER_RESET",
    "CMD_SUN_SENSOR_ON","CMD_SUN_SENSOR_OFF",
    "CMD_ADCS_RESET","CMD_ADCS_MODE_SAFE","CMD_ADCS_MODE_NOMINAL",
    "CMD_EULER_ANGLE_SET","CMD_QUATERNION_SET",
    # Communication commands
    "CMD_RF_TX_ON","CMD_RF_TX_OFF","CMD_RF_RX_ON","CMD_RF_RX_OFF",
    "CMD_ANTENNA_DEPLOY","CMD_ANTENNA_RETRACT","CMD_ANTENNA_POINT",
    "CMD_DOWNLINK_START","CMD_DOWNLINK_STOP","CMD_DOWNLINK_SCHEDULE",
    "CMD_UPLINK_ENABLE","CMD_UPLINK_DISABLE",
    "CMD_COMM_FREQ_CHANGE","CMD_TX_POWER_HIGH","CMD_TX_POWER_LOW",
    "CMD_MODULATION_BPSK","CMD_MODULATION_QPSK",
    "CMD_PACKET_STORE_FLUSH","CMD_COMM_RESET",
    # OBC / Software commands
    "CMD_REBOOT_OBC","CMD_SOFT_RESET","CMD_HARD_RESET","CMD_SAFE_MODE_ENTER",
    "CMD_SAFE_MODE_EXIT","CMD_NOMINAL_MODE_ENTER","CMD_EMERGENCY_MODE",
    "CMD_SOFTWARE_UPDATE","CMD_PARAMETER_UPDATE","CMD_CONFIG_UPLOAD",
    "CMD_CONFIG_DOWNLOAD","CMD_LOG_DOWNLOAD","CMD_LOG_CLEAR",
    "CMD_MEMORY_DUMP","CMD_MEMORY_CLEAR","CMD_MEMORY_CHECK",
    "CMD_TIME_SYNC","CMD_EPOCH_SET","CMD_WATCHDOG_RESET","CMD_WATCHDOG_DISABLE",
    "CMD_FAULT_CLEAR","CMD_FAULT_LOG_DOWNLOAD","CMD_HEALTH_CHECK",
    "CMD_DIAG_RUN","CMD_CPU_THROTTLE_50","CMD_CPU_THROTTLE_100",
    "CMD_FLASH_ERASE","CMD_FIRMWARE_VERIFY",
    # Payload commands
    "CMD_PAYLOAD_ON","CMD_PAYLOAD_OFF","CMD_PAYLOAD_RESET","CMD_PAYLOAD_STANDBY",
    "CMD_CAMERA_ON","CMD_CAMERA_OFF","CMD_CAMERA_CAPTURE","CMD_CAMERA_EXPOSURE_SET",
    "CMD_CAMERA_GAIN_SET","CMD_CAMERA_MODE_STILL","CMD_CAMERA_MODE_VIDEO",
    "CMD_PAYLOAD_CAL","CMD_PAYLOAD_SELFTEST","CMD_PAYLOAD_DATA_DOWNLOAD",
    "CMD_PAYLOAD_CONFIG_SET","CMD_IMAGING_START","CMD_IMAGING_STOP",
    "CMD_PAYLOAD_HEATER_ON","CMD_PAYLOAD_HEATER_OFF",
    # Propulsion commands
    "CMD_THRUSTER1_FIRE","CMD_THRUSTER2_FIRE","CMD_THRUSTER_ALL_FIRE",
    "CMD_THRUSTER1_OFF","CMD_THRUSTER2_OFF","CMD_THRUSTER_SAFE",
    "CMD_ORBIT_RAISE","CMD_ORBIT_LOWER","CMD_ORBIT_CIRCULARISE",
    "CMD_DEORBIT_BURN","CMD_STATION_KEEP","CMD_TANK_VENT",
    "CMD_VALVE1_OPEN","CMD_VALVE1_CLOSE","CMD_VALVE2_OPEN","CMD_VALVE2_CLOSE",
    "CMD_PROP_RESET","CMD_PROP_PRIME",
    # Mission / Operational
    "CMD_MISSION_START","CMD_MISSION_PAUSE","CMD_MISSION_RESUME","CMD_MISSION_ABORT",
    "CMD_SCHEDULE_UPLOAD","CMD_SCHEDULE_CLEAR","CMD_SCHEDULE_EXECUTE",
    "CMD_EVENT_LOG_CLEAR","CMD_TELEMETRY_RATE_HIGH","CMD_TELEMETRY_RATE_LOW",
    "CMD_TELEMETRY_RATE_NOMINAL","CMD_GROUND_PASS_PREP","CMD_ECLIPSE_MODE_ENTER",
    "CMD_ECLIPSE_MODE_EXIT","CMD_MANOEUVRE_START","CMD_MANOEUVRE_ABORT",
    "CMD_CONTINGENCY_1","CMD_CONTINGENCY_2","CMD_CONTINGENCY_3",
]

assert len(COMMANDS) >= 150, f"Only {len(COMMANDS)} commands defined"

# ── Generate Telemetry ─────────────────────────────────────────
print("Generating telemetry dataset (10 000 rows, 50 parameters)...")
rows = []
params_list = list(TELEMETRY_PARAMS.keys())
for i in range(N_TELEMETRY):
    ts    = START_TIME + timedelta(seconds=i * SAMPLE_RATE_S)
    param = params_list[i % len(params_list)]   # round-robin for uniform coverage
    lo, hi = TELEMETRY_PARAMS[param]
    mid    = (lo + hi) / 2.0
    amp    = (hi - lo) / 4.0
    phase  = 2 * np.pi * (i / N_TELEMETRY)
    val    = mid + amp * np.sin(phase + np.random.uniform(0, np.pi)) \
             + np.random.normal(0, (hi - lo) * 0.04)
    val    = round(float(np.clip(val, lo, hi)), 4)
    rows.append({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                 "parameter": param,
                 "value":     val})

tel_df = pd.DataFrame(rows)
tel_df.to_csv("telemetry_train.csv", index=False)
print(f"  telemetry_train.csv -> {len(tel_df):,} rows, "
      f"{tel_df['parameter'].nunique()} parameters")

# ── Generate Telecommands ──────────────────────────────────────
print("Generating telecommand dataset (175 commands)...")
cmd_rows = []
for i, cmd in enumerate(COMMANDS):
    ts  = START_TIME + timedelta(seconds=np.random.randint(0, N_TELEMETRY * SAMPLE_RATE_S))
    val = int(np.random.choice([0, 1], p=[0.04, 0.96]))
    cmd_rows.append({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                     "command":   cmd,
                     "value":     val})

cmd_df = pd.DataFrame(cmd_rows).sort_values("timestamp").reset_index(drop=True)
cmd_df.to_csv("telecommand_train.csv", index=False)
print(f"  telecommand_train.csv -> {len(cmd_df):,} rows, "
      f"{cmd_df['command'].nunique()} commands")

print("\nDone. Run the v2 notebook next.")
