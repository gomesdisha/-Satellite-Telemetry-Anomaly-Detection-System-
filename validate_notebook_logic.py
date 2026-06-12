"""
validate_notebook_logic.py
===========================
Runs all the core notebook logic as a plain Python script to validate
correctness without the Jupyter kernel overhead.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ── Plot style ──────────────────────────────────────────────────
plt.rcParams.update({'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
                     'axes.labelcolor': '#c9d1d9', 'text.color': '#c9d1d9',
                     'xtick.color': '#8b949e', 'ytick.color': '#8b949e'})
PALETTE = ['#58a6ff','#3fb950','#f78166','#d2a8ff',
           '#ffa657','#79c0ff','#56d364','#ff7b72']
os.makedirs('plots', exist_ok=True)
os.makedirs('processed', exist_ok=True)

print("=" * 60)
print("NOTEBOOK VALIDATION SCRIPT")
print("=" * 60)

# ── SECTION 2 — Load ───────────────────────────────────────────
print("\n[S2] Loading datasets...")
tel = pd.read_csv('telemetry_train.csv')
cmd = pd.read_csv('telecommand_train.csv')
print(f"  Telemetry shape    : {tel.shape}")
print(f"  Telecommand shape  : {cmd.shape}")

# ── SECTION 3 — Quality ────────────────────────────────────────
print("\n[S3] Data quality checks...")
assert tel.isnull().sum().sum() == 0, "Missing values in telemetry!"
assert cmd.isnull().sum().sum() == 0, "Missing values in telecommand!"
assert tel.duplicated().sum() == 0, "Duplicates in telemetry!"
print("  All quality checks passed.")

# ── SECTION 5 — Timestamps ────────────────────────────────────
print("\n[S5] Timestamp processing...")
tel['timestamp'] = pd.to_datetime(tel['timestamp'])
cmd['timestamp'] = pd.to_datetime(cmd['timestamp'])
tel['hour']        = tel['timestamp'].dt.hour
tel['minute']      = tel['timestamp'].dt.minute
tel['day']         = tel['timestamp'].dt.day
tel['weekday']     = tel['timestamp'].dt.weekday
tel['is_weekend']  = (tel['weekday'] >= 5).astype(int)
tel['elapsed_sec'] = (tel['timestamp'] - tel['timestamp'].min()).dt.total_seconds()
tel['minute_of_day'] = tel['timestamp'].dt.hour * 60 + tel['timestamp'].dt.minute
print(f"  Temporal features added. Columns: {list(tel.columns)}")

# ── SECTION 4 — EDA plots ─────────────────────────────────────
print("\n[S4] Generating EDA plots...")

# 4.1 frequency
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
tel['parameter'].value_counts().plot(kind='barh', ax=axes[0], color=PALETTE[0])
axes[0].set_title('Parameter Frequency')
cmd['command'].value_counts().plot(kind='barh', ax=axes[1], color=PALETTE[1])
axes[1].set_title('Command Frequency')
plt.tight_layout()
plt.savefig('plots/01_frequency_distributions.png', dpi=120,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved plots/01_frequency_distributions.png")

# 4.2 histograms
params = sorted(tel['parameter'].unique())
ncols, nrows = 3, (len(params)+2)//3
fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows*4))
for i, p in enumerate(params):
    d = tel.loc[tel['parameter']==p,'value']
    ax = axes.flatten()[i]
    ax.hist(d, bins=25, color=PALETTE[i%len(PALETTE)], alpha=0.8)
    ax.axvline(d.mean(), color='#ffa657', linestyle='--', linewidth=1.2)
    ax.set_title(p, fontsize=8)
for j in range(i+1, len(axes.flatten())):
    axes.flatten()[j].set_visible(False)
plt.tight_layout()
plt.savefig('plots/02_value_distributions.png', dpi=120,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved plots/02_value_distributions.png")

# 4.3 boxplot
pivot_box = tel.pivot_table(
    index=tel.groupby('parameter').cumcount(),
    columns='parameter', values='value')
fig, ax = plt.subplots(figsize=(16, 6))
ax.boxplot([pivot_box[c].dropna().values for c in pivot_box.columns],
           labels=pivot_box.columns, patch_artist=True,
           medianprops=dict(color='#ffa657', linewidth=2))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/03_boxplots.png', dpi=120,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved plots/03_boxplots.png")

# 4.4 time series
tel_s = tel.sort_values(['parameter','timestamp'])
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
for ax, param, col in zip(axes.flatten(),
                           ['BATT_VOLTAGE','CPU_TEMP','GYRO_X',
                            'SOLAR_POWER','RF_SIGNAL_STRENGTH','ATTITUDE_ROLL'],
                           PALETTE):
    sub = tel_s[tel_s['parameter']==param]
    ax.plot(sub['timestamp'], sub['value'], color=col, linewidth=0.8)
    ax.set_title(param, fontsize=9)
plt.tight_layout()
plt.savefig('plots/04_time_series.png', dpi=120,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved plots/04_time_series.png")

# ── SECTION 6 — Feature Engineering ──────────────────────────
print("\n[S6] Feature engineering...")
tel_fe = tel.sort_values(['parameter','timestamp']).copy()
frames = []
for param, g in tel_fe.groupby('parameter'):
    g = g.copy()
    g['rolling_mean_5']      = g['value'].rolling(5, min_periods=1).mean()
    g['rolling_std_5']       = g['value'].rolling(5, min_periods=1).std().fillna(0)
    g['rolling_mean_10']     = g['value'].rolling(10, min_periods=1).mean()
    g['deviation_from_mean'] = g['value'] - g['rolling_mean_5']
    g['change_rate']         = g['value'].diff().fillna(0)
    g['abs_change_rate']     = g['change_rate'].abs()
    g['lag_1']               = g['value'].shift(1).bfill()
    g['lag_2']               = g['value'].shift(2).bfill()
    g['lag_3']               = g['value'].shift(3).bfill()
    std_s                    = g['rolling_std_5'].replace(0, np.nan)
    g['z_score']             = ((g['value'] - g['rolling_mean_5']) / std_s).fillna(0)
    frames.append(g)
tel_fe = pd.concat(frames).sort_values('timestamp').reset_index(drop=True)
print(f"  Engineered feature matrix shape: {tel_fe.shape}")
print(f"  Columns: {list(tel_fe.columns)}")

# Feature plot
batt = tel_fe[tel_fe['parameter']=='BATT_VOLTAGE'].sort_values('timestamp')
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
axes[0].plot(batt['timestamp'], batt['value'], color='#58a6ff', linewidth=0.8, label='Raw')
axes[0].plot(batt['timestamp'], batt['rolling_mean_5'], color='#ffa657', linewidth=1.5, label='RM5')
axes[0].legend(); axes[0].set_title('Raw + Rolling Means')
axes[1].plot(batt['timestamp'], batt['rolling_std_5'], color='#d2a8ff')
axes[1].set_title('Rolling Std')
axes[2].bar(batt['timestamp'], batt['change_rate'],
            color=['#3fb950' if v>=0 else '#f78166' for v in batt['change_rate']])
axes[2].set_title('Change Rate')
axes[3].plot(batt['timestamp'], batt['z_score'], color='#79c0ff', linewidth=0.8)
axes[3].axhline(2, color='#f78166', linestyle='--', linewidth=1)
axes[3].axhline(-2, color='#f78166', linestyle='--', linewidth=1)
axes[3].set_title('Z-Score')
plt.tight_layout()
plt.savefig('plots/06_engineered_features.png', dpi=120,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved plots/06_engineered_features.png")

# ── SECTION 7 — Correlation & Variance ───────────────────────
print("\n[S7] Feature selection analysis...")
numeric_cols = ['value','hour','minute','day','weekday','elapsed_sec',
                'minute_of_day','rolling_mean_5','rolling_std_5',
                'rolling_mean_10','deviation_from_mean',
                'change_rate','abs_change_rate','lag_1','lag_2','lag_3','z_score']
batt_num = tel_fe[tel_fe['parameter']=='BATT_VOLTAGE'][numeric_cols].dropna()
corr = batt_num.corr()
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt='.1f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5, annot_kws={'size': 6})
ax.set_title('Correlation Matrix — BATT_VOLTAGE')
plt.tight_layout()
plt.savefig('plots/07_correlation_matrix.png', dpi=120,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved plots/07_correlation_matrix.png")

var_df = batt_num.var().sort_values(ascending=False).reset_index()
var_df.columns = ['Feature','Variance']
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(var_df['Feature'], var_df['Variance'], color=PALETTE*3)
ax.set_xscale('log'); ax.set_title('Feature Variance')
plt.tight_layout()
plt.savefig('plots/08_variance_analysis.png', dpi=120,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved plots/08_variance_analysis.png")

# ── SECTION 8 — Pivot ─────────────────────────────────────────
print("\n[S8] Long -> Wide format transformation...")
tel_wide = tel.pivot_table(index='timestamp', columns='parameter',
                            values='value', aggfunc='mean').reset_index()
tel_wide.columns.name = None
tel_wide = tel_wide.sort_values('timestamp').reset_index(drop=True)
param_cols = [c for c in tel_wide.columns if c != 'timestamp']
tel_wide[param_cols] = (tel_wide[param_cols].ffill().bfill())
print(f"  Wide format shape: {tel_wide.shape}  (NaN after fill: {tel_wide[param_cols].isnull().sum().sum()})")

# ── SECTION 9 — Scaling ───────────────────────────────────────
print("\n[S9] Applying scalers...")
X_raw = tel_wide[param_cols].values
std_scaler = StandardScaler()
X_std  = std_scaler.fit_transform(X_raw)
mm_scaler  = MinMaxScaler()
X_mm   = mm_scaler.fit_transform(X_raw)

df_std = pd.DataFrame(X_std, columns=param_cols)
df_mm  = pd.DataFrame(X_mm,  columns=param_cols)
print(f"  StandardScaler — mean range : [{df_std.mean().min():.3f}, {df_std.mean().max():.3f}]")
print(f"  StandardScaler — std range  : [{df_std.std().min():.3f}, {df_std.std().max():.3f}]")
print(f"  MinMaxScaler   — min range  : [{df_mm.min().min():.3f}, {df_mm.min().max():.3f}]")
print(f"  MinMaxScaler   — max range  : [{df_mm.max().min():.3f}, {df_mm.max().max():.3f}]")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sel = ['BATT_VOLTAGE','SOLAR_POWER','CPU_TEMP','RF_SIGNAL_STRENGTH','GYRO_X']
for ax, data, title in zip(axes,
                            [tel_wide[sel], df_std[sel], df_mm[sel]],
                            ['Raw','StandardScaler','MinMaxScaler']):
    for col, c in zip(sel, PALETTE):
        ax.hist(data[col], bins=25, alpha=0.5, label=col, color=c)
    ax.set_title(title); ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig('plots/09_scaling_comparison.png', dpi=120,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved plots/09_scaling_comparison.png")

# ── SECTION 11 — Save processed files ─────────────────────────
print("\n[S11] Saving processed datasets...")
tel_fe.to_csv('processed/telemetry_engineered.csv', index=False)
tel_wide.to_csv('processed/telemetry_wide.csv', index=False)
df_std_out = pd.DataFrame(X_std, columns=param_cols)
df_std_out.insert(0, 'timestamp', tel_wide['timestamp'].values)
df_std_out.to_csv('processed/telemetry_wide_standard_scaled.csv', index=False)

df_mm_out = pd.DataFrame(X_mm, columns=param_cols)
df_mm_out.insert(0, 'timestamp', tel_wide['timestamp'].values)
df_mm_out.to_csv('processed/telemetry_wide_minmax_scaled.csv', index=False)
cmd['timestamp'] = pd.to_datetime(cmd['timestamp'])
cmd.to_csv('processed/telecommand_processed.csv', index=False)

print("\nFiles saved:")
for f in sorted(os.listdir('processed')):
    fpath = os.path.join('processed', f)
    print(f"  {f:<50} {os.path.getsize(fpath):>8,} bytes")

print("\nPlots generated:")
for f in sorted(os.listdir('plots')):
    fpath = os.path.join('plots', f)
    print(f"  {f:<50} {os.path.getsize(fpath):>8,} bytes")

print("\n" + "="*60)
print("ALL VALIDATION CHECKS PASSED. Notebook is ready.")
print("="*60)
