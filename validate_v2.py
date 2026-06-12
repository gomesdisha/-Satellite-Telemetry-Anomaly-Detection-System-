"""
validate_v2.py  —  Headless validation of all notebook logic (light theme)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'#f8f9fa'})
PALETTE = ['#1f77b4','#2ca02c','#d62728','#9467bd','#8c564b',
           '#e377c2','#7f7f7f','#bcbd22','#17becf','#ff7f0e']

os.makedirs('plots_v2', exist_ok=True)
os.makedirs('processed_v2', exist_ok=True)

print("="*60)
print("VALIDATION — v2 Notebook Logic")
print("="*60)

# ── S2 Load ───────────────────────────────────────────────────
print("\n[S2] Loading...")
tel = pd.read_csv('telemetry_train.csv')
cmd = pd.read_csv('telecommand_train.csv')
assert tel.shape == (10000, 3), f"Unexpected shape: {tel.shape}"
print(f"  Telemetry  : {tel.shape}  | parameters: {tel['parameter'].nunique()}")
print(f"  Telecommand: {cmd.shape} | commands  : {cmd['command'].nunique()}")

# ── S3 Quality ────────────────────────────────────────────────
print("\n[S3] Quality checks...")
assert tel.isnull().sum().sum() == 0
assert cmd.isnull().sum().sum() == 0
assert tel.duplicated().sum() == 0
print("  All checks passed.")

# ── S5 Timestamps ─────────────────────────────────────────────
print("\n[S5] Timestamps...")
tel['timestamp'] = pd.to_datetime(tel['timestamp'])
cmd['timestamp'] = pd.to_datetime(cmd['timestamp'])
for df in [tel, cmd]:
    ts = df['timestamp']
    df['hour']          = ts.dt.hour
    df['minute']        = ts.dt.minute
    df['second']        = ts.dt.second
    df['day']           = ts.dt.day
    df['month']         = ts.dt.month
    df['weekday']       = ts.dt.weekday
    df['is_weekend']    = (ts.dt.weekday >= 5).astype(int)
    df['minute_of_day'] = ts.dt.hour * 60 + ts.dt.minute
    df['elapsed_sec']   = (ts - ts.min()).dt.total_seconds()
print(f"  Columns: {list(tel.columns)}")

# ── S4 EDA Plots ──────────────────────────────────────────────
print("\n[S4] EDA plots...")
params = sorted(tel['parameter'].unique())

# 4.1 frequency
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
tel['parameter'].value_counts().sort_values().plot(
    kind='barh', ax=axes[0], color='#1f77b4')
axes[0].set_title('Parameter Frequency')
cmd['command'].value_counts().sort_values().plot(
    kind='barh', ax=axes[1], color='#2ca02c')
axes[1].set_title('Command Frequency')
plt.tight_layout()
plt.savefig('plots_v2/01_frequency.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved 01_frequency.png")

# 4.2 distributions
ncols, nrows = 5, (len(params)+4)//5
fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows*3.5))
for idx, p in enumerate(params):
    ax = axes.flatten()[idx]
    d  = tel.loc[tel['parameter']==p,'value']
    ax.hist(d, bins=25, color=PALETTE[idx%len(PALETTE)], alpha=0.8, edgecolor='white')
    ax.axvline(d.mean(), color='red', lw=1.2, ls='--')
    ax.set_title(p, fontsize=7)
for j in range(idx+1, len(axes.flatten())):
    axes.flatten()[j].set_visible(False)
plt.tight_layout()
plt.savefig('plots_v2/02_distributions.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved 02_distributions.png")

# 4.3 boxplots by subsystem
SUBSYSTEMS = {
    'Power':      [p for p in params if any(k in p for k in ['BATT','SOLAR','BUS'])],
    'Thermal':    [p for p in params if 'TEMP' in p or 'RADIATOR' in p],
    'ADCS':       [p for p in params if any(k in p for k in
                   ['GYRO','MAG','REACTION','ATTITUDE','STAR','SUN_SENSOR'])],
    'Comms':      [p for p in params if any(k in p for k in
                   ['RF','TX','LINK','DATA_RATE','PACKET'])],
    'OBC':        [p for p in params if any(k in p for k in
                   ['CPU','MEMORY','FLASH','WATCHDOG'])],
    'Propulsion': [p for p in params if any(k in p for k in
                   ['TANK','THRUSTER_VALVE','THRUSTER_TEMP'])],
}
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
for ax, (subsys, ps), col in zip(axes.flatten(), SUBSYSTEMS.items(), PALETTE):
    valid_ps  = [p for p in ps if (tel['parameter']==p).any()]
    data_list = [tel.loc[tel['parameter']==p,'value'].values for p in valid_ps]
    bp = ax.boxplot(data_list, labels=valid_ps, patch_artist=True,
                    medianprops=dict(color='red', lw=1.5))
    for patch in bp['boxes']:
        patch.set_facecolor(col); patch.set_alpha(0.5)
    ax.set_title(subsys)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
plt.tight_layout()
plt.savefig('plots_v2/03_boxplots.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved 03_boxplots.png")

# 4.4 time-series
TS_PARAMS = ['BATT_VOLTAGE_1','SOLAR_POWER_TOTAL','OBC_TEMP','GYRO_X',
             'RF_SIGNAL_STRENGTH','ATTITUDE_ROLL','TANK_PRESSURE','MEMORY_USAGE']
fig, axes = plt.subplots(4, 2, figsize=(16, 12))
for ax, p, col in zip(axes.flatten(), TS_PARAMS, PALETTE):
    sub = tel[tel['parameter']==p].sort_values('timestamp')
    ax.plot(sub['timestamp'], sub['value'], color=col, lw=0.8, alpha=0.75)
    ax.plot(sub['timestamp'], sub['value'].rolling(10, center=True).mean(),
            color='red', lw=1.5, ls='--')
    ax.set_title(p, fontsize=9)
plt.tight_layout()
plt.savefig('plots_v2/04_timeseries.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved 04_timeseries.png")

# 5 temporal
feats = ['hour','minute','day','weekday','minute_of_day']
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, f, col in zip(axes, feats, PALETTE):
    ax.hist(tel[f], bins=24, color=col, alpha=0.8, edgecolor='white')
    ax.set_title(f)
plt.tight_layout()
plt.savefig('plots_v2/05_temporal.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved 05_temporal.png")

# ── S6 Feature Engineering ────────────────────────────────────
print("\n[S6] Feature engineering...")
tel_fe = tel.sort_values(['parameter','timestamp']).copy()
frames = []
for param, g in tel_fe.groupby('parameter'):
    g = g.copy()
    g['rolling_mean_5']      = g['value'].rolling(5,  min_periods=1).mean()
    g['rolling_mean_10']     = g['value'].rolling(10, min_periods=1).mean()
    g['rolling_std_5']       = g['value'].rolling(5,  min_periods=1).std().fillna(0)
    g['deviation_from_mean'] = g['value'] - g['rolling_mean_5']
    g['change_rate']         = g['value'].diff().fillna(0)
    g['abs_change_rate']     = g['change_rate'].abs()
    g['lag_1'] = g['value'].shift(1).bfill()
    g['lag_2'] = g['value'].shift(2).bfill()
    g['lag_3'] = g['value'].shift(3).bfill()
    std_safe   = g['rolling_std_5'].replace(0, np.nan)
    g['z_score'] = ((g['value'] - g['rolling_mean_5']) / std_safe).fillna(0)
    frames.append(g)
tel_fe = pd.concat(frames).sort_values('timestamp').reset_index(drop=True)
print(f"  Shape: {tel_fe.shape}  | Columns: {list(tel_fe.columns)}")

# Feature plot
batt = tel_fe[tel_fe['parameter']=='BATT_VOLTAGE_1'].sort_values('timestamp')
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
axes[0].plot(batt['timestamp'], batt['value'], color='#1f77b4', lw=0.9)
axes[0].plot(batt['timestamp'], batt['rolling_mean_5'], color='red', lw=1.5, ls='--')
axes[0].set_title('Raw + Rolling Means')
axes[1].fill_between(batt['timestamp'], batt['rolling_std_5'], alpha=0.4, color='#9467bd')
axes[1].set_title('Rolling Std (5)')
colors = ['#2ca02c' if v>=0 else '#d62728' for v in batt['change_rate']]
axes[2].bar(batt['timestamp'], batt['change_rate'], color=colors, alpha=0.7, width=0.007)
axes[2].set_title('Change Rate')
axes[3].plot(batt['timestamp'], batt['z_score'], color='#1f77b4', lw=0.8)
axes[3].axhline(2, color='red', ls='--', lw=1.2)
axes[3].axhline(-2, color='red', ls='--', lw=1.2)
axes[3].set_title('Z-Score')
plt.tight_layout()
plt.savefig('plots_v2/06_features.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved 06_features.png")

# Z-score dist
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(tel_fe['z_score'].clip(-5,5), bins=80, color='#1f77b4', alpha=0.8, edgecolor='white')
ax.axvline(2, color='red', ls='--', lw=1.5)
ax.axvline(-2, color='red', ls='--', lw=1.5)
ax.set_title('Z-Score Distribution — All Parameters')
plt.tight_layout()
plt.savefig('plots_v2/06b_zscore.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved 06b_zscore.png")

# Save engineered
tel_fe.to_csv('processed_v2/telemetry_engineered.csv', index=False)
print("  Saved processed_v2/telemetry_engineered.csv")

# ── S7 Correlation ────────────────────────────────────────────
print("\n[S7] Feature selection...")
NUMERIC_COLS = ['value','hour','minute','weekday','elapsed_sec','minute_of_day',
                'rolling_mean_5','rolling_std_5','rolling_mean_10',
                'deviation_from_mean','change_rate','abs_change_rate',
                'lag_1','lag_2','lag_3','z_score']
sample = tel_fe[tel_fe['parameter']=='OBC_TEMP'][NUMERIC_COLS].dropna()
corr   = sample.corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.4,
            annot_kws={'size':7}, cbar_kws={'shrink':0.8})
ax.set_title('Correlation Matrix')
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig('plots_v2/07a_correlation.png', dpi=120, bbox_inches='tight')
plt.close()

var_df = sample.var().sort_values(ascending=False).reset_index()
var_df.columns = ['Feature','Variance']
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(var_df['Feature'], var_df['Variance'],
        color=[PALETTE[i%len(PALETTE)] for i in range(len(var_df))])
ax.set_xscale('log')
ax.set_title('Feature Variance (log scale)')
plt.tight_layout()
plt.savefig('plots_v2/07b_variance.png', dpi=120, bbox_inches='tight')
plt.close()

# Cross-param correlation
wide_tmp = tel_fe.pivot_table(index='timestamp', columns='parameter',
                               values='value', aggfunc='mean')
wide_tmp.columns.name = None
wide_tmp = wide_tmp.ffill().bfill()
cross = wide_tmp.corr()
fig, ax = plt.subplots(figsize=(18, 14))
sns.heatmap(cross, annot=False, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=ax, linewidths=0.2, cbar_kws={'shrink':0.6})
ax.set_title('Cross-Parameter Correlation — 50 Parameters')
plt.xticks(rotation=90, fontsize=6)
plt.yticks(fontsize=6)
plt.tight_layout()
plt.savefig('plots_v2/07c_cross_corr.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Correlation & variance plots saved.")

# ── S8 Pivot ──────────────────────────────────────────────────
print("\n[S8] Dataset transformation...")
tel_wide = tel.pivot_table(index='timestamp', columns='parameter',
                            values='value', aggfunc='mean').reset_index()
tel_wide.columns.name = None
tel_wide = tel_wide.sort_values('timestamp').reset_index(drop=True)
param_cols = [c for c in tel_wide.columns if c != 'timestamp']
print(f"  Before fill — NaN: {tel_wide[param_cols].isnull().sum().sum()}")
tel_wide[param_cols] = tel_wide[param_cols].ffill().bfill()
print(f"  After fill  — NaN: {tel_wide[param_cols].isnull().sum().sum()}")
print(f"  Wide shape: {tel_wide.shape}")

# NaN pattern heatmap
tel_sparse = tel.pivot_table(index='timestamp', columns='parameter',
                              values='value', aggfunc='mean').reset_index()
tel_sparse.columns.name = None
sparse_cols = [c for c in tel_sparse.columns if c != 'timestamp']
null_mat = tel_sparse[sparse_cols].iloc[:60].isnull().astype(int)
fig, ax = plt.subplots(figsize=(18, 7))
sns.heatmap(null_mat.T, cmap=['#f8f9fa','#d62728'], cbar=False,
            ax=ax, linewidths=0.2)
ax.set_title('NaN Pattern — First 60 Timestamps Before Fill')
plt.tight_layout()
plt.savefig('plots_v2/08_nan_pattern.png', dpi=120, bbox_inches='tight')
plt.close()

tel_wide.to_csv('processed_v2/telemetry_wide.csv', index=False)
print("  Saved processed_v2/telemetry_wide.csv")

# ── S9 Scaling ────────────────────────────────────────────────
print("\n[S9] Scaling...")
X_raw = tel_wide[param_cols].values

std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X_raw)
df_std = pd.DataFrame(X_std, columns=param_cols)
print(f"  StandardScaler — mean: [{df_std.mean().min():.3f},{df_std.mean().max():.3f}]"
      f" | std: [{df_std.std().min():.3f},{df_std.std().max():.3f}]")

mm_scaler = MinMaxScaler()
X_mm = mm_scaler.fit_transform(X_raw)
df_mm = pd.DataFrame(X_mm, columns=param_cols)
print(f"  MinMaxScaler  — min: [{df_mm.min().min():.3f},{df_mm.min().max():.3f}]"
      f" | max: [{df_mm.max().min():.3f},{df_mm.max().max():.3f}]")

SEL = ['BATT_VOLTAGE_1','SOLAR_POWER_TOTAL','OBC_TEMP','RF_SIGNAL_STRENGTH','GYRO_X']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, data, title in zip(axes,
                            [tel_wide[SEL], df_std[SEL], df_mm[SEL]],
                            ['Raw','StandardScaler','MinMaxScaler']):
    for col, c in zip(SEL, PALETTE):
        ax.hist(data[col], bins=25, alpha=0.55, label=col.replace('_',' '), color=c)
    ax.set_title(title); ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig('plots_v2/09_scaling.png', dpi=120, bbox_inches='tight')
plt.close()

df_std_out = df_std.copy()
df_std_out.insert(0, 'timestamp', tel_wide['timestamp'].values)
df_std_out.to_csv('processed_v2/telemetry_standard_scaled.csv', index=False)

df_mm_out = df_mm.copy()
df_mm_out.insert(0, 'timestamp', tel_wide['timestamp'].values)
df_mm_out.to_csv('processed_v2/telemetry_minmax_scaled.csv', index=False)
print("  Scaled CSVs saved.")

# Roadmap
stages = [
    ('Stage 1\n(Done)', 'Data Engineering\n& Preprocessing', '#2ca02c'),
    ('Stage 2', 'Isolation Forest\n& One-Class SVM', '#1f77b4'),
    ('Stage 3', 'GRU / TCN\nAutoencoder', '#9467bd'),
    ('Stage 4', 'NCDE\n(Advanced)', '#d62728'),
]
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(-0.5, len(stages)-0.5)
ax.set_ylim(-0.6, 1.2)
ax.axis('off')
ax.set_title('Development Roadmap', fontsize=12, fontweight='bold')
for i, (stage, desc, color) in enumerate(stages):
    rect = mpatches.FancyBboxPatch((i-0.42,-0.45), 0.84, 1.35,
                                    boxstyle='round,pad=0.05',
                                    facecolor=color, edgecolor='white', alpha=0.85)
    ax.add_patch(rect)
    ax.text(i, 0.7, stage, ha='center', va='center', fontsize=9,
            fontweight='bold', color='white')
    ax.text(i, 0.1, desc, ha='center', va='center', fontsize=8, color='white')
    if i < len(stages)-1:
        ax.annotate('', xy=(i+0.5,0.2), xytext=(i+0.42,0.2),
                    arrowprops=dict(arrowstyle='->', color='#495057', lw=2))
plt.tight_layout()
plt.savefig('plots_v2/10_roadmap.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved 10_roadmap.png")

# ── Final summary ─────────────────────────────────────────────
print("\nFiles in plots_v2/:")
for f in sorted(os.listdir('plots_v2')):
    fpath = os.path.join('plots_v2', f)
    print(f"  {f:<40} {os.path.getsize(fpath):>8,} bytes")

print("\nFiles in processed_v2/:")
for f in sorted(os.listdir('processed_v2')):
    fpath = os.path.join('processed_v2', f)
    print(f"  {f:<50} {os.path.getsize(fpath):>10,} bytes")

print("\n" + "="*60)
print("ALL VALIDATION CHECKS PASSED.")
print("="*60)
