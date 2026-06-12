"""
validate_phase2.py — Headless validation of all Phase 2 notebook logic
Runs: Anomaly Injection → Isolation Forest → One-Class SVM → Comparison
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.patches as mpatches
import os, warnings, time, json
warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              f1_score, precision_score, recall_score)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'#f8f9fa'})
PALETTE = ['#1f77b4','#2ca02c','#d62728','#9467bd','#8c564b',
           '#e377c2','#7f7f7f','#bcbd22','#17becf','#ff7f0e']

os.makedirs('plots_v2', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 65)
print("  PHASE 2 VALIDATION — Spacecraft Telemetry Anomaly Detection")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════
# SECTION 10 — ANOMALY INJECTION
# ══════════════════════════════════════════════════════════════════
print("\n[S10] Anomaly Injection...")

tel = pd.read_csv('telemetry_train.csv')
tel['timestamp'] = pd.to_datetime(tel['timestamp'])
tel = tel.sort_values(['parameter','timestamp']).reset_index(drop=True)
tel['is_anomaly']   = 0
tel['anomaly_type'] = 'none'

param_stats = (tel.groupby('parameter')['value']
               .agg(['mean','std','min','max'])
               .rename(columns={'mean':'p_mean','std':'p_std',
                                'min':'p_min','max':'p_max'}))

# ── Type 1: Point anomalies ──────────────────────────────────────
POINT_PARAMS = {
    'Power':'BATT_VOLTAGE_1', 'Thermal':'OBC_TEMP', 'ADCS':'GYRO_X',
    'Comms':'RF_SIGNAL_STRENGTH', 'OBC':'CPU_USAGE', 'Propulsion':'TANK_PRESSURE'
}
N_POINT = 8
for subsys, param in POINT_PARAMS.items():
    mask   = tel['parameter'] == param
    idx    = tel[mask].index.tolist()
    chosen = np.random.choice(idx, size=N_POINT, replace=False)
    mu, sig = param_stats.loc[param,'p_mean'], param_stats.loc[param,'p_std']
    for i in chosen:
        d = np.random.choice([-1,1])
        m = np.random.uniform(4, 6)
        tel.at[i,'value']        = mu + d * m * sig
        tel.at[i,'is_anomaly']   = 1
        tel.at[i,'anomaly_type'] = 'point'

n_point = (tel['anomaly_type']=='point').sum()
print(f"  Point anomalies injected: {n_point}")

# ── Type 2: Contextual anomalies ─────────────────────────────────
CONTEXTUAL_CASES = [
    ('SOLAR_POWER_TOTAL', list(range(7,17)), lambda mu,sig: np.random.uniform(0.5,3.0)),
    ('BATT_VOLTAGE_2',
     list(range(0,5))+list(range(20,24)),
     lambda mu,sig: mu + np.random.uniform(2.5,4.0)*sig),
    ('RF_SIGNAL_STRENGTH', list(range(0,24)),
     lambda mu,sig: mu + np.random.uniform(3.0,4.5)*sig),
]
N_CONTEXTUAL = 10
tel['hour'] = tel['timestamp'].dt.hour
for param, anom_hours, val_fn in CONTEXTUAL_CASES:
    if param not in param_stats.index: continue
    mu, sig = param_stats.loc[param,'p_mean'], param_stats.loc[param,'p_std']
    mask    = (tel['parameter']==param) & (tel['hour'].isin(anom_hours))
    idx     = tel[mask & (tel['is_anomaly']==0)].index.tolist()
    if len(idx) >= N_CONTEXTUAL:
        chosen = np.random.choice(idx, size=N_CONTEXTUAL, replace=False)
        for i in chosen:
            tel.at[i,'value']        = val_fn(mu, sig)
            tel.at[i,'is_anomaly']   = 1
            tel.at[i,'anomaly_type'] = 'contextual'
tel.drop(columns=['hour'], inplace=True)
n_contextual = (tel['anomaly_type']=='contextual').sum()
print(f"  Contextual anomalies injected: {n_contextual}")

# ── Type 3: Collective anomalies ─────────────────────────────────
COLLECTIVE_CASES = [
    ('TANK_PRESSURE',  'drift_down', 25, 3),
    ('GYRO_Y',         'step_shift', 20, 3),
    ('MEMORY_USAGE',   'drift_up',   30, 2),
    ('RADIATOR_TEMP',  'oscillate',  20, 3),
    ('BATT_VOLTAGE_1', 'drift_down', 20, 2),
]

def inject_collective(tel, param, pattern, window_size, n_windows):
    mask    = (tel['parameter']==param) & (tel['is_anomaly']==0)
    indices = tel[mask].index.tolist()
    if param not in param_stats.index: return 0
    mu, sig = param_stats.loc[param,'p_mean'], param_stats.loc[param,'p_std']
    count   = 0
    for _ in range(n_windows):
        if len(indices) < window_size + 10: break
        sp     = np.random.randint(5, len(indices)-window_size-5)
        window = indices[sp: sp+window_size]
        if pattern == 'drift_down':
            drift = np.linspace(0,-3.0*sig,window_size)
            for j,i in enumerate(window):
                if tel.at[i,'is_anomaly']==0:
                    tel.at[i,'value']+=drift[j]; tel.at[i,'is_anomaly']=1
                    tel.at[i,'anomaly_type']='collective'; count+=1
        elif pattern == 'drift_up':
            drift = np.linspace(0,+3.0*sig,window_size)
            for j,i in enumerate(window):
                if tel.at[i,'is_anomaly']==0:
                    tel.at[i,'value']+=drift[j]; tel.at[i,'is_anomaly']=1
                    tel.at[i,'anomaly_type']='collective'; count+=1
        elif pattern == 'step_shift':
            offset = np.random.choice([-1,1])*np.random.uniform(2.5,3.5)*sig
            for i in window:
                if tel.at[i,'is_anomaly']==0:
                    tel.at[i,'value']+=offset; tel.at[i,'is_anomaly']=1
                    tel.at[i,'anomaly_type']='collective'; count+=1
        elif pattern == 'oscillate':
            freq  = np.random.uniform(0.5,1.5); amp=np.random.uniform(2.0,3.0)*sig
            phase = np.linspace(0,2*np.pi*freq,window_size)
            for j,i in enumerate(window):
                if tel.at[i,'is_anomaly']==0:
                    tel.at[i,'value']+=amp*np.sin(phase[j]); tel.at[i,'is_anomaly']=1
                    tel.at[i,'anomaly_type']='collective'; count+=1
        injected_set = set(window)
        indices = [i for i in indices if i not in injected_set]
    return count

n_col = 0
for param, pattern, win, n_win in COLLECTIVE_CASES:
    n_col += inject_collective(tel, param, pattern, win, n_win)
print(f"  Collective anomalies injected: {n_col}")

total_anom  = tel['is_anomaly'].sum()
anom_rate   = total_anom / len(tel) * 100
print(f"  Total anomalies: {total_anom}  |  Rate: {anom_rate:.2f}%")
assert 1 < anom_rate < 10, f"Anomaly rate {anom_rate:.2f}% out of expected range"

# Save
tel.to_csv('data/telemetry_with_anomalies.csv', index=False)
print("  Saved: data/telemetry_with_anomalies.csv")

# Injection plot
tel_clean = pd.read_csv('telemetry_train.csv')
tel_clean['timestamp'] = pd.to_datetime(tel_clean['timestamp'])
SHOW = ['BATT_VOLTAGE_1','TANK_PRESSURE','GYRO_Y',
        'SOLAR_POWER_TOTAL','MEMORY_USAGE','RADIATOR_TEMP']
fig, axes = plt.subplots(3,2,figsize=(16,12))
fig.suptitle('Anomaly Injection — Before vs After',fontsize=12,fontweight='bold')
anom_colors = {'point':'#d62728','contextual':'#ff7f0e','collective':'#9467bd'}
for ax, param in zip(axes.flatten(), SHOW):
    cs = tel_clean[tel_clean['parameter']==param].sort_values('timestamp')
    as_ = tel[tel['parameter']==param].sort_values('timestamp')
    ax.plot(cs['timestamp'],cs['value'],color='#1f77b4',lw=0.8,alpha=0.5,label='Normal')
    for atype,color in anom_colors.items():
        sub = as_[as_['anomaly_type']==atype]
        if len(sub)>0:
            ax.scatter(sub['timestamp'],sub['value'],color=color,s=25,zorder=5,label=atype)
    ax.set_title(param,fontweight='bold',fontsize=9)
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('plots_v2/10_anomaly_injection.png',dpi=120,bbox_inches='tight')
plt.close()
print("  Saved: plots_v2/10_anomaly_injection.png")

# ══════════════════════════════════════════════════════════════════
# SHARED DATA PREP (used by both models)
# ══════════════════════════════════════════════════════════════════
print("\n[Shared] Preparing wide-format data for models...")

tel_wide = tel.pivot_table(index='timestamp', columns='parameter',
                            values='value', aggfunc='mean').reset_index()
tel_wide.columns.name = None
tel_wide = tel_wide.sort_values('timestamp').reset_index(drop=True)

label_wide = tel.pivot_table(index='timestamp', columns='parameter',
                              values='is_anomaly', aggfunc='max').reset_index()
label_wide.columns.name = None
label_wide = label_wide.sort_values('timestamp').reset_index(drop=True)

param_cols = [c for c in tel_wide.columns if c != 'timestamp']
tel_wide[param_cols]   = tel_wide[param_cols].ffill().bfill()
label_wide[param_cols] = label_wide[param_cols].ffill().fillna(0)

y_true = label_wide[param_cols].max(axis=1).astype(int).values
X_all  = tel_wide[param_cols].values

# Train on the clean normal rows from the first 70% of the dataset
cutoff     = int(0.70 * len(X_all))
train_mask = (y_true[:cutoff] == 0)
X_train    = X_all[:cutoff][train_mask]

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_all_scaled   = scaler.transform(X_all)

print(f"  Train (normal rows, first 70%): {len(X_train_scaled):,} | Score (full): {len(X_all_scaled):,}")
print(f"  True anomaly rows             : {y_true.sum()}")

# Anomaly type per row
type_wide = tel.pivot_table(index='timestamp', columns='parameter',
                             values='anomaly_type', aggfunc='first').reset_index()
type_wide.columns.name = None
type_wide = type_wide.sort_values('timestamp').reset_index(drop=True)
def row_type(row):
    for v in row.values:
        if pd.notna(v) and v != 'none': return v
    return 'none'
type_col = type_wide.drop(columns='timestamp').apply(row_type, axis=1)


# ══════════════════════════════════════════════════════════════════
# SECTION 11 — ISOLATION FOREST
# ══════════════════════════════════════════════════════════════════
print("\n[S11] Isolation Forest...")

t0  = time.time()
# contamination=0.03 matches the ~3.73% raw injection rate.
# In real deployment: use contamination='auto' and tune threshold per operator tolerance.
ifo_model = IsolationForest(n_estimators=200, contamination=0.03,
                             random_state=42, n_jobs=-1)
ifo_model.fit(X_train_scaled)
t_train_if = time.time()-t0

t0         = time.time()
scores_if  = ifo_model.decision_function(X_all_scaled)
y_pred_if  = ifo_model.predict(X_all_scaled)
t_score_if = time.time()-t0

y_pred_if_bin = (y_pred_if==-1).astype(int)
scores_if_inv = -scores_if

prec_if = precision_score(y_true,y_pred_if_bin,zero_division=0)
rec_if  = recall_score(y_true,y_pred_if_bin,zero_division=0)
f1_if   = f1_score(y_true,y_pred_if_bin,zero_division=0)
auc_if  = roc_auc_score(y_true,scores_if_inv)
cm_if   = confusion_matrix(y_true,y_pred_if_bin)
tn,fp,fn,tp = cm_if.ravel()
fpr_if,tpr_if,_ = roc_curve(y_true,scores_if_inv)

print(f"  Precision:{prec_if:.4f} Recall:{rec_if:.4f} F1:{f1_if:.4f} AUC:{auc_if:.4f}")
print(f"  TP:{tp}  FP:{fp}  FN:{fn}  TN:{tn}  | Train:{t_train_if:.2f}s")

# Per-type recall
type_results_if = []
for atype in ['point','contextual','collective']:
    mask = type_col==atype
    if mask.sum()>0:
        yt = (type_col[mask]!='none').astype(int).values
        yp = y_pred_if_bin[mask.values]
        type_results_if.append({'Anomaly Type':atype,'Total':int(mask.sum()),
                                 'Detected':int((yp==1).sum()),
                                 'Recall':round(recall_score(yt,yp,zero_division=0),3),
                                 'Precision':round(precision_score(yt,yp,zero_division=0),3)})

# Plots
thresh = np.percentile(scores_if_inv,97)

fig,axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle('Isolation Forest — Evaluation',fontsize=11,fontweight='bold')
# score dist
axes[0].hist(scores_if_inv[y_true==0],bins=50,alpha=0.7,color='#1f77b4',label='Normal',density=True)
axes[0].hist(scores_if_inv[y_true==1],bins=50,alpha=0.7,color='#d62728',label='Anomaly',density=True)
axes[0].axvline(thresh,color='black',ls='--',lw=1.5,label='Threshold')
axes[0].set_title('Score Distribution'); axes[0].legend(fontsize=8)
# CM
sns.heatmap(cm_if,annot=True,fmt='d',cmap='Blues',
            xticklabels=['Pred N','Pred A'],yticklabels=['True N','True A'],
            ax=axes[1],cbar=False,annot_kws={'size':13,'weight':'bold'})
axes[1].set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig('plots_v2/11_iforest_eval.png',dpi=120,bbox_inches='tight'); plt.close()

# ROC
fig,ax=plt.subplots(figsize=(7,6))
ax.plot(fpr_if,tpr_if,color='#1f77b4',lw=2,label=f'IF AUC={auc_if:.3f}')
ax.plot([0,1],[0,1],'k--',lw=1)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend(); ax.set_title('ROC — Isolation Forest')
plt.tight_layout()
plt.savefig('plots_v2/11_roc_iforest.png',dpi=120,bbox_inches='tight'); plt.close()
print("  Plots saved.")

# Save results
ifo_results = {
    'model':'Isolation Forest','precision':round(prec_if,4),'recall':round(rec_if,4),
    'f1':round(f1_if,4),'auc':round(auc_if,4),
    'train_time':round(t_train_if,3),'score_time':round(t_score_if,3),
    'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn),
    'by_type':type_results_if,'fpr':fpr_if.tolist(),'tpr':tpr_if.tolist()
}
with open('models/iforest_results.json','w') as f: json.dump(ifo_results,f,indent=2)
print("  Saved: models/iforest_results.json")

# ══════════════════════════════════════════════════════════════════
# SECTION 12 — ONE-CLASS SVM
# ══════════════════════════════════════════════════════════════════
print("\n[S12] One-Class SVM...")

t0 = time.time()
# nu=0.03 matches the ~3.73% raw injection rate (same reasoning as contamination above).
ocsvm = OneClassSVM(kernel='rbf', nu=0.03, gamma='scale')
ocsvm.fit(X_train_scaled)
t_train_svm = time.time()-t0

t0          = time.time()
scores_svm  = ocsvm.decision_function(X_all_scaled)
y_pred_svm  = ocsvm.predict(X_all_scaled)
t_score_svm = time.time()-t0

y_pred_svm_bin = (y_pred_svm==-1).astype(int)
scores_svm_inv = -scores_svm

prec_svm = precision_score(y_true,y_pred_svm_bin,zero_division=0)
rec_svm  = recall_score(y_true,y_pred_svm_bin,zero_division=0)
f1_svm   = f1_score(y_true,y_pred_svm_bin,zero_division=0)
auc_svm  = roc_auc_score(y_true,scores_svm_inv)
cm_svm   = confusion_matrix(y_true,y_pred_svm_bin)
tn_s,fp_s,fn_s,tp_s = cm_svm.ravel()
fpr_svm,tpr_svm,_ = roc_curve(y_true,scores_svm_inv)

print(f"  Precision:{prec_svm:.4f} Recall:{rec_svm:.4f} F1:{f1_svm:.4f} AUC:{auc_svm:.4f}")
print(f"  TP:{tp_s}  FP:{fp_s}  FN:{fn_s}  TN:{tn_s}  | Train:{t_train_svm:.2f}s")

# Per-type
type_results_svm = []
for atype in ['point','contextual','collective']:
    mask = type_col==atype
    if mask.sum()>0:
        yt = (type_col[mask]!='none').astype(int).values
        yp = y_pred_svm_bin[mask.values]
        type_results_svm.append({'Anomaly Type':atype,'Total':int(mask.sum()),
                                  'Detected':int((yp==1).sum()),
                                  'Recall':round(recall_score(yt,yp,zero_division=0),3),
                                  'Precision':round(precision_score(yt,yp,zero_division=0),3)})

# PCA 2D plot
pca    = PCA(n_components=2,random_state=42)
X_2d   = pca.fit_transform(X_all_scaled)
var_ex = pca.explained_variance_ratio_.sum()*100

fig,ax=plt.subplots(figsize=(9,7))
mask_nn=(y_pred_svm_bin==0)&(y_true==0)
mask_tp=(y_pred_svm_bin==1)&(y_true==1)
mask_fn=(y_pred_svm_bin==0)&(y_true==1)
mask_fp=(y_pred_svm_bin==1)&(y_true==0)
ax.scatter(X_2d[mask_nn,0],X_2d[mask_nn,1],c='#1f77b4',s=6,alpha=0.3,label='TN')
ax.scatter(X_2d[mask_tp,0],X_2d[mask_tp,1],c='#d62728',s=40,alpha=0.9,label='TP',marker='*')
ax.scatter(X_2d[mask_fn,0],X_2d[mask_fn,1],c='#ff7f0e',s=40,alpha=0.9,label='FN (missed)',marker='v')
ax.scatter(X_2d[mask_fp,0],X_2d[mask_fp,1],c='#bcbd22',s=20,alpha=0.7,label='FP',marker='x')
ax.set_title(f'OCSVM PCA 2D ({var_ex:.1f}% variance)',fontweight='bold')
ax.legend(fontsize=8); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
plt.tight_layout()
plt.savefig('plots_v2/12_pca_ocsvm.png',dpi=120,bbox_inches='tight'); plt.close()

# CM plot
fig,ax=plt.subplots(figsize=(6,5))
sns.heatmap(cm_svm,annot=True,fmt='d',cmap='Purples',
            xticklabels=['Pred N','Pred A'],yticklabels=['True N','True A'],
            ax=ax,cbar=False,annot_kws={'size':13,'weight':'bold'})
ax.set_title('One-Class SVM — Confusion Matrix')
plt.tight_layout()
plt.savefig('plots_v2/12_cm_ocsvm.png',dpi=120,bbox_inches='tight'); plt.close()
print("  Plots saved.")

svm_results = {
    'model':'One-Class SVM','precision':round(prec_svm,4),'recall':round(rec_svm,4),
    'f1':round(f1_svm,4),'auc':round(auc_svm,4),
    'train_time':round(t_train_svm,3),'score_time':round(t_score_svm,3),
    'tp':int(tp_s),'fp':int(fp_s),'fn':int(fn_s),'tn':int(tn_s),
    'by_type':type_results_svm,'fpr':fpr_svm.tolist(),'tpr':tpr_svm.tolist()
}
with open('models/ocsvm_results.json','w') as f: json.dump(svm_results,f,indent=2)
print("  Saved: models/ocsvm_results.json")

# ══════════════════════════════════════════════════════════════════
# SECTION 13 — COMPARISON
# ══════════════════════════════════════════════════════════════════
print("\n[S13] Model Comparison...")

# Overlaid ROC
fig,ax=plt.subplots(figsize=(8,7))
ax.plot(fpr_if,tpr_if,color='#1f77b4',lw=2.5,label=f'Isolation Forest (AUC={auc_if:.3f})')
ax.plot(fpr_svm,tpr_svm,color='#9467bd',lw=2.5,ls='--',label=f'One-Class SVM (AUC={auc_svm:.3f})')
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
ax.fill_between(fpr_if,tpr_if,alpha=0.07,color='#1f77b4')
ax.fill_between(fpr_svm,tpr_svm,alpha=0.07,color='#9467bd')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend(fontsize=9)
ax.set_title('ROC Comparison — Both Models',fontweight='bold')
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig('plots_v2/13_roc_comparison.png',dpi=120,bbox_inches='tight'); plt.close()

# Metric bar chart
compare_metrics = ['Precision','Recall','F1 Score','ROC-AUC']
if_v  = [prec_if,rec_if,f1_if,auc_if]
svm_v = [prec_svm,rec_svm,f1_svm,auc_svm]
x=np.arange(4); w=0.35
fig,ax=plt.subplots(figsize=(10,5))
b1=ax.bar(x-w/2,if_v, w,label='Isolation Forest',color='#1f77b4',alpha=0.85,edgecolor='white')
b2=ax.bar(x+w/2,svm_v,w,label='One-Class SVM',  color='#9467bd',alpha=0.85,edgecolor='white')
ax.bar_label(b1,fmt='%.3f',padding=3,fontsize=9)
ax.bar_label(b2,fmt='%.3f',padding=3,fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(compare_metrics)
ax.set_ylim(0,1.15); ax.legend(fontsize=9)
ax.set_title('Performance Comparison',fontweight='bold')
plt.tight_layout()
plt.savefig('plots_v2/13_metric_comparison.png',dpi=120,bbox_inches='tight'); plt.close()

# Side-by-side CM
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,cm,title,cmap in zip(axes,[cm_if,cm_svm],
                              ['Isolation Forest','One-Class SVM'],['Blues','Purples']):
    sns.heatmap(cm,annot=True,fmt='d',cmap=cmap,
                xticklabels=['Pred N','Pred A'],yticklabels=['True N','True A'],
                ax=ax,cbar=False,annot_kws={'size':13,'weight':'bold'})
    ax.set_title(title,fontweight='bold')
plt.tight_layout()
plt.savefig('plots_v2/13_cm_comparison.png',dpi=120,bbox_inches='tight'); plt.close()

# Per-type recall chart
if_type_df  = pd.DataFrame(type_results_if)
svm_type_df = pd.DataFrame(type_results_svm)
if 'Anomaly Type' in if_type_df.columns and 'Anomaly Type' in svm_type_df.columns:
    type_cmp = if_type_df[['Anomaly Type','Recall']].merge(
        svm_type_df[['Anomaly Type','Recall']], on='Anomaly Type', suffixes=(' IF',' SVM'))
    atypes = type_cmp['Anomaly Type'].values
    x2 = np.arange(len(atypes))
    fig,ax=plt.subplots(figsize=(8,5))
    b1=ax.bar(x2-0.2,type_cmp['Recall IF'], 0.35,label='IF', color='#1f77b4',alpha=0.85,edgecolor='white')
    b2=ax.bar(x2+0.2,type_cmp['Recall SVM'],0.35,label='OCSVM',color='#9467bd',alpha=0.85,edgecolor='white')
    ax.bar_label(b1,fmt='%.3f',padding=3,fontsize=9)
    ax.bar_label(b2,fmt='%.3f',padding=3,fontsize=9)
    ax.set_xticks(x2); ax.set_xticklabels(atypes)
    ax.set_ylim(0,1.2); ax.set_ylabel('Recall')
    ax.set_title('Recall by Anomaly Type',fontweight='bold'); ax.legend()
    plt.tight_layout()
    plt.savefig('plots_v2/13_per_type_recall.png',dpi=120,bbox_inches='tight'); plt.close()

print("  Comparison plots saved.")

# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  PHASE 2 VALIDATION COMPLETE")
print("="*65)
print(f"\n  Anomaly injection  : {total_anom} anomalies ({anom_rate:.2f}%)")
print(f"                        Point={n_point} | Contextual={n_contextual} | Collective={n_col}")
print(f"\n  Isolation Forest   : Precision={prec_if:.4f}  Recall={rec_if:.4f}  F1={f1_if:.4f}  AUC={auc_if:.4f}")
print(f"  One-Class SVM      : Precision={prec_svm:.4f}  Recall={rec_svm:.4f}  F1={f1_svm:.4f}  AUC={auc_svm:.4f}")
print(f"\n  Plots saved to plots_v2/  (10_*, 11_*, 12_*, 13_*)")
print(f"  Data  saved to data/telemetry_with_anomalies.csv")
print(f"  Models saved to models/iforest_results.json, ocsvm_results.json")
print("\n" + "="*65)
