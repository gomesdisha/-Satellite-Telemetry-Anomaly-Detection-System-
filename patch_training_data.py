import json

# ══════════════════════════════════════════════════════════
# New Section 11.1 cell — clean separation of train vs test
# ══════════════════════════════════════════════════════════
NEW_11_DATA_CELL = """\
# ── TRAINING DATA: original clean telemetry (guaranteed 0 anomalies) ──────────
tel_clean = pd.read_csv('telemetry_train.csv', parse_dates=['timestamp'])

tel_train_wide = tel_clean.pivot_table(index='timestamp', columns='parameter',
                                        values='value', aggfunc='mean').reset_index()
tel_train_wide.columns.name = None
tel_train_wide = tel_train_wide.sort_values('timestamp').reset_index(drop=True)

train_param_cols = [c for c in tel_train_wide.columns if c != 'timestamp']
tel_train_wide[train_param_cols] = tel_train_wide[train_param_cols].ffill().bfill()

X_train_raw = tel_train_wide[train_param_cols].values
print('Training data  (clean, 0 anomalies) :', tel_clean.shape)
print('Training matrix shape               :', X_train_raw.shape)

# ── TEST/SCORE DATA: injected dataset (has anomalies — used only for evaluation) ─
tel_test = pd.read_csv('data/telemetry_with_anomalies.csv', parse_dates=['timestamp'])
print('\\nScoring data   (with anomalies)     :', tel_test.shape)
print(f'  Anomalies: {tel_test[\"is_anomaly\"].sum()} ({tel_test[\"is_anomaly\"].mean()*100:.2f}%)')

tel_test_wide = tel_test.pivot_table(index='timestamp', columns='parameter',
                                      values='value', aggfunc='mean').reset_index()
tel_test_wide.columns.name = None
tel_test_wide = tel_test_wide.sort_values('timestamp').reset_index(drop=True)

label_wide = tel_test.pivot_table(index='timestamp', columns='parameter',
                                   values='is_anomaly', aggfunc='max').reset_index()
label_wide.columns.name = None
label_wide = label_wide.sort_values('timestamp').reset_index(drop=True)

param_cols = [c for c in tel_test_wide.columns if c != 'timestamp']
tel_test_wide[param_cols] = tel_test_wide[param_cols].ffill().bfill()
label_wide[param_cols]    = label_wide[param_cols].ffill().fillna(0)

X_all  = tel_test_wide[param_cols].values
y_true = label_wide[param_cols].max(axis=1).astype(int).values

print('\\nScore matrix shape:', X_all.shape)
print('Anomaly rows      :', y_true.sum(), '/', len(y_true))
"""

# ══════════════════════════════════════════════════════════
# New Section 11.2 — just scaling now (no more train/test split logic)
# ══════════════════════════════════════════════════════════
NEW_11_SCALE_CELL = """\
# StandardScaler — fitted on 100% clean training data, applied to test data
# This is the correct order: learn the scale from normal, apply to everything else
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)   # fit on clean original data
X_all_scaled   = scaler.transform(X_all)              # transform injected data (no refit)

print('Scaler fitted on clean training data (no anomaly leakage).')
print(f'  Training rows   : {X_train_scaled.shape[0]:,} (all normal)')
print(f'  Scoring rows    : {X_all_scaled.shape[0]:,} (includes injected anomalies)')
print(f'  Features        : {X_train_scaled.shape[1]}')
print()
print('Why this is important:')
print('  fit_transform on clean data  → scaler learns what normal ranges look like')
print('  transform only on test data  → anomalies are NOT used to set the scale')
print('  If we refitted on test data  → anomalies would shift mean/std and hide themselves')
"""

# ══════════════════════════════════════════════════════════
# Patch notebooks
# ══════════════════════════════════════════════════════════

def patch_nb(path, old_data_snippet, new_data_cell, new_scale_cell):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    found_data  = False
    found_scale = False

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])

        # Replace the data loading cell
        if old_data_snippet in src and not found_data:
            cell['source'] = new_data_cell
            found_data = True
            print(f'  [PATCHED] data cell in {path}')

        # Replace the train/test split / scaling cell
        if 'X_train_scaled = scaler.fit_transform' in src and not found_scale:
            cell['source'] = new_scale_cell
            found_scale = True
            print(f'  [PATCHED] scaling cell in {path}')

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    if not found_data:
        print(f'  [WARN] data cell not found in {path}')
    if not found_scale:
        print(f'  [WARN] scale cell not found in {path}')


print('Patching Section 11...')
patch_nb(
    'sections/11_isolation_forest.ipynb',
    "# Load the labelled anomaly dataset from Section 10",
    NEW_11_DATA_CELL,
    NEW_11_SCALE_CELL
)

print('Patching Section 12...')
patch_nb(
    'sections/12_one_class_svm.ipynb',
    "# Load labelled dataset",
    NEW_11_DATA_CELL,
    NEW_11_SCALE_CELL
)

# ══════════════════════════════════════════════════════════
# Patch the shared data prep cell in phase2_combined.ipynb
# ══════════════════════════════════════════════════════════
NEW_COMBINED_DATA = """\
# ── TRAINING DATA: original clean telemetry (100% normal, 0 anomalies) ─────────
tel_clean = pd.read_csv('telemetry_train.csv', parse_dates=['timestamp'])

tel_train_wide = tel_clean.pivot_table(index='timestamp', columns='parameter',
                                        values='value', aggfunc='mean').reset_index()
tel_train_wide.columns.name = None
tel_train_wide = tel_train_wide.sort_values('timestamp').reset_index(drop=True)
train_param_cols = [c for c in tel_train_wide.columns if c != 'timestamp']
tel_train_wide[train_param_cols] = tel_train_wide[train_param_cols].ffill().bfill()
X_train_raw = tel_train_wide[train_param_cols].values

# ── TEST DATA: injected dataset (anomalies present — for scoring + evaluation) ──
tel_test_wide = tel.pivot_table(index='timestamp', columns='parameter',
                                 values='value', aggfunc='mean').reset_index()
tel_test_wide.columns.name = None
tel_test_wide = tel_test_wide.sort_values('timestamp').reset_index(drop=True)

label_wide = tel.pivot_table(index='timestamp', columns='parameter',
                              values='is_anomaly', aggfunc='max').reset_index()
label_wide.columns.name = None
label_wide = label_wide.sort_values('timestamp').reset_index(drop=True)

param_cols = [c for c in tel_test_wide.columns if c != 'timestamp']
tel_test_wide[param_cols] = tel_test_wide[param_cols].ffill().bfill()
label_wide[param_cols]    = label_wide[param_cols].ffill().fillna(0)

y_true = label_wide[param_cols].max(axis=1).astype(int).values
X_all  = tel_test_wide[param_cols].values

# Scale: fit on clean, transform on test
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_all_scaled   = scaler.transform(X_all)

print(f'Training rows (clean normal only) : {X_train_scaled.shape[0]:,}')
print(f'Scoring rows  (full injected)     : {X_all_scaled.shape[0]:,}')
print(f'True anomaly rows                 : {y_true.sum()}')

# Row-level anomaly type label
type_wide = tel.pivot_table(index='timestamp', columns='parameter',
                             values='anomaly_type', aggfunc='first').reset_index()
type_wide.columns.name = None
type_wide = type_wide.sort_values('timestamp').reset_index(drop=True)
def row_type(row):
    for v in row.values:
        if pd.notna(v) and v != 'none': return v
    return 'none'
type_col = type_wide.drop(columns='timestamp').apply(row_type, axis=1)
"""

print('Patching phase2_combined.ipynb...')
with open('phase2_combined.ipynb', 'r', encoding='utf-8') as f:
    nb3 = json.load(f)

for cell in nb3['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if 'X_train_scaled  = scaler.fit_transform(X_train)' in src or \
       "X_train_scaled = scaler.fit_transform(X_train)" in src:
        cell['source'] = NEW_COMBINED_DATA
        print('  [PATCHED] combined shared data cell')
        break

with open('phase2_combined.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb3, f, indent=1)

print('\nAll notebooks patched.')
