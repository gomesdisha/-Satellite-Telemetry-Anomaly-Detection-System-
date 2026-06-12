import json, sys
sys.stdout.reconfigure(encoding='utf-8')

ORIGINAL_DATA_CELL = (
    "# Load the labelled anomaly dataset from Section 10\n"
    "tel = pd.read_csv('data/telemetry_with_anomalies.csv', parse_dates=['timestamp'])\n"
    "tel = tel.sort_values(['parameter','timestamp']).reset_index(drop=True)\n"
    "\n"
    "tel_wide = tel.pivot_table(index='timestamp', columns='parameter',\n"
    "                            values='value', aggfunc='mean').reset_index()\n"
    "tel_wide.columns.name = None\n"
    "tel_wide = tel_wide.sort_values('timestamp').reset_index(drop=True)\n"
    "\n"
    "label_wide = tel.pivot_table(index='timestamp', columns='parameter',\n"
    "                              values='is_anomaly', aggfunc='max').reset_index()\n"
    "label_wide.columns.name = None\n"
    "label_wide = label_wide.sort_values('timestamp').reset_index(drop=True)\n"
    "\n"
    "param_cols = [c for c in tel_wide.columns if c != 'timestamp']\n"
    "tel_wide[param_cols]   = tel_wide[param_cols].ffill().bfill()\n"
    "label_wide[param_cols] = label_wide[param_cols].ffill().fillna(0)\n"
    "\n"
    "y_true = label_wide[param_cols].max(axis=1).astype(int).values\n"
    "X_all  = tel_wide[param_cols].values\n"
    "\n"
    "print(f'Dataset loaded: {tel.shape}')\n"
    "print(f'Wide-format shape: {X_all.shape}')\n"
    "print(f'Anomaly rows (wide): {y_true.sum()} / {len(y_true)}')"
)

ORIGINAL_SCALE_CELL = (
    "# Train on clean normal rows from the first 70% of timestamps\n"
    "# (simulates having historical clean data before anomalies started appearing)\n"
    "cutoff     = int(0.70 * len(X_all))\n"
    "train_mask = (y_true[:cutoff] == 0)   # keep only rows with no anomaly\n"
    "X_train    = X_all[:cutoff][train_mask]\n"
    "\n"
    "scaler         = StandardScaler()\n"
    "X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on clean training rows\n"
    "X_all_scaled   = scaler.transform(X_all)          # apply same scale to full dataset\n"
    "\n"
    "print(f'Training rows (normal only): {len(X_train_scaled):,}')\n"
    "print(f'Scoring rows  (full data)  : {len(X_all_scaled):,}')\n"
    "\n"
    "# Row-level anomaly type label\n"
    "type_wide = tel.pivot_table(index='timestamp', columns='parameter',\n"
    "                             values='anomaly_type', aggfunc='first').reset_index()\n"
    "type_wide.columns.name = None\n"
    "type_wide = type_wide.sort_values('timestamp').reset_index(drop=True)\n"
    "def row_type(row):\n"
    "    for v in row.values:\n"
    "        if pd.notna(v) and v != 'none': return v\n"
    "    return 'none'\n"
    "type_col = type_wide.drop(columns='timestamp').apply(row_type, axis=1)"
)

files = ['sections/11_isolation_forest.ipynb', 'sections/12_one_class_svm.ipynb']

for path in files:
    # Read as raw text to avoid json control char issues
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    nb = json.loads(content)

    data_patched  = False
    scale_patched = False

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])

        if ('X_train_raw' in src or 'tel_clean_train' in src or
                'tel_test_wide' in src or 'y_long' in src):
            cell['source'] = ORIGINAL_DATA_CELL
            data_patched = True

        elif 'X_train_scaled = scaler.fit_transform' in src:
            cell['source'] = ORIGINAL_SCALE_CELL
            scale_patched = True

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f'{path}: data={data_patched}, scale={scale_patched}')

print('Done.')
