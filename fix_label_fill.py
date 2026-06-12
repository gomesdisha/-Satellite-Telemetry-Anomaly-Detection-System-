import json

def fix_label_fill(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    count = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        # Fix the label ffill bug — both variations
        fixed = src.replace(
            "label_wide[param_cols] = label_wide[param_cols].ffill().fillna(0)",
            "label_wide[param_cols] = label_wide[param_cols].fillna(0)      # labels: NO ffill — missing = 0 (not anomalous)"
        )
        if fixed != src:
            cell['source'] = fixed
            count += 1
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"  {path}: {count} cells patched")

fix_label_fill('sections/11_isolation_forest.ipynb')
fix_label_fill('sections/12_one_class_svm.ipynb')
fix_label_fill('phase2_combined.ipynb')
print("Done.")
