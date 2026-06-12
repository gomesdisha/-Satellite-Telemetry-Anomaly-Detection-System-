import json, re

def patch_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    changed = 0
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            new_src = src.replace('RADIATOR_TEMP_1', 'RADIATOR_TEMP')
            new_src = new_src.replace("list(range(2, 5)),\n     lambda mu, sig: mu + np.random.uniform(3.0, 4.5) * sig",
                                      "list(range(0, 24)),\n     lambda mu, sig: mu + np.random.uniform(3.0, 4.5) * sig")
            new_src = new_src.replace("'RF_SIGNAL_STRENGTH',list(range(2,5))",
                                      "'RF_SIGNAL_STRENGTH',list(range(0,24))")
            if new_src != src:
                cell['source'] = new_src
                changed += 1
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"  Patched {path} ({changed} cells changed)")

patch_notebook('sections/10_anomaly_injection.ipynb')
patch_notebook('phase2_combined.ipynb')
print("Done.")
