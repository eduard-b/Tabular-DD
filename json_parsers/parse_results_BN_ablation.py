import os
import json
import pandas as pd

RESULTS_DIR = "./results_embedder_ablation/"

EMB_COLS = ["LN", "BN", "BNDeep", "BNWide", "BNRes", "BNCascade"]

def detect_embedder(fname):
    base = os.path.basename(fname)
    # Example: before: adult_trueDM_ipc10_BNRes_h256...
    # extract substring between "ipc10_" and "_h"
    try:
        right = base.split("ipc10_")[1]
        emb = right.split("_h")[0]
        return emb   # "BN", "BNRes", "LN", etc.
    except:
        return None


rows = []

for dataset in sorted(os.listdir(RESULTS_DIR)):
    dataset_dir = os.path.join(RESULTS_DIR, dataset)
    if not os.path.isdir(dataset_dir):
        continue

    row = {
        "Dataset": dataset,
        "Full AUC": None,
        "Rand AUC": None,
        "Herd AUC": None,
    }

    for col in EMB_COLS:
        row[f"{col} AUC"] = None

    for fname in os.listdir(dataset_dir):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(dataset_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)

        # write baselines once
        if row["Full AUC"] is None:
            row["Full AUC"] = data["results"]["full"]["test_auc"]
            row["Rand AUC"] = data["results"]["random"]["test_auc"]
            row["Herd AUC"] = data["results"]["herd"]["test_auc"]

        emb = detect_embedder(fname)

        if emb in EMB_COLS:
            row[f"{emb} AUC"] = data["results"]["dm"]["test_auc"]

    rows.append(row)

df = pd.DataFrame(rows)
df = df.sort_values("Dataset").reset_index(drop=True)
df.to_csv("embedder_summary_table.csv", index=False)

print(df)
print("\nSaved to embedder_summary_table.csv")
