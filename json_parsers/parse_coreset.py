import os
import json
import pandas as pd

ROOT_DIR = "results/results_coreset"
OUT_CSV = "results/results_coreset/coreset_summary.csv"

METHODS = ["full", "random", "vq", "voronoi", "gonzalez", "herding"]

rows = []

for db in sorted(os.listdir(ROOT_DIR)):
    db_dir = os.path.join(ROOT_DIR, db)
    if not os.path.isdir(db_dir):
        continue

    # find json file
    json_files = [f for f in os.listdir(db_dir) if f.endswith(".json")]
    if not json_files:
        print(f"⚠️ No JSON found for {db}")
        continue

    json_path = os.path.join(db_dir, json_files[0])

    with open(json_path, "r") as f:
        results = json.load(f)

    row = {"dataset": db}

    for method in METHODS:
        acc = results.get(method, {}).get("acc", None)
        auc = results.get(method, {}).get("auc", None)

        row[f"{method}_acc"] = acc
        row[f"{method}_auc"] = auc

    rows.append(row)

df = pd.DataFrame(rows)
df = df.sort_values("dataset")

df.to_csv(OUT_CSV, index=False)
print(f"✅ Saved CSV to {OUT_CSV}")
