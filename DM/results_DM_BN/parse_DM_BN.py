import os
import json
import pandas as pd

# directory where your DM-BN jsons are
RESULTS_DIR = "./results_DM_BN/"

rows = []

for fname in os.listdir(RESULTS_DIR):
    if not fname.endswith(".json"):
        continue

    path = os.path.join(RESULTS_DIR, fname)

    with open(path, "r") as f:
        data = json.load(f)

    dataset = data["dataset"]
    res     = data["results"]

    row = {
        "Dataset": dataset,
        "Full AUC": res["full"]["test_auc"],
        "Rand AUC": res["random"]["test_auc"],
        "Herd AUC": res["herd"]["test_auc"],
        "DM-BN AUC": res["dm_bn"]["test_auc"],
        "file": fname
    }

    rows.append(row)

# convert to dataframe
df = pd.DataFrame(rows)

# sort alphabetically
df = df.sort_values("Dataset").reset_index(drop=True)

print(df)

# produce a markdown table
print("\n--- MARKDOWN TABLE ---\n")
print(df.to_markdown(index=False))
