import os
import json
import pandas as pd

ROOT = "./results_embedder_ablation_LN/"

EMBEDDER_LIST = ["Rand AUC", "Herd AUC",
                 "LN AUC", "LNDeep AUC", "LNWide AUC",
                 "LNRes AUC", "LNCascade AUC"]

rows = {}

# --------------------------------------------
# Load all results
# --------------------------------------------
for dataset in os.listdir(ROOT):
    dataset_dir = os.path.join(ROOT, dataset)
    if not os.path.isdir(dataset_dir):
        continue

    rows[dataset] = {
        "Dataset": dataset,
        "Full AUC": None,
        "Rand AUC": None,
        "Herd AUC": None,
        "LN AUC": None,
        "LNDeep AUC": None,
        "LNWide AUC": None,
        "LNRes AUC": None,
        "LNCascade AUC": None,
    }

    for fname in os.listdir(dataset_dir):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(dataset_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        res = data["results"]
        rows[dataset]["Full AUC"] = res["full"]["test_auc"]
        rows[dataset]["Rand AUC"] = res["random"]["test_auc"]
        rows[dataset]["Herd AUC"] = res["herd"]["test_auc"]

        # detect embedder
        for emb in ["LN", "LNDeep", "LNWide", "LNRes", "LNCascade"]:
            if f"_{emb}_" in fname:
                rows[dataset][f"{emb} AUC"] = res["dm"]["test_auc"]

# --------------------------------------------
# Convert to DataFrame
# --------------------------------------------
df = pd.DataFrame(list(rows.values()))
df = df.sort_values("Dataset").reset_index(drop=True)

# --------------------------------------------
# Markdown formatting: Bold best, Italic second
# --------------------------------------------
md_df = df.copy()

for idx, row in md_df.iterrows():
    # extract only the candidate columns
    aucs = row[EMBEDDER_LIST]

    # compute ranking
    sorted_vals = aucs.sort_values(ascending=False)

    best_col = sorted_vals.index[0]
    second_col = sorted_vals.index[1]

    # format table values
    md_df.at[idx, best_col] = f"**{row[best_col]:.6f}**"
    md_df.at[idx, second_col] = f"*{row[second_col]:.6f}*"

    # format others to 6 decimals
    for col in aucs.index:
        if col not in (best_col, second_col):
            md_df.at[idx, col] = f"{row[col]:.6f}"

# Full AUC only printed normally
for idx, row in md_df.iterrows():
    md_df.at[idx, "Full AUC"] = f"{row['Full AUC']:.6f}"

# --------------------------------------------
# Print
# --------------------------------------------
print("\n--- MARKDOWN TABLE ---\n")
print(md_df.to_markdown(index=False))
