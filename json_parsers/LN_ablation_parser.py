import os
import json
import pandas as pd

ROOT = "./results_embedder_ablation_LN/"

EMBEDDER_LIST = ["Rand acc", "Herd acc",
                 "LN acc", "LNDeep acc", "LNWide acc",
                 "LNRes acc", "LNCascade acc"]

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
        "Full acc": None,
        "Rand acc": None,
        "Herd acc": None,
        "LN acc": None,
        "LNDeep acc": None,
        "LNWide acc": None,
        "LNRes acc": None,
        "LNCascade acc": None,
    }

    for fname in os.listdir(dataset_dir):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(dataset_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        res = data["results"]
        rows[dataset]["Full acc"] = res["full"]["test_acc"]
        rows[dataset]["Rand acc"] = res["random"]["test_acc"]
        rows[dataset]["Herd acc"] = res["herd"]["test_acc"]

        # detect embedder
        for emb in ["LN", "LNDeep", "LNWide", "LNRes", "LNCascade"]:
            if f"_{emb}_" in fname:
                rows[dataset][f"{emb} acc"] = res["dm"]["test_acc"]

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
    accs = row[EMBEDDER_LIST]

    # compute ranking
    sorted_vals = accs.sort_values(ascending=False)

    best_col = sorted_vals.index[0]
    second_col = sorted_vals.index[1]

    # format table values
    md_df.at[idx, best_col] = f"**{row[best_col]:.6f}**"
    md_df.at[idx, second_col] = f"*{row[second_col]:.6f}*"

    # format others to 6 decimals
    for col in accs.index:
        if col not in (best_col, second_col):
            md_df.at[idx, col] = f"{row[col]:.6f}"

# Full acc only printed normally
for idx, row in md_df.iterrows():
    md_df.at[idx, "Full acc"] = f"{row['Full acc']:.6f}"

# --------------------------------------------
# Print
# --------------------------------------------
print("\n--- MARKDOWN TABLE ---\n")
print(md_df.to_markdown(index=False))
