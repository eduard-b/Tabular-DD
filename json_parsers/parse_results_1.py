import os
import json
import pandas as pd

# -------------------------------------------------------------
# Helper: Extract metrics from binary or multiclass JSON
# -------------------------------------------------------------
def parse_result_file(path):
    with open(path, "r") as f:
        data = json.load(f)

    config = data.get("config", {})
    name = config.get("dataset_name", os.path.basename(path))

    # ---------------------------------------------------------
    # Determine number of classes (rough tag only)
    # ---------------------------------------------------------
    num_classes = 2  # default
    
    # Binary-format JSON (presence of AUC fields)
    if "full_data_test_auc" in data:
        num_classes = 2

    # Multiclass-format JSON (presence of dict "full_data": {...})
    elif "full_data" in data:
        num_classes = ">2"

    row = {
        "dataset": name,
        "ipc": config.get("ipc", None),
        "num_classes": num_classes,
        "full_acc": None,
        "full_auc_f1": None,
        "rand_acc": None,
        "rand_auc_f1": None,
        "herd_acc": None,
        "herd_auc_f1": None,
        "dm_acc": None,
        "dm_auc_f1": None,
    }

    # ---------------------------------------------------------
    # Case A — Binary JSON
    # ---------------------------------------------------------
    if "full_data_test_auc" in data:
        row["full_auc_f1"] = data.get("full_data_test_auc")
        row["rand_auc_f1"] = data.get("random_ipc_test_auc")
        row["herd_auc_f1"] = data.get("herding_ipc_test_auc")
        row["dm_auc_f1"]   = data.get("dm_test_auc")

    # ---------------------------------------------------------
    # Case B — Multiclass JSON (acc & f1)
    # ---------------------------------------------------------
    elif "full_data" in data:
        for key, prefix in [
            ("full_data", "full"),
            ("rand_ipc", "rand"),
            ("herd_ipc", "herd"),
            ("dm_ipc",  "dm")
        ]:
            if key in data:
                row[f"{prefix}_acc"] = data[key].get("acc")
                row[f"{prefix}_auc_f1"] = data[key].get("f1")

    return row

# -------------------------------------------------------------
# Scan multiple results folders
# -------------------------------------------------------------
def parse_all_results(directories):
    rows = []
    for d in directories:
        if not os.path.exists(d):
            continue
        for fname in os.listdir(d):
            if fname.endswith(".json"):
                path = os.path.join(d, fname)
                print("Parsing:", path)
                rows.append(parse_result_file(path))
    return pd.DataFrame(rows)

# -------------------------------------------------------------
# Convert DataFrame to Markdown table for README.md
# -------------------------------------------------------------
def df_to_markdown(df):
    return df.to_markdown(index=False)

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
df = parse_all_results(["./results", "./results_multiclass"])

print("\n=== SUMMARY TABLE ===")
print(df)

# Save CSV
df.to_csv("all_results_summary.csv", index=False)

# Save JSON
df.to_json("all_results_summary.json", orient="records", indent=2)

# Save Markdown table (README-ready)
markdown_table = df_to_markdown(df)
with open("results_table.md", "w") as f:
    f.write(markdown_table)

print("\nSaved:")
print("  → all_results_summary.csv")
print("  → all_results_summary.json")
print("  → results_table.md (Markdown table for README)")
