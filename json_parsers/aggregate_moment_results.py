import os
import json
import csv

ROOT_DIR = "results_moments"
OUTPUT_CSV = "moment_accuracy_summary.csv"
MOMENTS = [1, 2, 3, 4]

rows = []

for dataset in sorted(os.listdir(ROOT_DIR)):
    dataset_dir = os.path.join(ROOT_DIR, dataset)
    if not os.path.isdir(dataset_dir):
        continue

    row = {"dataset": dataset}

    # initialize all moments as missing
    for m in MOMENTS:
        row[f"acc_M{m}"] = None

    for m in MOMENTS:

        moment_dir = os.path.join(dataset_dir, 'LNRes', f"M{m}")
        if not os.path.isdir(moment_dir):
            continue

        # read all jsons in this moment folder
        for fname in os.listdir(moment_dir):
            print("F", fname)
            if not fname.endswith(".json"):
                continue

            json_path = os.path.join(moment_dir, fname)

            print(json_path)

            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                # overwrite if multiple exist (last one wins)
                row[f"acc_M{m}"] = data.get("test_acc")

            except Exception as e:
                print(f"⚠️ Failed to read {json_path}: {e}")

    rows.append(row)

# write CSV
fieldnames = ["dataset"] + [f"acc_M{m}" for m in MOMENTS]

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Saved CSV to {OUTPUT_CSV}")
