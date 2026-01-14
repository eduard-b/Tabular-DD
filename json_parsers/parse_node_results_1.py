import json
from pathlib import Path
from collections import defaultdict

ROOT = Path("results_NODE_v_LNRES")

EMBEDDERS = ["ln_res", "node"]
MOMENTS = [1, 2, 3, 4]

# data[metric][dataset][embedder][moment] = value
data = {
    "acc": defaultdict(lambda: defaultdict(dict)),
    "auc": defaultdict(lambda: defaultdict(dict)),
}

# -------- parse files --------
for dataset_dir in ROOT.iterdir():
    if not dataset_dir.is_dir():
        continue

    dataset = dataset_dir.name

    for embedder in EMBEDDERS:
        emb_dir = dataset_dir / embedder
        if not emb_dir.exists():
            continue

        for m_dir in emb_dir.iterdir():
            if not m_dir.is_dir():
                continue

            result_file = m_dir / "result.json"
            if not result_file.exists():
                continue

            with open(result_file, "r") as f:
                res = json.load(f)

            moment = res["max_moment"]
            data["acc"][dataset][embedder][moment] = res["test_acc"]
            data["auc"][dataset][embedder][moment] = res["test_auc"]

# -------- markdown table helper --------
def make_md_table(metric_name, metric_key):
    headers = ["Dataset"]
    for emb in EMBEDDERS:
        for m in MOMENTS:
            headers.append(f"{emb}/M{m}")

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for dataset in sorted(data[metric_key].keys()):
        row = [dataset]
        for emb in EMBEDDERS:
            for m in MOMENTS:
                val = data[metric_key][dataset][emb].get(m, "")
                if isinstance(val, float):
                    val = f"{val:.4f}"
                row.append(val)
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)

# -------- output --------
print("\n## Test Accuracy\n")
print(make_md_table("Test Accuracy", "acc"))

print("\n## Test AUC\n")
print(make_md_table("Test AUC", "auc"))
