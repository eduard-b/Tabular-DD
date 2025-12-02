import os
import json
import argparse
from glob import glob

def parse_filename(fname):
    """
    Expected pattern:
    {dataset}_trueDM_ipc{ipc}_{Norm}embed_h{hidden}_e{embed}_it{iters}_lr{lr}_{timestamp}.json
    """
    base = os.path.basename(fname).replace(".json", "")
    parts = base.split("_")

    dataset = parts[0]
    ipc = int(parts[2].replace("ipc", ""))
    norm = parts[3]                 # LayerNorm / BatchNorm
    hidden = int(parts[4].replace("h", ""))
    embed_dim = int(parts[5].replace("e", ""))
    iters = int(parts[6].replace("it", ""))
    lr = float(parts[7].replace("lr", ""))

    return dataset, ipc, norm, hidden, embed_dim, iters, lr


def main(results_dir):
    files = sorted(glob(os.path.join(results_dir, "*.json")))
    if not files:
        print("No JSON files found in directory.")
        return

    rows = []

    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)

        dataset, ipc, norm, hdim, edim, iters, lr = parse_filename(f)

        # Try reading number of classes (optional)
        try:
            n_classes = data["full_config"]["num_classes"]
        except:
            n_classes = "-"

        R = data["results"]

        row = {
            "Dataset": dataset,
            "IPC": ipc,
            "Classes": n_classes,
            "Norm": norm.replace("embed", ""),  # cleaner
            "Full AUC": R["full"]["test_auc"],
            "Rand AUC": R["random"]["test_auc"],
            "Herd AUC": R["herd"]["test_auc"],
            "DM AUC": R["dm"]["test_auc"],
            "embed_hidden": hdim,
            "embed_dim": edim,
            "iters": iters,
            "lr": lr,
            "file": os.path.basename(f),
        }

        rows.append(row)

    # Write Markdown summary
    out_md = os.path.join(results_dir, "summary_auc_only.md")
    with open(out_md, "w") as md:

        md.write("| Dataset | IPC | Classes | Norm | Full AUC | Rand AUC | Herd AUC | DM AUC | embed_hidden | embed_dim | iters | lr | file |\n")
        md.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")

        for r in rows:
            md.write(
                f"| {r['Dataset']} | {r['IPC']} | {r['Classes']} | {r['Norm']} | "
                f"{r['Full AUC']:.4f} | {r['Rand AUC']:.4f} | {r['Herd AUC']:.4f} | {r['DM AUC']:.4f} | "
                f"{r['embed_hidden']} | {r['embed_dim']} | {r['iters']} | {r['lr']} | {r['file']} |\n"
            )

    print(f"Markdown table saved to: {out_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./results_trueDM_BN_2", help="Directory with JSON runs.")
    args = parser.parse_args()

    main(args.dir)
