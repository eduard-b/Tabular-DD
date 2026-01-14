import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Optional, List


# ----------------------------
# Config (edit these if needed)
# ----------------------------
DEFAULT_METRICS = ["test_acc", "test_auc"]  # keys inside result.json
JSON_NAME = "result.json"

# Folder names that imply a "variant" (moment/repeat/etc.)
# We’ll try to parse an order key from these (e.g., M3 -> 3).
VARIANT_PATTERNS = [
    re.compile(r"^M(\d+)$", re.IGNORECASE),        # M1, M2, ...
    re.compile(r"^moment[_-]?(\d+)$", re.IGNORECASE),
    re.compile(r"^(\d+)$"),                        # 1, 2, 3, 4
]


def parse_variant_name(name: str) -> Tuple[str, Optional[int]]:
    """
    Return (variant_label, variant_order) from a directory name.
    If it doesn't match known patterns, return label=name, order=None.
    """
    for pat in VARIANT_PATTERNS:
        m = pat.match(name)
        if m:
            num = int(m.group(1))
            # canonical label: M<num> if it came from M or numeric
            label = f"M{num}" if not name.lower().startswith("moment") else f"M{num}"
            return label, num
    return name, None


def try_read_json(p: Path) -> Optional[Dict]:
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def discover_results(root: Path, json_name: str = JSON_NAME) -> Dict:
    """
    Returns nested dict:
    results[dataset][embedder][variant_label] = json_dict
    where variant_label is "__root__" if embedder has direct result.json.
    """
    results = defaultdict(lambda: defaultdict(dict))

    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for embedder_dir in dataset_dir.iterdir():
            if not embedder_dir.is_dir():
                continue
            embedder = embedder_dir.name

            # Case A: direct result.json under embedder/
            direct = embedder_dir / json_name
            if direct.exists():
                js = try_read_json(direct)
                if js is not None:
                    results[dataset][embedder]["__root__"] = js

            # Case B: variant folders under embedder/ (e.g. M1/M2/..., 1/2/..., moment_1/...)
            for child in embedder_dir.iterdir():
                if not child.is_dir():
                    continue
                jf = child / json_name
                if not jf.exists():
                    continue
                js = try_read_json(jf)
                if js is None:
                    continue

                variant_label, _ = parse_variant_name(child.name)
                results[dataset][embedder][variant_label] = js

    return results


def sort_variants(variants: List[str]) -> List[str]:
    """
    Sort variants with numeric order when possible: M1, M2, ... then others.
    """
    parsed = []
    for v in variants:
        if v == "__root__":
            parsed.append((0, -1, v))
            continue
        label, order = parse_variant_name(v)
        if order is None:
            parsed.append((2, 10**9, v))
        else:
            parsed.append((1, order, f"M{order}"))
    # keep label stable
    parsed = sorted(set(parsed))
    return [t[2] for t in parsed]


def fmt_val(x):
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return f"{x:.4f}"
    return str(x)


def make_md_table(results: Dict, metric: str, bold_best_per_row: bool = True) -> str:
    # collect global columns
    datasets = sorted(results.keys())
    embedders = sorted({e for d in results.values() for e in d.keys()})

    # collect variants per embedder (union across datasets)
    variants_by_embedder = {}
    for e in embedders:
        vs = set()
        for ds in datasets:
            vs |= set(results[ds].get(e, {}).keys())
        variants_by_embedder[e] = sort_variants(list(vs))

    # build column headers (embedder/variant) (if __root__ -> embedder)
    columns = []
    for e in embedders:
        for v in variants_by_embedder[e]:
            col = e if v == "__root__" else f"{e}/{v}"
            columns.append((e, v, col))

    header = ["Dataset"] + [c[2] for c in columns]

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for ds in datasets:
        # gather row values
        row_vals = []
        numeric_vals = []

        for (e, v, _) in columns:
            js = results[ds].get(e, {}).get(v)
            val = js.get(metric) if isinstance(js, dict) else None
            row_vals.append(val)
            numeric_vals.append(val if isinstance(val, (int, float)) else None)

        # compute best index (max) among numeric values
        best_idx = None
        if bold_best_per_row:
            best_val = None
            for i, val in enumerate(numeric_vals):
                if val is None:
                    continue
                if best_val is None or val > best_val:
                    best_val = val
                    best_idx = i

        # render row
        rendered = [ds]
        for i, val in enumerate(row_vals):
            s = fmt_val(val) if val is not None else ""
            if bold_best_per_row and best_idx is not None and i == best_idx and s != "":
                s = f"**{s}**"
            rendered.append(s)

        lines.append("| " + " | ".join(rendered) + " |")

    return "\n".join(lines)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=str, help="Path to results directory")
    ap.add_argument("--json-name", type=str, default=JSON_NAME, help="Results JSON filename (default: result.json)")
    ap.add_argument("--metrics", type=str, nargs="+", default=DEFAULT_METRICS,
                    help="Metric keys to tabulate (default: test_acc test_auc)")
    ap.add_argument("--no-bold", action="store_true", help="Disable bolding best-per-row")
    args = ap.parse_args()

    root = Path(args.results_dir)
    if not root.exists():
        raise FileNotFoundError(f"Results dir not found: {root}")

    results = discover_results(root, json_name=args.json_name)

    if not results:
        print("No results found. Expected structure: results_dir/<dataset>/<embedder>/[variant]/result.json")
        return

    for metric in args.metrics:
        title = metric.replace("_", " ").title()
        print(f"\n## {title}\n")
        print(make_md_table(results, metric=metric, bold_best_per_row=not args.no_bold))


if __name__ == "__main__":
    main()
