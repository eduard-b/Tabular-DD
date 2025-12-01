# Tabular Dataset Distillation (DM / DC) Experiments  
*A systematic evaluation of Dataset Condensation on real tabular benchmarks*

---

## Overview

This repository implements **Dataset Condensation / Dataset Distillation (DC/DD)** for **tabular datasets**, following the **Distribution Matching (DM / DM-DC)** paradigm:

- Synthetic samples are optimized so that **random networks embed them similarly to real data**.
- Works for **binary and multiclass** tabular datasets.
- Includes baselines:
  - **Random subset**
  - **Herding (k-means centroids per class)**
  - **Full-data upper bound**

We evaluate across **seven real-world classification datasets** (OpenML-based), covering binary, multi-class, balanced, and highly imbalanced scenarios.

---

## Supported Datasets

### Binary  
- Adult (Census Income)  
- Bank Marketing  
- Credit Default  

### Multiclass  
- Airlines  
- Covertype (Forest Cover Type)  
- Dry Bean  
- HIGGS (reduced)

---

## Method Summary

Each experiment compares:

1. **Full Data Training**  
   - A strong MLP classifier trained on all real samples  

2. **Random IPC Baseline**  
   - Uniformly sample *IPC (= images per class)* real points  
   - Serves as a naive lower baseline  

3. **Herding Baseline (k-means centroids)**  
   - Compute k-means clusters per class  
   - Use centroids as synthetic representatives  

4. **Distribution Matching Dataset Condensation (DM)**  
   - Initialize synthetic data as Gaussian noise  
   - Optimize features to match real distribution via:
     - Embeddings from multiple **random MLPs**
     - Matching per-class **mean and variance** in embedding space  
   - IPC = number of synthetic samples per class  

All synthetic datasets are then used to train the same classifier architecture.

---

## Architecture

### Classifier
A multi-layer perceptron with BatchNorm:
```
input_dim → 128 → 64 → 1/num_classes
```

### Embedder Networks
10 randomly initialized MLPs:
```
input_dim → 256 → 256 → 128
```
Embedder weights are **frozen**.

### Training
- Optimizer: Adam  
- LR: 0.05 for DM  
- Epochs: 20 for classifier  
- Metrics:
  - **AUC** for binary datasets  
  - **Accuracy / Macro-F1** for multiclass datasets  

---

## Results

Below is the **complete benchmark table** across all datasets.  
Results use **IPC=10** unless noted.

---

## Summary of Results Across Tabular Datasets

| Dataset    | IPC | #Classes | Full Acc | Full AUC/F1 | Rand Acc | Rand AUC/F1 | Herd Acc | Herd AUC/F1 | DM Acc | DM AUC/F1 |
|-----------|-----|----------|----------|--------------|----------|--------------|----------|--------------|---------|------------|
| adult     | 10   | 2        | —        | **0.8947**   | —        | 0.7319       | —        | **0.8402**   | —       | 0.8250     |
| bank      | 10  | 2        | —        | **0.9155**   | —        | 0.6922       | —        | **0.8461**   | —       | 0.6888     |
| credit    | 10  | 2        | —        | **0.7701**   | —        | 0.6701       | —        | **0.6921**   | —       | 0.6700     |
| airlines  | 10  | >2       | **0.6545** | 0.6344     | 0.5058   | 0.5055       | **0.6133** | 0.6120     | 0.5718 | 0.5655     |
| covertype | 10  | >2       | **0.9103** | 0.8683     | 0.3340   | 0.2804       | **0.5148** | 0.3557     | 0.4753 | 0.3658     |
| drybean   | 10  | >2       | **0.9799** | 0.9569     | 0.7958   | 0.6824       | **0.8271** | 0.7122     | 0.7693 | 0.6678     |
| higgs     | 10  | >2       | **0.7081** | 0.7042     | 0.4988   | 0.4962       | **0.5851** | 0.5842     | 0.5221 | 0.5062     |

(— indicates the dataset is binary, so accuracy isn't directly logged in the original runs.)

---

###  Observations

- Tabular data lacks local structure → harder than CV  
- High imbalance (e.g., Covertype class 4) makes IPC=10 unrealistic  
- DM struggles when:
  - Distributions are highly overlapping
  - Classes are very imbalanced
  - Input dimension is large but non-linear relationships dominate

---

## How to Run

Run a binary dataset:
```bash
python general_DM.py --dataset adult
```

Run a multiclass dataset:
```bash
python general_DM_multiclass.py --dataset drybean
```

---

## Generated Files

- `results/` — JSON results for binary datasets  
- `results_multiclass/` — JSON results for multiclass  
- `all_results_summary.json` — unified summary  
- `results_table.md` — Markdown table for README

---

## Citation

If you use this work in your research, consider citing the original DM/DC papers:
- **Distribution Matching for Dataset Condensation** (Zhao et al., 2021)
