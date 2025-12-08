# True Dataset Condensation for Tabular Data (True_DM)

This repository contains an implementation of a principled, high-fidelity variant of Dataset Condensation, adapted for tabular data. The method implemented here is referred to as **True_DM**, and is inspired directly by the logic of the original Dataset Condensation paper (Zhao et al., 2021) while correcting several limitations in the commonly circulated PyTorch reference implementation for image data.

The goal of Dataset Condensation is to produce a *very small synthetic dataset* that, when used to train a classifier, yields accuracy comparable to the full training set. For tabular tasks—especially with heterogeneous feature types—careful adaptation is required to obtain meaningful condensed datasets. This README provides an overview of the method, its rationale, and explains why True_DM is a more accurate translation of the original algorithm than the DM script that the project initially relied on.

---

## 1. Overview

True_DM is a full re-implementation of the core Dataset Condensation logic for tabular datasets. It includes:

• Classifier-agnostic synthetic sample optimization.  
• Fully differentiable condensation of per-class sample statistics.  
• Fresh random neural networks (embedders) sampled at every optimization step.  
• Mean feature matching in embedder space, following the original DM objective.  
• Per-class synthetic batches optimized using SGD with momentum.  
• Multiclass-safe baselines for Random IPC and Herding.  
• Support for both LayerNorm and BatchNorm embedders.  
• Multi-dataset evaluation loop with unified result logging.

The method condenses a dataset of size *N* into a synthetic dataset of size *C × IPC* where *C* is the number of classes and *IPC* is the number of synthetic samples per class.

---

## 2. Method: How True_DM Works

True_DM mimics the logic of the original Dataset Condensation algorithm, but removes several shortcuts and simplifications that were present in prior PyTorch implementations meant for image data.

For each condensation iteration:

1. A **fresh randomly initialized embedder network** is sampled.  
2. For each real class *c*:  
   - A batch of real samples is drawn.  
   - The embedder extracts features for the real batch.  
3. For each synthetic class *c*:  
   - The embedder extracts features from the current synthetic samples.  
4. The loss is computed as the **mean squared error between the class-wise feature means** of real vs synthetic samples.  
5. The synthetic data is updated with SGD through the embedder (which remains frozen).

This process is repeated for a fixed number of iterations, producing a condensed dataset that transfers well to downstream classifiers.

---

## 3. Why True_DM Is More Accurate Than the Earlier DM Script

The original project used a script adapted from the public GitHub implementation of Dataset Condensation for image classification (the code that trains ConvNets on CIFAR-10). That code works well for images, but its internal logic included several assumptions that are not appropriate for tabular data or even for faithful reproduction of the method’s theory.

True_DM corrects these issues:

### 3.1. Fresh Embedders per Iteration
The original algorithm samples a *new random network every iteration*.  
Public DM implementations often reuse a fixed pool of networks or even a single network for all updates, reducing variability and weakening the theoretical justification for feature matching.

True_DM restores the intended “fresh random function per iteration” logic.

### 3.2. Proper Logit Flow, Loss Functions, and Label Handling
Earlier scripts mixed:
• BCE instead of BCEWithLogitsLoss  
• Sigmoid in the forward pass  
• Float labels for multiclass data  
• Non-logit outputs for CE loss

True_DM normalizes all classification and evaluation paths:  
• logits only  
• BCEWithLogitsLoss for binary  
• CrossEntropyLoss for multiclass  
• correct probabilities for ROC-AUC computation  

This prevents numerical errors and CUDA asserts.

### 3.3. Image-specific Assumptions Removed
The reference implementation uses:
• ConvNets  
• BN behavior tuned for images  
• augmentations that do not exist for tabular data  
• hard-coded embedder shapes  

True_DM implements tabular-appropriate embedders with LayerNorm or BatchNorm and avoids any image-only operations.

### 3.4. Multiclass-Safe Baselines
The original baselines (IPC-random, IPC-herding) assumed binary or image-class datasets.  
True_DM generalizes both to arbitrary class counts.

### 3.5. Correct Synthetic Initialization Strategy
The reference implementation includes complex initialization options ("real", "noise") but also makes assumptions about 2D spatial structure.  
True_DM simplifies this to tabular-appropriate initialization while maintaining faithfulness to the DM objective.

---

## 4. Usage

True_DM is driven by a single experiment engine: run_dm_true_experiment(config)

A multi-dataset loop is provided in `__main__`, automatically running the algorithm on:

• adult  
• bank marketing  
• credit default  
• drybean  
• covertype  
• airlines  
• higgs  

All results are written as `*.json` files into `results_trueDM/`.

---

## 5. Switching Between LayerNorm and BatchNorm

The embedder used in the condensation step can be switched in the configuration:

- "dm_use_batchnorm": False # LayerNorm (recommended for tabular)
-   "dm_use_batchnorm": True # BatchNorm (closer to original DM)

LayerNorm typically performs better for tabular data because BatchNorm relies on batch statistics that may not be representative in low-IPC settings.

---

## 6. Results

Below is the space where the aggregated results table can be pasted.  
The parser script `parse_results_trueDM.py` produces a Markdown summary from all JSON files in the results directory.





| Dataset | IPC | Classes | Norm | Full AUC | Rand AUC | Herd AUC | DM AUC | embed_hidden | embed_dim | iters | lr | file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adult | 10 | - | LayerNorm | 0.8951 | 0.7320 | 0.8403 | 0.8412 | 256 | 128 | 2000 | 0.05 | adult_trueDM_ipc10_LayerNorm_h256_e128_it2000_lr0.05_20251201-203506.json |
| airlines | 10 | - | LayerNorm | 0.7006 | 0.5262 | 0.6513 | 0.6051 | 256 | 128 | 2000 | 0.05 | airlines_trueDM_ipc10_LayerNorm_h256_e128_it2000_lr0.05_20251201-204026.json |
| bank | 10 | - | LayerNorm | 0.9151 | 0.6922 | 0.8461 | 0.8112 | 256 | 128 | 2000 | 0.05 | bank_trueDM_ipc10_LayerNorm_h256_e128_it2000_lr0.05_20251201-203526.json |
| covertype | 10 | - | LayerNorm | 0.9901 | 0.7323 | 0.8237 | 0.8189 | 256 | 128 | 2000 | 0.05 | covertype_trueDM_ipc10_LayerNorm_h256_e128_it2000_lr0.05_20251201-203815.json |
| credit | 10 | - | LayerNorm | 0.7703 | 0.6701 | 0.6921 | 0.7281 | 256 | 128 | 2000 | 0.05 | credit_trueDM_ipc10_LayerNorm_h256_e128_it2000_lr0.05_20251201-203539.json |
| drybean | 10 | - | LayerNorm | 0.9996 | 0.9695 | 0.9771 | 0.9735 | 256 | 128 | 2000 | 0.05 | drybean_trueDM_ipc10_LayerNorm_h256_e128_it2000_lr0.05_20251201-203444.json |
| higgs | 10 | - | LayerNorm | 0.7865 | 0.5080 | 0.6237 | 0.5770 | 256 | 128 | 2000 | 0.05 | higgs_trueDM_ipc10_LayerNorm_h256_e128_it2000_lr0.05_20251201-204057.json |

| Dataset | IPC | Classes | Norm | Full AUC | Rand AUC | Herd AUC | DM AUC | embed_hidden | embed_dim | iters | lr | file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adult | 10 | - | BatchNorm | 0.8951 | 0.7320 | 0.8403 | 0.8293 | 256 | 128 | 2000 | 0.05 | adult_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251201-205306.json |
| airlines | 10 | - | BatchNorm | 0.7006 | 0.5262 | 0.6513 | 0.5349 | 256 | 128 | 2000 | 0.05 | airlines_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251201-205854.json |
| bank | 10 | - | BatchNorm | 0.9151 | 0.6922 | 0.8461 | 0.7077 | 256 | 128 | 2000 | 0.05 | bank_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251201-205325.json |
| covertype | 10 | - | BatchNorm | 0.9901 | 0.7323 | 0.8238 | 0.7964 | 256 | 128 | 2000 | 0.05 | covertype_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251201-205634.json |
| credit | 10 | - | BatchNorm | 0.7703 | 0.6701 | 0.6921 | 0.6571 | 256 | 128 | 2000 | 0.05 | credit_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251201-205340.json |
| drybean | 10 | - | BatchNorm | 0.9996 | 0.9695 | 0.9770 | 0.9757 | 256 | 128 | 2000 | 0.05 | drybean_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251201-205245.json |
| higgs | 10 | - | BatchNorm | 0.7865 | 0.5080 | 0.6237 | 0.5310 | 256 | 128 | 2000 | 0.05 | higgs_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251201-205928.json |

### Alternative BN

| Dataset | IPC | Classes | Norm | Full AUC | Rand AUC | Herd AUC | DM AUC | embed_hidden | embed_dim | iters | lr | file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adult | 10 | - | BatchNorm | 0.8951 | 0.7320 | 0.8403 | 0.8266 | 256 | 128 | 2000 | 0.05 | adult_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251202-180557.json |
| airlines | 10 | - | BatchNorm | 0.7006 | 0.5262 | 0.6513 | 0.5345 | 256 | 128 | 2000 | 0.05 | airlines_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251202-181113.json |
| bank | 10 | - | BatchNorm | 0.9151 | 0.6922 | 0.8461 | 0.7098 | 256 | 128 | 2000 | 0.05 | bank_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251202-180613.json |
| covertype | 10 | - | BatchNorm | 0.9901 | 0.7323 | 0.8236 | 0.7969 | 256 | 128 | 2000 | 0.05 | covertype_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251202-180900.json |
| credit | 10 | - | BatchNorm | 0.7703 | 0.6701 | 0.6921 | 0.6554 | 256 | 128 | 2000 | 0.05 | credit_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251202-180626.json |
| drybean | 10 | - | BatchNorm | 0.9996 | 0.9695 | 0.9771 | 0.9756 | 256 | 128 | 2000 | 0.05 | drybean_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251202-180539.json |
| higgs | 10 | - | BatchNorm | 0.7865 | 0.5080 | 0.6237 | 0.5295 | 256 | 128 | 2000 | 0.05 | higgs_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251202-181143.json |

---

## Batch-Norm statistics

Recently, dataset condensation via matching internal model statistics has gained traction. For example, SRe²L (Squeeze, Recover and Relabel) – a NeurIPS 2023 method — demonstrates that by first training a neural network on the full dataset, then synthesizing new data by aligning the network’s BatchNorm running mean/variance on synthetic inputs with those from real data, one can recover a compact distilled dataset that — despite being orders of magnitude smaller — preserves sufficient information for training high-accuracy models. 

In the “Recover” phase, SRe²L optimizes synthetic inputs (initialized from noise) under a loss combining BN-stat matching, soft-label consistency, and simple regularizers (e.g. ℓ₂ / total-variation), producing a condensed set that yields strong performance even on large datasets such as ImageNet-1K. 

Our tabular DM implementation follows a similar high-level principle — by embedding data via a neural network (optionally using BatchNorm), and matching class-wise feature moments between real and synthetic sets — adapted to the case of tabular data rather than images. This domain adaptation avoids image-specific regularization and augmentation, but retains the essential idea of distributional matching in a learned embedding space

## Embedder Ablation Study

To assess how the choice of feature extractor (“embedder”) affects tabular dataset distillation, we conduct an ablation over six architectures. During DM synthesis, the embedder is a fixed, randomly initialized network used only to compute the feature-matching loss; its parameters are not trained.

We evaluate the following embedders:

- LN — 2-layer MLP with LayerNorm (baseline)
- BN — same architecture with BatchNorm
- BNDeep — deeper BN stack (more nonlinear layers)
- BNWide — wide BN network (higher capacity)
- BNRes — residual MLP with BN blocks
- BNCascade — BN-heavy cascade (most expressive)

These models differ in width, depth, normalization, and residual structure, providing a range of embedding geometries.

For each dataset and each embedder, we run the standard Distribution Matching (DM) objective using only per-class feature-mean matching. No BatchNorm-statistics loss is used here; this isolates the effect of embedder architecture alone. Each distilled dataset is then used to train a classifier, and test AUC is reported. Full-data, Random IPC, and Herding IPC baselines are included for comparison.

Results from all runs are saved as JSON files and aggregated into a unified table with columns:

Dataset, IPC, Classes, Norm, Full AUC, Rand AUC, Herd AUC, DM AUC, embed_hidden, embed_dim, iters, lr, file.

This ablation reveals how the choice of embedder influences distilled-data quality on tabular datasets, providing a clean foundation for evaluating more advanced objectives such as BatchNorm-statistics matching.

| Dataset   | Full AUC  | Rand AUC  | Herd AUC  | LN AUC   | BN AUC   | BNDeep AUC | BNWide AUC | BNRes AUC | BNCascade AUC |
|-----------|-----------|-----------|-----------|----------|----------|------------|------------|-----------|----------------|
| adult     | 0.895058  | 0.731950  | 0.840260  | 0.843080 | 0.828890 | 0.823710   | 0.821695   | 0.839648  | 0.821203       |
| airlines  | 0.700636  | 0.526157  | 0.651348  | 0.601879 | 0.533718 | 0.498597   | 0.495686   | 0.595988  | 0.491306       |
| bank      | 0.915145  | 0.692192  | 0.846062  | 0.811742 | 0.710006 | 0.675850   | 0.676163   | 0.802547  | 0.668721       |
| covertype | 0.990084  | 0.732284  | 0.823773  | 0.817537 | 0.796904 | 0.786684   | 0.786699   | 0.822952  | 0.784098       |
| credit    | 0.770316  | 0.670140  | 0.692166  | 0.731576 | 0.657193 | 0.633681   | 0.633513   | 0.696522  | 0.630608       |
| drybean   | 0.999642  | 0.969517  | 0.977058  | 0.973469 | 0.975577 | 0.978515   | 0.978541   | 0.973593  | 0.978011       |
| higgs     | 0.786457  | 0.508003  | 0.623686  | 0.577478 | 0.531963 | 0.512929   | 0.512947   | 0.562216  | 0.512469       |

-----

BN DM results

| Dataset   |   Full AUC |   Rand AUC |   Herd AUC |   DM-BN AUC | file                      |
|:----------|-----------:|-----------:|-----------:|------------:|:--------------------------|
| adult     |   0.895058 |   0.73195  |   0.840265 |    0.737701 | adult_dmBN_ipc10.json     |
| airlines  |   0.700636 |   0.526157 |   0.651348 |    0.523843 | airlines_dmBN_ipc10.json  |
| bank      |   0.915145 |   0.692192 |   0.846069 |    0.594989 | bank_dmBN_ipc10.json      |
| covertype |   0.990084 |   0.732284 |   0.823724 |    0.691984 | covertype_dmBN_ipc10.json |
| credit    |   0.770316 |   0.67014  |   0.692119 |    0.620799 | credit_dmBN_ipc10.json    |
| drybean   |   0.999642 |   0.969517 |   0.977057 |    0.936522 | drybean_dmBN_ipc10.json   |
| higgs     |   0.786457 |   0.508003 |   0.623692 |    0.535829 | higgs_dmBN_ipc10.json     |

LN ablation results

| Dataset   |   Full AUC |   Rand AUC | Herd AUC     | LN AUC     | LNDeep AUC   | LNWide AUC   | LNRes AUC    | LNCascade AUC   |
|:----------|-----------:|-----------:|:-------------|:-----------|:-------------|:-------------|:-------------|:----------------|
| adult     |   0.895058 |   0.73195  | 0.840271     | 0.842904   | 0.844155     | *0.844245*   | **0.845593** | 0.840814        |
| airlines  |   0.700636 |   0.526157 | **0.651348** | 0.605764   | 0.595468     | 0.603687     | *0.616599*   | 0.586157        |
| bank      |   0.915145 |   0.692192 | **0.846056** | *0.810129* | 0.809082     | 0.805788     | 0.808435     | 0.783345        |
| covertype |   0.990084 |   0.732284 | **0.823756** | 0.817764   | 0.815849     | 0.817965     | *0.821923*   | 0.809592        |
| credit    |   0.770316 |   0.67014  | 0.692139     | 0.727118   | **0.731637** | 0.728462     | *0.730588*   | 0.726402        |
| drybean   |   0.999642 |   0.969517 | **0.977028** | 0.973334   | 0.973476     | 0.973606     | 0.973151     | *0.974667*      |
| higgs     |   0.786457 |   0.508003 | **0.623686** | 0.579255   | 0.578725     | 0.578672     | *0.582558*   | 0.577789        |