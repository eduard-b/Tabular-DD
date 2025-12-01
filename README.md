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




| Dataset   | IPC | Classes | DM-LN AUC | DM-BN AUC | Full AUC | Rand AUC | Herd AUC | embed_hidden | embed_dim | iters | lr   | file-LN | file-BN |
|-----------|-----|---------|-----------|-----------|----------|----------|----------|--------------|-----------|-------|------|---------|---------|
| adult     | 10  | -       | 0.8412    | 0.8293    | 0.8951   | 0.7320   | 0.8403   | 256          | 128       | 2000  | 0.05 | adult_trueDM_ipc10_LayerNorm_h256_e128_it2000_lr0.05_20251201-203506.json | adult_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251201-205306.json |
| airlines  | 10  | -       | 0.6051    | 0.5349    | 0.7006   | 0.5262   | 0.6513   | 256          | 128       | 2000  | 0.05 | airlines_trueDM_ipc10_LayerNorm_h256_e128_it2000_lr0.05_20251201-204026.json | airlines_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251201-205854.json |
| bank      | 10  | -       | 0.8112    | 0.7077    | 0.9151   | 0.6922   | 0.8461   | 256          | 128       | 2000  | 0.05 | bank_trueDM_ipc10_LayerNorm_h256_e128_it2000_lr0.05_20251201-203526.json | bank_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251201-205325.json |
| covertype | 10  | -       | 0.8189    | 0.7964    | 0.9901   | 0.7323   | 0.8237   | 256          | 128       | 2000  | 0.05 | covertype_trueDM_ipc10_LayerNorm_h256_e128_it2000_lr0.05_20251201-203815.json | covertype_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251201-205634.json |
| credit    | 10  | -       | 0.7281    | 0.6571    | 0.7703   | 0.6701   | 0.6921   | 256          | 128       | 2000  | 0.05 | credit_trueDM_ipc10_LayerNorm_h256_e128_it2000_lr0.05_20251201-203539.json | credit_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251201-205340.json |
| drybean   | 10  | -       | 0.9735    | 0.9757    | 0.9996   | 0.9695   | 0.9771   | 256          | 128       | 2000  | 0.05 | drybean_trueDM_ipc10_LayerNorm_h256_e128_it2000_lr0.05_20251201-203444.json | drybean_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251201-205245.json |
| higgs     | 10  | -       | 0.5770    | 0.5310    | 0.7865   | 0.5080   | 0.6237   | 256          | 128       | 2000  | 0.05 | higgs_trueDM_ipc10_LayerNorm_h256_e128_it2000_lr0.05_20251201-204057.json | higgs_trueDM_ipc10_BatchNorm_h256_e128_it2000_lr0.05_20251201-205928.json |
