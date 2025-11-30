import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit

def print_class_distribution(name, y):
    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    print(f"\n=== Class Distribution: {name} ===")
    for c in np.unique(y_np):
        print(f"  Class {c}: {np.sum(y_np == c)} samples")


def stratified_train_val_test_split(X, y, test_size=0.3, val_size=0.5, seed=42, max_tries=20):
    rng = np.random.RandomState(seed)

    for attempt in range(max_tries):
        # first: train vs temp
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=rng.randint(0, 10**6))
        train_idx, temp_idx = next(sss1.split(X, y))

        # then: val vs test from temp
        y_temp = y[temp_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=rng.randint(0, 10**6))
        val_idx, test_idx = next(sss2.split(X[temp_idx], y_temp))

        y_train, y_val, y_test = y[train_idx], y_temp[val_idx], y_temp[test_idx]

        # ensure all splits contain both classes
        if (np.unique(y_train).shape[0] == 2 and
            np.unique(y_val).shape[0] == 2 and
            np.unique(y_test).shape[0] == 2):
            X_train = X[train_idx]
            X_val   = X[temp_idx][val_idx]
            X_test  = X[temp_idx][test_idx]
            return X_train, X_val, X_test, y_train, y_val, y_test

    # If we fail max_tries times, just fall back to the last attempt
    print("[prepare_bank] WARNING: could not guarantee both classes in all splits.")
    X_train = X[train_idx]
    X_val   = X[temp_idx][val_idx]
    X_test  = X[temp_idx][test_idx]
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_db(config, name):

    seed = config["random_seed"]
    device = config["device"]

    if name == "adult":
        return prepare_adult(random_seed=seed, device=device)

    elif name in ["bank", "bank_marketing", "bank-marketing"]:
        return prepare_bank(random_seed=seed, device=device)

    elif name in ["credit", "credit_default", "default_credit"]:
        return prepare_credit_default(random_seed=seed, device=device)

    elif name.lower() == "drybean":
        return prepare_drybean(seed, device)

    else:
        raise ValueError(f"Unknown dataset: {name}")

def prepare_credit_default(random_seed=42, device="cpu"):
    print("Loading Credit Default dataset...")

    credit = fetch_openml("default-of-credit-card-clients", version=1, as_frame=True)
    df = credit.frame

    # Target is simply "y"
    y = df["y"].astype(np.float32).values
    X_df = df.drop(columns=["y"])

    # No real categorical columns preserved, so no one-hot:
    # OpenML version is already numeric-encoded!
    X = X_df.values.astype(np.float32)
    feature_dim = X.shape[1]

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_seed, stratify=y_temp
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device),
        "X_val":   torch.tensor(X_val,   device=device),
        "y_val":   torch.tensor(y_val,   device=device),
        "X_test":  torch.tensor(X_test,  device=device),
        "y_test":  torch.tensor(y_test,  device=device),
        "input_dim": feature_dim,
        "num_classes": 2,
    }


def prepare_bank(random_seed=42, device="cpu"):
    print("Loading Bank Marketing dataset ...")
    bank = fetch_openml("bank-marketing", version=1, as_frame=True)

    X_df = bank.data
    y_series = bank.target

    y = (y_series.astype(str) == "yes").astype(int).values

    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns
    X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=False)
    X = X_df.values.astype(np.float32)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
        X, y, test_size=0.3, val_size=0.5, seed=random_seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.float32),
        "X_val":   torch.tensor(X_val,   device=device),
        "y_val":   torch.tensor(y_val,   device=device, dtype=torch.float32),
        "X_test":  torch.tensor(X_test,  device=device),
        "y_test":  torch.tensor(y_test,  device=device, dtype=torch.float32),
        "input_dim": X_train.shape[1],
        "num_classes": 2,
    }


def prepare_drybean(random_seed=42, device="cpu"):
    """
    Loads the anonymized Dry Bean dataset from OpenML (ID 43585).
    Fixes categorical one-hot label columns, removes missing classes,
    converts to integer labels, stratifies splits, standardizes features,
    and returns PyTorch tensors in your standard format.
    """

    print("Loading Dry Bean dataset (OpenML)…")
    from sklearn.datasets import fetch_openml
    dry = fetch_openml("beans", as_frame=True)

    df = dry.frame.copy()
    print("Columns:", df.columns.tolist())

    # ---------------------------------------
    # Identify feature columns (A1..A16)
    # ---------------------------------------
    feature_cols = [c for c in df.columns if c.startswith("A")]

    # ---------------------------------------
    # Identify label columns (L1..L7)
    # ---------------------------------------
    label_cols = [c for c in df.columns if c.startswith("L")]
    if len(label_cols) == 0:
        raise ValueError("Dry Bean dataset: no L* label columns found.")

    # ---------------------------------------
    # FIX: Convert categorical one-hot -> numeric (0/1)
    # ---------------------------------------
    for c in label_cols:
        df[c] = df[c].astype(int)

    # ---------------------------------------
    # Remove empty label columns (missing classes)
    # ---------------------------------------
    real_label_cols = [c for c in label_cols if df[c].sum() > 0]

    missing = [c for c in label_cols if c not in real_label_cols]
    if missing:
        print("⚠ Warning: Missing label classes removed:", missing)

    print("Using label columns:", real_label_cols)

    # ---------------------------------------
    # Extract X and y
    # ---------------------------------------
    X = df[feature_cols].values.astype(np.float32)

    # One-hot → class index
    Y_onehot = df[real_label_cols].values.astype(np.int64)
    y = np.argmax(Y_onehot, axis=1).astype(np.int64)

    num_classes = len(real_label_cols)
    input_dim = X.shape[1]

    print(f"Dry Bean: {len(X)} samples | {num_classes} classes | dim={input_dim}")

    # ---------------------------------------
    # Train/val/test split (stratified)
    # ---------------------------------------
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_seed, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_seed, stratify=y_temp
    )

    # ---------------------------------------
    # Standardize
    # ---------------------------------------
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # ---------------------------------------
    # Torch tensors
    # ---------------------------------------
    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)

    X_val_t   = torch.tensor(X_val, device=device)
    y_val_t   = torch.tensor(y_val, device=device)

    X_test_t  = torch.tensor(X_test, device=device)
    y_test_t  = torch.tensor(y_test, device=device)

    return {
        "X_train": X_train_t,
        "y_train": y_train_t,
        "X_val":   X_val_t,
        "y_val":   y_val_t,
        "X_test":  X_test_t,
        "y_test":  y_test_t,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "class_names": real_label_cols,  # L-columns actually present
    }



def prepare_adult(random_seed=42, device="cpu"):
    print("Loading Adult dataset (OpenML)...")
    adult = fetch_openml("adult", version=2, as_frame=True)

    X_df: pd.DataFrame = adult.data
    y_series: pd.Series = adult.target

    y = (y_series.astype(str).str.contains(">50K")).astype(np.float32).values

    print("One-hot encoding...")
    X_df = pd.get_dummies(X_df, drop_first=False)
    X = X_df.values.astype(np.float32)

    feature_dim = X.shape[1]
    print(f"Feature dim after one-hot = {feature_dim}")

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_seed, stratify=y_temp
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # To tensors
    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)
    X_val_t   = torch.tensor(X_val,   device=device)
    y_val_t   = torch.tensor(y_val,   device=device)
    X_test_t  = torch.tensor(X_test,  device=device)
    y_test_t  = torch.tensor(y_test,  device=device)

    return {
        "X_train": X_train_t,
        "y_train": y_train_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_test": X_test_t,
        "y_test": y_test_t,
        "input_dim": feature_dim,
        "num_classes": 2,
    }