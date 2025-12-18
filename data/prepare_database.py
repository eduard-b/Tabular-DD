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

def prepare_bank_marketing(random_seed=42, device="cpu"):
    """
    Loads the Bank Marketing dataset from OpenML.
    Handles both textual and numeric label formats.
    Returns normalized tensors ready for the DM pipeline.
    """

    print("Loading Bank Marketing dataset ...")
    bank = fetch_openml("bank-marketing", version=1, as_frame=True)

    X_df = bank.data
    y_series = bank.target

    # -----------------------------------------------------
    # FIX: Robust label parser (handles yes/no OR 1/2)
    # -----------------------------------------------------
    raw = y_series.astype(str)
    uniques = set(np.unique(raw))
    print("Raw label values:", uniques)

    # Text labels ("yes"/"no")
    if uniques <= {"yes", "no"}:
        print("Detected yes/no → mapping: yes=1, no=0")
        y = (raw == "yes").astype(int).values

    # Numeric labels ("1"/"2")
    elif uniques <= {"1", "2"}:
        print("Detected numeric labels 1/2 → mapping: 2=1, 1=0")
        y = (raw == "2").astype(int).values

    else:
        raise ValueError(f"Unknown bank-marketing label format: {uniques}")

    # -----------------------------------------------------
    # Handle categorical features → One-hot encode
    # -----------------------------------------------------
    print("One-hot encoding categorical features...")
    X_df = pd.get_dummies(X_df, drop_first=False)
    X = X_df.values.astype(np.float32)
    input_dim = X.shape[1]
    num_classes = 2

    print(f"Bank Marketing: {len(X)} samples | dim = {input_dim}")

    # -----------------------------------------------------
    # Train/Val/Test split (stratified)
    # -----------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_seed, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_seed, stratify=y_temp
    )

    # Show class balance
    print("Train class distribution:", np.bincount(y_train))
    print("Val class distribution:  ", np.bincount(y_val))
    print("Test class distribution: ", np.bincount(y_test))

    # -----------------------------------------------------
    # Standardize features
    # -----------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # -----------------------------------------------------
    # Convert to tensors
    # -----------------------------------------------------
    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)

    X_val_t   = torch.tensor(X_val,   device=device)
    y_val_t   = torch.tensor(y_val,   device=device)

    X_test_t  = torch.tensor(X_test,  device=device)
    y_test_t  = torch.tensor(y_test,  device=device)

    # -----------------------------------------------------
    # Output standardized dictionary
    # -----------------------------------------------------
    return {
        "X_train": X_train_t,
        "y_train": y_train_t,
        "X_val":   X_val_t,
        "y_val":   y_val_t,
        "X_test":  X_test_t,
        "y_test":  y_test_t,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "class_names": ["no", "yes"],  # your canonical mapping
    }

def prepare_drybean(random_seed=42, device="cpu"):
    """
    Loads the anonymized Dry Bean dataset from OpenML (ID="beans").
    Fixes one-hot labels, removes unused label columns, compresses class indices,
    stratifies train/val/test splits, standardizes features,
    and returns tensors in your standard format.
    """

    print("Loading Dry Bean dataset (OpenML)…")
    dry = fetch_openml("beans", as_frame=True)
    df = dry.frame.copy()

    # ---------------------------------------
    # Feature columns (A1..A16)
    # ---------------------------------------
    feature_cols = [c for c in df.columns if c.startswith("A")]
    X = df[feature_cols].astype(np.float32).values

    # ---------------------------------------
    # Label columns (L1..L7, but some are empty)
    # ---------------------------------------
    label_cols = [c for c in df.columns if c.startswith("L")]

    # Force numeric 0/1
    for c in label_cols:
        df[c] = df[c].astype(int)

    # Drop columns corresponding to *missing classes*
    real_label_cols = [c for c in label_cols if df[c].sum() > 0]
    missing = [c for c in label_cols if c not in real_label_cols]
    if missing:
        print("⚠ Removed empty label columns:", missing)

    print("Using label columns:", real_label_cols)

    # ---------------------------------------
    # Convert one-hot → raw class labels
    # ---------------------------------------
    Y_onehot = df[real_label_cols].values.astype(np.int64)
    y_raw = np.argmax(Y_onehot, axis=1)   # but indices correspond to L3,L4,... not 0..K-1

    # ---------------------------------------
    # FIX: Compress labels to 0..num_classes-1
    # ---------------------------------------
    unique_raw = np.unique(y_raw)
    mapping = {old: new for new, old in enumerate(unique_raw)}
    y = np.array([mapping[v] for v in y_raw], dtype=np.int64)

    num_classes = len(unique_raw)
    input_dim = X.shape[1]

    print(f"Dry Bean: {len(X)} samples | {num_classes} classes | dim={input_dim}")
    print("Compressed class mapping:", mapping)
    print("Unique labels (after compression):", np.unique(y))

    # ---------------------------------------
    # Stratified train/val/test
    # ---------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    # ---------------------------------------
    # Standardize
    # ---------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # ---------------------------------------
    # Torch tensors
    # ---------------------------------------
    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long),

        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),

        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.long),

        "input_dim": input_dim,
        "num_classes": num_classes,
        "class_names": real_label_cols,  # original L* names
        "class_mapping": mapping,        # compressed mapping
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

def prepare_covertype(random_seed=42, device="cpu"):
    """
    Prepares the Forest Cover Type dataset (OpenML id=150).
    Normalizes features, converts labels 1–7 → 0–6,
    returns tensors for multiclass DM.
    """

    print("Loading Forest Cover Type dataset (OpenML id=150)...")
    from sklearn.datasets import fetch_openml
    ds = fetch_openml("covertype", version=3, as_frame=True)  # v3=normalized

    df = ds.frame.copy()

    print("Columns:", df.columns.tolist())

    # -----------------------------------------
    # Extract features + label
    # -----------------------------------------
    feature_cols = df.columns[:-1]
    label_col = df.columns[-1]

    X = df[feature_cols].values.astype(np.float32)

    # class labels are categorical strings '1'..'7'
    y_raw = df[label_col].astype(str)
    y = y_raw.astype(int) - 1    # ensure 0..6
    y = y.astype(np.int64)

    num_classes = len(np.unique(y))
    input_dim   = X.shape[1]

    print(f"Covertype: {len(X)} samples | {num_classes} classes | dim={input_dim}")
    print("Class distribution:", np.bincount(y))

    # -----------------------------------------
    # Train/val/test split
    # -----------------------------------------
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_seed, stratify=y_temp
    )

    y_train = np.array(y_train, dtype=np.int64)
    y_val   = np.array(y_val, dtype=np.int64)
    y_test  = np.array(y_test, dtype=np.int64)

    # -----------------------------------------
    # Standardize anyway (DM likes standardized inputs)
    # -----------------------------------------
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # -----------------------------------------
    # Convert to tensors
    # -----------------------------------------
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
        "class_names": [f"class_{i}" for i in range(num_classes)],
    }

def prepare_airlines(random_seed=42, device="cpu"):
    """
    Prepares Airlines Delay dataset (OpenML ID 1169)
    for binary classification: Delay in {0,1}.
    """

    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("Loading Airlines dataset (OpenML id=1169)...")
    ds = fetch_openml(data_id=1169, as_frame=True)
    df = ds.frame.copy()

    print("Columns:", df.columns.tolist())

    # --------------------------
    # Target: Delay ("0" / "1")
    # --------------------------
    y_raw = df["Delay"].astype(str)
    print("Target unique:", y_raw.unique())

    y = (y_raw == "1").astype(np.int64).values  # 0/1 labels

    # --------------------------
    # Features: one-hot encode categoricals
    # --------------------------
    X_df = df.drop(columns=["Delay"])

    # Identify categorical columns
    cat_cols = ["Airline", "Flight", "AirportFrom", "AirportTo", "DayOfWeek"]

    # Everything else is numeric
    # (Time, Length)
    print("Categorical columns:", cat_cols)

    # One-hot encode categoricals, keep all dummy cols (no drop_first)
    X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=False)

    X = X_df.values.astype(np.float32)
    input_dim = X.shape[1]
    num_classes = 2

    print(f"Airlines: {len(X)} samples | dim={input_dim}")

    # --------------------------
    # Split
    # --------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_seed, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_seed, stratify=y_temp
    )

    # --------------------------
    # Standardize numeric features
    # --------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # --------------------------
    # Convert to tensors
    # --------------------------
    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device, dtype=torch.long)

    X_val_t   = torch.tensor(X_val, device=device)
    y_val_t   = torch.tensor(y_val, device=device, dtype=torch.long)

    X_test_t  = torch.tensor(X_test, device=device)
    y_test_t  = torch.tensor(y_test, device=device, dtype=torch.long)

    return {
        "X_train": X_train_t,
        "y_train": y_train_t,
        "X_val":   X_val_t,
        "y_val":   y_val_t,
        "X_test":  X_test_t,
        "y_test":  y_test_t,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "class_names": ["no_delay", "delay"],
    }

def prepare_airlines_optimized(random_seed=42, device="cpu"):
    print("Loading Airlines dataset (OpenML id=1169)...")
    ds = fetch_openml(data_id=1169, as_frame=True)
    df = ds.frame.copy()

    # TARGET
    y = (df["Delay"].astype(str) == "1").astype(np.int64).values
    df = df.drop(columns=["Delay"])

    # CATEGORICAL SPLIT
    low_card = ["Airline", "DayOfWeek"]
    high_card = ["Flight", "AirportFrom", "AirportTo"]

    # LOW CARD = ONE-HOT
    df = pd.get_dummies(df, columns=low_card, drop_first=False)

    # HIGH CARD = CATEGORY CODES
    for col in high_card:
        df[col] = df[col].astype("category").cat.codes.astype(np.int32)

    # All remaining columns numeric
    X = df.values.astype(np.float32)
    print("Final feature dim:", X.shape[1])

    # SPLIT + SCALE
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # TORCH
    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.long),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.long),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.long),
        "input_dim": X_train.shape[1],
        "num_classes": 2,
    }

def prepare_higgs(random_seed=42, device="cpu"):
    """
    Loads the HIGGS dataset (OpenML ID 23512).
    Drops NaN rows because KMeans (herding) cannot handle NaN.
    """
    print("Loading HIGGS dataset (OpenML id=23512)...")

    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=23512, as_frame=True)
    df = ds.frame.copy()

    print("Original shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Drop NaNs early
    df = df.dropna()
    print("After dropna:", df.shape)

    # ---------------------------------------
    # Features and target
    # ---------------------------------------
    y = df["class"].astype(int).values
    X = df.drop(columns=["class"]).values.astype(np.float32)

    print(f"HIGGS: {len(X)} samples | dim={X.shape[1]}")
    print("Target distribution:", np.bincount(y))

    # ---------------------------------------
    # Train/val/test split
    # ---------------------------------------
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_seed
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
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
    return {
        "X_train": torch.tensor(X_train, device=device),
        "y_train": torch.tensor(y_train, device=device, dtype=torch.float32),
        "X_val":   torch.tensor(X_val, device=device),
        "y_val":   torch.tensor(y_val, device=device, dtype=torch.float32),
        "X_test":  torch.tensor(X_test, device=device),
        "y_test":  torch.tensor(y_test, device=device, dtype=torch.float32),
        "input_dim": X_train.shape[1],
        "num_classes": 2,
    }

DATASET_REGISTRY = {
    "adult": prepare_adult,
    "bank": prepare_bank_marketing,
    "credit": prepare_credit_default,
    "drybean": prepare_drybean,
    "covertype": prepare_covertype,
    "airlines": prepare_airlines_optimized,
    "higgs": prepare_higgs,
}

def prepare_db(config, name):
    seed = config["random_seed"]
    device = config["device"]

    key = name.lower()

    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )

    prepare_fn = DATASET_REGISTRY[key]

    # Normalize call signature if needed
    try:
        return prepare_fn(random_seed=seed, device=device)
    except TypeError:
        # fallback for functions that use positional args
        return prepare_fn(seed, device)
