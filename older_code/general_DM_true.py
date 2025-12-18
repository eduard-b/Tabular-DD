import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn

from sklearn.cluster import KMeans

from prepare_database import prepare_db

from utils.utils import set_seed

from models.embedders import sample_random_embedder

from models.classifiers import ClassifierMLP

from utils.utils import ensure_dir

def run_dm_moment_experiment(config):

    device   = config["device"]
    save_dir = config["save_dir"]
    ensure_dir(save_dir)

    # ---------- Load dataset ----------
    data = prepare_db(config, name=config["dataset_name"])
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]

    input_dim   = data["input_dim"]
    num_classes = data["num_classes"]

    # ---------- DM synthesis ----------
    print(f"\n--- DM SYNTHESIS (M{config['dm_max_moment']}) ---")

    X_syn, y_syn = dm_moment_synthesize(
        X_train, y_train,
        ipc=config["ipc"],
        iters=config["dm_iters"],
        lr=config["dm_lr"],
        batch_real=config["dm_batch_real"],
        input_dim=input_dim,
        num_classes=num_classes,
        device=device,
        embedder_type=config["dm_embedder_type"],
        embed_dim=config["dm_embed_dim"],
        embed_hidden=config["dm_embed_hidden"],
        max_moment=config["dm_max_moment"],
    )

    torch.save(
        {"X_syn": X_syn.cpu(), "y_syn": y_syn.cpu()},
        os.path.join(save_dir, "synthetic.pt")
    )

    # ---------- Train classifier ----------
    model, _ = train_classifier(
        X_syn, y_syn, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )

    test_acc, test_auc = evaluate_classifier(
        model, X_test, y_test, device, num_classes
    )

    # ---------- Save results ----------
    result = {
        "dataset": config["dataset_name"],
        "embedder": config["dm_embedder_type"],
        "ipc": config["ipc"],
        "max_moment": config["dm_max_moment"],
        "test_acc": test_acc,
        "test_auc": test_auc,
    }

    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

def run_dm_bn_experiment(config):

    print("===== RUNNING DM-BN EXPERIMENT =====")
    print(json.dumps(config, indent=2))

    device   = config["device"]
    save_dir = config["save_dir"]
    ensure_dir(save_dir)

    # ---------------- Load dataset ----------------
    data = prepare_db(config, name=config["dataset_name"])
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    input_dim   = data["input_dim"]
    num_classes = data["num_classes"]

    # ---------------- Full baseline ----------------
    print("\n--- FULL-DATA BASELINE ---")
    model_full, _ = train_classifier(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    full_acc, full_auc = evaluate_classifier(
        model_full, X_test, y_test, device, num_classes
    )

    # ---------------- Random IPC baseline ----------------
    print("\n--- RANDOM IPC BASELINE ---")
    X_rand, y_rand = make_random_ipc_subset(
        X_train, y_train,
        config["ipc"],
        seed=config["random_seed"],
        device=device
    )
    model_rand, _ = train_classifier(
        X_rand, y_rand, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    rand_acc, rand_auc = evaluate_classifier(
        model_rand, X_test, y_test, device, num_classes
    )

    # ---------------- Herding IPC baseline ----------------
    print("\n--- HERDING IPC BASELINE ---")
    X_herd, y_herd = make_herding_ipc_subset(
        X_train, y_train,
        config["ipc"],
        seed=config["random_seed"],
        device=device
    )
    model_herd, _ = train_classifier(
        X_herd, y_herd, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    herd_acc, herd_auc = evaluate_classifier(
        model_herd, X_test, y_test, device, num_classes
    )

    # ---------------- DM-BN synthesis ----------------
    print("\n--- DM-BN SYNTHESIS ---")
    X_syn, y_syn = dm_bn_synthesize(
        model_full,
        X_train, y_train,
        ipc=config["ipc"],
        iters=config["dm_iters"],
        lr_img=config["dm_lr"],
        batch_real=config["dm_batch_real"],
        input_dim=input_dim,
        num_classes=num_classes,
        device=device,
    )

    # save synthetic dataset
    syn_path = os.path.join(
        save_dir,
        f"{config['dataset_name']}_dmBN_ipc{config['ipc']}.pt"
    )
    torch.save({"X_syn": X_syn.cpu(), "y_syn": y_syn.cpu()}, syn_path)
    print(f"Saved DM-BN synthetic data to {syn_path}")

    # ---------------- Train classifier on DM-BN synthetic ----------------
    print("\n--- CLASSIFIER ON DM-BN SYNTHETIC DATA ---")
    model_dm, _ = train_classifier(
        X_syn, y_syn, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    dm_acc, dm_auc = evaluate_classifier(
        model_dm, X_test, y_test, device, num_classes
    )

    # ---------------- Save result JSON ----------------
    result = {
        "dataset": config["dataset_name"],
        "method": "DM-BN",
        "ipc": config["ipc"],
        "results": {
            "full":  {"test_acc": full_acc,  "test_auc": full_auc},
            "random":{"test_acc": rand_acc,  "test_auc": rand_auc},
            "herd":  {"test_acc": herd_acc,  "test_auc": herd_auc},
            "dm_bn": {"test_acc": dm_acc,    "test_auc": dm_auc},
        }
    }

    fname = f"{config['dataset_name']}_dmBN_ipc{config['ipc']}.json"
    json_path = os.path.join(save_dir, fname)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved DM-BN results to {json_path}")
    print("===== DM-BN EXPERIMENT FINISHED =====")

def run_dm_true_experiment(config):

    print("===== RUNNING TRUE DM-TABULAR EXPERIMENT =====")
    print(json.dumps(config, indent=2))

    device   = config["device"]
    save_dir = config["save_dir"]
    ensure_dir(save_dir)

    # ---------------- Load dataset ----------------
    data = prepare_db(config, name=config["dataset_name"])
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    input_dim        = data["input_dim"]
    num_classes      = data["num_classes"]

    # ---------------- Full baseline ----------------
    print("\n--- FULL-DATA BASELINE ---")
    model_full, val_auc_full = train_classifier(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    test_acc_full, test_auc_full = evaluate_classifier(
        model_full, X_test, y_test, device, num_classes
    )

    # ---------------- Random IPC baseline ----------------
    print(f"\n--- RANDOM IPC BASELINE (IPC={config['ipc']}) ---")
    X_rand, y_rand = make_random_ipc_subset(X_train, y_train, config["ipc"],
                                           seed=config["random_seed"],
                                           device=device)

    model_rand, val_auc_rand = train_classifier(
        X_rand, y_rand, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    test_acc_rand, test_auc_rand = evaluate_classifier(
        model_rand, X_test, y_test, device, num_classes
    )

    # ---------------- Herding IPC baseline ----------------
    print(f"\n--- HERDING BASELINE (IPC={config['ipc']}) ---")
    X_herd, y_herd = make_herding_ipc_subset(
        X_train, y_train, config["ipc"],
        seed=config["random_seed"],
        device=device,
    )
    model_herd, val_auc_herd = train_classifier(
        X_herd, y_herd, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    test_acc_herd, test_auc_herd = evaluate_classifier(
        model_herd, X_test, y_test, device, num_classes
    )

    # ---------------- TRUE DM Synthesis ----------------
    print("\n--- TRUE DM SYNTHESIS ---")
    X_syn, y_syn = dm_true_synthesize(
        X_train, y_train,
        ipc=config["ipc"],
        iters=config["dm_iters"],
        lr_img=config["dm_lr"],
        batch_real=config["dm_batch_real"],
        input_dim=input_dim,
        num_classes=num_classes,
        embed_hidden=config["dm_embed_hidden"],
        embed_dim=config["dm_embed_dim"],
        embedder_type=config["dm_embedder_type"],   # <-- NEW
        device=device,
    )


    syn_path = os.path.join(
        save_dir,
        f"{config['dataset_name']}_trueDM_ipc{config['ipc']}_seed{config['dm_seed']}.pt"
    )
    torch.save({"X_syn": X_syn.cpu(), "y_syn": y_syn.cpu()}, syn_path)
    print(f"Saved synthetic data to {syn_path}")

    # ---------------- Train classifier on synthetic ----------------
    print("\n--- CLASSIFIER ON TRUE DM SYNTHETIC DATA ---")
    model_dm, val_auc_dm = train_classifier(
        X_syn, y_syn, X_val, y_val,
        input_dim=input_dim,
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=device,
        num_classes=num_classes,
    )
    test_acc_dm, test_auc_dm = evaluate_classifier(
        model_dm, X_test, y_test, device, num_classes
    )

    # ----------------------------------------------------------------------
    # Save JSON with all results
    # ----------------------------------------------------------------------
    run_id = time.strftime("%Y%m%d-%H%M%S")
    embedder_type = config["dm_embedder_type"]

    result = {
        "run_id": run_id,
        "dataset": config["dataset_name"],
        "method": "True DM (Tabular)",
        "ipc": config["ipc"],
        "embedder_type": embedder_type,
        "embed_hidden": config["dm_embed_hidden"],
        "embed_dim": config["dm_embed_dim"],
        "dm_iters": config["dm_iters"],
        "dm_lr": config["dm_lr"],
        "dm_batch_real": config["dm_batch_real"],
        "results": {
            "full":   {"val_auc": val_auc_full, "test_acc": test_acc_full, "test_auc": test_auc_full},
            "random": {"val_auc": val_auc_rand, "test_acc": test_acc_rand, "test_auc": test_auc_rand},
            "herd":   {"val_auc": val_auc_herd, "test_acc": test_acc_herd, "test_auc": test_auc_herd},
            "dm":     {"val_auc": val_auc_dm,  "test_acc": test_acc_dm,  "test_auc": test_auc_dm},
        }
    }

    fname = (
        f"{config['dataset_name']}_trueDM_"
        f"ipc{config['ipc']}_{embedder_type}_"
        f"h{config['dm_embed_hidden']}_e{config['dm_embed_dim']}_"
        f"it{config['dm_iters']}_lr{config['dm_lr']}_"
        f"{run_id}.json"
    )

    json_path = os.path.join(save_dir, fname)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved results to {json_path}")
    print("===== EXPERIMENT FINISHED =====")



if __name__ == "__main__":

    DB_LIST = [
        "drybean",
        "adult",
        "bank",
        "credit",
        "covertype",
        "airlines",
        "higgs",
    ]

    EMBEDDER_LIST = [
        # "LN",
        # "LNDeep",
        # "LNWide",
        "LNRes",
        # "LNCascade",
    ]

    MOMENT_EXPERIMENTS = [
    {"name": "M1", "max_moment": 1},
    {"name": "M2", "max_moment": 2},
    {"name": "M3", "max_moment": 3},
    {"name": "M4", "max_moment": 4},
]


    RESULTS_DIR = "./results_moments/"
    ensure_dir(RESULTS_DIR)

    for db in DB_LIST:
        for emb in EMBEDDER_LIST:
            for exp in MOMENT_EXPERIMENTS:

                config = {
                    "dataset_name": db,
                    "save_dir": os.path.join(RESULTS_DIR, db, emb, exp["name"]),
                    "device": "cuda" if torch.cuda.is_available() else "cpu",

                    # DM
                    "ipc": 10,
                    "dm_iters": 2000,
                    "dm_lr": 0.05,
                    "dm_batch_real": 256,
                    "dm_seed": 2025,

                    # Embedder
                    "dm_embedder_type": emb,
                    "dm_embed_hidden": 256,
                    "dm_embed_dim": 128,

                    # Moment control
                    "dm_max_moment": exp["max_moment"],

                    # Classifier
                    "classifier_hidden": [128, 64],
                    "classifier_epochs": 20,

                    "random_seed": 42,
                }

                run_dm_moment_experiment(config)

