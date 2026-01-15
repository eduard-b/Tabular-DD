import json
import os
import torch

from utils.utils import ensure_dir
from data.prepare_database import prepare_db
from synth.registry import synthesize
from models.classifiers import train_classifier
from eval.eval_classifiers import evaluate_classifier


def run_dm_moment_experiment(config):
    device = config["device"]
    ensure_dir(config["save_dir"])

    # ---------------- Load dataset ----------------
    data = prepare_db(config, name=config["dataset_name"])

    # ---------------- Synthesize ----------------
    X_syn, y_syn = synthesize(
        synth_type=config["synth_type"],  # IMPORTANT: use config, not hardcoded
        data=data,
        config=config,
    )

    # ---------------- Train (UNIFIED WRAPPER) ----------------
    train_data = {
        "X_train": X_syn,
        "y_train": y_syn,
        "X_val": data["X_val"],
        "y_val": data["y_val"],
        "input_dim": data["input_dim"],
        "num_classes": data["num_classes"],
    }

    model = train_classifier(train_data, config)

    # ---------------- Evaluate ----------------
    acc, auc = evaluate_classifier(model, data, device)

    # ---------------- Save ----------------
    result = {
        "dataset": config["dataset_name"],
        "synth_type": config["synth_type"],
        "embedder": config["dm_embedder_type"],
        "ipc": config["ipc"],
        "max_moment": config.get("max_moment", None),
        "cov_weight": config.get("cov_weight", None),
        "classifier": config.get("classifier", "mlp"),
        "test_acc": acc,
        "test_auc": auc,
    }

    out_path = os.path.join(config["save_dir"], "result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)


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

    EMBEDDERS = ["ln_res", "ln_res_xl", "node"]  # add others if you want, e.g. 

    # Compare old diagonal M2 vs new full-covariance M2
    SYNTH_EXPERIMENTS = [
        {"name": "M2_fullcov", "synth_type": "dm_moments_cov2", "max_moment": 2},
    ]

    RESULTS_DIR = "./results_cov2_ablation_ipc_variable"
    ensure_dir(RESULTS_DIR)

    IPC_LIST = [1, 5, 25, 50, 100]

    for db in DB_LIST:
        for embedder in EMBEDDERS:
            for exp in SYNTH_EXPERIMENTS:
                for ipc in IPC_LIST:

                    save_dir = os.path.join(RESULTS_DIR, db, embedder, str(ipc), exp["name"])
                    ensure_dir(save_dir)

                    config = {
                        "dataset_name": db,
                        "save_dir": save_dir,
                        "device": "cuda" if torch.cuda.is_available() else "cpu",

                        # ---------------- Synthesis ----------------
                        "synth_type": exp["synth_type"],
                        "ipc": ipc,
                        "dm_iters": 2000,
                        "dm_lr": 0.05,
                        "dm_batch_real": 256,

                        # ---------------- Embedder ----------------
                        "dm_embedder_type": embedder,
                        "dm_embed_hidden": 256,
                        "dm_embed_dim": 128,

                        # ---------------- Moments ----------------
                        "max_moment": exp["max_moment"],
                        "cov_weight": 1.0,  # only used by dm_moments_cov2 (safe to keep)

                        # ---------------- Classifier ----------------
                        "classifier": "mlp",
                        "classifier_hidden": [128, 64],
                        "classifier_epochs": 20,

                        # ---------------- Reproducibility ----------------
                        "random_seed": 42,
                    }

                    run_dm_moment_experiment(config)
