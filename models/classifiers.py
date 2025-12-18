import torch.nn as nn
from utils.utils import set_seed
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

def train_classifier(data, config):
    """
    Unified classifier training entry point.

    Expects:
      config["classifier"] in CLASSIFIER_REGISTRY
    """
    clf_name = config.get("classifier", "mlp")

    if clf_name not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier '{clf_name}'. "
            f"Available: {list(CLASSIFIER_REGISTRY.keys())}"
        )

    train_fn = CLASSIFIER_REGISTRY[clf_name]
    return train_fn(data, config)


def train_rf(data, config):
    X = data["X_train"].cpu().numpy()
    y = data["y_train"].cpu().numpy()

    model = RandomForestClassifier(
        n_estimators=config.get("rf_n_estimators", 200),
        random_state=config["random_seed"],
        n_jobs=-1,
    )
    model.fit(X, y)
    return model

def train_svm(data, config):
    X = data["X_train"].cpu().numpy()
    y = data["y_train"].cpu().numpy()

    model = SVC(
        kernel="rbf",
        probability=True,
        random_state=config["random_seed"],
    )
    model.fit(X, y)
    return model

class ClassifierMLP(nn.Module):
    def __init__(self, input_dim, hidden=[128, 64], num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h

        # Output logits only
        if num_classes == 2:
            layers.append(nn.Linear(prev, 1))
        else:
            layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.net(x)
        if self.num_classes == 2:
            return logits.view(-1)  # shape [N]
        else:
            return logits            # shape [N, C]
        
def train_mlp_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim,
    hidden,
    epochs,
    seed,
    device,
    num_classes,
):
    set_seed(seed)

    X_train = X_train.to(device).float()
    X_val   = X_val.to(device).float()
    y_train = y_train.to(device)
    y_val   = y_val.to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=1024, shuffle=False)

    model = ClassifierMLP(input_dim, hidden, num_classes).to(device)

    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_auc = -1.0
    best_state = None

    for ep in range(1, epochs + 1):

        # -------------- TRAIN ----------------
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device)

            opt.zero_grad()

            logits = model(xb)
            if num_classes == 2:
                loss = criterion(logits, yb.float())
            else:
                loss = criterion(logits, yb.long())

            loss.backward()
            opt.step()

        # -------------- VALID ----------------
        model.eval()
        probs_all = []
        trues_all = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float()
                yb = yb.to(device)

                logits = model(xb)
                if num_classes == 2:
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, dim=1)

                probs_all.append(probs.cpu().numpy())
                trues_all.append(yb.cpu().numpy())

        probs_all = np.concatenate(probs_all)
        trues_all = np.concatenate(trues_all)

        if num_classes == 2:
            val_auc = roc_auc_score(trues_all, probs_all)
        else:
            val_auc = roc_auc_score(trues_all, probs_all, multi_class="ovr", average="macro")

        print(f"[Classifier] Epoch {ep:02d} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model, best_auc        

def train_mlp(data, config):
    model, _ = train_mlp_classifier(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        input_dim=data["input_dim"],
        hidden=config["classifier_hidden"],
        epochs=config["classifier_epochs"],
        seed=config["random_seed"],
        device=config["device"],
        num_classes=data["num_classes"],
    )
    return model

CLASSIFIER_REGISTRY = {
    "mlp": train_mlp,
    "rf": train_rf,
    "svm": train_svm,
}
