import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_classifier(model, data, device):
    X_test = data["X_test"]
    y_test = data["y_test"]
    num_classes = data["num_classes"]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test.cpu().numpy())
    else:
        with torch.no_grad():
            model.eval()
            logits = model(X_test.to(device).float())
            if num_classes == 2:
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()

    y_true = y_test.cpu().numpy()

    if num_classes == 2:
        auc = roc_auc_score(y_true, probs)
        acc = ((probs > 0.5).astype(int) == y_true).mean()
    else:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        acc = (probs.argmax(1) == y_true).mean()

    return acc, auc
