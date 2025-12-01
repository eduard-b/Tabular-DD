import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import numpy as np

def set_seed(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def create_random_subset(X, y, keep_ratio=0.5, seed=42):
    set_seed(seed)
    n = len(X)
    k = int(n * keep_ratio)

    idx = np.random.choice(n, k, replace=False)
    return X[idx], y[idx]

# -----------------------------------------------------
# 1. Load dataset
# -----------------------------------------------------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
cols = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "label"
]
df = pd.read_csv(url, names=cols)

X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.float32)

# -----------------------------------------------------
# 2. Train/Val/Test split
# -----------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# -----------------------------------------------------
# 3. Standardize features
# -----------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# -----------------------------------------------------
# 4. Convert to PyTorch tensors
# -----------------------------------------------------
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_ds   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
test_ds  = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)
test_loader  = DataLoader(test_ds, batch_size=32)

# -----------------------------------------------------
# 5. Define simple MLP
# -----------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------------------------------
# 6. Training + Validation loop
# -----------------------------------------------------
def evaluate(model, loader):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb).squeeze()
            preds.append(out.numpy())
            trues.append(yb.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    pred_labels = (preds > 0.5).astype(int)
    acc = accuracy_score(trues, pred_labels)
    auc = roc_auc_score(trues, preds)

    return acc, auc

epochs = 40
best_val_auc = 0

def train_model(X_train, y_train, X_val, y_val, epochs=40, seed=42):
    set_seed(seed)

    # create loaders
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32)

    # init model
    model = MLP()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_auc = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb).squeeze()
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        val_acc, val_auc = evaluate(model, val_loader)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model, best_val_auc

def test_model(model, X_test, y_test):
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_loader = DataLoader(test_ds, batch_size=32)

    acc, auc = evaluate(model, test_loader)
    return acc, auc

def one_sample_per_class(X, y, seed=42):
    set_seed(seed)
    X_small = []
    y_small = []

    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0]
        chosen = np.random.choice(idx, 1, replace=False)
        X_small.append(X[chosen][0])
        y_small.append(y[chosen][0])

    return np.array(X_small, dtype=np.float32), np.array(y_small, dtype=np.float32)


print("\n=== BASELINE (100% TRAINING DATA) ===")
model_full, val_auc_full = train_model(X_train, y_train, X_val, y_val)
test_acc_full, test_auc_full = test_model(model_full, X_test, y_test)
print(f"Val AUC:  {val_auc_full:.4f} | Test AUC: {test_auc_full:.4f}")

# ----------------------------------------------------------
print("\n=== RANDOM 50% SUBSET ===")
X_50, y_50 = create_random_subset(X_train, y_train, keep_ratio=0.5)
model_50, val_auc_50 = train_model(X_50, y_50, X_val, y_val)
test_acc_50, test_auc_50 = test_model(model_50, X_test, y_test)
print(f"Val AUC:  {val_auc_50:.4f} | Test AUC: {test_auc_50:.4f}")

# ----------------------------------------------------------
print("\n=== RANDOM 10% SUBSET ===")
X_10, y_10 = create_random_subset(X_train, y_train, keep_ratio=0.1)
model_10, val_auc_10 = train_model(X_10, y_10, X_val, y_val)
test_acc_10, test_auc_10 = test_model(model_10, X_test, y_test)
print(f"Val AUC:  {val_auc_10:.4f} | Test AUC: {test_auc_10:.4f}")

# ----------------------------------------------------------
print("\n=== ONE SAMPLE PER CLASS (IPC = 1) ===")
X_ipc1, y_ipc1 = one_sample_per_class(X_train, y_train, seed=42)

model_ipc1, val_auc_ipc1 = train_model(X_ipc1, y_ipc1, X_val, y_val)
test_acc_ipc1, test_auc_ipc1 = test_model(model_ipc1, X_test, y_test)

print(f"Val AUC:  {val_auc_ipc1:.4f} | Test AUC: {test_auc_ipc1:.4f}")
