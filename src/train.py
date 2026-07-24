import copy

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from src.models import MLPClassifier


_MIN_ROWS_FOR_VAL_SPLIT = 50


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    batch_size: int = 256,
    patience: int = 15,
    max_epochs: int = 300,
) -> MLPClassifier:
    # Below _MIN_ROWS_FOR_VAL_SPLIT rows, a 10% validation split would leave fewer than
    # 5 samples for monitoring, making early stopping unreliable; training runs to max_epochs instead.
    use_val_split = len(X_train) >= _MIN_ROWS_FOR_VAL_SPLIT

    def to_tensors(X, y):
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    if use_val_split:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.10, random_state=42, stratify=y_train
        )
        X_val_t, y_val_t = to_tensors(X_val, y_val)
    else:
        print(f"  [train_mlp] Only {len(X_train)} rows skipping validation split, early stopping disabled.")
        X_tr, y_tr = X_train, y_train

    X_tr_t, y_tr_t = to_tensors(X_tr, y_tr)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True
    )

    model = MLPClassifier(input_dim)
    optimizer = model.get_optimizer()

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = model.criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(y_batch)

        train_loss = train_loss_sum / len(y_tr_t)

        if use_val_split:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t)
                val_loss = model.criterion(val_preds, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch % 10 == 0:
                print(
                    f"  Epoch {epoch:4d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}"
                )

            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break
        else:
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:4d} | train loss: {train_loss:.4f}")

    # Restore weights from the best validation epoch rather than the final epoch,
    # so the returned model reflects the point of lowest generalisation loss.
    # Without a validation split there is no signal to select a best epoch, so
    # the final trained weights are kept instead of the initial random weights.
    if use_val_split:
        model.load_state_dict(best_weights)
    return model


def evaluate_mlp(model: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray):
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        probs = model(X_t).numpy()

    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)

    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")

    return auc, acc
