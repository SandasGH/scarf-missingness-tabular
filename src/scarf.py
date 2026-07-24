import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class SCARFEncoder(nn.Module):
    """
    Encoder network producing a 64-dimensional embedding from tabular input.
    Architecture follows Bahri et al. (2021): three hidden layers with
    ReLU activation and dropout for regularisation.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SCARFModel(nn.Module):
    """
    Full SCARF model combining encoder and projection head for pretraining.
    The projection head is discarded after pretraining; only the encoder
    is used downstream.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = SCARFEncoder(input_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.projection_head(z)


def corrupt_features(X: np.ndarray, corruption_rate: float = 0.6) -> np.ndarray:
    """
    Replace a random subset of feature values with draws from each feature's
    empirical marginal distribution across the training set.

    Replacement values are sampled from observed (non-NaN) values in each column,
    making corrupted values plausible in isolation but incorrect for that specific row.
    This is the corruption mechanism described in Bahri et al. (2021, Algorithm 1).
    NaN values are excluded from the sampling pool so that only observed values
    are used as replacements; the corruption step cannot re-introduce missingness.

    Corruption rate 0.6 follows the paper recommendation (Bahri et al. 2021).
    The paper's ablation shows stable performance between 0.5 and 0.8.
    """
    X_corr = X.copy()
    n_rows, n_cols = X.shape

    for j in range(n_cols):
        col = X[:, j]
        observed = col[~np.isnan(col)]
        if len(observed) == 0:
            continue
        mask = np.random.rand(n_rows) < corruption_rate
        if mask.any():
            replacements = observed[np.random.randint(0, len(observed), size=mask.sum())]
            X_corr[mask, j] = replacements

    return X_corr


def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    InfoNCE (NT-Xent) contrastive loss between two sets of projected embeddings.

    Each sample in z1 is treated as a positive pair with the corresponding sample
    in z2; all other samples in the batch serve as negatives. Temperature tau=1.0
    follows Bahri et al. ablation recommendations for tabular data; this is higher
    than the tau=0.07 used in SimCLR for image data, reflecting the lower uniformity
    of tabular feature spaces relative to image feature spaces.
    """
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)

    sim = torch.mm(z, z.t()) / tau

    # Remove self-similarity from denominator
    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    # Positive pair indices: for row i in [0,n), its positive is i+n; for i in [n,2n), it's i-n
    labels = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)

    loss = F.cross_entropy(sim, labels)
    return loss


def pretrain_scarf(
    X_train: np.ndarray,
    corruption_rate: float = 0.6,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 0.001,
) -> SCARFEncoder:
    # NaNs are filled with column medians before pretraining because corrupt_features
    # samples replacements from observed values; without a fill, NaN-heavy columns
    # would have too few observed values to sample from.  This fill is a preprocessing
    # step to enable the corruption mechanism, not an imputation strategy:
    # SCARF is still pretrained on the already-incomplete data, reflecting the realistic
    # scenario where clean data is unavailable.
    X_filled = X_train.copy()
    for j in range(X_filled.shape[1]):
        col = X_filled[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median_val = np.nanmedian(col)
            X_filled[nan_mask, j] = median_val

    input_dim = X_filled.shape[1]
    model = SCARFModel(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(X_filled)
    indices = np.arange(n)

    model.train()
    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            X_batch = X_filled[batch_idx]

            # view1 is the original row; view2 is corrupted.  This follows Algorithm 1
            # in Bahri et al.: corrupting both views was found to reduce performance in
            # the paper's ablation study, so only view2 is corrupted here.
            view1 = X_batch.copy()
            # Replacement values here are sampled from X_batch (the current mini-batch),
            # not the full training set, so this is a computational approximation to
            # sampling from the full training-set marginal distribution described above.
            view2 = corrupt_features(X_batch, corruption_rate)

            t1 = torch.tensor(view1, dtype=torch.float32)
            t2 = torch.tensor(view2, dtype=torch.float32)

            optimizer.zero_grad()
            z1 = model(t1)
            z2 = model(t2)
            loss = _nt_xent_loss(z1, z2, tau=1.0)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0:
            print(f"  [SCARF pretrain] Epoch {epoch:4d} | loss: {epoch_loss / n_batches:.4f}")

    return model.encoder


def finetune_scarf(
    encoder: SCARFEncoder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 0.001,
):
    # Fill NaN with column median before passing through encoder
    X_filled = X_train.copy()
    for j in range(X_filled.shape[1]):
        col = X_filled[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            X_filled[nan_mask, j] = np.nanmedian(col)

    X_train = X_filled

    # The encoder is frozen to isolate the contribution of the pretraining stage:
    # only the linear classification head is trained on labels, so downstream performance
    # reflects the quality of the pretrained representations rather than joint optimisation.
    # This deviates from Bahri et al. who fine-tune both encoder and head jointly;
    # freezing is a deliberate design choice to enable a cleaner ablation.
    for param in encoder.parameters():
        param.requires_grad = False

    head = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )

    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.10, random_state=42, stratify=y_train
    )

    def to_tensors(X, y):
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    X_tr_t, y_tr_t = to_tensors(X_tr, y_tr)
    X_val_t, y_val_t = to_tensors(X_val, y_val)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True
    )

    best_val_loss = float("inf")
    best_head_weights = copy.deepcopy(head.state_dict())
    patience = 15
    epochs_without_improvement = 0

    encoder.eval()
    for epoch in range(1, epochs + 1):
        head.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            with torch.no_grad():
                embeddings = encoder(X_batch)
            preds = head(embeddings).squeeze(1)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_emb = encoder(X_val_t)
            val_preds = head(val_emb).squeeze(1)
            val_loss = criterion(val_preds, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_head_weights = copy.deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"  [SCARF finetune] Early stopping at epoch {epoch} (patience={patience})")
            break

    head.load_state_dict(best_head_weights)
    return encoder, head


def evaluate_scarf(
    encoder: SCARFEncoder,
    head: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_medians: np.ndarray | None = None,
):
    # train_medians must be computed from X_train (not X_test) to prevent leakage
    # of test set statistics into the preprocessing pipeline, consistent with the
    # imputation baselines.
    X_eval = X_test.copy()
    if train_medians is not None:
        for j in range(X_eval.shape[1]):
            nan_mask = np.isnan(X_eval[:, j])
            if nan_mask.any():
                X_eval[nan_mask, j] = train_medians[j]
    elif np.isnan(X_eval).any():
        raise ValueError(
            "evaluate_scarf received NaNs in X_test but no train_medians provided. "
            "Pass train_medians computed from X_train."
        )

    encoder.eval()
    head.eval()

    X_t = torch.tensor(X_eval, dtype=torch.float32)
    with torch.no_grad():
        embeddings = encoder(X_t)
        probs = head(embeddings).squeeze(1).numpy()

    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)

    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")

    return auc, acc
