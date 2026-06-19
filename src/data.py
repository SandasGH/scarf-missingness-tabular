import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Feature groups for feature-dependent missingness injection
UCI_CORRELATED_GROUPS = [
    ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
    ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"],
    ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"],
]

# Note: these column names refer to the encoded (0/1) versions produced below
TELCO_CORRELATED_GROUPS = [
    [
        "StreamingTV",
        "StreamingMovies",
        "OnlineBackup",
        "OnlineSecurity",
        "DeviceProtection",
        "TechSupport",
    ]
]

# Populated on first call to _load_adult; contains one group of column indices
# covering all workclass_* and occupation_* OHE columns combined.
ADULT_CORRELATED_GROUPS = []

# Paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_RAW  = os.path.join(_HERE, "..", "Data", "Raw")

_UCI_PATH   = os.path.join(_RAW, "default_credit_card.csv")
_TELCO_PATH = os.path.join(_RAW, "telco_clean.csv")
_ADULT_PATH = os.path.join(_RAW, "adult_income_dataset.csv")

# Internal helpers
def _load_uci():
    df = pd.read_csv(_UCI_PATH)

    df = df.drop(columns=["ID"])

    y = df["default payment next month"].values
    X = df.drop(columns=["default payment next month"])

    feature_names = X.columns.tolist()
    return X.values.astype(np.float64), y.astype(np.int64), feature_names


def _load_telco():
    df = pd.read_csv(_TELCO_PATH)

    df = df.drop(columns=["customerID"])

    #  Binary Yes/No columns → 0/1 
    yes_no_cols = [
        "gender",        # Female=0, Male=1
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]
    binary_map = {"No": 0, "Yes": 1, "Female": 0, "Male": 1}
    for col in yes_no_cols:
        df[col] = df[col].map(binary_map).astype(np.float64)

    # Columns with three values: No / No [phone|internet] service / Yes 
    # "No X service" is treated as 0 (same as No) since the add-on is absent.
    three_val_cols = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    three_val_map = {
        "No": 0,
        "No phone service": 0,
        "No internet service": 0,
        "Yes": 1,
    }
    for col in three_val_cols:
        df[col] = df[col].map(three_val_map).astype(np.float64)

    #  Target 
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1}).astype(np.int64)

    # One-hot encode multi-class categoricals
    ohe_cols = ["InternetService", "Contract", "PaymentMethod"]# These are the only remaining object columns, so we can OHE them all at once
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)# Keep all OHE columns (don't drop first) since we want to test feature-dependent missingness on them

    # Cast the new boolean/object OHE columns to float
    for col in df.columns:
        if df[col].dtype == bool or str(df[col].dtype) == "boolean":
            df[col] = df[col].astype(np.float64)
    y = df["Churn"].values
    X = df.drop(columns=["Churn"])

    feature_names = X.columns.tolist()
    return X.values.astype(np.float64), y.astype(np.int64), feature_names



def _load_adult():
    df = pd.read_csv(_ADULT_PATH)

    # Step 1: drop non-personal / redundant columns
    df = df.drop(columns=["fnlwgt", "education"])

    # Step 2: structural missingness — raw data encodes missing as "?"
    # Replace with NaN, then impute with mode computed from training rows only.
    # This is data cleaning, separate from the experimental missingness injected later.
    for col in ["workclass", "occupation"]:
        df[col] = df[col].replace("?", np.nan)

    # Step 3: encode label first so we can stratify the preliminary split
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1}).astype(np.int64)

    # Preliminary split (same seed/stratification as load_dataset) to obtain
    # training indices — ensures mode is derived from training data only.
    train_idx, _ = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=42,
        stratify=df["income"].values,
    )
    for col in ["workclass", "occupation"]:
        mode_val = df[col].iloc[train_idx].mode()[0]
        df[col] = df[col].fillna(mode_val)

    # Step 4: binary-encode gender
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).astype(np.float64)

    # Step 5: one-hot encode remaining categoricals (keep all categories)
    ohe_cols = [
        "workclass", "marital-status", "occupation",
        "relationship", "race", "native-country",
    ]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

    # Cast boolean OHE columns to float64
    for col in df.columns:
        if df[col].dtype == bool or str(df[col].dtype) == "boolean":
            df[col] = df[col].astype(np.float64)

    y = df["income"].values
    X = df.drop(columns=["income"])
    feature_names = X.columns.tolist()

    # Build correlated-group indices: workclass_* and occupation_* columns combined.
    # Both relate to employment and would plausibly originate from the same source.
    wc_occ_indices = [
        i for i, name in enumerate(feature_names)
        if name.startswith("workclass_") or name.startswith("occupation_")
    ]
    # Mutate in-place so the module-level reference stays valid after import.
    ADULT_CORRELATED_GROUPS.clear()
    ADULT_CORRELATED_GROUPS.append(wc_occ_indices)

    return X.values.astype(np.float64), y.astype(np.int64), feature_names


# Public API


def inject_missingness(X, mechanism, rate, feature_groups=None):
    """
    Inject missing values (NaN) into a feature array.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input feature matrix. Not modified in place.
    mechanism : {"MCAR", "feature_dependent"}
        Missing-data mechanism to simulate.
    rate : float in (0, 1]
        For MCAR: probability that any individual value is set to NaN.
        For feature_dependent: proportion of rows whose selected group is erased.
    feature_groups : list of list of int, optional
        Required for ``mechanism="feature_dependent"``.
        Each inner list is a set of column indices that form a correlated group.
        One group is chosen at random and all its values are set to NaN for
        the selected rows.

    Returns
    -------
    X_miss : np.ndarray
        Copy of X with NaN values injected.
    """
    if mechanism not in ("MCAR", "feature_dependent"):
        raise ValueError("mechanism must be 'MCAR' or 'feature_dependent'.")
    if not (0 < rate <= 1):
        raise ValueError("rate must be in (0, 1].")

    X_miss = X.astype(np.float64, copy=True)
    n_rows, n_cols = X_miss.shape
    total_values = n_rows * n_cols

    if mechanism == "MCAR":
        mask = np.random.rand(n_rows, n_cols) < rate
        X_miss[mask] = np.nan
        n_missing = int(mask.sum())

    elif mechanism == "feature_dependent":
        if not feature_groups:
            raise ValueError("feature_groups must be provided for 'feature_dependent'.")
        n_affected = max(1, int(np.round(rate * n_rows)))
        affected_rows = np.random.choice(n_rows, size=n_affected, replace=False)
        n_missing_before = int(np.isnan(X_miss).sum())
        for row_idx in affected_rows:
            group = feature_groups[np.random.randint(len(feature_groups))]
            for col_idx in group:
                X_miss[row_idx, col_idx] = np.nan
        n_missing = int(np.isnan(X_miss).sum()) - n_missing_before

    pct = 100.0 * n_missing / total_values
    print(f"  [inject_missingness] mechanism={mechanism}, rate={rate}"
          f"  ->  {n_missing:,} / {total_values:,} values set to NaN  ({pct:.2f}%)")
    return X_miss


def load_dataset(name: str):
    """
    Load and preprocess a dataset.

    Parameters
    ----------
    name : {"uci", "telco", "adult"}

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
        Features are standardised (scaler fit on training set only).
        Targets are integer class labels (0 / 1).
    """
    if name == "uci":
        X, y, feature_names = _load_uci()
    elif name == "telco":
        X, y, feature_names = _load_telco()
    elif name == "adult":
        X, y, feature_names = _load_adult()
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose 'uci', 'telco', or 'adult'.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[{name}] X_train: {X_train.shape}  X_test: {X_test.shape}"
          f"  y_train: {y_train.shape}  y_test: {y_test.shape}")
    print(f"[{name}] Features ({len(feature_names)}): {feature_names}")

    return X_train, X_test, y_train, y_test



# Quick test


if __name__ == "__main__":
    print("=" * 60)
    print("UCI Credit Card Default")
    print("=" * 60)
    X_train, X_test, y_train, y_test = load_dataset("uci")

    print()
    print("=" * 60)
    print("Telco Customer Churn")
    print("=" * 60)
    X_train, X_test, y_train, y_test = load_dataset("telco")

    
    # inject_missingness tests on UCI training set
    
    print()
    print("=" * 60)
    print("inject_missingness tests — UCI X_train")
    print("=" * 60)

    np.random.seed(0)

    print("\n--- MCAR ---")
    for rate in (0.10, 0.20, 0.30):
        X_miss = inject_missingness(X_train, mechanism="MCAR", rate=rate)
        assert X_miss.shape == X_train.shape, "Shape changed"
        assert not np.isnan(X_train).any(), "Original X_train was mutated"

    print("\n--- feature_dependent (rate=0.30) ---")
    # Column indices in UCI after ID is dropped (0-indexed):
    #   PAY group      : 5-10
    #   BILL_AMT group : 11-16
    #   PAY_AMT group  : 17-22
    uci_groups = [
        list(range(5, 11)),   # PAY_0, PAY_2, ..., PAY_6
        list(range(11, 17)),  # BILL_AMT1 ... BILL_AMT6
        list(range(17, 23)),  # PAY_AMT1 ... PAY_AMT6
    ]
    X_miss_fd = inject_missingness(
        X_train, mechanism="feature_dependent", rate=0.30,
        feature_groups=uci_groups,
    )
    assert X_miss_fd.shape == X_train.shape, "Shape changed"
    assert not np.isnan(X_train).any(), "Original X_train was mutated"

    print("\nAll inject_missingness tests passed.")
