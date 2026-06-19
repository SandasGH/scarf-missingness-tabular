import numpy as np
from sklearn.impute import KNNImputer


def median_imputation(X_train, X_test):
    """
    Impute missing values using per-feature medians computed from X_train.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_samples, n_features)
        Training feature matrix, may contain NaN.
    X_test : np.ndarray, shape (n_samples, n_features)
        Test feature matrix. Medians from X_train are applied here too.

    Returns
    -------
    X_train_imp : np.ndarray
        X_train with NaN replaced by training medians.
    X_test_imp : np.ndarray
        X_test with NaN replaced by training medians.
    """
    train_medians = np.nanmedian(X_train, axis=0)

    X_train_imp = X_train.copy()
    X_test_imp = X_test.copy()

    for col in range(X_train.shape[1]):
        train_nan = np.isnan(X_train_imp[:, col])
        test_nan = np.isnan(X_test_imp[:, col])
        if train_nan.any():
            X_train_imp[train_nan, col] = train_medians[col]
        if test_nan.any():
            X_test_imp[test_nan, col] = train_medians[col]

    n_train_filled = int(np.isnan(X_train).sum())
    n_test_filled = int(np.isnan(X_test).sum())
    print(
        f"  [median_imputation] filled {n_train_filled} NaNs in X_train,"
        f" {n_test_filled} NaNs in X_test using training medians"
    )

    return X_train_imp, X_test_imp


def knn_imputation(X_train, X_test, k=5):
    """
    Impute missing values using k-nearest neighbours, fit on X_train only.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_samples, n_features)
        Training feature matrix, may contain NaN.
    X_test : np.ndarray, shape (n_samples, n_features)
        Test feature matrix. Imputed using the imputer fitted on X_train.
    k : int
        Number of neighbours for KNNImputer.

    Returns
    -------
    X_train_imp : np.ndarray
    X_test_imp  : np.ndarray
    """
    n_train_missing = int(np.isnan(X_train).sum())
    n_test_missing = int(np.isnan(X_test).sum())

    imputer = KNNImputer(n_neighbors=k)
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    print(
        f"  [knn_imputation] k={k}  |  filled {n_train_missing} NaNs in X_train,"
        f" {n_test_missing} NaNs in X_test"
    )

    return X_train_imp, X_test_imp


def complete_case(X, y):
    """
    Drop any row from X (and the corresponding y label) where at least one
    feature value is NaN.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix, may contain NaN.
    y : np.ndarray, shape (n_samples,)
        Class labels corresponding to rows of X.

    Returns
    -------
    X_clean : np.ndarray
        Subset of X with no NaN values.
    y_clean : np.ndarray
        Corresponding subset of y.
    """
    complete_mask = ~np.isnan(X).any(axis=1)
    n_dropped = int((~complete_mask).sum())
    n_remain = int(complete_mask.sum())

    print(f"  [complete_case] rows dropped: {n_dropped}  |  rows remaining: {n_remain}")

    return X[complete_mask], y[complete_mask]
