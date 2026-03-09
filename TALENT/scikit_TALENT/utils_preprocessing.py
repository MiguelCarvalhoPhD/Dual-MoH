import numpy as np
import sys
import os
from typing import List, Tuple, Union, Any

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import fbeta_score
from imblearn.under_sampling import RandomUnderSampler


def get_best_alphas_for_betas(pred_balanced, pred_empirical, y_true):
    target_betas = np.arange(0.25, 3, 0.25)
    possible_alphas = np.arange(0.0, 1.01, 0.01)
    
    alpha_predictions = []
    for alpha in possible_alphas:
        combined_preds = (alpha * pred_balanced + (1 - alpha) * pred_empirical)
        predictions_final = np.argmax(combined_preds, axis=1)
        alpha_predictions.append((alpha, predictions_final))

    best_alphas = []

    for beta in target_betas:
        results = []
        
        for alpha, preds in alpha_predictions:
            score = fbeta_score(y_true=y_true, y_pred=preds, beta=beta, average='macro')
            results.append((alpha, score))

        results = np.array(results) 
        
        top_indices = np.argsort(results[:, 1])[::-1][:5]
        top_5_results = results[top_indices]
        
        avg_alpha = np.mean(top_5_results[:, 0])
        best_alphas.append(avg_alpha)

    return np.array(best_alphas)


def apply_maximal_subset(X, y, target_ir, min_cardinality=10, seed=42):
    classes, counts = np.unique(y, return_counts=True)
    min_class = classes[np.argmin(counts)]
    maj_class = classes[np.argmax(counts)]
    n_min, n_maj = min(counts), max(counts)
    
    natural_ir = n_maj / n_min
    
    if natural_ir < target_ir:
        new_n_maj, new_n_min = n_maj, int(n_maj / target_ir)
    else:
        new_n_min, new_n_maj = n_min, int(n_min * target_ir)
    
    if new_n_min < min_cardinality:
        return None, None, None, n_min, n_maj
    
    rus = RandomUnderSampler(sampling_strategy={
        int(maj_class): int(new_n_maj), 
        int(min_class): int(new_n_min)
    }, random_state=seed)
    
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res, (new_n_maj / new_n_min), n_min, n_maj


def preprocess_data(
    X,
    y,
    categorical_indicator: List[bool],
    n_neighbors: int = 5,
    ordinal_encode: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    X = np.asarray(X).copy()
    y = np.asarray(y)
    categorical_indicator = np.asarray(categorical_indicator, dtype=bool)

    if categorical_indicator.shape[0] != X.shape[1]:
        raise ValueError(
            f"categorical_indicator length ({categorical_indicator.shape[0]}) "
            f"does not match number of features in X ({X.shape[1]})."
        )

    X_obj = np.asarray(X, dtype=object)
    n_samples, n_features = X_obj.shape

    all_nan_cols = np.zeros(n_features, dtype=bool)
    for j in range(n_features):
        col = X_obj[:, j]
        nan_mask = [v != v for v in col]
        all_nan_cols[j] = all(nan_mask)

    keep_feature_mask = ~all_nan_cols
    
    print(f"Dropping {np.sum(all_nan_cols)} all-NaN features out of {n_features} total features.")

    X = X[:, keep_feature_mask]
    categorical_indicator = categorical_indicator[keep_feature_mask]

    cat_indices = [i for i, is_cat in enumerate(categorical_indicator) if is_cat]
    num_indices = [i for i, is_cat in enumerate(categorical_indicator) if not is_cat]

    if len(cat_indices) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[:, cat_indices] = cat_imputer.fit_transform(X[:, cat_indices])
        
        if ordinal_encode:
            ord_encoder = OrdinalEncoder()
            X[:, cat_indices] = ord_encoder.fit_transform(X[:, cat_indices])

    if len(num_indices) > 0:
        num_block = X[:, num_indices].astype(float)
        num_imputer = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
        X[:, num_indices] = num_imputer.fit_transform(num_block)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y).astype(int)

    return X, y_encoded, categorical_indicator


def calculate_imbalance_ratio(
    y: np.ndarray,
    X: np.ndarray,
    imbalance_thr: float = 3.0,
    target_interval: Tuple[float, float] = (3.0, 4.0),
    random_state: int = 42,
) -> Tuple[bool, float, np.ndarray, np.ndarray]:
    
    y = np.asarray(y)
    X = np.asarray(X)

    if y.ndim != 1:
        raise ValueError("`y` must be a 1D array of shape (n_samples,)")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    low, high = target_interval
    if not (0 < low < high):
        raise ValueError("`target_interval` must be (low, high) with 0 < low < high")

    classes, counts = np.unique(y, return_counts=True)
    if classes.shape[0] != 2:
        return False, float("nan"), y, X

    maj_idx = int(np.argmax(counts))
    min_idx = int(np.argmin(counts))
    maj_label = classes[maj_idx]
    min_label = classes[min_idx]
    n_maj = int(counts[maj_idx])
    n_min = int(counts[min_idx])

    if n_min == 0:
        return True, float("inf"), y, X

    current_ratio = n_maj / n_min

    if current_ratio >= imbalance_thr:
        return True, current_ratio, y, X

    rng = np.random.default_rng(random_state)
    target_ratio = float(rng.uniform(low, high))

    target_min = int(np.floor(n_maj / target_ratio))
    target_min = max(1, min(target_min, n_min))

    if target_min == n_min:
        return True, current_ratio, y, X

    min_mask = (y == min_label)
    maj_mask = (y == maj_label)

    min_idx_all = np.flatnonzero(min_mask)
    maj_idx_all = np.flatnonzero(maj_mask)

    chosen_min_idx = rng.choice(min_idx_all, size=target_min, replace=False)

    keep_idx = np.concatenate([maj_idx_all, chosen_min_idx])
    keep_idx_shuffled = rng.permutation(keep_idx)

    y_out = y[keep_idx_shuffled]
    X_out = X[keep_idx_shuffled, :]

    new_classes, new_counts = np.unique(y_out, return_counts=True)
    if new_classes.shape[0] < 2:
        final_ratio = float("inf")
    else:
        new_n_maj = int(np.max(new_counts))
        new_n_min = int(np.min(new_counts))
        final_ratio = new_n_maj / new_n_min if new_n_min > 0 else float("inf")

    return True, final_ratio, y_out, X_out


def random_one_vs_all(
    X: np.ndarray,
    y: Union[np.ndarray, list],
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Any]:
    
    X = np.asarray(X)
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("`y` must be a 1D array of shape (n_samples,)")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("Need at least 2 distinct classes in y for One-vs-All.")

    rng = np.random.default_rng(random_state)
    chosen_class = rng.choice(classes)

    y_bin = (y == chosen_class).astype(int)

    return X, y_bin, chosen_class


def difficulty_aware_binary_decomposition(
    X: np.ndarray,
    y: Union[np.ndarray, list],
    random_state: int = 42,
    *,
    n_candidates: int = 64,
    difficulty_window: Tuple[float, float] = (0.6, 0.9),
) -> Tuple[np.ndarray, np.ndarray, Any]:
    
    X = np.asarray(X)
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("`y` must be a 1D array of shape (n_samples,)")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    classes = np.unique(y)
    n_classes = classes.size
    if n_classes < 2:
        raise ValueError("Need at least 2 distinct classes in y.")

    if n_candidates <= 0:
        raise ValueError("`n_candidates` must be a positive integer.")

    min_auc, max_auc = difficulty_window
    if not (0.0 <= min_auc < max_auc <= 1.0):
        raise ValueError("`difficulty_window` must satisfy 0 <= min_auc < max_auc <= 1.")

    rng = np.random.default_rng(random_state)

    candidates = []
    seen = set()
    attempts = 0
    max_attempts = n_candidates * 10

    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1
        size = rng.integers(1, n_classes)
        pos_classes = tuple(sorted(rng.choice(classes, size=size, replace=False)))

        if pos_classes in seen:
            continue
        seen.add(pos_classes)
        candidates.append(pos_classes)

    if not candidates:
        chosen_class = rng.choice(classes)
        y_bin = (y == chosen_class).astype(int)
        return X, y_bin, chosen_class

    ref_models = [
        (
            "log_reg",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "gb",
            GradientBoostingClassifier(random_state=random_state),
        ),
        (
            "mlp",
            make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 64),
                    max_iter=200,
                    random_state=random_state),
            ),
        ),
    ]

    cv = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=random_state
    )

    positive_classes = None
    best_candidate = None
    best_auc = np.inf

    for pos_classes in candidates:
        
        print("Evaluating candidate positive classes:", pos_classes)
        
        y_bin = np.isin(y, pos_classes).astype(int)

        if np.unique(y_bin).size < 2:
            continue

        aucs = []
        bad = False

        for _, model in ref_models:
            try:
                scores = cross_val_score(
                    model,
                    X,
                    y_bin,
                    scoring="roc_auc",
                    cv=cv,
                    n_jobs=None,
                )
                aucs.append(scores.mean())
            except Exception:
                bad = True
                break

        if bad or not aucs:
            continue

        mean_auc = float(np.mean(aucs))

        if mean_auc < best_auc:
            best_auc = mean_auc
            best_candidate = pos_classes

        if min_auc <= mean_auc <= max_auc:
            positive_classes = pos_classes
            break

    if positive_classes is None:
        if best_candidate is None:
            chosen_class = rng.choice(classes)
            y_bin = (y == chosen_class).astype(int)
            return X, y_bin, chosen_class
        positive_classes = best_candidate

    y_bin_full = np.isin(y, positive_classes).astype(int)
    
    print("Selected positive classes:", positive_classes)

    return X, y_bin_full, positive_classes


class SuppressStdoutStderr:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._null = open(os.devnull, "w")
        sys.stdout = self._null
        sys.stderr = self._null
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._null.close()
