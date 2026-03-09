import os
import random
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Any

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler

# ---------------------------
# Preprocessing Functions
# ---------------------------

def preprocess_data(
    X,
    y,
    categorical_indicator: List[bool],
    n_neighbors: int = 5,
    ordinal_encode: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess features and target.
    """
    X = np.asarray(X).copy()
    y = np.asarray(y)
    categorical_indicator = np.asarray(categorical_indicator, dtype=bool)

    if categorical_indicator.shape[0] != X.shape[1]:
        raise ValueError(
            f"categorical_indicator length ({categorical_indicator.shape[0]}) "
            f"does not match number of features in X ({X.shape[1]})."
        )

    # ---- 1) Drop features that are all-NaN, irrespective of type ----
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

    # ---- 2) Split into categorical and numerical ----
    cat_indices = [i for i, is_cat in enumerate(categorical_indicator) if is_cat]
    num_indices = [i for i, is_cat in enumerate(categorical_indicator) if not is_cat]

    # ---- 3) Categorical: most frequent imputation ----
    if len(cat_indices) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[:, cat_indices] = cat_imputer.fit_transform(X[:, cat_indices])
        
        if ordinal_encode:
            ord_encoder = OrdinalEncoder()
            X[:, cat_indices] = ord_encoder.fit_transform(X[:, cat_indices])

    # ---- 4) Numerical: KNN imputation ----
    if len(num_indices) > 0:
        num_block = X[:, num_indices].astype(float)
        num_imputer = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
        X[:, num_indices] = num_imputer.fit_transform(num_block)

    # ---- 5) Label-encode target ----
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y).astype(int)

    return X, y_encoded, categorical_indicator

def difficulty_aware_binary_decomposition(
    X: np.ndarray,
    y: Union[np.ndarray, list],
    random_state: int = 42,
    *,
    n_candidates: int = 64,
    difficulty_window: Tuple[float, float] = (0.6, 0.9),
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Given a multi-class tabular dataset (X, y), construct a binary task by
    selecting a subset of classes as the positive label in a principled way.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    classes = np.unique(y)
    n_classes = classes.size

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
        ("log_reg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state))),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(random_state=random_state)),
        ("mlp", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=200, random_state=random_state))),
    ]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    min_auc, max_auc = difficulty_window
    positive_classes = None
    best_candidate = None
    best_auc = np.inf

    for k, pos_classes in enumerate(candidates):
        
        print(f"{k+1}/{len(candidates)}-- Testing candidates: {pos_classes}")
        
        y_bin = np.isin(y, pos_classes).astype(int)

        if np.unique(y_bin).size < 2:
            continue

        aucs = []
        bad = False

        for _, model in ref_models:
            try:
                scores = cross_val_score(model, X, y_bin, scoring="roc_auc", cv=cv, n_jobs=None)
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

# ---------------------------------------------------------
# Helper: Maximal Subset Method
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------
path_files = "/home/miguel_arvalho/ICL_hypernetwork/cc18_datasets_with_metadata/"

# Safe handling if directory doesn't exist during test runs
if os.path.exists(path_files):
    files = [f for f in os.listdir(path_files) if f.endswith(".npy")]
    files.sort()
else:
    files = []
    
print(f"Found {len(files)} datasets.")

min_minority_size = 10

n_datasets_binary = 0
n_datasets_multiclass = 0
valid_dataset_idx = 0

imbalance_ratio_list = []
dataset_size_list = []
n_features_list = []

print(f"\n{'='*80}\nEXPERIMENTAL PIPELINE START\n{'='*80}")

for file_idx, file in enumerate(files):
    # 1. Load and Preprocess
    data = np.load(os.path.join(path_files, file), allow_pickle=True)
    data, categorical_indicator, feature_names = data[0], data[1], data[2]
    X, y = data.iloc[:,:-1], data.iloc[:,-1]

    
    X_processed, y_processed, categorical_indicator = preprocess_data(X, y, categorical_indicator, ordinal_encode=True)
    
        # Calculate natural_ir correctly BEFORE printing
    u_cls, u_counts = np.unique(y_processed, return_counts=True)
    natural_ir = max(u_counts) / min(u_counts)

    print(f"\n[DATASET LOADED] {file} | Natural IR: {natural_ir:.2f}")

    if np.sum(categorical_indicator)> 50:
        continue

    # 2. CONDITIONAL DOWNSAMPLING
    if X_processed.shape[0] > 10000:
        print(f"|-- Action: Large dataset but limit reached. Skipping.")
        continue
    
    # 3. Handle Multiclass / Binary
    if len(np.unique(y_processed)) > 2:
        n_datasets_multiclass += 1
        X_processed, y_processed, chosen = difficulty_aware_binary_decomposition(X_processed, y_processed, random_state=42)
        # Recalculate IR after binarization
        u_cls, u_counts = np.unique(y_processed, return_counts=True)
        natural_ir = max(u_counts) / min(u_counts)
    else:
        n_datasets_binary += 1
        
    n_features_list.append(X_processed.shape[1])
    valid_dataset_idx += 1
    

    # ---------------------------------------------------------
    # IR SELECTION LOGIC
    # ---------------------------------------------------------
    if natural_ir > 3:
        target_irs = [natural_ir]
        is_natural = True
        print(f"{valid_dataset_idx}|-- Action: Using Natural IR (Direct usage, no sampling)")
    else:
        random.seed(42 + file_idx)  
        target_irs = [random.uniform(3, 50) for _ in range(5)]
        is_natural = False
        print(f"|-- Action: Sampling 5 random IRs between 3 and 50")
    
    for run_idx, target_ir in enumerate(target_irs):
        if is_natural:
            X_std, y_std = X_processed, y_processed
            final_ratio = natural_ir
        else:   
            X_std, y_std, final_ratio, _, _ = apply_maximal_subset(
                X_processed, y_processed, target_ir, 
                min_cardinality=min_minority_size, seed=42 + run_idx + file_idx
            )
        
        if X_std is None:
            continue
        
        # Track statistics for valid runs
        dataset_size_list.append(X_std.shape[0])
        imbalance_ratio_list.append(final_ratio)


# ---------------------------------------------------------
# Print Statistics & Save Results
# ---------------------------------------------------------
print(f"\n{'='*80}\nFINAL PIPELINE STATISTICS\n{'='*80}")

def print_stat_block(name, data_list):
    if not data_list:
        print(f"{name}: No data available")
        return
    mean_val = np.mean(data_list)
    std_val = np.std(data_list)
    median_val = np.median(data_list)
    min_val = np.min(data_list)
    max_val = np.max(data_list)
    
    print(f"{name}:")
    print(f"  Mean ± Std: {mean_val:.2f} ± {std_val:.2f}")
    print(f"  Median:     {median_val:.2f}")
    print(f"  [Min - Max]: [{min_val:.2f} - {max_val:.2f}]\n")

print_stat_block("Imbalance Ratios (Majority / Minority)", imbalance_ratio_list)
print_stat_block("Sample Sizes", dataset_size_list)
print_stat_block("Number of Features", n_features_list)

print(f"Dataset Types Evaluated:")
print(f"  Binary Datasets:     {n_datasets_binary}")
print(f"  Multiclass Datasets: {n_datasets_multiclass}\n")

# Save the tracked lists to a CSV
if dataset_size_list:
    # Adjust n_features_list to match length of the other lists (since n_features scales per dataset, while sizes/IR scale per run)
    # The simplest way to save uneven lists is to pad them or save them as separate files/JSON. 
    # Here we save them into a dictionary and serialize via Pandas.
    
    summary_df = pd.DataFrame({
        "Metric": ["Mean", "Std", "Median", "Min", "Max"],
        "Imbalance_Ratio": [np.mean(imbalance_ratio_list), np.std(imbalance_ratio_list), np.median(imbalance_ratio_list), np.min(imbalance_ratio_list), np.max(imbalance_ratio_list)],
        "Sample_Size": [np.mean(dataset_size_list), np.std(dataset_size_list), np.median(dataset_size_list), np.min(dataset_size_list), np.max(dataset_size_list)],
        "N_Features": [np.mean(n_features_list), np.std(n_features_list), np.median(n_features_list), np.min(n_features_list), np.max(n_features_list)],
    })
    
    try:
        summary_df.to_csv("dataset_experiment_summary.csv", index=False)
        print("Summary statistics successfully saved to 'dataset_experiment_summary.csv'.")
        
        # Optionally, save raw run data
        raw_runs_df = pd.DataFrame({
            "Run_Sample_Size": dataset_size_list,
            "Run_Imbalance_Ratio": imbalance_ratio_list
        })
        raw_runs_df.to_csv("dataset_experiment_raw_runs.csv", index=False)
        print("Raw run data successfully saved to 'dataset_experiment_raw_runs.csv'.")
    except Exception as e:
        print(f"Could not save results: {e}")
else:
    print("No valid runs to save.")