import os
import sys
import time
import requests
import warnings
import gc
import random
from typing import Tuple, Optional, List, Union, Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.decomposition import PCA

from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedBaggingClassifier

from talent_classifier import DeepClassifier, SuppressPrint
from BalancedBagging import UnderSamplingBaggingClassifier
from mixture_hypernetworks import *
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from ticl.prediction import MotherNetClassifier, tabflex
from data_loader import *
from utils_preprocessing import *


path_files = "/home/miguel/Desktop/TALENT/cc18_datasets_with_metadata/"

files = [f for f in os.listdir(path_files) if f.endswith(".npy")]
files.sort()
print(f"Found {len(files)} datasets.")


def test_deep_classifiers(X_train, y_train, X_test, y_test, categorical_indicator, to_resample=False, resampler=None, to_ordinal_encode=False, to_time=False):
    
    models = [
        "realmlp",
        "LogReg",
        "RandomForest",
        "knn",
        "xgboost",
        "mlp",
        "lightgbm",
        "tabtransformer",
        "modernNCA", 
        "resnet"
    ]
    
    if to_resample:
        if to_ordinal_encode:
            ord_encoder = OrdinalEncoder()
            X_train = ord_encoder.fit_transform(X_train)
            X_test = ord_encoder.transform(X_test)
        
        resampler_instance = resampler()
        X_train, y_train = resampler_instance.fit_resample(X_train, y_train)
        
        print(f"After resampling, training data shape: X: {X_train.shape}, y: {y_train.shape}")
    
    results = np.zeros((len(models),4)) 
    times = np.zeros((len(models),2)) 
    
    for i, model in enumerate(models):
        print(f"Training TALENT model: {model}")

        clf = DeepClassifier(model_type=model)
        
        if to_time:
            start_time = time.time()
        
        clf.fit(X_train, y_train, categorical_indicator=categorical_indicator)
        
        if to_time:
            end_time = time.time()
            times[i,0] = end_time - start_time
            start_time = time.time()
            
        y_pred = clf.predict(X_test)
        
        if to_time:
            end_time = time.time()
            times[i,1] = end_time - start_time
        
        results[i,0] = accuracy_score(y_test, y_pred)
        results[i,1] = precision_score(y_test, y_pred, average='macro')
        results[i,2] = recall_score(y_test, y_pred, average='macro')
        results[i,3] = f1_score(y_test, y_pred, average='macro')
        
        print(f"Accuracy: {results[i,0]:.4f}, Precision: {results[i,1]:.4f}, Recall: {results[i,2]:.4f}, F1-score: {results[i,3]:.4f}")

        torch.cuda.empty_cache()
        gc.collect()
    
    return results, len(models), models, times

def icl_models(X_train, y_train, X_test, y_test, categorical_indicator):
    
    def _resample_data(X, y, value, rnd_state=42):
        rng = np.random.default_rng(rnd_state)
        indices = rng.choice(X.shape[0], size=value, replace=False)
        X_icl = X[indices]
        y_icl = y[indices]
        print(f"Subsampled training data to 10000 samples for ICL models.")
        return X_icl, y_icl
    
    train_data = split_train_val(X_train, y_train, categorical_features=categorical_indicator, task_type="binclass", val_size=0.2, random_state=0)

    cat_indices = [
        i for i, is_cat in enumerate(categorical_indicator) if is_cat
    ]
    
    numerical_indices = [
        i for i, is_cat in enumerate(categorical_indicator) if not is_cat
    ]
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)  
    
    if len(cat_indices) > 0:
        X_cat = encoder.fit_transform(train_data[1]["train"]).astype(float)
        X_test_cat = encoder.transform(X_test[:, cat_indices]).astype(float)
        
        if len(numerical_indices) > 0:
            X_icl = np.hstack((train_data[0]["train"], X_cat)).astype(float)
            X_test = np.hstack((X_test[:, numerical_indices], X_test_cat)).astype(float)
        else:
            X_icl = X_cat
    else:
        X_icl = train_data[0]["train"].astype(float)
        
    y_icl = train_data[2]["train"]

    if X_icl.shape[1] > 100:
        pca = PCA(n_components=99, random_state=42)
        X_icl = pca.fit_transform(X_icl)
        X_test = pca.transform(X_test)
        print(f"Applied PCA to reduce features to 100 dimensions for ICL models.") 
        
    if X_test.dtype == object:
        X_test = X_test.astype(np.float32)
    
    icl_models = [
        ("MOH", Dual_MoH(m=6, overlap=0.15, verbose=False, minority_cluster="KMeans", majority_cluster='KMeans')),
        ("MOH_mothernet", Dual_MoH(m=6, overlap=0.15, verbose=False, minority_cluster='KMeans', majority_cluster='KMeans', classifier_type='MLP')),
        ("EnsembleMixtutre", CustomEnsembleClassifier(n_estimators=10, base_model_class=Dual_MoH, model_params={"m":6, "overlap":0.15, "verbose":False, "random_state":10, "minority_cluster":'KMeans', "majority_cluster":'KMeans'}, mode="regular")),
        ("TabPFN", TabPFNClassifier.create_default_for_version(ModelVersion.V2)),
        ("MotherNet", MotherNetClassifier(device='cuda')),
        ("TabFlex", tabflex.TabFlex()),
    ]
    
    results = np.zeros((len(icl_models), 4)) 
    times = np.zeros((len(icl_models), 2)) 
    
    already_resampled = False
    Thr = 10000
    
    for i, (model_name, model) in enumerate(icl_models):
        
        if X_icl.shape[0] > Thr and model_name not in ["TabICL","MOH"] and not already_resampled:
            already_resampled = True
            X_icl, y_icl = _resample_data(X_icl, y_icl, Thr-1, rnd_state=42)
            
        if model_name == "MOH" or model_name == "MOH_mothernet":
            
            time_start = time.time()
            model.fit_mixture_hypernetworks(X_icl.astype(np.float32), y_icl)
            torch.cuda.synchronize()
            time_end = time.time()
            times[i,0] = time_end - time_start
            
            time_start = time.time()
            
            _, _, y_pred, predictions_balanced, predictions_empirical, alpha, _ = model.predict_2(X_test)
            
            best_alpha = get_best_alphas_for_betas(predictions_balanced, predictions_empirical, y_test)
            
            torch.cuda.synchronize()
            time_end = time.time()
            times[i,1] = time_end - time_start
            
        else:
            
            time_start = time.time()

            if model_name == "EnsembleMixtutre":
                model.fit(X_icl, y_icl)
                torch.cuda.synchronize()
                time_end = time.time()
            
                times[i,0] = time_end - time_start
                time_start = time.time()
                y_pred = model.predict(X_test)
                
                time_end = time.time()
                times[i,1] = time_end - time_start
            else:
                resampler = RandomUnderSampler()
                X_res, y_res = resampler.fit_resample(X_icl, y_icl)
                model.fit(X_res, y_res)
                torch.cuda.synchronize()
                time_end = time.time()
            
                times[i,0] = time_end - time_start
                time_start = time.time()
                y_pred = model.predict(X_test)
                
                time_end = time.time()
                times[i,1] = time_end - time_start

        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        results[i,0] = accuracy_score(y_test, y_pred)
        results[i,1] = precision_score(y_test, y_pred, average='macro')
        results[i,2] = recall_score(y_test, y_pred, average='macro')
        results[i,3] = f1_score(y_test, y_pred, average='macro')
        
    return results, len(icl_models), [model_name for model_name, _ in icl_models], alpha, best_alpha ,times


n_models = 16
random_states = [0, 1, 2, 3]
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
min_minority_size = 10

cumulative_ranks = np.zeros((n_models, 4)) 
total_successful_runs = 0
raw_results_storage = []
valid_dataset_idx = 0
all_runs = 0
n_missing_datasets = 4
alphas = []
n_errors = 0

print(f"\n{'='*80}\nEXPERIMENTAL PIPELINE START\n{'='*80}")

for file_idx, file in enumerate(files):
    data = np.load(os.path.join(path_files, file), allow_pickle=True)
    data, categorical_indicator, feature_names = data[0], data[1], data[2]
    X, y = data.iloc[:,:-1], data.iloc[:,-1]
    
    X_processed, y_processed, categorical_indicator = preprocess_data(X, y, categorical_indicator, ordinal_encode=True)
    
    if np.sum(categorical_indicator) > 50:
        continue

    if X_processed.shape[0] > 10000:
        if n_missing_datasets < 4:
            print(f"|-- Action: Large dataset ({X_processed.shape[0]} samples). Downsampling.")
            rng = np.random.default_rng(42 + file_idx)
            indices = rng.choice(X_processed.shape[0], size=10000, replace=False)
            X_processed = X_processed[indices]
            y_processed = y_processed[indices]
            n_missing_datasets += 1 
        else:
            print(f"|-- Action: Large dataset but limit reached. Skipping.")
            continue
    
    u_cls, u_counts = np.unique(y_processed, return_counts=True)
    natural_ir = max(u_counts) / min(u_counts)
    
    print(f"\n[DATASET LOADED] {file} | Natural IR: {natural_ir:.2f}")

    if len(np.unique(y_processed)) > 2:
        X_processed, y_processed, chosen = difficulty_aware_binary_decomposition(X_processed, y_processed, random_state=42)
        
    valid_dataset_idx += 1

    if natural_ir > 3:
        target_irs = [natural_ir]
        n_repeats = 1
        is_natural = True
        print(f"|-- Action: Using Natural IR (Direct usage, no sampling)")
    else:
        random.seed(42 + file_idx)  
        target_irs = [random.uniform(3, 50) for _ in range(5)]
        n_repeats = 5
        is_natural = False
        print(f"|-- Action: Sampling 5 random IRs between 3 and 20")

    dataset_ir_ranks = []
    dataset_ir_metrics = []
    
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

        print(f"  >>> Run {run_idx+1}/{n_repeats} | Target IR: {target_ir:.2f} | Shape: {X_std.shape}")

        results_all_states = np.zeros((len(random_states), n_models, 4))
        
        for rs_idx, rs in enumerate(random_states):
            try:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
                train_idx, test_idx = next(skf.split(X_std, y_std))
                X_train, X_test = X_std[train_idx], X_std[test_idx]
                y_train, y_test = y_std[train_idx], y_std[test_idx]
                
                with SuppressStdoutStderr():
                    res_d, _, m_names, _ = test_deep_classifiers(X_train, y_train, X_test, y_test, categorical_indicator, to_resample=True, resampler=RandomUnderSampler)
                    res_i, _, i_names, alpha, best_alpha, _ = icl_models(X_train, y_train, X_test, y_test, categorical_indicator)
                    
                    all_model_names = m_names + i_names
                    curr_res = np.vstack([res_d, res_i])
                    results_all_states[rs_idx] = curr_res
                    alphas.append([alpha, best_alpha, file_idx, run_idx, np.mean(curr_res), target_ir])
                
                f1_scores = curr_res[:, 3]
                best_idx = np.nanargmax(f1_scores)
                print(f"    [RS {rs}] Best Model: {all_model_names[best_idx]} (F1: {f1_scores[best_idx]:.4f}) | Used Alpha (if MH): {alpha:.2f} | Best Alpha (if MH): {best_alpha}")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                results_all_states[rs_idx] =  np.ones((n_models,4))
                print(f"    [RS {rs}] Error: {e}")
                n_errors += 1
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        mean_metrics_ir = np.nanmean(results_all_states, axis=0)
        
        ir_ranks = np.zeros((n_models, 4))
        for m_idx in range(4):
            ir_ranks[:, m_idx] = rankdata(-mean_metrics_ir[:, m_idx], method='average')
        
        dataset_ir_ranks.append(ir_ranks)
        dataset_ir_metrics.append(mean_metrics_ir)
        all_runs += 1
         
    if len(dataset_ir_ranks) > 0:
        avg_dataset_ranks = np.mean(dataset_ir_ranks, axis=0)
        avg_dataset_metrics = np.mean(dataset_ir_metrics, axis=0)
        
        total_successful_runs += 1
        valid_dataset_idx += 1
        cumulative_ranks += avg_dataset_ranks
        avg_running_ranks = cumulative_ranks / total_successful_runs

        print(f"\n    {'#'*20} DATASET SUMMARY {'#'*20} | after {total_successful_runs} valid datasets | total successful runs: {all_runs} | n_errors: {n_errors}")
        print(f"    {'Model':<18} | {'F1 (Avg IR)':<12} | {'Rank (Avg IR)':<15} | {'Running Rank':<15}")
        for i in range(n_models):
            print(f"    {all_model_names[i]:<18} | {avg_dataset_metrics[i,3]:>12.4f} | {avg_dataset_ranks[i,3]:>15.1f} | {avg_running_ranks[i,3]:>15.2f}")

        row = {"dataset_id": file, "natural_ir": natural_ir, "is_sampled": not is_natural}
        for m_idx, model_name in enumerate(all_model_names):
            for met_idx, metric_name in enumerate(metrics):
                row[f"{model_name}_{metric_name}"] = avg_dataset_metrics[m_idx, met_idx]
        raw_results_storage.append(row)


results_df = pd.DataFrame(raw_results_storage)
results_df.to_csv("Final_benchmark_56_datasest_undersampling_no_swap_no_LSA.csv", index=False)

alphas_df = pd.DataFrame(alphas, columns=["used_alpha","best_alpha","dataset_idx","run_idx","mean_f1","target_ir"])
alphas_df.to_csv("benchmark_alphas_final_beta_all.csv", index=False)