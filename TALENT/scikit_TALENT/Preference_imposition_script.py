#!/usr/bin/env python3
# ============================================================
# Benchmark: each (dataset, run_idx, random_state) is treated as a "trial"
# - Compute ALL test F_beta scores for 6 approaches and all betas
# - Convert trial scores -> ranks (1=best) per beta
# - Mean ranks are averaged over random states (and IRs if natural IR < 3) 
#   so each dataset contributes equally.
# - Print running ranks every PRINT_EVERY datasets
# - Save results tensor + metadata + long CSV
# ============================================================

#%%

import os
import gc
import random
import numpy as np
import pandas as pd
import torch

from typing import Tuple, List, Union, Any
from scipy.stats import rankdata

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import fbeta_score

from imblearn.under_sampling import RandomUnderSampler


from data_loader import * # provides split_train_val
from mixture_hypernetworks import *
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from ticl.prediction import MotherNetClassifier, tabflex
from utils_preprocessing import * 


# ============================================================
# Post-processing selection per beta
# ============================================================

def select_best_postproc_binary_from_dict_fbeta(
    predictions: dict,
    y_val,
    y_test,
    beta: float,
    step: float = 0.01,
    average: str = "macro",
    moh_keys=("MOH",),
    ensemble_keys=("EnsembleMixtutre",),
):
    def _to01(y):
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 2:
            return np.argmax(y, axis=1).astype(int)
        return y.reshape(-1).astype(int)

    def _pos_score(p):
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] == 2:
            return p[:, 1].astype(float)
        if p.ndim == 2 and p.shape[1] == 1:
            return p[:, 0].astype(float)
        if p.ndim == 1:
            return p.astype(float)
        raise ValueError("Prediction must be shape (n,2) or (n,) or (n,1).")

    def _fb(y_true, y_pred):
        return float(
            fbeta_score(y_true=y_true, y_pred=y_pred, beta=beta, average=average, zero_division=0)
        )

    def _minority_only_temp_scale_binary_probs_old(probs2, t, minority_idx=1, eps=1e-12):
        probs2 = np.asarray(probs2, dtype=float)
        p_m = np.clip(probs2[:, minority_idx], eps, 1.0 - eps)
        p_m_t = np.exp(t * np.log(p_m))  # p_m^t
        denom = p_m_t + (1.0 - p_m)
        p_m_new = p_m_t / np.clip(denom, eps, None)
        out = probs2.copy()
        out[:, minority_idx] = p_m_new
        out[:, 1 - minority_idx] = 1.0 - p_m_new
        return out
    
    def _minority_only_temp_scale_binary_probs(probs2, delta, minority_idx=1, eps=1e-12):
        probs2 = np.asarray(probs2, dtype=float)
        
        # Extract minority probability and clip for stability
        p = np.clip(probs2[:, minority_idx], eps, 1.0 - eps)
        
        # Compute logit
        logit_p = np.log(p) - np.log(1.0 - p)
        
        # Apply log-odds shift
        logit_shifted = logit_p + delta
        
        # Convert back via sigmoid
        p_new = 1.0 / (1.0 + np.exp(-logit_shifted))
        
        # Reassemble probability matrix
        out = probs2.copy()
        out[:, minority_idx] = p_new
        out[:, 1 - minority_idx] = 1.0 - p_new
        
        return out

    y_val = _to01(y_val)
    y_test = _to01(y_test)

    thr_grid = np.round(np.arange(0.0, 1.0 + 1e-12, step), 10)
    alpha_grid = np.round(np.arange(0.0, 1.0 + 1e-12, step * 5), 10)
    t_grid = np.concatenate([np.linspace(-2, 0, num=20), np.linspace(0, 2, num=20)])

    scores = {}

    def _opt_alpha_t(pred_bal_val, pred_emp_val, pred_bal_test, pred_emp_test):
        pred_bal_val = np.asarray(pred_bal_val)
        pred_emp_val = np.asarray(pred_emp_val)
        pred_bal_test = np.asarray(pred_bal_test)
        pred_emp_test = np.asarray(pred_emp_test)

        best_alpha, best_t, best_val = 0.0, 1.0, -np.inf
        for alpha in alpha_grid:
            comb_val = alpha * pred_bal_val + (1.0 - alpha) * pred_emp_val
            for t in t_grid:
                comb_val_t = _minority_only_temp_scale_binary_probs(comb_val, t, minority_idx=1)
                yhat_val = (comb_val_t[:, 1] > 0.5).astype(int)
                v = _fb(y_val, yhat_val)
                if v > best_val:
                    best_val, best_alpha, best_t = v, float(alpha), float(t)

        comb_test = best_alpha * pred_bal_test + (1.0 - best_alpha) * pred_emp_test
        comb_test_t = _minority_only_temp_scale_binary_probs(comb_test, best_t, minority_idx=1)
        yhat_test = (comb_test_t[:, 1] > 0.5).astype(int)
        test_s = _fb(y_test, yhat_test)
        return {"val_fbeta": best_val, "test_fbeta": test_s, "params": {"type": "alpha_t", "alpha": best_alpha, "t": best_t, "thr": 0.5}}

    def _opt_thr_only(val_scores, test_scores, fixed_params=None):
        val_scores = _pos_score(val_scores)
        test_scores = _pos_score(test_scores)

        best_thr, best_val = 0.5, -np.inf
        for thr in thr_grid:
            yhat_val = (val_scores > thr).astype(int)
            v = _fb(y_val, yhat_val)
            if v > best_val:
                best_val, best_thr = v, float(thr)

        yhat_test = (test_scores > best_thr).astype(int)
        test_s = _fb(y_test, yhat_test)
        params = {"type": "thr", "thr": best_thr}
        if fixed_params:
            params.update(fixed_params)
        return {"val_fbeta": best_val, "test_fbeta": test_s, "params": params}

    moh_keys = set(moh_keys)
    ensemble_keys = set(ensemble_keys)

    for model_name, payload in predictions.items():
        if payload is None:
            continue

        is_moh = model_name in moh_keys
        is_ens = model_name in ensemble_keys

        if is_moh or is_ens:
            entry = _opt_alpha_t(
                payload["pred_bal_val"],
                payload["pred_emp_val"],
                payload["pred_bal_test"],
                payload["pred_emp_test"],
            )
            scores[f"{model_name}__alpha_t"] = entry

        if is_ens:
            comb_val = 0.5 * np.asarray(payload["pred_bal_val"]) + 0.5 * np.asarray(payload["pred_emp_val"])
            comb_test = 0.5 * np.asarray(payload["pred_bal_test"]) + 0.5 * np.asarray(payload["pred_emp_test"])
            entry_thr = _opt_thr_only(
                val_scores=comb_val[:, 1],
                test_scores=comb_test[:, 1],
                fixed_params={"alpha": 0.5, "t": 1.0},
            )
            scores[f"{model_name}__thr_only_fixed_mix"] = entry_thr

        if (not is_moh) and (not is_ens):
            entry = _opt_thr_only(payload["pred_emp_val"], payload["pred_emp_test"])
            scores[f"{model_name}__thr"] = entry

    return scores

# ============================================================
# ICL models
# ============================================================
def icl_models(X_train, y_train, X_test, y_test, categorical_indicator):
    train_data = split_train_val(
        X_train,
        y_train,
        categorical_features=categorical_indicator,
        task_type="binclass",
        val_size=0.2,
        random_state=0,
    )

    cat_idx = [i for i, is_cat in enumerate(categorical_indicator) if is_cat]
    num_idx = [i for i, is_cat in enumerate(categorical_indicator) if not is_cat]

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    if len(cat_idx) > 0:
        X_cat_tr = encoder.fit_transform(train_data[1]["train"]).astype(float)
        X_cat_val = encoder.transform(train_data[1]["val"]).astype(float)
        X_cat_test = encoder.transform(X_test[:, cat_idx]).astype(float)

        if len(num_idx) > 0:
            X_icl = np.hstack((train_data[0]["train"], X_cat_tr)).astype(float)
            X_icl_val = np.hstack((train_data[0]["val"], X_cat_val)).astype(float)
            X_test_icl = np.hstack((X_test[:, num_idx], X_cat_test)).astype(float)
        else:
            X_icl = X_cat_tr
            X_icl_val = X_cat_val
            X_test_icl = X_cat_test
    else:
        X_icl = train_data[0]["train"].astype(float)
        X_icl_val = train_data[0]["val"].astype(float)
        X_test_icl = X_test.astype(float)

    y_icl = train_data[2]["train"]
    y_icl_val = train_data[2]["val"]

    if X_icl.shape[1] > 100:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=99, random_state=42)
        X_icl = pca.fit_transform(X_icl)
        X_icl_val = pca.transform(X_icl_val)
        X_test_icl = pca.transform(X_test_icl)

    X_icl = X_icl.astype(np.float32, copy=False)
    X_icl_val = X_icl_val.astype(np.float32, copy=False)
    X_test_icl = X_test_icl.astype(np.float32, copy=False)
    
    from xgboost import XGBClassifier

    models = [
        ("MOH", Dual_MoH(
            m=6, overlap=0.15, verbose=False,
            minority_cluster="BalancedKMeansLSA", majority_cluster="BalancedKMeansLSA"
        )),
        ("EnsembleMixtutre", CustomEnsembleClassifier(
            n_estimators=10,
            base_model_class=Dual_MoH,
            model_params={
                "m": 6, "overlap": 0.15, "verbose": False, "random_state": 10,
                "minority_cluster": "KMeans", "majority_cluster": "BalancedKMeansLSA",
            },
            mode="regular",
        )),
        ("TabPFN", TabPFNClassifier.create_default_for_version(ModelVersion.V2)),
        ("MotherNet", MotherNetClassifier(device="cuda")),
        ("XGBoost",XGBClassifier())
    ]

    preds = {}
    for name, model in models:
        if name == "MOH":
            model.fit_mixture_hypernetworks(X_icl, y_icl)
            _, _, _, p_bal_test, p_emp_test, _, _ = model.predict_2(X_test_icl)
            _, _, _, p_bal_val, p_emp_val, _, _ = model.predict_2(X_icl_val)
            preds[name] = {
                "pred_bal_val": p_bal_val,
                "pred_emp_val": p_emp_val,
                "pred_bal_test": p_bal_test,
                "pred_emp_test": p_emp_test,
            }
        elif name == "EnsembleMixtutre":
            model.fit(X_icl, y_icl)
            _, p_bal_val, p_emp_val = model.predict_proba(X_icl_val)
            _, p_bal_test, p_emp_test = model.predict_proba(X_test_icl)
            preds[name] = {
                "pred_bal_val": p_bal_val,
                "pred_emp_val": p_emp_val,
                "pred_bal_test": p_bal_test,
                "pred_emp_test": p_emp_test,
            }
        else:
            model.fit(X_icl, y_icl)
            
            # Use predict_proba to get probabilities for thresholding
            prob_val = model.predict_proba(X_icl_val)
            prob_test = model.predict_proba(X_test_icl)
            
            # Extract the probability of the positive class (index 1)
            y_pred_val = prob_val[:, 1] if prob_val.ndim == 2 else prob_val
            y_pred_test = prob_test[:, 1] if prob_test.ndim == 2 else prob_test
            
            preds[name] = {"pred_emp_val": y_pred_val, "pred_emp_test": y_pred_test}

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    model_names = [n for n, _ in models]
    return preds, model_names, y_icl_val


# ============================================================
# Helper: print running ranks
# ============================================================
def print_running_ranks(ranks_sum, n_datasets, betas, approaches):
    # ranks_sum is now accumulating dataset-level averages
    avg_ranks = ranks_sum / max(1, n_datasets)

    col_names = [f"b={b:.1f}" for b in betas]
    df = pd.DataFrame(avg_ranks.T, index=approaches, columns=col_names)

    print(
        f"\n[RUNNING AVERAGE RANKS] after {n_datasets} datasets (1=best; averaged across RS/IR appropriately) "
    )
    print(df.to_string(float_format=lambda x: f"{x:.2f}"))


# ============================================================
# Main Experiment
# ============================================================
def main():
    path_files = "/home/miguel/Desktop/TALENT/cc18_datasets_with_metadata"
    files = sorted([f for f in os.listdir(path_files) if f.endswith(".npy")])
    print(f"Found {len(files)} datasets.")

    BETAS = np.arange(0.5, 5.5, 0.5)
    BETAS = np.append(BETAS, 10)
    APPROACHES = [
        "MOH__alpha_t",
        "EnsembleMixtutre__alpha_t",
        "EnsembleMixtutre__thr_only_fixed_mix",
        "TabPFN__thr",
        "MotherNet__thr",
        "XGBoost__thr",
    ]
    random_states = [0, 1, 2, 3]
    min_minority_size = 10

    PRINT_EVERY = 1  # prints after every PRINT_EVERY datasets

    # Each entry is ONE TRIAL (dataset, run_idx, random_state): (n_betas, 6)
    scores_list = []
    best_list = []
    alpha_list = [] # <--- Added Tracker for alphas
    meta_list = []

    # Dataset-level trackers for the global averages
    counter_per_beta_alpha = {b: 0.0 for b in BETAS}
    counter_per_beta_thr = {b: 0.0 for b in BETAS}

    # running rank accumulator across DATASETS (average over RS and, if natural_ir < 3, IRs)
    ranks_sum = np.zeros((len(BETAS), len(APPROACHES)), dtype=float)
    n_datasets_processed = 0

    print(f"\n{'='*80}\nEXPERIMENTAL PIPELINE START\n{'='*80}")

    for file_idx, file in enumerate(files):
        arr = np.load(os.path.join(path_files, file), allow_pickle=True)
        data, categorical_indicator, _ = arr[0], arr[1], arr[2]

        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        Xp, yp, categorical_indicator = preprocess_data(X, y, categorical_indicator, ordinal_encode=True)

        # Skip rules
        if np.sum(categorical_indicator) > 50:
            continue
        if Xp.shape[0] > 10000:
            continue

        if len(np.unique(yp)) > 2:
            Xp, yp, _ = difficulty_aware_binary_decomposition(Xp, yp, random_state=42)
            
        _, counts = np.unique(yp, return_counts=True)
        natural_ir = float(np.max(counts) / np.min(counts))

        print(f"\n[DATASET LOADED] {file} | Natural IR: {natural_ir:.2f} | shape={Xp.shape}")

        # IR plan
        if natural_ir > 3.0:
            target_irs = [natural_ir]
            is_natural = True
            print("|-- Action: Using Natural IR (no sampling)")
        else:
            random.seed(42 + file_idx)
            target_irs = [random.uniform(3.0, 50.0) for _ in range(5)]
            is_natural = False
            print("|-- Action: Sampling 5 random IRs between 3 and 50")

        # Trackers strictly for this single dataset
        dataset_ranks_list = []
        dataset_wins_alpha = {b: [] for b in BETAS}
        dataset_wins_thr = {b: [] for b in BETAS}

        for run_idx, target_ir in enumerate(target_irs):
            if is_natural:
                X_std, y_std, final_ir = Xp, yp, natural_ir
            else:
                X_std, y_std, final_ir = apply_maximal_subset(
                    Xp, yp, target_ir, min_cardinality=min_minority_size, seed=42 + file_idx + run_idx
                )
                if X_std is None:
                    continue

            print(
                f"  >>> Run {run_idx+1}/{len(target_irs)} | Target IR: {target_ir:.2f} | "
                f"Final IR: {final_ir:.2f} | Shape: {X_std.shape}"
            )

            rs_scores = np.full((len(random_states), len(BETAS), len(APPROACHES)), np.nan, dtype=float)
            rs_alphas = np.full((len(random_states), len(BETAS)), np.nan, dtype=float) # <--- Added Array for alphas

            for rs_i, rs in enumerate(random_states):
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
                train_idx, test_idx = next(skf.split(X_std, y_std))

                X_train, X_test = X_std[train_idx], X_std[test_idx]
                y_train, y_test = y_std[train_idx], y_std[test_idx]

                with SuppressStdoutStderr():
                    pred_dict, _, y_val = icl_models(X_train, y_train, X_test, y_test, categorical_indicator)

                for b_i, beta in enumerate(BETAS):
                    sd = select_best_postproc_binary_from_dict_fbeta(pred_dict, y_val, y_test, beta=beta)
                    for m_i, name in enumerate(APPROACHES):
                        if name in sd:
                            rs_scores[rs_i, b_i, m_i] = float(sd[name]["test_fbeta"])
                    
                    # <--- Extract the Alpha value specifically for the ensemble
                    if "EnsembleMixtutre__alpha_t" in sd:
                        rs_alphas[rs_i, b_i] = float(sd["EnsembleMixtutre__alpha_t"]["params"]["alpha"])

                # ------------------------------------------------------------
                # Save scores per trial + Determine per-trial ranks & counts
                # ------------------------------------------------------------
                trial_scores = rs_scores[rs_i]                # (n_betas, 6)
                trial_best = np.nanmax(trial_scores, axis=1)  # (n_betas,)
                trial_alphas = rs_alphas[rs_i]                # (n_betas,) <--- Define trial_alphas
                
                ranks_trial = np.zeros((len(BETAS), len(APPROACHES)), dtype=float)
                
                for b_i, beta in enumerate(BETAS):
                    vals = trial_scores[b_i]
                    safe = np.where(np.isnan(vals), -np.inf, vals)  # nans -> worst
                    ranks_trial[b_i] = rankdata(-safe, method="average")  # 1=best
                    
                    # Accumulate dataset-level win comparisons for EnsembleMixture
                    a = trial_scores[b_i, APPROACHES.index("EnsembleMixtutre__alpha_t")]
                    t = trial_scores[b_i, APPROACHES.index("EnsembleMixtutre__thr_only_fixed_mix")]
                    if np.isfinite(a) and np.isfinite(t):
                        if a > t:
                            dataset_wins_alpha[beta].append(1.0)
                            dataset_wins_thr[beta].append(0.0)
                        elif a < t:
                            dataset_wins_alpha[beta].append(0.0)
                            dataset_wins_thr[beta].append(1.0)
                        else:
                            dataset_wins_alpha[beta].append(0.0)
                            dataset_wins_thr[beta].append(0.0)

                dataset_ranks_list.append(ranks_trial)

                scores_list.append(trial_scores)
                best_list.append(trial_best)
                alpha_list.append(trial_alphas)               # <--- Append alphas to the master list
                meta_list.append(
                    dict(
                        dataset_id=file,
                        file_idx=int(file_idx),
                        run_idx=int(run_idx),
                        random_state=int(rs),
                        rs_i=int(rs_i),
                        is_natural=bool(is_natural),
                        target_ir=float(target_ir),
                        final_ir=float(final_ir),
                        n_samples=int(X_std.shape[0]),
                        n_features=int(X_std.shape[1]),
                    )
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # ------------------------------------------------------------
        # Commit Dataset-Level aggregations 
        # ------------------------------------------------------------
        if len(dataset_ranks_list) > 0:
            avg_dataset_ranks = np.mean(dataset_ranks_list, axis=0)
            ranks_sum += avg_dataset_ranks
            
            for beta in BETAS:
                if len(dataset_wins_alpha[beta]) > 0:
                    # FIX 1: Use np.sum instead of np.mean to accumulate exact integer counts across trials
                    counter_per_beta_alpha[beta] += np.sum(dataset_wins_alpha[beta])
                    counter_per_beta_thr[beta] += np.sum(dataset_wins_thr[beta])

            n_datasets_processed += 1

            # Print running ranks periodically
            if (n_datasets_processed % PRINT_EVERY) == 0:
                print_running_ranks(ranks_sum, n_datasets_processed, BETAS, APPROACHES)

            # Print the updated average win counts per beta
            # FIX 1 continued: Cast counts to integers for terminal readability
            alpha_str = ", ".join(f"{k:.1f}: {int(v)}" for k, v in counter_per_beta_alpha.items())
            thr_str = ", ".join(f"{k:.1f}: {int(v)}" for k, v in counter_per_beta_thr.items())
            
            print(f"  |-- EnsembleMixture__alpha_t win counts: {{{alpha_str}}}")
            print(f"  |-- EnsembleMixture__thr win counts: {{{thr_str}}}")

    # ------------------------------------------------------------
    # End of pipeline
    # ------------------------------------------------------------
    if n_datasets_processed > 0 and (n_datasets_processed % PRINT_EVERY) != 0:
        print_running_ranks(ranks_sum, n_datasets_processed, BETAS, APPROACHES)

    if not scores_list:
        raise RuntimeError("No successful trials recorded; nothing to save.")

    # Note: Saved data (CSV and npz files) still output RAW trial results for deep analysis.
    scores_tensor = np.stack(scores_list, axis=0)  # (n_trials, n_betas, 6)
    best_matrix = np.stack(best_list, axis=0)      # (n_trials, n_betas)
    alpha_matrix = np.stack(alpha_list, axis=0)    # (n_trials, n_betas) <--- Stack the matrix

    meta_df = pd.DataFrame(meta_list)

    np.savez_compressed(
        "benchmark_trial_as_dataset_results_valpha.npz",
        scores_tensor=scores_tensor,
        best_matrix=best_matrix,
        alpha_matrix=alpha_matrix,                 # <--- Save to NPZ
        betas=BETAS,
        approaches=np.array(APPROACHES, dtype=object),
        random_states=np.array(random_states, dtype=int),
    )
    meta_df.to_csv("benchmark_trial_as_dataset_metadata_valpha.csv", index=False)

    # long-form CSV (easy analysis)
    rows = []
    for trial_i in range(scores_tensor.shape[0]):
        for b_i, beta in enumerate(BETAS):
            row = {
                **meta_list[trial_i],
                "beta": float(beta),
                "best_test_fbeta": float(best_matrix[trial_i, b_i]),
                "ensemble_alpha": float(alpha_matrix[trial_i, b_i]), # <--- Add alpha to CSV rows
            }
            for m_i, name in enumerate(APPROACHES):
                row[name] = float(scores_tensor[trial_i, b_i, m_i])
            rows.append(row)
    pd.DataFrame(rows).to_csv("benchmark_trial_as_dataset_scores_long_valpha.csv", index=False)

    # FIX 2: Save the global alpha vs thr win counts to a dedicated CSV file
    counts_df = pd.DataFrame({
        "Beta": list(BETAS),
        "Alpha_Wins": [int(counter_per_beta_alpha[b]) for b in BETAS],
        "Thr_Wins": [int(counter_per_beta_thr[b]) for b in BETAS]
    })
    counts_df.to_csv("benchmark_win_counts_alpha_vs_thr_valpha.csv", index=False)

    print("\nSaved:")
    print("  - benchmark_trial_as_dataset_results_valpha.npz")
    print("  - benchmark_trial_as_dataset_metadata_valpha.csv")
    print("  - benchmark_trial_as_dataset_scores_long_valpha.csv")
    print("  - benchmark_win_counts_alpha_vs_thr_valpha.csv")

if __name__ == "__main__":
    main()