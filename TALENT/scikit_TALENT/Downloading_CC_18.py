# Python script to load the OpenML-CC18 benchmark suite.
# This script uses the OpenML Python library to fetch the suite and download all datasets.
# Requirements: pip install openml
# Note: Downloading all datasets may take time and require significant storage space.

import openml
import pandas as pd
from typing import List, Dict
import os

def load_cc18_suite() -> Dict[int, pd.DataFrame]:
    """
    Loads the OpenML-CC18 benchmark suite by fetching the study and downloading each dataset.
    
    Returns:
        A dictionary where keys are dataset IDs and values are pandas DataFrames.
    """
    # Get the OpenML-CC18 study (suite ID: 99)
    study = openml.study.get_suite("OpenML-CC18")
    
    # Extract dataset IDs from the tasks in the suite
    dataset_ids: List[int] = []
    for task_id in study.tasks:
        task = openml.tasks.get_task(task_id)
        dataset_id = task.dataset_id
        if dataset_id not in dataset_ids:
            dataset_ids.append(dataset_id)
    
    print(f"Found {len(dataset_ids)} unique datasets in the CC18 suite.")
    
    # Download and load each dataset as a pandas DataFrame
    datasets: Dict[int, list] = {}
    for did in dataset_ids:
        try:
            # Download the dataset
            dataset = openml.datasets.get_dataset(did, download_data=True)
            # Convert to DataFrame
            X, y, categorical_atributes, attributes_names = dataset.get_data(
                dataset_format="dataframe",
                target=dataset.default_target_attribute
            )
            # Combine X and y into a single DataFrame
            df = X.copy()
            df[dataset.default_target_attribute] = y
            datasets[did] = [df,categorical_atributes, attributes_names]
            print(f"Loaded dataset {did}: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"Class distribution:\n{df[dataset.default_target_attribute].value_counts()}")
        except Exception as e:
            print(f"Error loading dataset {did}: {e}")
    
    return datasets

if __name__ == "__main__":
    # Load the suite
    import numpy as np
    cc18_datasets = load_cc18_suite()
    
    # Example: Print info about the first dataset
    if cc18_datasets:
        first_did = next(iter(cc18_datasets))
        print(f"\nExample dataset {first_did}:")
        print(cc18_datasets[first_did][0].head())
    
    # save datasets to disk
    output_dir = "cc18_datasets_with_metadata"
    os.makedirs(output_dir, exist_ok=True)
    for did, df in cc18_datasets.items():
        np.save(os.path.join(output_dir, f"dataset_{did}.npy"), np.array(df,dtype=object), allow_pickle=True)
        print(f"Saved dataset {did} to CSV.")