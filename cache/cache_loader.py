import os
import numpy as np

from utils.data_loader import data_loader

def load_or_generate_data(dataset, miss_rate, miss_modality, seed, save_dir="data_cache"):
    """
    Wrapper for data_loader that caches results to disk.

    Parameters:
        dataset: any - dataset object or name
        miss_rate: float - missing data rate
        miss_modality: str - missing modality spec
        seed: int - random seed
        save_dir: str - directory to store cached numpy arrays

    Returns:
        data_x, miss_data_x, data_mask: numpy arrays
    """
    os.makedirs(save_dir, exist_ok=True)

    # Generate a unique key for this combination of parameters
    key = f"{dataset}_{miss_rate}_{miss_modality}_{seed}"
    key = key.replace("/", "_")  # sanitize for file names

    # Define file paths
    file_paths = {
        "data_x": os.path.join(save_dir, f"{key}_data_x.npy"),
        "miss_data_x": os.path.join(save_dir, f"{key}_miss_data_x.npy"),
        "data_mask": os.path.join(save_dir, f"{key}_data_mask.npy"),
    }

    # Check if all files exist
    if all(os.path.exists(path) for path in file_paths.values()):
        data_x = np.load(file_paths["data_x"], allow_pickle=False)
        miss_data_x = np.load(file_paths["miss_data_x"], allow_pickle=False)
        data_mask = np.load(file_paths["data_mask"], allow_pickle=False)
        print("Loaded cached data from disk.")
    else:
        # Call original data loader
        data_x, miss_data_x, data_mask = data_loader(dataset, miss_rate, miss_modality, seed)
        # Save to disk
        np.save(file_paths["data_x"], data_x, allow_pickle=False)
        np.save(file_paths["miss_data_x"], miss_data_x, allow_pickle=False)
        np.save(file_paths["data_mask"], data_mask, allow_pickle=False)
        print("Generated data and saved to disk.")

    return data_x, miss_data_x, data_mask
