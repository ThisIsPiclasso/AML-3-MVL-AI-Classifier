import os
import sys
import h5py
from h5py import Dataset
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from MVL_AI_Classifier.model_configuration import MODEL_CONFIGURATION
from MVL_AI_Classifier.constants import PARQUET_FILE, BATCH_SIZE, NUM_WORKERS, CACHE_DIR


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MVL_AI_Classifier.data.dataclass import DataClass

from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def compile_split_cache(
    parquet_file: str, view_config: dict, split: str, save_path: str
):
    """
    Extracts features from the raw dataset and caches them into an HDF5 matrix layout.
    Supports seamless resuming if interrupted and skips corrupted image clusters.
    Args:
        parquet_file: Path to the original dataset in Parquet format.
        view_config: Dictionary specifying which views to include in the dataset.
        split: One of "train", "val", or "test" to specify which dataset split to process.
        save_path: Path where the HDF5 cache file will be saved. If the file already exists, the function will attempt to resume from where it left off.
    """

    original_dataset = DataClass(
        parquet_file=parquet_file, view_configuration=view_config, split=split
    )

    if split == "train":
        original_dataset.set_epoch(0)

    num_samples = len(original_dataset)
    active_view_keys = list(view_config.keys())
    first_sample = original_dataset[0]

    start_idx = 0
    file_mode = "w"

    if os.path.exists(save_path):
        try:
            with h5py.File(save_path, "r") as f_check:
                if "aps" in f_check:
                    print("adding to existing cache file")

                    aps_dataset = f_check["aps"]

                    chunk_size = 50000
                    total_allocated = aps_dataset.shape[0]
                    last_written_row = -1

                    print("scanning blocks")
                    for i in range(0, total_allocated, chunk_size):
                        end_chunk = min(i + chunk_size, total_allocated)
                        chunk_data = aps_dataset[i:end_chunk]

                        nonzero_in_chunk = np.any(chunk_data != 0, axis=1)
                        if np.any(nonzero_in_chunk):
                            last_written_row = i + np.max(np.where(nonzero_in_chunk)[0])

                    if last_written_row != -1:
                        start_idx = (
                            int((last_written_row // BATCH_SIZE) * BATCH_SIZE)
                            + BATCH_SIZE
                        )

                        if start_idx < num_samples:
                            file_mode = (
                                "r+"  # Convert file open settings to append/update mode
                            )
                            print(f"real data found at row {last_written_row}.")
                            print(
                                f"skipping to: {start_idx} ({start_idx/num_samples*100:.2f}%)"
                            )
                        else:
                            print(
                                f"cache file for {split.upper()} is  complete ({last_written_row+1} rows)"
                            )
                            return
                    else:
                        print("file exists but is empty or not initialized properly")
                        file_mode = "w"
        except Exception as e:
            print(f"file analysis failed ({e})")
            file_mode = "w"

    loader = DataLoader(
        original_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        drop_last=False,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with h5py.File(save_path, file_mode) as f:
        dataset_storage: dict[str, Dataset] = {}
        label_storage: Dataset

        if file_mode == "w":
            print("allocating space for new cache file")
            for view_name in active_view_keys:
                feature_shape = first_sample["views"][view_name].shape
                full_shape = (num_samples,) + feature_shape

                # lzf compression minimizes CPU execution overhead during streaming
                dataset_storage[view_name] = f.create_dataset(
                    view_name, shape=full_shape, dtype="float32", compression="lzf"
                )
            label_storage = f.create_dataset(
                "label", shape=(num_samples,), dtype="int64"
            )
        else:
            print(
                "opening existing cache file in update mode and mapping datasets for writing"
            )
            for view_name in active_view_keys:
                dataset_storage[view_name] = f[view_name]  # type: ignore
            label_storage = f["label"]  # type: ignore

        current_idx = 0
        pbar = tqdm(loader, desc=f"caching {split.upper()} to file", total=len(loader))

        for batch in pbar:
            actual_batch_size = batch["label"].shape[0]
            end_idx = current_idx + actual_batch_size

            # Fast-forward past batches that have already been completely committed to disk
            if current_idx < start_idx:
                current_idx = end_idx
                continue

            views_dict = batch["views"]
            label_tensor = batch["label"]

            try:
                for view_name in active_view_keys:
                    dataset_storage[view_name][current_idx:end_idx] = views_dict[
                        view_name
                    ].numpy()
                label_storage[current_idx:end_idx] = label_tensor.numpy()
            except Exception as e:
                print(
                    f"\nskipped corrupted input range [{current_idx}:{end_idx}]. exception: {e}"
                )

            current_idx = end_idx


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)

    train_save_path = os.path.join(CACHE_DIR, "train_features.h5")
    compile_split_cache(
        parquet_file=PARQUET_FILE,
        view_config=MODEL_CONFIGURATION,
        split="train",
        save_path=train_save_path,
    )

    val_save_path = os.path.join(CACHE_DIR, "val_features.h5")
    compile_split_cache(
        parquet_file=PARQUET_FILE,
        view_config=MODEL_CONFIGURATION,
        split="val",
        save_path=val_save_path,
    )

    test_save_path = os.path.join(CACHE_DIR, "test_features.h5")
    compile_split_cache(
        parquet_file=PARQUET_FILE,
        view_config=MODEL_CONFIGURATION,
        split="test",
        save_path=test_save_path,
    )

    print("caching complete")
