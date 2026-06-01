import os
import sys
import h5py
from h5py import Dataset
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from ..constants import MODEL_CONFIGURATION

# Resolve paths to ensure imports from your project directory work flawlessly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MVL_AI_Classifier.data.dataclass import DataClass
from MVL_AI_Classifier.constants import PARQUET_FILE
from PIL import ImageFile

# 🚀 Allow PIL to open and load broken/truncated images without crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True
# ─── HARDWARE TARGET CONFIGURATION ──────────────────────────────────────────
# Define where the pristine HDF5 arrays will sit on your Unraid cache pool
CACHE_DIR = "/workspace/AML-3-MVL-AI-Classifier/data/data_cache"
BATCH_SIZE = 64
NUM_WORKERS = 6

# Mirror your exact network vision input configurations

# ─────────────────────────────────────────────────────────────────────────────


def compile_split_cache(
    parquet_file: str, view_config: dict, split: str, save_path: str
):
    """
    Extracts features from the raw dataset and caches them into an HDF5 matrix layout.
    Supports seamless resuming if interrupted and skips corrupted image clusters.
    """
    print("\n──────────────────────────────────────────────────")
    print(f"🎬 Initializing Data Caching Engine for: {split.upper()}")
    print("──────────────────────────────────────────────────")

    original_dataset = DataClass(
        parquet_file=parquet_file, view_configuration=view_config, split=split
    )

    if split == "train":
        original_dataset.set_epoch(0)

    num_samples = len(original_dataset)
    active_view_keys = list(view_config.keys())
    first_sample = original_dataset[0]

    start_idx = 0
    file_mode = "w"  # Default to fresh initialization

    # ─── RESUME DETECTION LOGIC ──────────────────────────────────────────────
    if os.path.exists(save_path):
        try:
            # Open strictly in read-only first to evaluate if we can salvage past work
            with h5py.File(save_path, "r") as f_check:
                if "aps" in f_check:
                    print(
                        "🔍 Existing cache file detected. Inspecting contents for progress..."
                    )

                    # Pull a lightweight map of the aps block to see where real data stops
                    aps_dataset = f_check["aps"]

                    # Read in chunks to avoid blowing up memory during the check
                    chunk_size = 50000
                    total_allocated = aps_dataset.shape[0]
                    last_written_row = -1

                    print("🎚️ Scanning array matrix blocks...")
                    for i in range(0, total_allocated, chunk_size):
                        end_chunk = min(i + chunk_size, total_allocated)
                        chunk_data = aps_dataset[i:end_chunk]

                        # Find non-zero indices within this slice
                        nonzero_in_chunk = np.any(chunk_data != 0, axis=1)
                        if np.any(nonzero_in_chunk):
                            last_written_row = i + np.max(np.where(nonzero_in_chunk)[0])

                    if last_written_row != -1:
                        # Snap progress back to the closest clean batch boundary marker
                        start_idx = (
                            int((last_written_row // BATCH_SIZE) * BATCH_SIZE)
                            + BATCH_SIZE
                        )

                        if start_idx < num_samples:
                            file_mode = (
                                "r+"  # Convert file open settings to append/update mode
                            )
                            print(
                                f"📍 Resuming enabled! Found real data up to row {last_written_row}."
                            )
                            print(
                                f"🚀 Progress bar will jump forward to index: {start_idx} ({start_idx/num_samples*100:.2f}%)"
                            )
                        else:
                            print(
                                f"✅ Cache file for {split.upper()} is already 100% complete ({last_written_row+1} rows). Skipping file generation entirely."
                            )
                            return
                    else:
                        print(
                            "⚠️ File exists but contains only zero placeholders. Overwriting fresh."
                        )
                        file_mode = "w"
        except Exception as e:
            print(
                f"⚠️ Existing file analysis failed ({e}). Re-initializing layout from scratch."
            )
            file_mode = "w"
    # ─────────────────────────────────────────────────────────────────────────

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
            print("📦 Allocating pristine binary shells inside HDF5...")
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
                "🔗 Linking runtime descriptors back to current file layout indices..."
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

            # ─── RUNTIME SAFETY EMBEDDING ────────────────────────────────────
            # Prevents corrupted images at index 700,000+ from crashing your run
            try:
                for view_name in active_view_keys:
                    dataset_storage[view_name][current_idx:end_idx] = views_dict[
                        view_name
                    ].numpy()
                label_storage[current_idx:end_idx] = label_tensor.numpy()
            except Exception as e:
                print(
                    f"\n❌ Catch block tripped! Skipped corrupted input range [{current_idx}:{end_idx}]. Exception: {e}"
                )
            # ─────────────────────────────────────────────────────────────────

            current_idx = end_idx


if __name__ == "__main__":
    # Ensure our save folder path exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Compile the main Training split file (Will automatically trigger resume calculation)
    train_save_path = os.path.join(CACHE_DIR, "train_features.h5")
    compile_split_cache(
        parquet_file=PARQUET_FILE,
        view_config=MODEL_CONFIGURATION,
        split="train",
        save_path=train_save_path,
    )

    # 2. Compile the Validation split file
    val_save_path = os.path.join(CACHE_DIR, "val_features.h5")
    compile_split_cache(
        parquet_file=PARQUET_FILE,
        view_config=MODEL_CONFIGURATION,
        split="val",
        save_path=val_save_path,
    )

    # 3. Compile the Test split file
    test_save_path = os.path.join(CACHE_DIR, "test_features.h5")
    compile_split_cache(
        parquet_file=PARQUET_FILE,
        view_config=MODEL_CONFIGURATION,
        split="test",
        save_path=test_save_path,
    )

    print(
        "\n🎉 [ALL SPLITS COMPLETE] Your optimized master cache files are baked and ready!"
    )
