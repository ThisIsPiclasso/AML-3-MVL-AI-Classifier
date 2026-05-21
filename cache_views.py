import os
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader

from MVL_AI_Classifier.data.dataclass import DataClass


def compile_split_cache(
    parquet_file: str, view_config: dict, split: str, save_path: str
):
    original_dataset = DataClass(
        parquet_file=parquet_file, view_configuration=view_config, split=split
    )

    if split == "train":
        original_dataset.set_epoch(0)

    extraction_batch_size = 64
    loader = DataLoader(
        original_dataset,
        batch_size=extraction_batch_size,
        num_workers=12,
        shuffle=False,
        drop_last=False,
    )

    num_samples = len(original_dataset)
    active_view_keys = list(view_config.keys())

    first_sample = original_dataset[0]

    with h5py.File(save_path, "w") as f:
        dataset_storage = {}
        for view_name in active_view_keys:
            feature_shape = first_sample["views"][view_name].shape
            full_shape = (num_samples,) + feature_shape

            dataset_storage[view_name] = f.create_dataset(
                view_name, shape=full_shape, dtype="float32", compression="gzip"
            )

        label_storage = f.create_dataset("label", shape=(num_samples,), dtype="int64")

        current_idx = 0
        pbar = tqdm(loader, desc=f"caching{split.upper()} to file")

        for batch in pbar:
            views_dict = batch["views"]
            label_tensor = batch["label"]

            actual_batch_size = label_tensor.shape[0]
            end_idx = current_idx + actual_batch_size

            for view_name in active_view_keys:
                dataset_storage[view_name][current_idx:end_idx] = views_dict[
                    view_name
                ].numpy()

            label_storage[current_idx:end_idx] = label_tensor.numpy()

            current_idx = end_idx


def main():
    PARQUET_PATH = "/workspace/AML-3-MVL-AI-Classifier/data/dataset.parquet"
    CACHE_DIR = "/workspace/AML-3-MVL-AI-Classifier/data/data_cache"
    os.makedirs(CACHE_DIR, exist_ok=True)

    from train import MODEL_CONFIGURATION

    splits_to_process = {
        "train": os.path.join(CACHE_DIR, "train_features.h5"),
        "val": os.path.join(CACHE_DIR, "val_features.h5"),
        "test": os.path.join(CACHE_DIR, "test_features.h5"),
    }

    # Loop over and process each file sequentially
    for split_name, file_save_path in splits_to_process.items():
        if os.path.exists(file_save_path):
            print(f"{file_save_path} already exists. skipping")
            continue

        compile_split_cache(
            parquet_file=PARQUET_PATH,
            view_config=MODEL_CONFIGURATION,
            split=split_name,
            save_path=file_save_path,
        )
        print(f"archived {split_name} cache file data structure.")


if __name__ == "__main__":
    main()
