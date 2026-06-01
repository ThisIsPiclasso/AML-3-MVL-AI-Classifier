import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CachedDataClass(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        view_keys: list,
        start_idx: int = 0,
        end_idx: int = None,
        chunk_size: int = 20000,
    ):
        """
        A PyTorch Dataset class that loads preprocessed multi-view features from an HDF5 file into memory in a memory-efficient way.
        It reads the data in chunks to avoid memory overflow and allows for specifying a range of samples
        to load, which is useful for training in sections or resuming from a specific point.
        Args:
            hdf5_path (str): Path to the HDF5 file containing the cached features and labels.
            view_keys (list): List of view names corresponding to the datasets in the HDF5 file.
            start_idx (int): The starting index of samples to load from the HDF5 file. Defaults to 0.
            end_idx (int): The ending index of samples to load from the HDF5 file. If None, it loads until the end of the dataset. Defaults to None.
            chunk_size (int): The number of samples to read in each chunk when loading from the HDF5 file. Defaults to 20,000.
        """
        super().__init__()
        self.view_keys = view_keys

        with h5py.File(hdf5_path, "r") as f:
            total_rows = f["label"].shape[0]
            if end_idx is None:
                end_idx = total_rows

            total_slice_samples = end_idx - start_idx
            self.dataset_length = total_slice_samples

            self.labels = torch.from_numpy(f["label"][start_idx:end_idx]).long()

            self.in_memory_views = {}
            for view_name in self.view_keys:
                feature_shape = f[view_name].shape[1:]
                self.in_memory_views[view_name] = torch.empty(
                    (total_slice_samples, *feature_shape), dtype=torch.float32
                )

            print(f"loading {total_slice_samples:,} samples ")

            with tqdm(
                total=total_slice_samples,
                desc="allocating RAM for tensors",
                unit="rows",
                leave=False,
            ) as pbar:
                current_offset = 0
                while current_offset < total_slice_samples:
                    read_len = min(chunk_size, total_slice_samples - current_offset)

                    source_start = start_idx + current_offset
                    source_end = source_start + read_len

                    for view_name in self.view_keys:
                        disk_chunk = f[view_name][source_start:source_end]

                        self.in_memory_views[view_name][
                            current_offset : current_offset + read_len
                        ] = torch.from_numpy(disk_chunk).float()

                    current_offset += read_len
                    pbar.update(read_len)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        Returns:
            int: The total number of samples available in the specified range of the dataset.
        """
        return self.dataset_length

    def set_epoch(self, epoch: int):
        """
        Unused method for compatibiulity with normal dataclass.
            epoch (int): The epoch number.
        """
        pass

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a single sample from the dataset.
        Args:
            idx (int): The index of the sample to retrieve.
        Returns:
            dict: A dictionary containing the views and label for the specified sample.
        """
        view_outputs = {}
        for view_name in self.view_keys:
            view_outputs[view_name] = self.in_memory_views[view_name][idx]

        return {"views": view_outputs, "label": self.labels[idx]}
