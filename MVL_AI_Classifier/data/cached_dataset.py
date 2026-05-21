import h5py
import torch
from torch.utils.data import Dataset


class CachedMultiViewDataset(Dataset):
    """
    A high-performance dataset that bypasses all CPU math by reading
    precomputed multi-view tensors directly from a single HDF5 file.
    """

    def __init__(
        self,
        hdf5_path: str = "/workspace/AML-3-MVL-AI-Classifier/data_cache/features.h5",
        view_keys: list = None,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.file_handle = None  # Opened lazily per worker thread

        self.view_keys = view_keys

        # Read the total length of the dataset directly from the file metadata
        with h5py.File(self.hdf5_path, "r") as f:
            self.dataset_length = len(f["label"])

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict:
        # CRITICAL FOR MULTI-PROCESSING: HDF5 file handles cannot be shared across
        # separate CPU worker threads. We must open the file inside the worker
        # the very first time it requests a sample.
        if self.file_handle is None:
            self.file_handle = h5py.File(self.hdf5_path, "r")

        # 1. Dynamically rebuild the view_outputs dictionary using your configuration keys
        view_outputs = {}
        for view_name in self.view_keys:
            # Read straight from the raw binary array via disk pointer offsets
            raw_array = self.file_handle[view_name][idx]
            view_outputs[view_name] = torch.from_numpy(raw_array).float()

        # 2. Extract the corresponding ground-truth classification label
        label = self.file_handle["label"][idx]

        # 3. Return the identical structure your original training loop expects
        return {"views": view_outputs, "label": torch.tensor(label, dtype=torch.long)}
