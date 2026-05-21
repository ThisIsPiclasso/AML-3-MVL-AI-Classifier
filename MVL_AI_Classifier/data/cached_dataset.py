import h5py
import torch
from torch.utils.data import Dataset


class CachedMultiViewDataset(Dataset):
    """
    This is a caching class that allows us to precompute the tensors for each view beforehand and store them.
    This speeds up training time by a lot since we are not CPU bottlenecked.
    drawback is that we lose the random patching per epoch, but a small price for the speedup

    important note is that any changes to view configuration requires a rebuild of this cache.
    """

    def __init__(
        self,
        hdf5_path: str = "/workspace/AML-3-MVL-AI-Classifier/data_cache/features.h5",
        view_keys: list = None,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.file_handle = None

        self.view_keys = view_keys

        # read file
        with h5py.File(self.hdf5_path, "r") as f:
            self.dataset_length = len(f["label"])

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict:
        # fix for multi threading
        if self.file_handle is None:
            self.file_handle = h5py.File(self.hdf5_path, "r")

        # build the views as saved in the cache
        view_outputs = {}
        for view_name in self.view_keys:
            # read from cache
            raw_array = self.file_handle[view_name][idx]
            view_outputs[view_name] = torch.from_numpy(raw_array).float()

        # extract labels from cache
        label = self.file_handle["label"][idx]

        # return in correct format
        return {"views": view_outputs, "label": torch.tensor(label, dtype=torch.long)}
