import h5py
import torch
from torch.utils.data import Dataset


class CachedDataClass(Dataset):
    def __init__(
        self, hdf5_path: str, view_keys: list, start_idx: int = 0, end_idx: int = None
    ):
        super().__init__()
        self.view_keys = view_keys

        with h5py.File(hdf5_path, "r") as f:
            total_rows = f["label"].shape[0]
            if end_idx is None:
                end_idx = total_rows

            self.labels = torch.from_numpy(f["label"][start_idx:end_idx]).long()
            self.dataset_length = len(self.labels)

            self.in_memory_views = {}
            for view_name in self.view_keys:
                raw_numpy_array = f[view_name][start_idx:end_idx]
                self.in_memory_views[view_name] = torch.from_numpy(
                    raw_numpy_array
                ).float()

    def __len__(self) -> int:
        return self.dataset_length

    def set_epoch(self, epoch: int):
        pass

    def __getitem__(self, idx: int) -> dict:
        view_outputs = {}
        for view_name in self.view_keys:
            view_outputs[view_name] = self.in_memory_views[view_name][idx]

        return {"views": view_outputs, "label": self.labels[idx]}
