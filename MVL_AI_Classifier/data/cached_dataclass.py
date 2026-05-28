import h5py
import torch
from torch.utils.data import Dataset


class CachedDataClass(Dataset):
    def __init__(self, hdf5_path: str, view_keys: list):
        super().__init__()
        self.view_keys = view_keys

        print(f"🧠 Loading HDF5 data matrix completely into RAM from: {hdf5_path}")

        # Open the file once, read everything into memory arrays, and close the file handle immediately!
        with h5py.File(hdf5_path, "r") as f:
            # Reconstruct classification labels directly as a continuous PyTorch Tensor
            self.labels = torch.from_numpy(f["label"][:]).long()
            self.dataset_length = len(self.labels)

            # Read the feature views entirely into memory matrices
            self.in_memory_views = {}
            for view_name in self.view_keys:
                print(f"   -> Sucking '{view_name}' matrix into RAM layout...")
                # The [:] operator forces h5py to load the entire binary block into a NumPy array
                raw_numpy_array = f[view_name][:]
                self.in_memory_views[view_name] = torch.from_numpy(
                    raw_numpy_array
                ).float()

        print(
            "✅ System RAM array allocation completed successfully. File handle closed safely."
        )

    def __len__(self) -> int:
        return self.dataset_length

    def set_epoch(self, epoch: int):
        pass

    def __getitem__(self, idx: int) -> dict:
        # Pull slices straight out of your ultra-fast system RAM matrix allocations
        view_outputs = {}
        for view_name in self.view_keys:
            view_outputs[view_name] = self.in_memory_views[view_name][idx]

        return {"views": view_outputs, "label": self.labels[idx]}
