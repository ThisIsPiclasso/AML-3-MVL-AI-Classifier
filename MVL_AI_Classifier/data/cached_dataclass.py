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

            # 🚀 3. THE LIVE CHUNK BUFFER INGESTION LOGIC
            # Read rows from disk sequentially and slide them into the pre-allocated RAM structures
            print(f"loading {total_slice_samples:,} samples ")

            # Setup a master progress bar tracking rows allocated
            with tqdm(
                total=total_slice_samples,
                desc="allocating RAM for tensors",
                unit="rows",
                leave=False,
            ) as pbar:
                current_offset = 0
                while current_offset < total_slice_samples:
                    # Calculate chunk size boundary limits
                    read_len = min(chunk_size, total_slice_samples - current_offset)

                    # Compute relative index pointers matching the source file coordinates
                    source_start = start_idx + current_offset
                    source_end = source_start + read_len

                    # Read from disk and paste into the pre-allocated tensor structure
                    for view_name in self.view_keys:
                        # Pull numpy block from disk
                        disk_chunk = f[view_name][source_start:source_end]
                        # Transfer it directly onto our pre-allocated memory view slice
                        self.in_memory_views[view_name][
                            current_offset : current_offset + read_len
                        ] = torch.from_numpy(disk_chunk).float()

                    current_offset += read_len
                    pbar.update(read_len)  # Bump progress display bar forward

        print(
            "✅ System RAM allocation finalized. Binary HDF5 file handle safely unlinked.\n"
        )

    def __len__(self) -> int:
        return self.dataset_length

    def set_epoch(self, epoch: int):
        pass

    def __getitem__(self, idx: int) -> dict:
        view_outputs = {}
        for view_name in self.view_keys:
            view_outputs[view_name] = self.in_memory_views[view_name][idx]

        return {"views": view_outputs, "label": self.labels[idx]}
