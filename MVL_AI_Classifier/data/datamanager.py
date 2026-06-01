"""
This file is for the datamanager class and will handle which dataclass to use and how to translate it to the train and tuning files
"""

import os
import h5py
import torch
import gc
from MVL_AI_Classifier.constants import (
    TRAIN_CACHE,
    VAL_CACHE,
    TEST_CACHE,
    PARQUET_FILE,
)
from MVL_AI_Classifier.data.cached_dataclass import CachedDataClass
from MVL_AI_Classifier.data.dataclass import DataClass
from torch.utils.data import DataLoader


class DataManager:
    def __init__(
        self,
        view_configuration: dict,
        batch_size: int,
        use_cache: bool = True,
        num_sections: int = 2,
    ):
        """Initialize the DataManager with the given view configuration, batch size, cache usage, and number of sections for splitting.
        Args:
            view_configuration (dict): A dictionary specifying which views to include in the dataset.
            batch_size (int): The number of samples per batch to load.
            use_cache (bool): Whether to use cached HDF5 files for loading data.
            num_sections (int): The number of sections to split the training data into for memory management.
        """
        self.view_configuration = view_configuration
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.num_sections = max(1, num_sections)
        self.active_keys = list(view_configuration.keys())

        # check if cached file exists
        if self.use_cache and os.path.exists(TRAIN_CACHE):
            with h5py.File(TRAIN_CACHE, "r") as file:
                self.total_samples = file["label"].shape[0]
        self.sections = self._get_sections()

    def _get_sections(self):
        """Determine how to split the training data into sections based on the total number of samples and the specified number of sections.
        Returns:
            list: A list of tuples, where each tuple contains the start and end indices for a section of the training data.
        """
        # in case splitting is not needed, return as 1 section
        if not self.use_cache or self.num_sections == 1:
            return [(0, self.total_samples)]

        section_size = self.total_samples // self.num_sections
        sections = []
        for i in range(self.num_sections):
            start_idx = i * section_size
            if i == self.num_sections - 1:
                end_idx = self.total_samples
            else:
                end_idx = (i + 1) * section_size
            sections.append((start_idx, end_idx))
        return sections

    def get_train_loaders(self, only_first_section: bool = False):
        """Generate DataLoader objects for the training data, optionally only for the first section.
        Args:
            only_first_section (bool): If True, only generate a DataLoader for the first section of the training data.
        Yields:
            DataLoader: A DataLoader object for a section of the training data.
        """
        active_sections = [self.sections[0]] if only_first_section else self.sections
        for idx, (start_idx, end_idx) in enumerate(active_sections):
            print(f"loading section {idx+1} from {start_idx} until {end_idx}")
            if self.use_cache and os.path.exists(TRAIN_CACHE):
                train_data = CachedDataClass(
                    hdf5_path=TRAIN_CACHE,
                    view_keys=self.active_keys,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
            else:
                train_data = DataClass(
                    parquet_file=PARQUET_FILE,
                    view_configuration=self.view_configuration,
                    split="train",
                )

            train_loader = DataLoader(
                train_data,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                persistent_workers=False,
            )

            yield train_loader
            if (
                hasattr(train_loader, "_iterator")
                and train_loader._iterator is not None
            ):
                train_loader._iterator._shutdown_workers()
            del train_loader, train_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import ctypes

            try:
                # This calls malloc_trim, clearing out memory fragmentation fragmentation
                ctypes.CDLL("libc.so.6").malloc_trim(0)
                print("clearing ram")
            except Exception:
                pass

    def get_val_loader(self):
        """Generate a DataLoader object for the validation data.
        Returns:
            DataLoader: A DataLoader object for the validation data.
        """
        print("loading validation set")
        if self.use_cache and os.path.exists(VAL_CACHE):
            val_data = CachedDataClass(hdf5_path=VAL_CACHE, view_keys=self.active_keys)
        else:
            val_data = DataClass(
                parquet_file=PARQUET_FILE,
                view_configuration=self.view_configuration,
                split="val",
            )

        val_loader = DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        return val_loader

    def get_test_loader(self):
        """Generate a DataLoader object for the test data.
        Returns:
            DataLoader: A DataLoader object for the test data.
        """
        if self.use_cache and os.path.exists(TEST_CACHE):
            test_data = CachedDataClass(
                hdf5_path=TEST_CACHE, view_keys=self.active_keys
            )
        else:
            test_data = DataClass(
                parquet_file=PARQUET_FILE,
                view_configuration=self.view_configuration,
                split="test",
            )

        test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        return test_loader
