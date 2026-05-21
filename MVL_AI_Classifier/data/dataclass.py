from torch.utils.data import Dataset
import torch
import pandas as pd
from PIL import Image
import numpy as np

from MVL_AI_Classifier.constants import PATCH_SIZE


class DataClass(Dataset):
    """
    A custom dataset class for loading and preprocessing images based on a given configuration. The dataset reads from a parquet file, filters the data based on the specified split (train, val, test), and applies the appropriate preprocessors to the images.
    """

    def __init__(self, parquet_file: str, view_configuration: dict, split="train"):
        """ "Initializes the dataset with the given parquet file, view configuration, and split.
        Args:
            parquet_file (str): Path to the parquet file containing the dataset information.
            view_configuration (dict): A dictionary mapping view names to their corresponding preprocessors.
            split (str, optional): The dataset split to use (train, val, test). Defaults to "train".
        """
        super().__init__()
        self.view_configuration = view_configuration
        self.split = split
        self.current_epoch = 0
        self.df = pd.read_parquet(parquet_file)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.preprocessors = {
            view_name: config_["preprocessor"]
            for view_name, config_ in view_configuration.items()
        }

    def __len__(self):
        """Returns the length of the dataset, which is the number of samples in the filtered DataFrame."""
        return len(self.df)

    def set_epoch(self, epoch: int):
        """
        Setter for the epoch
        Args:
            epoch (int): The current epoch number, used for deterministic random cropping during training.
        """
        self.current_epoch = epoch

    def _get_patch(self, image_: Image.Image, idx: int) -> Image.Image:
        """Extracts a patch from the given image based on the current split and epoch. For validation and testing, it extracts a centered patch, while for training, it extracts a random patch using a deterministic random seed.
        Args:
            image_ (PIL.Image.Image): The input image from which to extract the patch.
            idx (int): The index of the current sample, used for generating a deterministic random seed during training.
        Returns:
            PIL.Image.Image: The extracted patch from the input image.
        """
        width, height = image_.size
        patch_size = PATCH_SIZE
        if self.split in ["val", "test"]:
            x = (width - patch_size) // 2
            y = (height - patch_size) // 2
        else:
            seed = int(self.current_epoch * 997 + idx)
            rng = np.random.default_rng(np.random.PCG64(seed))

            max_x = width - patch_size
            max_y = height - patch_size

            x = rng.integers(0, max_x) if max_x > 0 else 0
            y = rng.integers(0, max_y) if max_y > 0 else 0

        return image_.crop((x, y, x + patch_size, y + patch_size))

    def __getitem__(self, idx) -> dict:
        """Retrieves the sample at the specified index, applies the appropriate preprocessors to the image, and returns a dictionary containing the processed views and the corresponding label.
        Args:
            idx (int): The index of the sample to retrieve.
        Returns:
            dict: A dictionary containing the processed views (as tensors) and the corresponding label (as a tensor).
        """
        item = self.df.iloc[idx]
        image_ = Image.open(item["path"]).convert("RGB")
        image_ = self._get_patch(image_, idx)
        label = item["label"]

        view_outputs = {}
        for view_name, preprocessor in self.preprocessors.items():
            # this will run through all preprocessors, convert the outputs into a torch tensor and add the result back in a dictionary
            view_outputs[view_name] = torch.from_numpy(preprocessor(image_)).float()
        return {"views": view_outputs, "label": torch.tensor(label, dtype=torch.long)}
