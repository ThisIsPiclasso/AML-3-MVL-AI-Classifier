"""Scanner script that runs through target path and subfolders and indexes all files into a dataframe and parquet file"""

import pandas as pd
import os
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

from ..constants import (
    DEFAULT_EPSILON,
    DEFAULT_AI_PATH,
    DEFAULT_NATURE_PATH,
    DEFAULT_SEED,
    DEFAULT_TRAIN_SPLIT,
    DEFAULT_VAL_SPLIT,
    DEFAULT_TEST_SPLIT,
    ALLOWED_EXTENSIONS,
    NAME_MAP,
    PARQUET_FILE,
    SAMPLE_SIZE,
)


class DatasetBuilder:
    def __init__(
        self,
        size: int = SAMPLE_SIZE,
        ai_path: str = DEFAULT_AI_PATH,
        nature_path: str = DEFAULT_NATURE_PATH,
    ) -> None:
        """Initialize the DatasetBuilder with a specified size and root directory.
        Args:
            size (int): The number of samples to include in the final dataset after undersampling.
            ai_path (str): The path to the directory containing AI-generated images.
            nature_path (str): The path to the directory containing natural images.
        """
        self.ai_path = Path(ai_path).resolve()
        self.nature_path = Path(nature_path).resolve()
        self.data = pd.DataFrame()
        self.size = size

    def _get_paths(self) -> list:
        """Recursively scan the root directory for image files with allowed extensions and return their paths.
        Returns:
            list: A list of file paths for images found in the root directory and its subdirectories.
        """
        file_paths = []
        combined_paths = [self.ai_path, self.nature_path]
        for target_path in combined_paths:
            if not target_path.exists():
                print(
                    f"Warning: Path {target_path} does not exist. Please check your structure."
                )
                continue
            for root, dirs, files in os.walk(target_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
                        file_paths.append(os.path.join(root, file))
        print(f"{len(file_paths):,} images found")
        return file_paths

    def scan(self):
        """Scan the root directory for images, extract metadata, and store it in a DataFrame.
        This method processes each image file found in the root directory and its subdirectories,
        extracting metadata such as dimensions, file size, and format.
        """
        errors = []
        image_records = []
        file_paths = self._get_paths()
        for file_path in tqdm(file_paths, desc="parsing img"):
            try:
                with Image.open(file_path) as img:
                    w, h = img.size
                    path_obj = Path(file_path)
                    folder_name = path_obj.parent.name
                    model_type = NAME_MAP.get(folder_name, folder_name)
                    parts = [p.lower() for p in path_obj.parts]
                    label = 1 if "ai" in parts else 0

                    image_records.append(
                        {
                            "path": file_path,
                            "filename": path_obj.name,
                            "model_type": model_type,
                            "label": label,
                        }
                    )
            except Exception as e:
                errors.append((file_path, str(e)))
                continue

        print(f"Skipped {len(errors)} files due to errors.")
        self.data = pd.DataFrame(image_records)

    def save(self, name: str = PARQUET_FILE):
        """Save the dataset to a Parquet file.
        Args:
            name (str): The file path where the dataset will be saved. Defaults to "data/dataset.parquet".
        """
        if self.data.empty:
            print("No data to save.")
            return
        self.data.to_parquet(name, engine="pyarrow", index=False)

    def filter(self):
        """
        Filter the dataset to exclude files with specific prefixes in their filenames.
        """
        if self.data.empty:
            print("No data to filter.")
            return
        self.data = self.data[~self.data["filename"].str.startswith(("[DUPE]", "[LQ]"))]

    def undersample(self):
        """Undersample the dataset to ensure a balanced representation of categories and model types."""
        if self.data.empty:
            print("No data to sample.")
            return
        self.data = self.data.groupby(["label", "model_type"]).sample(
            n=self.size, random_state=DEFAULT_SEED
        )

    def read(self, name: str = PARQUET_FILE):
        """Read the dataset from a Parquet file.
        Args:
            name (str): The file path from which to read the dataset. Defaults to "data/dataset.parquet".
        """
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(f"No dataset file found at {path.resolve()}")
        self.data = pd.read_parquet(path)

    def split(
        self,
        train_size: float = DEFAULT_TEST_SPLIT,
        val_size: float = DEFAULT_VAL_SPLIT,
        test_size: float = DEFAULT_TEST_SPLIT,
    ) -> None:
        """
        Split the dataset into training, validation, and test sets based on specified ratios.
        Args:
            train_size (float): The proportion of the dataset to include in the training set. Defaults to 0.8.
            val_size (float): The proportion of the dataset to include in the validation set. Defaults to 0.1.
            test_size (float): The proportion of the dataset to include in the test set. Defaults to 0.1.
        Raises:
            ValueError: If the split ratios do not sum to 1.0.
        """
        if self.data.empty:
            print("No data to split.")
            return

        total_size = train_size + val_size + test_size
        if not abs(total_size - 1.0) < DEFAULT_EPSILON:
            raise ValueError(
                f"The split ratios must sum to 1.0. Currently they sum to {total_size} "
                f"(Train: {train_size}, Val: {val_size}, Test: {test_size})"
            )

        # Shuffeling the data in order to create a random split
        df = self.data.sample(frac=1, random_state=DEFAULT_SEED).reset_index(drop=True)
        total_len = len(df)

        # Setting boundries for the split
        train_end = int(total_len * train_size)
        val_end = train_end + int(total_len * val_size)

        df["split"] = "train"

        if val_size > 0:
            val_indices = df.index[train_end:val_end]
            df.loc[val_indices, "split"] = "val"

        if test_size > 0:
            test_indices = df.index[val_end:]
            df.loc[test_indices, "split"] = "test"

        self.data = df

    def __len__(self):
        """Magic method to get the number of records in the dataset.
        Returns:
            int: The number of records in the dataset.
        """
        return len(self.data)

    def __call__(
        self,
        output_path: str = PARQUET_FILE,
        train_size: float = DEFAULT_TRAIN_SPLIT,
        val_size: float = DEFAULT_VAL_SPLIT,
        test_size: float = DEFAULT_TEST_SPLIT,
    ) -> None:
        """
        Execute the full dataset building process, including scanning for images, filtering, undersampling,
        splitting, and saving the final dataset.
        Args:
            output_path (str): The file path where the final dataset will be saved. Defaults to "data/dataset.parquet".
            train_size (float): The proportion of the dataset to include in the training set. Defaults to 0.8.
            val_size (float): The proportion of the dataset to include in the validation set. Defaults to 0.1.
            test_size (float): The proportion of the dataset to include in the test set. Defaults to 0.1.
        """
        # Get raw data from the storage path
        self.scan()

        # Filter out low quality data and duplicates
        self.filter()

        # Balance dataset representation
        self.undersample()

        # Split the data into train/validation/test
        self.split(train_size=train_size, val_size=val_size, test_size=test_size)

        # Save the final data frame
        self.save(name=output_path)
