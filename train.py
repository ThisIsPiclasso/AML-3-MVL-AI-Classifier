import torch
from torch.utils.data import DataLoader

from MVL_AI_Classifier.data.dataclass import DataClass
from MVL_AI_Classifier.models.custom_loss import MultiViewLoss
from MVL_AI_Classifier.models.multi_view_manager_concat import MultiViewNet
from MVL_AI_Classifier.constants import (
    PARQUET_FILE,
    BATCH_SIZE,
    NUM_WORKERS,
    DEFAULT_N_BINS,
    DEFAULT_N_LEVELS,
    PATCH_SIZE,
    NUM_EPOCHS,
    VAL_CACHE,
    TRAIN_CACHE,
    N_TRIALS,
)
from MVL_AI_Classifier.features.aps_pipeline import AzimuthalPowerSpectrumPreprocessor
from MVL_AI_Classifier.features.dct_pipeline import DCTDistributionPreprocessor
from MVL_AI_Classifier.data.cached_dataset import CachedDataClass
from MVL_AI_Classifier.features.glcm_pipeline import GLCMPreprocessor
from MVL_AI_Classifier.features.noise_residuals_pipeline import (
    NoiseResidualPreprocessor,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


MODEL_CONFIGURATION = {
    "aps": {
        "model_type": "mlp",
        "input_shape": (DEFAULT_N_BINS,),
        "preprocessor": AzimuthalPowerSpectrumPreprocessor(),
    },
    "dct": {
        "model_type": "cnn",
        "input_shape": (12, 8, 8),  # this needs defaults to set input shape
        "preprocessor": DCTDistributionPreprocessor(),
    },
    "glcm": {
        "model_type": "cnn",
        "input_shape": (4, DEFAULT_N_LEVELS, DEFAULT_N_LEVELS),
        "preprocessor": GLCMPreprocessor(),
    },
    "noise": {
        "model_type": "cnn",
        "input_shape": (
            3,
            PATCH_SIZE // 16,
            PATCH_SIZE // 16,
        ),  # fix this with good defaults
        "preprocessor": NoiseResidualPreprocessor(),
    },
}


def get_dataloaders(
    view_configuration: dict, batch_size: int, use_cache: bool = False
) -> tuple[DataLoader, DataLoader]:
    """
    Utility function to get dataloaders for training and validation.
    Args:
        view_configuration (dict): The configuration dictionary for the views, used to determine which features to load.
        batch_size (int): The batch size for the dataloaders.
        use_cache (bool): Whether to use cached preprocessed features or to preprocess on the fly. Defaults to False.
    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
    """
    if use_cache:
        # Cached data Pipeline
        active_views = list(view_configuration.keys())
        train_data = CachedDataClass(hdf5_path=TRAIN_CACHE, view_keys=active_views)
        val_data = CachedDataClass(hdf5_path=VAL_CACHE, view_keys=active_views)
    else:
        # Old preprocessing pipeline
        train_data = DataClass(
            parquet_file=PARQUET_FILE,
            view_configuration=view_configuration,
            split="train",
        )
        val_data = DataClass(
            parquet_file=PARQUET_FILE,
            view_configuration=view_configuration,
            split="val",
        )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    return train_loader, val_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    writer = SummaryWriter(log_dir="runs/multiview_experiment_1")

    train_loader, val_loader = get_dataloaders(
        view_configuration=MODEL_CONFIGURATION, batch_size=BATCH_SIZE, use_cache=True
    )

    print(
        f"training on: {len(train_loader)} samples, validating on: {len(val_loader)} samples"
    )

    model = MultiViewNet(MODEL_CONFIGURATION).to(device)
    loss_function = MultiViewLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, loss_function, device
        )
        train_loss = 0.0
        val_loss, val_acc = validate(model, val_loader, loss_function, device)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_multiview_model.pt")
    writer.close()
    print("training done")


def train_epoch(model, dataloader, optimizer, loss_function, device):
    progress = tqdm(dataloader, desc="Training...", leave=True)
    model.train()
    # intitalize trackers for this epoch
    running_loss = 0.0
    correct_fusion = 0
    total_samples = 0

    for batch_idx, batch in enumerate(progress):
        # the dataloader takes a batch of images to process at the same time
        # moves all preproccessed views, and all labels into vram
        x_dict = {key: tensor.to(device) for key, tensor in batch["views"].items()}
        y = batch["label"].to(device)
        # clearing optimizer from previous batches
        optimizer.zero_grad()
        # get output guesses from model
        outputs = model(x_dict)
        # calculate losses according to the custom loss function
        losses = loss_function(outputs, y)
        total_loss = losses["total_loss"]
        # do a back prop based on total loss
        total_loss.backward()
        # update model weights
        optimizer.step()
        # add loss to running total, batch loss is scaled to batch size for the last possible incomplete batch
        running_loss += total_loss.item() * y.size(0)
        # actual final prediction made by model
        predictions = torch.argmax(outputs["fusion"], dim=1)
        # count how many of the batch was correcly labeled
        correct_fusion += (predictions == y).sum().item()
        # add the current amount of processed images to runnning total
        total_samples += y.size(0)
        current_loss = total_loss.item()
        current_acc = (correct_fusion / total_samples) * 100
        progress.set_postfix(
            {"Loss": f"{current_loss:.4f}", "Fusion_Acc": f"{current_acc:.2f}%"}
        )
    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_fusion / total_samples) * 100
    return epoch_loss, epoch_acc


def validate(model, dataloader, loss_function, device):
    model.eval()
    # again init trackers
    running_loss = 0.0
    correct_fusion = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # the dataloader takes a batch of images to process at the same time
            # moves all preproccessed views, and all labels into vram
            x_dict = {key: tensor.to(device) for key, tensor in batch["views"].items()}
            y = batch["label"].to(device)
            # get output guesses from model
            outputs = model(x_dict)
            # calculate losses according to the custom loss function
            losses = loss_function(outputs, y)
            total_loss = losses["total_loss"]
            running_loss += total_loss.item() * y.size(0)
            # actual final prediction made by model
            predictions = torch.argmax(outputs["fusion"], dim=1)
            # count how many of the batch was correcly labeled
            correct_fusion += (predictions == y).sum().item()
            # add the current amount of processed images to runnning total
            total_samples += y.size(0)
    val_loss = running_loss / total_samples
    val_acc = (correct_fusion / total_samples) * 100
    return val_loss, val_acc


if __name__ == "__main__":
    # The training is commented out for now.
    # After we get the best hyperparameters, we will run the main training.
    # main()

    tuning_network = MultiViewNet(MODEL_CONFIGURATION)

    best_hyperparameters = tuning_network.tune(n_trials=N_TRIALS)

    print(f"Optimized Parameters: {best_hyperparameters}")
