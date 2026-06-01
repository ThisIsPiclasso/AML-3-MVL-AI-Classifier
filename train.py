import torch

from MVL_AI_Classifier.models.custom_loss import MultiViewLoss
from MVL_AI_Classifier.models.multi_view_manager_concat import MultiViewNet
from MVL_AI_Classifier.constants import (
    BATCH_SIZE,
    DEFAULT_N_BINS,
    DEFAULT_N_LEVELS,
    PATCH_SIZE,
    NUM_EPOCHS,
    DEFAULT_LEARNING_RATE,
)
from MVL_AI_Classifier.features.aps_pipeline import AzimuthalPowerSpectrumPreprocessor
from MVL_AI_Classifier.features.dct_pipeline import DCTDistributionPreprocessor
from MVL_AI_Classifier.features.glcm_pipeline import GLCMPreprocessor
from MVL_AI_Classifier.features.noise_residuals_pipeline import (
    NoiseResidualPreprocessor,
)
from MVL_AI_Classifier.data.datamanager import DataManager
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    writer = SummaryWriter(log_dir="runs/multiview_final_training")

    data_manager = DataManager(
        view_configuration=MODEL_CONFIGURATION,
        batch_size=BATCH_SIZE,
        use_cache=True,
        num_sections=3,
    )
    val_loader = data_manager.get_val_loader()
    model = MultiViewNet(MODEL_CONFIGURATION).to(device)
    loss_function = MultiViewLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=DEFAULT_LEARNING_RATE, weight_decay=1e-2
    )

    best_val_loss = float("inf")
    print("starting training")
    for epoch in range(NUM_EPOCHS):
        # init global trackers. this is for this epoch and all sections
        global_loss = 0.0
        global_correct = 0
        global_samples = 0
        for train_loader in data_manager.get_train_loaders():
            section_loss, section_correct, section_samples = train_epoch(
                model, train_loader, optimizer, loss_function, device
            )
            global_loss += section_loss
            global_correct += int(section_correct)
            global_samples += int(section_samples)
            del train_loader, section_loss, section_correct, section_samples

        val_loss, val_acc = validate(model, val_loader, loss_function, device)
        train_loss = global_loss / global_samples
        train_acc = (global_correct / global_samples) * 100
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
            torch.save(
                model.state_dict(), f"checkpoints/multiview_model_epoch_{epoch+1}.pt"
            )
    writer.close()
    print("training done")


def train_epoch(model, dataloader, optimizer, loss_function, device):
    progress = tqdm(dataloader, desc="Training section...", leave=False)
    model.train()
    # intitalize trackers for this epoch abd specific section
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

    return running_loss, correct_fusion, total_samples


def tune_hyperparameters():
    import optuna

    tuning_data_manager = DataManager(
        view_configuration=MODEL_CONFIGURATION,
        batch_size=BATCH_SIZE,
        use_cache=True,
        num_sections=3,
    )
    tuning_network = MultiViewNet(MODEL_CONFIGURATION)
    study = optuna.create_study(
        study_name="multiview_hyperparameter_sweep",
        storage="sqlite:///optuna_tuning.db",
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=1, interval_steps=1
        ),
    )
    best_hyperparameters = tuning_network.tune(
        data_manager=tuning_data_manager, study=study
    )

    print(f"optimized Parameters: {best_hyperparameters}")


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
    main()
    # tune_hyperparameters()
