from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # prevent opening GUI windows for plots
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

# Project imports
from MVL_AI_Classifier.data.cached_dataset import CachedDataClass
from MVL_AI_Classifier.data.dataclass import DataClass
from MVL_AI_Classifier.models.multi_view_manager_concat import MultiViewNet
from MVL_AI_Classifier.constants import (
    PARQUET_FILE,
    BATCH_SIZE,
    NUM_WORKERS,
    DEFAULT_N_BINS,
    DEFAULT_N_LEVELS,
    PATCH_SIZE,
    TRAIN_CACHE, 
    VAL_CACHE, 
    TEST_CACHE,
    BASELINE_EPOCHS,
    BASELINE_BATCH_SIZE,
    BASELINE_LR
)

from MVL_AI_Classifier.features.aps_pipeline import AzimuthalPowerSpectrumPreprocessor
from MVL_AI_Classifier.features.dct_pipeline import DCTDistributionPreprocessor
from MVL_AI_Classifier.features.glcm_pipeline import GLCMPreprocessor
from MVL_AI_Classifier.features.noise_residuals_pipeline import NoiseResidualPreprocessor


RESULTS_DIR = Path("evaluation_results")
CLASS_NAMES = ["Real", "AI-Generated"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRANCH_NAMES = ["aps", "dct", "glcm", "noise"]


MODEL_CONFIGURATION = {
    "aps": {
        "model_type": "mlp",
        "input_shape": (DEFAULT_N_BINS,),
        "preprocessor": AzimuthalPowerSpectrumPreprocessor(),
    },
    "dct": {
        "model_type": "cnn",
        "input_shape": (12, 8, 8),
        "preprocessor": DCTDistributionPreprocessor(),
    },
    "glcm": {
        "model_type": "cnn",
        "input_shape": (4, DEFAULT_N_LEVELS, DEFAULT_N_LEVELS),
        "preprocessor": GLCMPreprocessor(),
    },
    "noise": {
        "model_type": "cnn",
        "input_shape": (3, PATCH_SIZE // 16, PATCH_SIZE // 16),
        "preprocessor": NoiseResidualPreprocessor(),
    },
}

# Path to the saved model weights.
BEST_MODEL_PATH = Path("best_multiview_model.pt")
CACHE_DIR = Path("/workspace/AML-3-MVL-AI-Classifier/data/data_cache")
TEST_CACHE_PATH = CACHE_DIR / "test_features.h5"

class RawPixelDataset(DataClass):
    """DataClass subclass that returns raw pixel patches instead of preprocessed views.

    Reuses DataClass's parquet loading, split filtering, and center-crop
    patch extraction. It returns the raw (3, PATCH_SIZE, PATCH_SIZE) pixel tensor
    normalised to [0, 1] without the view preprocessing. Used by the baseline CNN
    """

    def __init__(self, parquet_file: str, split: str = "test"):
        super().__init__(
            parquet_file=parquet_file,
            view_configuration={}, # no view preprocessors needed for raw pixels
            split=split,
        )

    def __getitem__(self, index) -> dict:
        # Load image and extract patch
        item = self.df.iloc[index]  # get row from filtered dataframe
        image = Image.open(item["path"]) # get image 
        patch = self._get_patch(image, index) # get patch

        # Convert PIL patch to float32 tensor in [0, 1], shape (3, H, W).
        pixel_array = np.array(patch, dtype=np.float32) / 255.0
        pixel_tensor = torch.from_numpy(pixel_array.transpose(2, 0, 1)) # (H, W, C) -> (3, H, W)

        label = torch.tensor(int(item["label"]), dtype=torch.long) # convert label to tensor
        return {"image": pixel_tensor, "label": label}

# Baseline CNN definition and training function 

class BaselineCNN(nn.Module):
    """Simple CNN operating on raw (3, 256, 256) pixel patches."""

    def __init__(self):
        super().__init__()
        # Simple 3-layer CNN 
        self.features = nn.Sequential( 
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))

def train_and_predict_baseline(
    train_loader: DataLoader,
    eval_loader: DataLoader,
) -> dict:
    """Train baseline CNN and collect predictions.

    Args:
        train_loader: DataLoader yielding {"image": tensor, "label": tensor}.
            Shuffled for training.
        eval_loader: Same dataset but not shuffled, for collecting predictions.

    Returns:
        Dict with "probabilities" and "predictions" numpy arrays.
    """
    model = BaselineCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=BASELINE_LR)

    model.train()
    for epoch in range(BASELINE_EPOCHS):
        epoch_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"    Epoch {epoch+1}/{BASELINE_EPOCHS}  "
              f"loss={epoch_loss / len(train_loader):.4f}")

    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in eval_loader:
            images = batch["image"].to(DEVICE)
            probs = torch.softmax(model(images), dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["label"].numpy())

    probabilities = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return {
        "probabilities": probabilities,
        "predictions": (probabilities >= 0.5).astype(np.int64),
        "labels": labels,
    }
    
    
def collect_multiview_predictions(
    model: MultiViewNet, 
    test_loader: DataLoader,
) -> dict:
    """Run multi-view model on the test set and collect all predictions.

    Args:
        model: Trained MultiViewNet on DEVICE.
        test_loader: DataLoader from CachedDataClass.

    Returns:
        Dict with true_labels, fusion preds, and per-branch preds.
    """
    model.eval() # evaluation mode 
    all_labels = [] # true labels for all samples
    all_fusion_probs = [] # fusion head probabilities for all samples
    all_branch_probs = {name: [] for name in BRANCH_NAMES} # dict of branch probabilities

    with torch.no_grad(): # no gradients needed for evaluation
        for batch in test_loader: # iterate over test batches
            # Same batch format as training.
            views = {
                name: tensor.to(DEVICE)
                for name, tensor in batch["views"].items() # batch["views"] is a dict of view_name: tensor
            }
            labels = batch["label"].to(DEVICE)

            outputs = model(views) 
            # outputs["fusion"]: (B, 2) logits from the fusion head. B is batch size, 2 is number of classes.
            # outputs["branches"]: dict of {view_name: (B, 2) logits} from each branch's individual classifier 

            # Fusion probabilities.
            fusion_p = torch.softmax(outputs["fusion"], dim=1)[:, 1] # (B, 2) -> (B,) probabilities for class 1 (AI-generated)
            all_fusion_probs.append(fusion_p.cpu().numpy()) # move to CPU and convert to numpy for later concatenation

            # Branch probabilities.
            if "branches" in outputs:
                for name in BRANCH_NAMES:
                    if name in outputs["branches"]:
                        branch_prob = torch.softmax(outputs["branches"][name], dim=1)[:, 1] # (B, 2) -> (B,) probabilities for class 1 from this branch
                        all_branch_probs[name].append(branch_prob.cpu().numpy()) # move to CPU and store in dict

            all_labels.append(labels.cpu().numpy()) # store true labels for this batch

    true_labels = np.concatenate(all_labels) # concatenate all batches of true labels into one array
    fusion_probs = np.concatenate(all_fusion_probs) # concatenate all batches of fusion probabilities into one array

    result = {
        "true_labels": true_labels,
        "fusion": {
            "probabilities": fusion_probs,
            "predictions": (fusion_probs >= 0.5).astype(np.int64), #0.5 threshold for binary classification
        },
        "branches": {},
    }
    for name in BRANCH_NAMES:
        if all_branch_probs[name]:
            probs = np.concatenate(all_branch_probs[name])
            result["branches"][name] = {
                "probabilities": probs,
                "predictions": (probs >= 0.5).astype(np.int64),
            }
    return result



# Metrics and plotting functions 

def compute_all_metrics(true_labels, predicted_labels, probabilities, model_name):
    """Compute accuracy, precision, recall, F1, ROC-AUC"""
    metrics = {
        "name": model_name,
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1": f1_score(true_labels, predicted_labels, zero_division=0),
        "roc_auc": roc_auc_score(true_labels, probabilities)
    }
    print(f"  {model_name}")
    for k, v in metrics.items(): # k = metric name, v = metric value
        if k != "name": # skip printing the name key
            print(f"    {k:<18s} {v:.4f}") # 18 spaces for metric name, 4 decimal places for value
    return metrics


def plot_confusion_matrix(true_labels, preds, name, save_dir):
    """Plot a normalised confusion matrix."""
    confusion_m = confusion_matrix(true_labels, preds, normalize="true")
    _, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=confusion_m, display_labels=CLASS_NAMES).plot(
        ax=axis, cmap="Blues", values_format=".2f")
    axis.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    path = save_dir / f"cm_{name.replace(' ', '_').lower()}.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"    Saved: {path}")


def plot_roc_curves(curve_data, save_dir):
    """Plot ROC curves for all models."""
    _, axis = plt.subplots(figsize=(8, 6))
    for name, dictionary in curve_data.items(): # d is the dict containing true_labels, probabilities and roc_auc for this model
        false_positive_rate, true_positive_rate, _ = roc_curve(dictionary["true_labels"], dictionary["probabilities"]) #
        axis.plot(false_positive_rate, true_positive_rate, lw=2, label=f"{name} (AUC={dictionary['roc_auc']:.3f})")
    axis.plot([0, 1], [0, 1], "k--", lw=1, label="Random") # random classifier line
    axis.set_xlabel("False Positive Rate"); axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curves"); axis.legend(fontsize=8); axis.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "roc_curves.png", dpi=150); plt.close()
    print(f"    Saved: {save_dir / 'roc_curves.png'}")


def plot_metrics_bar(all_metrics, save_dir):
    """Save grouped bar chart comparing all models."""
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    length = len(all_metrics)
    width = 0.7 / length
    x_axis = np.arange(len(keys)) #x = 

    _, ax = plt.subplots(figsize=(12, 6))
    for index, metrics_dict in enumerate(all_metrics):
        vals = [metrics_dict[k] for k in keys]
        bars = ax.bar(x_axis + (index - length/2 + 0.5) * width, vals, width, label=metrics_dict["name"])
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{value:.3f}", ha="center", fontsize=7, rotation=45)
    ax.set_xticks(x_axis); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("Metric Comparison"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "metrics_bar.png", dpi=150); plt.close()
    print(f"    Saved: {save_dir / 'metrics_bar.png'}")


def plot_branch_heatmap(branch_metrics, save_dir):
    """Save heatmap of branch performance."""
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    branches = list(branch_metrics.keys())
    data = np.array([[branch_metrics[b][k] for k in keys] for b in branches])

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(["Acc", "Prec", "Rec", "F1", "AUC"])
    ax.set_yticks(range(len(branches)))
    ax.set_yticklabels([b.upper() for b in branches])
    ax.set_title("Branch Heatmap")
    plt.colorbar(im, ax=ax, fraction=0.03)
    for r in range(len(branches)):
        for c in range(len(keys)):
            color = "white" if data[r, c] > 0.7 else "black"
            ax.text(c, r, f"{data[r,c]:.3f}", ha="center", va="center",
                    fontsize=10, color=color)
    plt.tight_layout()
    plt.savefig(save_dir / "branch_heatmap.png", dpi=150); plt.close()
    print(f"    Saved: {save_dir / 'branch_heatmap.png'}")


def plot_branch_vs_fusion(fusion_m, branch_m, save_dir):
    """Save bar chart comparing branches to fusion."""
    names, accs, f1s = [], [], []
    for bname, bm in branch_m.items():
        names.append(bname.upper())
        accs.append(bm["accuracy"])
        f1s.append(bm["f1"])
    names.append("FUSION"); accs.append(fusion_m["accuracy"]); f1s.append(fusion_m["f1"])

    x = np.arange(len(names)); w = 0.35
    _, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#4C72B0")
    b2 = ax.bar(x + w/2, f1s, w, label="F1 Score", color="#DD8452")
    for bar in [b1[-1], b2[-1]]:
        bar.set_edgecolor("red"); bar.set_linewidth(2)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("Branch vs Fusion"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "branch_vs_fusion.png", dpi=150); plt.close()
    print(f"    Saved: {save_dir / 'branch_vs_fusion.png'}")



def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    active_views = list(MODEL_CONFIGURATION.keys()) # get the list of view names from the model configuration
    #   batch["views"] = {"aps": tensor, "dct": tensor, ...}, returned by CachedDataClass, already on CPU
    #   batch["label"] = tensor
    test_data = CachedDataClass(
        hdf5_path=str(TEST_CACHE_PATH),
        view_keys=active_views,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    print(f"  Cached test set: {len(test_data)} samples")

    # Load raw images for baseline CNN
    raw_test_data = RawPixelDataset(parquet_file=PARQUET_FILE, split="test")

    # Shuffled loader for training the baseline.
    baseline_train_loader = DataLoader(
        raw_test_data,
        batch_size=BASELINE_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    # Unshuffled loader for collecting predictions in order.
    baseline_eval_loader = DataLoader(
        raw_test_data,
        batch_size=BASELINE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    print(f"  Raw test set: {len(raw_test_data)} images")


    model = MultiViewNet(MODEL_CONFIGURATION).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    mvl_results = collect_multiview_predictions(model, test_loader)
    true_labels = mvl_results["true_labels"]

    baseline_preds = train_and_predict_baseline(
        baseline_train_loader,
        baseline_eval_loader,
    )

    all_metrics = []

    # Fusion head.
    fusion_m = compute_all_metrics(
        true_labels,
        mvl_results["fusion"]["predictions"],
        mvl_results["fusion"]["probabilities"],
        "MVL Fusion",
    )
    all_metrics.append(fusion_m)

    # Individual branches.
    branch_m = {}
    for bname, bdata in mvl_results["branches"].items():
        bm = compute_all_metrics(
            true_labels, bdata["predictions"], bdata["probabilities"],
            f"Branch {bname.upper()}",
        )
        branch_m[bname] = bm
        all_metrics.append(bm)

    # Baseline CNN.
    baseline_m = compute_all_metrics(
        baseline_preds["labels"],
        baseline_preds["predictions"],
        baseline_preds["probabilities"],
        "Baseline CNN",
    )
    all_metrics.append(baseline_m)


    #  confusion matrix for baseline:
    plot_confusion_matrix(
        baseline_preds["labels"], baseline_preds["predictions"],
        "Baseline CNN", RESULTS_DIR)

    # curve data for baseline:
    curve_data["Baseline CNN"] = {
        "true_labels": baseline_preds["labels"],
        "probabilities": baseline_preds["probabilities"],
        "roc_auc": baseline_m["roc_auc"]
    }
    for bname in branch_m:
        curve_data[f"Branch {bname.upper()}"] = {
            "true_labels": true_labels,
            "probabilities": mvl_results["branches"][bname]["probabilities"],
            "roc_auc": branch_m[bname]["roc_auc"]
        }

    plot_roc_curves(curve_data, RESULTS_DIR)

    # 7c. Bar charts and heatmaps.
    plot_metrics_bar(all_metrics, RESULTS_DIR)
    if branch_m:
        plot_branch_heatmap(branch_m, RESULTS_DIR)
        plot_branch_vs_fusion(fusion_m, branch_m, RESULTS_DIR)


    print(f"  {'Model':<25s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'AUC':>7s}")
    for m in all_metrics:
        print(f"  {m['name']:<25s} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['roc_auc']:>7.4f}")

    # Fusion vs baseline.
    d_acc = fusion_m["accuracy"] - baseline_m["accuracy"]
    d_f1 = fusion_m["f1"] - baseline_m["f1"]
    print(f"     Accuracy: {d_acc:+.4f}")
    print(f"     F1:       {d_f1:+.4f}")

    if branch_m:
        best = max(branch_m.items(), key=lambda x: x[1]["f1"])
        print(f"\n  Best branch: {best[0].upper()} (F1={best[1]['f1']:.4f})")
        if all(fusion_m["f1"] >= bm["f1"] for bm in branch_m.values()):
            print("  Fusion outperforms all individual branches.")
        else:
            print(" Fusion does not beat all branches.")

    print(f"\n  All plots saved to: {RESULTS_DIR.resolve()}")

if __name__ == "__main__":
    main()
