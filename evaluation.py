from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Prevent images from opening in a GUI during runs.

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from PIL import Image, ImageFile

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from MVL_AI_Classifier.data.cached_dataclass import CachedDataClass
from MVL_AI_Classifier.data.dataclass import DataClass

from MVL_AI_Classifier.features.aps_pipeline import AzimuthalPowerSpectrumPreprocessor

from MVL_AI_Classifier.features.dct_pipeline import DiscreteCosineTransformPreprocessor

from MVL_AI_Classifier.features.glcm_pipeline import (
    GrayLevelCooccurrenceMatrixPreprocessor,
)

from MVL_AI_Classifier.features.nr_pipeline import NoiseResidualPreprocessor

from MVL_AI_Classifier.models.multi_view_manager_concat import MultiViewNet

from MVL_AI_Classifier.constants import (
    BASELINE_BATCH_SIZE,
    BASELINE_EPOCHS,
    BASELINE_LR,
    BATCH_SIZE,
    DEFAULT_N_BINS,
    DEFAULT_N_LEVELS,
    NUM_WORKERS,
    PARQUET_FILE,
    _SIZE,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("evaluation_results")

CLASS_NAMES = [
    "Real",  # 0
    "AI-Generated",  # 1
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BRANCH_NAMES = [
    "aps",
    "dct",
    "glcm",
    "noise",
]

MODEL_CONFIGURATION = {
    "aps": {
        "model_type": "mlp",
        "input_shape": (DEFAULT_N_BINS,),
        "preprocessor": AzimuthalPowerSpectrumPreprocessor(),
    },
    "dct": {
        "model_type": "cnn",
        "input_shape": (12, 8, 8),
        "preprocessor": DiscreteCosineTransformPreprocessor(),
    },
    "glcm": {
        "model_type": "cnn",
        "input_shape": (
            4,
            DEFAULT_N_LEVELS,
            DEFAULT_N_LEVELS,
        ),
        "preprocessor": GrayLevelCooccurrenceMatrixPreprocessor(),
    },
    "noise": {
        "model_type": "cnn",
        "input_shape": (
            3,
            _SIZE // 16,
            _SIZE // 16,
        ),
        "preprocessor": NoiseResidualPreprocessor(),
    },
}

BEST_MODEL_PATH = Path("models/trained_model.pt")
BASELINE_MODEL_PATH = Path("baseline_cnn.pt")

CACHE_DIR = Path("/workspace/AML-3-MVL-AI-Classifier/data/data_cache")

TEST_CACHE_PATH = CACHE_DIR / "test_features.h5"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RawPixelDataset(DataClass):
    """Dataset that returns raw RGB pixel es.

    This subclass reuses the parquet loading, split filtering, and 
    extraction logic provided by ``DataClass``. It skips all view
    preprocessors and only returns the raw  as a tensor

    Attributes:
        Inherited from ``DataClass``.
    """

    def __init__(
        self,
        parquet_file: str,
        split: str = "test",
    ) -> None:
        """Initialize the dataset for a specific parquet file and split.

        Args:
            parquet_file: Path to the parquet file containing metadata.
            split: Dataset split to load. Defaults to ``"test"``.
        """
        super().__init__(
            parquet_file=parquet_file,
            view_configuration={},
            split=split,
        )

    def __getitem__(self, idx: int) -> dict:
        """Return one sample as an image-label dictionary.

        Args:
            idx: Integer row index into the dataset.

        Returns:
            A dictionary with:
                - ``"image"``: Tensor of shape ``(3, H, W)`` in ``[0, 1]``.
                - ``"label"``: Long tensor containing the binary class label.
        """
        item = self.df.iloc[idx]

        with Image.open(item["path"]) as image:
            patch = self._get_patch(image, idx).convert("RGB")

        pixel_array = np.array(patch, dtype=np.float32) / 255.0
        pixel_tensor = torch.from_numpy(
            pixel_array.transpose(2, 0, 1)
        )  # from (Height, Width, Colour) to (Colour, Height, Width)

        label = torch.tensor(int(item["label"]), dtype=torch.long)

        return {
            "image": pixel_tensor,
            "label": label,
        }


# ---------------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------------


class BaselineCNN(nn.Module):
    """Three-block CNN for raw RGB pixel patches as baseline.

    The model consumes tensors of shape ``(B, 3, 256, 256)`` and produces
    binary logits of shape ``(B, 2)``
    Architecture:
        Block 1:
            Conv2d(3 -> 16, kernel_size=3, padding=1)
            BatchNorm2d(16)
            ReLU
            MaxPool2d(2)

        Block 2:
            Conv2d(16 -> 32, kernel_size=3, padding=1)
            BatchNorm2d(32)
            ReLU
            MaxPool2d(2)

        Block 3:
            Conv2d(32 -> 64, kernel_size=3, padding=1)
            ReLU
            AdaptiveAvgPool2d((4, 4))

        Head:
            Linear(1024 -> 128)
            ReLU
            Dropout(0.3)
            Linear(128 -> 2)
    """

    def __init__(self) -> None:
        """Build the CNN feature extractor and classifier head."""
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            Logits tensor of shape ``(B, 2)``.
        """
        return self.classifier(torch.flatten(self.features(x), 1))


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _probabilities_from_logits(logits: torch.Tensor) -> np.ndarray:
    """Convert 2-class logits to positive-class probabilities (AI-generated)

    Args:
        logits: Tensor of shape ``(B, 2)`` containing raw model outputs.

    Returns:
        NumPy array of shape ``(B,)`` containing the probability of class 1 (AI-generated)

    """
    return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


def _predictions_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Threshold probabilities at 0.5 for a binary predictions.

    Args:
        probabilities: NumPy array of positive-class probabilities.

    Returns:
        NumPy array of dtype ``int64`` with values 0 or 1.
    """
    return (probabilities >= 0.5).astype(np.int64)


# ---------------------------------------------------------------------------
# Training and prediction
# ---------------------------------------------------------------------------


def train_and_predict_baseline(
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> dict:
    """Train the baseline CNN and collect predictions on the evaluation split.


    Args:
        train_loader:  DataLoader yielding dictionaries with keys
            ``"image"`` and ``"label"``.
        test_loader:  DataLoader yielding dictionaries with keys
            ``"image"`` and ``"label"``.

    Returns:
        Dictionary with keys:
            - ``"labels"``: Ground-truth labels as a NumPy array.
            - ``"probabilities"``: Positive-class probabilities as a NumPy array.
            - ``"predictions"``: Binary predictions as a NumPy array.
    """
    model = BaselineCNN().to(DEVICE)
    if BASELINE_MODEL_PATH.exists():
        model.load_state_dict(
            torch.load(BASELINE_MODEL_PATH, map_location=DEVICE)
        )
    else: 
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=BASELINE_LR)

        model.train()
        for _epoch in range(BASELINE_EPOCHS):
            for batch in tqdm(train_loader, desc=f"Epoch {_epoch + 1}/{BASELINE_EPOCHS}"):
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), BASELINE_MODEL_PATH)


    model.eval()

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(DEVICE)

            all_probs.append(_probabilities_from_logits(model(images)))
            all_labels.append(batch["label"].cpu().numpy())

    probabilities = np.concatenate(all_probs)
    labels_array = np.concatenate(all_labels)

    return {
        "labels": labels_array,
        "probabilities": probabilities,
        "predictions": _predictions_from_probabilities(probabilities),
    }


def collect_multiview_predictions(
    model: MultiViewNet,
    test_loader: DataLoader,
) -> dict:
    """Run a trained multi-view model over the test set and collect outputs.

    Args:
        model: Trained ``MultiViewNet``
        test_loader: DataLoader built from ``CachedDataClass``.

    Returns:
        Dictionary with this structure::

            {
                "true_labels": np.ndarray,
                "fusion": {
                    "probabilities": np.ndarray,
                    "predictions": np.ndarray,
                },
                "branches": {
                    "<branch_name>": {
                        "probabilities": np.ndarray,
                        "predictions": np.ndarray,
                    },
                    ...
                },
            }
    """
    model.eval()

    all_labels: list[np.ndarray] = []
    all_fusion_probs: list[np.ndarray] = []
    all_branch_probs: dict[str, list[np.ndarray]] = {name: [] for name in BRANCH_NAMES}

    with torch.no_grad():
        for batch in test_loader:
            views = {name: tensor.to(DEVICE) for name, tensor in batch["views"].items()}
            labels = batch["label"].to(DEVICE)

            outputs = model(views)

            all_fusion_probs.append(_probabilities_from_logits(outputs["fusion"]))

            if "branches" in outputs:
                for name in BRANCH_NAMES:
                    if name in outputs["branches"]:
                        all_branch_probs[name].append(
                            _probabilities_from_logits(outputs["branches"][name])
                        )

            all_labels.append(labels.cpu().numpy())

    true_labels = np.concatenate(all_labels)
    fusion_probs = np.concatenate(all_fusion_probs)

    result: dict = {
        "true_labels": true_labels,
        "fusion": {
            "probabilities": fusion_probs,
            "predictions": _predictions_from_probabilities(fusion_probs),
        },
        "branches": {},
    }

    for name in BRANCH_NAMES:
        if all_branch_probs[name]:
            probs = np.concatenate(all_branch_probs[name])
            result["branches"][name] = {
                "probabilities": probs,
                "predictions": _predictions_from_probabilities(probs),
            }

    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_all_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
) -> dict:
    """Compute the standard binary-classification metrics.

    Args:
        true_labels: Ground-truth labels of shape ``(N,)``.
        predicted_labels: Binary predictions of shape ``(N,)``.
        probabilities: Positive-class probabilities of shape ``(N,)``.
        model_name: Human-readable identifier for the model.

    Returns:
        Dictionary containing the model name and all computed metrics.

    Raises:
        ValueError: If ROC AUC cannot be computed because only one class is
            present in ``true_labels``.
    """
    try:
        roc_auc = roc_auc_score(true_labels, probabilities)
    except ValueError:
        roc_auc = float("nan")

    return {
        "name": model_name,
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1": f1_score(true_labels, predicted_labels, zero_division=0),
        "roc_auc": roc_auc,
        "avg_precision": average_precision_score(true_labels, probabilities),
        "mcc": matthews_corrcoef(true_labels, predicted_labels),
        "brier_score": brier_score_loss(true_labels, probabilities),
    }


def compute_per_generator_accuracy(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    model_types: np.ndarray,
    model_name: str,
    save_dir: Path,
) -> pd.DataFrame:
    """Compute per-generator accuracy and save the result as CSV.

    For each unique source / generator in ``model_types``, the function
    computes accuracy and then reports either recall (for AI-generated groups)
    or specificity (for pure real-image groups).

    Args:
        true_labels: Ground-truth binary labels of shape ``(N,)``.
        predicted_labels: Binary predictions of shape ``(N,)``.
        model_types: String array of source / generator labels of shape ``(N,)``.
        model_name: Human-readable model identifier used in the output filename.
        save_dir: Directory where the CSV file will be written.

    Returns:
        DataFrame sorted by ascending accuracy, with one row per generator.

    Raises:
        OSError: If the CSV cannot be written to ``save_dir``.
    """
    rows: list[dict] = []

    for generator_name in sorted(np.unique(model_types)):
        mask = model_types == generator_name

        group_true = true_labels[mask]
        group_pred = predicted_labels[mask]

        n_samples = int(mask.sum())
        n_correct = int((group_true == group_pred).sum())

        group_accuracy = n_correct / n_samples if n_samples > 0 else 0.0

        if group_true.sum() == 0:
            # Pure real-image group: recall is not meaningful.
            group_recall = float("nan")
            group_specificity = float((group_pred == 0).sum()) / n_samples
        else:
            # AI-generated or mixed group: specificity is not the intended summary.
            group_recall = float(recall_score(group_true, group_pred, zero_division=0))
            group_specificity = float("nan")

        rows.append(
            {
                "generator": generator_name,
                "n_samples": n_samples,
                "n_correct": n_correct,
                "accuracy": group_accuracy,
                "recall": group_recall,
                "specificity": group_specificity,
            }
        )

    results_df = (
        pd.DataFrame(rows)
        .sort_values("accuracy", ascending=True)
        .reset_index(drop=True)
    )

    safe_name = model_name.replace(" ", "").lower()
    csv_path = save_dir / f"per_generator_{safe_name}.csv"

    results_df.to_csv(csv_path, index=False)

    return results_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _save_and_close(path: Path) -> None:
    """Save the current Matplotlib figure and close it.

    Args:
        path: Destination file path for the current figure.
    """
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    name: str,
    save_dir: Path,
) -> None:
    """Save a row-normalized confusion matrix for one model.

    Args:
        true_labels: Ground-truth binary labels of shape ``(N,)``.
        predicted_labels: Binary predictions of shape ``(N,)``.
        name: Model label used in the title and filename.
        save_dir: Output directory.
    """
    cm = confusion_matrix(true_labels, predicted_labels, normalize="true")

    _, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    ).plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f",
    )

    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()

    _save_and_close(save_dir / f"cm_{name.replace(' ', '').lower()}.png")


def plot_roc_curves(curve_data: dict, save_dir: Path) -> None:
    """Save overlaid ROC curves for all models.

    Args:
        curve_data: Dictionary mapping model names to dictionaries with keys
            ``true_labels``, ``probabilities``, and ``roc_auc``.
        save_dir: Output directory.
    """
    _, ax = plt.subplots(figsize=(8, 6))

    for name, d in curve_data.items():
        fpr, tpr, _ = roc_curve(d["true_labels"], d["probabilities"])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={d['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save_and_close(save_dir / "roc_curves.png")


def plot_pr_curves(curve_data: dict, save_dir: Path) -> None:
    """Save overlaid precision-recall curves for all models.

    Args:
        curve_data: Dictionary mapping model names to dictionaries with keys
            ``true_labels``, ``probabilities``, and ``avg_precision``.
        save_dir: Output directory.
    """
    _, ax = plt.subplots(figsize=(8, 6))

    for name, d in curve_data.items():
        prec, rec, _ = precision_recall_curve(d["true_labels"], d["probabilities"])
        ax.plot(rec, prec, lw=2, label=f"{name} (AP={d['avg_precision']:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    _save_and_close(save_dir / "pr_curves.png")


def plot_metrics_bar(all_metrics: list[dict], save_dir: Path) -> None:
    """Save a grouped bar chart comparing all models across all metrics.

    Args:
        all_metrics: List of metric dictionaries returned by
            ``compute_all_metrics``.
        save_dir: Output directory.
    """
    metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "avg_precision",
        "mcc",
        "brier_score",
    ]

    metric_labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC AUC",
        "Avg Prec",
        "MCC",
        "Brier",
    ]

    n_models = len(all_metrics)
    bar_width = 0.7 / n_models
    x_positions = np.arange(len(metric_keys))

    _, ax = plt.subplots(figsize=(14, 6))

    for model_idx, metrics in enumerate(all_metrics):
        values = [metrics[k] for k in metric_keys]

        offsets = x_positions + (model_idx - n_models / 2 + 0.5) * bar_width

        bars = ax.bar(offsets, values, bar_width, label=metrics["name"])

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{value:.3f}",
                ha="center",
                fontsize=7,
                rotation=45,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(-0.15, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Metric Comparison")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_and_close(save_dir / "metrics_bar.png")


def plot_branch_heatmap(branch_metrics: dict, save_dir: Path) -> None:
    """Save a heatmap of all metrics for each individual branch.

    Args:
        branch_metrics: Dictionary mapping branch names to metric dictionaries.
        save_dir: Output directory.
    """
    metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "avg_precision",
        "mcc",
        "brier_score",
    ]

    metric_labels = [
        "Acc",
        "Prec",
        "Rec",
        "F1",
        "AUC",
        "AP",
        "MCC",
        "Brier",
    ]

    branch_names = list(branch_metrics.keys())

    data = np.array([[branch_metrics[b][k] for k in metric_keys] for b in branch_names])

    _, ax = plt.subplots(figsize=(10, 4))

    image = ax.imshow(
        data,
        vmin=-0.1,
        vmax=1.0,
        cmap="YlOrRd",
        aspect="auto",
    )

    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(metric_labels)

    ax.set_yticks(range(len(branch_names)))
    ax.set_yticklabels([b.upper() for b in branch_names])

    ax.set_title("Branch Performance Heatmap")
    plt.colorbar(image, ax=ax, fraction=0.03)

    for row_idx in range(len(branch_names)):
        for col_idx in range(len(metric_keys)):
            cell_value = data[row_idx, col_idx]
            text_color = "white" if cell_value > 0.7 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{cell_value:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
            )

    plt.tight_layout()
    _save_and_close(save_dir / "branch_heatmap.png")


def plot_branch_vs_fusion(
    fusion_metrics: dict,
    branch_metrics: dict,
    save_dir: Path,
) -> None:
    """Save a grouped bar chart comparing branch metrics against fusion.

    The fusion bars are highlighted with a red border to distinguish them from
    the individual branches.

    Args:
        fusion_metrics: Metric dictionary for the fusion head.
        branch_metrics: Dictionary mapping branch names to metric dictionaries.
        save_dir: Output directory.
    """
    names = [name.upper() for name in branch_metrics]

    accuracies = [m["accuracy"] for m in branch_metrics.values()]
    f1_scores = [m["f1"] for m in branch_metrics.values()]

    names.append("FUSION")
    accuracies.append(fusion_metrics["accuracy"])
    f1_scores.append(fusion_metrics["f1"])

    x_positions = np.arange(len(names))
    bar_width = 0.35

    _, ax = plt.subplots(figsize=(10, 5))

    accuracy_bars = ax.bar(
        x_positions - bar_width / 2,
        accuracies,
        bar_width,
        label="Accuracy",
        color="#4C72B0",
    )

    f1_bars = ax.bar(
        x_positions + bar_width / 2,
        f1_scores,
        bar_width,
        label="F1 Score",
        color="#DD8452",
    )

    for bar in [accuracy_bars[-1], f1_bars[-1]]:
        bar.set_edgecolor("red")
        bar.set_linewidth(2)

    for bar in list(accuracy_bars) + list(f1_bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center",
            fontsize=8,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Branch vs Fusion")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_and_close(save_dir / "branch_vs_fusion.png")


def plot_per_generator_accuracy(
    generator_df: pd.DataFrame,
    model_name: str,
    save_dir: Path,
) -> None:
    """Save a horizontal bar chart of per-generator accuracy.

    Bars for real-image sources are colored blue; AI-generator bars are colored
    orange. The rows are assumed to arrive already sorted by accuracy.

    Args:
        generator_df: DataFrame produced by ``compute_per_generator_accuracy``.
        model_name: Model label used in the title and filename.
        save_dir: Output directory.
    """
    generator_names = generator_df["generator"].values
    accuracies = generator_df["accuracy"].values
    y_positions = np.arange(len(generator_names))

    colors = [
        "#4C72B0" if np.isnan(row["recall"]) else "#DD8452"
        for _, row in generator_df.iterrows()
    ]

    _, ax = plt.subplots(figsize=(10, max(4, len(generator_df) * 0.5)))

    bars = ax.barh(
        y_positions,
        accuracies,
        color=colors,
        edgecolor="white",
    )

    for bar, accuracy_value in zip(bars, accuracies):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{accuracy_value:.3f}",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(generator_names, fontsize=9)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1.12)

    ax.set_title(f"Per-Generator Accuracy — {model_name}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    safe_name = model_name.replace(" ", "").lower()
    _save_and_close(save_dir / f"per_generator{safe_name}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print_summary(
    all_metrics: list[dict],
    fusion_m: dict,
    baseline_m: dict,
    branch_m: dict,
) -> None:
    """Print the final results table and comparison summary to stdout.

    Args:
        all_metrics: List of metric dictionaries to print in table form.
        fusion_m: Metric dictionary for the fusion head.
        baseline_m: Metric dictionary for the raw-pixel baseline.
        branch_m: Dictionary mapping branch names to metric dictionaries.
    """
    header = (
        f"{'Model':<25s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s}"
        f" {'F1':>7s} {'AUC':>7s} {'AP':>7s} {'MCC':>7s} {'Brier':>7s}"
    )

    separator = "─" * 72

    print(f"\n{separator}")
    print(header)
    print(separator)

    for m in all_metrics:
        print(
            f"{m['name']:<25s}"
            f" {m['accuracy']:>7.4f}"
            f" {m['precision']:>7.4f}"
            f" {m['recall']:>7.4f}"
            f" {m['f1']:>7.4f}"
            f" {m['roc_auc']:>7.4f}"
            f" {m['avg_precision']:>7.4f}"
            f" {m['mcc']:>7.4f}"
            f" {m['brier_score']:>7.4f}"
        )

    print(separator)

    d_acc = fusion_m["accuracy"] - baseline_m["accuracy"]
    d_f1 = fusion_m["f1"] - baseline_m["f1"]
    d_mcc = fusion_m["mcc"] - baseline_m["mcc"]

    print(
        f"\nMVL Fusion vs Baseline "
        f"Acc {d_acc:+.4f} "
        f"F1 {d_f1:+.4f} "
        f"MCC {d_mcc:+.4f}"
    )

    if branch_m:
        best_branch, best_metrics = max(branch_m.items(), key=lambda x: x[1]["f1"])

        print(f"Best branch: {best_branch.upper()} " f"F1={best_metrics['f1']:.4f}")

        if all(fusion_m["f1"] >= bm["f1"] for bm in branch_m.values()):
            print("Fusion outperforms all individual branches.")
        else:
            print("Fusion does NOT outperform all individual branches.")


def main() -> None:
    """Run the full evaluation pipeline.

    The pipeline performs four main steps:
        1. Load cached multi-view test features.
        2. Build the raw-pixel baseline dataset.
        3. Run inference for the multi-view model and train/evaluate the baseline.
        4. Compute metrics, generate plots, and print a summary table.

    Raises:
        FileNotFoundError: If the cached test features file or model checkpoint
            is missing.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cached test features (multi-view model)
    if not TEST_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Test cache not found: {TEST_CACHE_PATH}. "
            "Run cache_features.py to generate it."
        )

    active_views = list(MODEL_CONFIGURATION.keys())

    test_data = CachedDataClass(
        hdf5_path=str(TEST_CACHE_PATH),
        view_keys=active_views,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    # 2. Raw pixel dataset (baseline CNN)
    raw_test_data = RawPixelDataset(
        parquet_file=PARQUET_FILE,
        split="test",
    )

    raw_train_data = RawPixelDataset(
        parquet_file=PARQUET_FILE,
        split="train",
    )

    baseline_train_loader = DataLoader(
        raw_train_data,
        batch_size=BASELINE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    baseline_test_loader = DataLoader(
        raw_test_data,
        batch_size=BASELINE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # 3. Load MultiViewNet
    mvl_model = MultiViewNet(MODEL_CONFIGURATION).to(DEVICE)

    mvl_model.load_state_dict(
        torch.load(
            BEST_MODEL_PATH,
            map_location=DEVICE,
        )
    )

    # 4. Multi-view inference

    mvl_results = collect_multiview_predictions(
        mvl_model,
        test_loader,
    )

    true_labels = mvl_results["true_labels"]

    # 5. Baseline training and inference
    baseline_preds = train_and_predict_baseline(
        baseline_train_loader,
        baseline_test_loader,
    )

    baseline_labels = baseline_preds["labels"]

    # 6. Metrics
    all_metrics: list[dict] = []

    fusion_m = compute_all_metrics(
        true_labels,
        mvl_results["fusion"]["predictions"],
        mvl_results["fusion"]["probabilities"],
        "MVL Fusion",
    )
    all_metrics.append(fusion_m)

    branch_m: dict[str, dict] = {}

    for branch_name, branch_data in mvl_results["branches"].items():
        branch_metrics = compute_all_metrics(
            true_labels,
            branch_data["predictions"],
            branch_data["probabilities"],
            f"Branch {branch_name.upper()}",
        )
        branch_m[branch_name] = branch_metrics
        all_metrics.append(branch_metrics)

    baseline_m = compute_all_metrics(
        baseline_labels,
        baseline_preds["predictions"],
        baseline_preds["probabilities"],
        "Baseline CNN",
    )
    all_metrics.append(baseline_m)

    # 6d. Per-generator breakdown
    full_df = pd.read_parquet(PARQUET_FILE)

    test_df = full_df[full_df["split"] == "test"].reset_index(drop=True)
    model_types = test_df["model_type"].values

    fusion_gen_df = compute_per_generator_accuracy(
        true_labels,
        mvl_results["fusion"]["predictions"],
        model_types,
        "MVL Fusion",
        RESULTS_DIR,
    )

    compute_per_generator_accuracy(
        baseline_labels,
        baseline_preds["predictions"],
        model_types,
        "Baseline CNN",
        RESULTS_DIR,
    )

    # 7. Plots
    plot_confusion_matrix(
        true_labels,
        mvl_results["fusion"]["predictions"],
        "MVL Fusion",
        RESULTS_DIR,
    )

    for branch_name, branch_data in mvl_results["branches"].items():
        plot_confusion_matrix(
            true_labels,
            branch_data["predictions"],
            f"Branch {branch_name.upper()}",
            RESULTS_DIR,
        )

    plot_confusion_matrix(
        baseline_labels,
        baseline_preds["predictions"],
        "Baseline CNN",
        RESULTS_DIR,
    )

    curve_data: dict = {
        "MVL Fusion": {
            "true_labels": true_labels,
            "probabilities": mvl_results["fusion"]["probabilities"],
            "roc_auc": fusion_m["roc_auc"],
            "avg_precision": fusion_m["avg_precision"],
        },
        "Baseline CNN": {
            "true_labels": baseline_labels,
            "probabilities": baseline_preds["probabilities"],
            "roc_auc": baseline_m["roc_auc"],
            "avg_precision": baseline_m["avg_precision"],
        },
    }

    for branch_name in branch_m:
        curve_data[f"Branch {branch_name.upper()}"] = {
            "true_labels": true_labels,
            "probabilities": mvl_results["branches"][branch_name]["probabilities"],
            "roc_auc": branch_m[branch_name]["roc_auc"],
            "avg_precision": branch_m[branch_name]["avg_precision"],
        }

    plot_roc_curves(curve_data, RESULTS_DIR)
    plot_pr_curves(curve_data, RESULTS_DIR)
    plot_metrics_bar(all_metrics, RESULTS_DIR)

    if branch_m:
        plot_branch_heatmap(branch_m, RESULTS_DIR)
        plot_branch_vs_fusion(
            fusion_m,
            branch_m,
            RESULTS_DIR,
        )

    plot_per_generator_accuracy(
        fusion_gen_df,
        "MVL Fusion",
        RESULTS_DIR,
    )

    # 8. Summary
    _print_summary(
        all_metrics,
        fusion_m,
        baseline_m,
        branch_m,
    )


if __name__ == "__main__":
    main()
