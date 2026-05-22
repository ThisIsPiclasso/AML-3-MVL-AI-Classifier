"""
Full evaluation pipeline for the Multi-View Learning AI Image Classifier.

Produces:
    - Per-branch and fusion classification metrics
    - Confusion matrices
    - ROC and Precision-Recall curves
    - Metric visualisations
    - Comparison against VanillaBaselineCNN
"""

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

# --- Project imports -----------------------------------------------------------
from MVL_AI_Classifier.models.multi_view_manager_concat import MultiViewNet
from MVL_AI_Classifier.loss import MultiViewLoss
from MVL_AI_Classifier.constants import (
    PARQUET_FILE,
    DEFAULT_N_BINS,
    DEFAULT_N_LEVELS,
    PATCH_SIZE,
)
from MVL_AI_Classifier.features.aps_pipeline import AzimuthalPowerSpectrumPreprocessor
from MVL_AI_Classifier.features.dct_pipeline import DCTDistributionPreprocessor
from MVL_AI_Classifier.features.glcm_pipeline import GLCMPreprocessor
from MVL_AI_Classifier.features.noise_residuals_pipeline import NoiseResidualPreprocessor

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

RESULTS_DIR = Path("evaluation_results")
CLASS_NAMES = ["Real", "AI-Generated"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

BRANCH_NAMES = list(MODEL_CONFIGURATION.keys())


# ==============================================================================
# 1. BASELINE MODEL
# ==============================================================================

class VanillaBaselineCNN(nn.Module):
    """Standalone CNN operating on raw pixel patches.

    Serves as the performance floor against which the multi-view
    framework is evaluated.

    Args:
        in_channels: Number of input image channels.
        num_classes: Number of output classes.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        return self.classifier(torch.flatten(self.features(x), 1))


# ==============================================================================
# 2. EVALUATION METRICS
# ==============================================================================

def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """Compute a complete set of classification metrics.

    Args:
        labels: Ground-truth integer labels, shape (N,).
        predictions: Predicted integer labels, shape (N,).
        probabilities: Predicted probabilities for the positive class,
            shape (N,). Used for AUC and PR calculations.
        model_name: Display name for logging.

    Returns:
        Dictionary containing all computed metric values.
    """
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "avg_precision": average_precision_score(labels, probabilities),
        # False positive rate at threshold optimised for F1.
        "fpr": 1.0 - precision_score(
            labels, predictions, pos_label=0, zero_division=0
        ),
    }

    print(f"\n{'─' * 55}")
    print(f"  {model_name}")
    print(f"{'─' * 55}")
    for key, value in metrics.items():
        if key != "model":
            print(f"  {key:<20} {value:.4f}")

    return metrics


# ==============================================================================
# 3. INFERENCE HELPERS
# ==============================================================================

def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_multiview: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run inference on a DataLoader and collect predictions.

    Args:
        model: Trained PyTorch model.
        loader: DataLoader yielding (inputs, labels) batches.
        device: Torch device to run on.
        is_multiview: If True, expects the model to return the
            ``outputs`` dictionary used by ``MultiViewLoss``, including
            ``"fusion"`` and ``"branches"`` keys.

    Returns:
        Tuple of:
            - ``labels``: Ground-truth numpy array, shape (N,).
            - ``fusion_probs``: Fusion-head positive-class probabilities,
              shape (N,).
            - ``branch_probs``: Dictionary mapping branch name to
              positive-class probability array, shape (N,).
              Empty dict when ``is_multiview=False``.
    """
    model.eval()
    all_labels = []
    all_fusion_probs = []
    all_branch_probs = {name: [] for name in BRANCH_NAMES}

    with torch.no_grad():
        for batch in loader:
            inputs, batch_labels = batch
            if isinstance(inputs, (list, tuple)):
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = inputs.to(device)

            batch_labels = batch_labels.to(device)

            if is_multiview:
                outputs = model(inputs)
                fusion_logits = outputs["fusion"]
                fusion_prob = torch.softmax(fusion_logits, dim=1)[:, 1]
                all_fusion_probs.append(fusion_prob.cpu().numpy())

                if "branches" in outputs:
                    for name in BRANCH_NAMES:
                        if name in outputs["branches"]:
                            branch_logits = outputs["branches"][name]
                            branch_prob = torch.softmax(
                                branch_logits, dim=1
                            )[:, 1]
                            all_branch_probs[name].append(
                                branch_prob.cpu().numpy()
                            )
            else:
                logits = model(inputs)
                prob = torch.softmax(logits, dim=1)[:, 1]
                all_fusion_probs.append(prob.cpu().numpy())

            all_labels.append(batch_labels.cpu().numpy())

    labels = np.concatenate(all_labels)
    fusion_probs = np.concatenate(all_fusion_probs)
    branch_probs = {
        name: np.concatenate(probs)
        for name, probs in all_branch_probs.items()
        if len(probs) > 0
    }

    return labels, fusion_probs, branch_probs


# ==============================================================================
# 4. VISUALISATION FUNCTIONS
# ==============================================================================

def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
    save_dir: Path,
) -> None:
    """Plot and save a normalised confusion matrix.

    Args:
        labels: Ground-truth integer labels.
        predictions: Predicted integer labels.
        model_name: Used as the figure title and filename.
        save_dir: Directory to save the PNG file.
    """
    cm = confusion_matrix(labels, predictions, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=True,
        values_format=".2f",
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    plt.tight_layout()
    filename = save_dir / f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_roc_curves(
    results: dict,
    save_dir: Path,
) -> None:
    """Plot ROC curves for all models on one figure.

    Args:
        results: Dictionary mapping model name to dict with keys
            ``"labels"``, ``"probs"`` (positive-class probabilities),
            and ``"auc"``.
        save_dir: Directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, data in results.items():
        fpr, tpr, _ = roc_curve(data["labels"], data["probs"])
        auc = data["auc"]
        ax.plot(fpr, tpr, lw=2, label=f"{model_name}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = save_dir / "roc_curves_all_models.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_precision_recall_curves(
    results: dict,
    save_dir: Path,
) -> None:
    """Plot Precision-Recall curves for all models on one figure.

    Args:
        results: Dictionary mapping model name to dict with keys
            ``"labels"``, ``"probs"``, and ``"ap"`` (average precision).
        save_dir: Directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, data in results.items():
        precision, recall, _ = precision_recall_curve(
            data["labels"], data["probs"]
        )
        ap = data["ap"]
        ax.plot(
            recall, precision, lw=2,
            label=f"{model_name}  (AP = {ap:.3f})"
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = save_dir / "pr_curves_all_models.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_metrics_bar(
    all_metrics: list[dict],
    save_dir: Path,
) -> None:
    """Plot grouped bar chart comparing all models across metrics.

    Args:
        all_metrics: List of metric dictionaries, one per model.
            Each dict must contain ``"model"`` plus numeric metric keys.
        save_dir: Directory to save the figure.
    """
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = [m["model"] for m in all_metrics]
    num_models = len(model_names)
    num_metrics = len(metric_keys)

    bar_width = 0.7 / num_models
    x = np.arange(num_metrics)

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, metrics in enumerate(all_metrics):
        values = [metrics[k] for k in metric_keys]
        offsets = x + (idx - num_models / 2 + 0.5) * bar_width
        bars = ax.bar(offsets, values, bar_width, label=metrics["model"])
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [k.replace("_", " ").title() for k in metric_keys], fontsize=11
    )
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Classification Metrics", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = save_dir / "metrics_bar_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_branch_heatmap(
    branch_metrics: dict,
    save_dir: Path,
) -> None:
    """Plot heatmap of metric values across individual branches.

    Args:
        branch_metrics: Dictionary mapping branch name to its
            metric dictionary.
        save_dir: Directory to save the figure.
    """
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    branch_names = list(branch_metrics.keys())

    data = np.array(
        [
            [branch_metrics[b][k] for k in metric_keys]
            for b in branch_names
        ]
    )

    fig, ax = plt.subplots(
        figsize=(len(metric_keys) * 1.4, len(branch_names) * 1.0 + 1.5)
    )
    im = ax.imshow(data, vmin=0.0, vmax=1.0, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(
        [k.replace("_", " ").title() for k in metric_keys], fontsize=11
    )
    ax.set_yticks(range(len(branch_names)))
    ax.set_yticklabels(
        [b.upper() for b in branch_names], fontsize=11
    )
    ax.set_title("Branch Performance Heatmap", fontsize=13, pad=12)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    for row in range(len(branch_names)):
        for col in range(len(metric_keys)):
            ax.text(
                col, row,
                f"{data[row, col]:.3f}",
                ha="center", va="center",
                fontsize=9,
                color="black" if data[row, col] < 0.7 else "white",
            )

    plt.tight_layout()
    path = save_dir / "branch_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_performance_overview(
    all_metrics: list[dict],
    save_dir: Path,
) -> None:
    """Plot radar (spider) chart comparing models across key metrics.

    Args:
        all_metrics: List of metric dicts. Each must contain
            ``"model"`` and numeric metric values.
        save_dir: Directory to save the figure.
    """
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    num_metrics = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(
        figsize=(7, 7), subplot_kw=dict(polar=True)
    )

    for metrics in all_metrics:
        values = [metrics[k] for k in metric_keys]
        values += values[:1]  # close the polygon
        ax.plot(angles, values, lw=2, label=metrics["model"])
        ax.fill(angles, values, alpha=0.07)

    ax.set_thetagrids(
        np.degrees(angles[:-1]),
        [k.replace("_", " ").title() for k in metric_keys],
        fontsize=10,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Radar — Model Overview", fontsize=13, y=1.08)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.1),
        fontsize=9,
    )
    plt.tight_layout()

    path = save_dir / "radar_overview.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ==============================================================================
# 5. BASELINE TRAINING HELPER
# ==============================================================================

def train_baseline(
    raw_images: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 2,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> tuple[VanillaBaselineCNN, np.ndarray, np.ndarray]:
    """Train VanillaBaselineCNN and return model and predictions.

    Args:
        raw_images: Float32 array of shape (N, C, H, W).
        labels: Integer label array of shape (N,).
        num_classes: Number of output classes.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.

    Returns:
        Tuple of:
            - Trained ``VanillaBaselineCNN``.
            - Predicted integer labels on the full dataset.
            - Predicted positive-class probabilities on the full dataset.
    """
    X_tensor = torch.tensor(raw_images, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VanillaBaselineCNN(
        in_channels=X_tensor.shape[1], num_classes=num_classes
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(
            f"  Baseline epoch {epoch + 1}/{epochs}  "
            f"loss={epoch_loss / len(loader):.4f}"
        )

    # Collect predictions.
    model.eval()
    all_preds, all_probs = [], []
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_x, _ in eval_loader:
            logits = model(batch_x.to(DEVICE))
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return (
        model,
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


# ==============================================================================
# 6. PLACEHOLDER: MULTI-VIEW DATA LOADER
# ==============================================================================

def load_evaluation_data(
    model: "MultiViewNet",          # type: ignore[name-defined]
) -> tuple[DataLoader, np.ndarray]:
    """Load the test split from the trained MultiViewNet.

    This function is a PLACEHOLDER. Replace the body with your actual
    data loading logic once the MultiViewNet data-access API is stable.

    Expected contract:
        - Returns a DataLoader that yields ``(views, labels)`` batches
          where ``views`` is whatever format ``MultiViewNet.forward``
          expects.
        - Also returns the corresponding raw images as a numpy array
          of shape ``(N, 3, H, W)`` for the baseline CNN.

    Args:
        model: Trained MultiViewNet instance.

    Returns:
        Tuple of:
            - ``DataLoader`` for the test split.
            - Raw image numpy array of shape ``(N, 3, H, W)``,
              float32, pixel values in [0, 1].
    """
    # ------------------------------------------------------------------ #
    # PLACEHOLDER — replace with real test-split loading.                 #
    # ------------------------------------------------------------------ #
    print("  ⚠️  load_evaluation_data: using synthetic placeholder data.")

    num_samples = 200
    rng = np.random.default_rng(0)

    # Synthetic raw images for the baseline CNN.
    raw_images = rng.random((num_samples, 3, PATCH_SIZE, PATCH_SIZE)).astype(
        np.float32
    )
    labels = rng.integers(0, 2, size=(num_samples,)).astype(np.int64)

    # Synthetic view tensors matching MODEL_CONFIGURATION input shapes.
    view_tensors = {
        name: torch.tensor(
            rng.random((num_samples, *cfg["input_shape"])).astype(np.float32)
        )
        for name, cfg in MODEL_CONFIGURATION.items()
    }
    label_tensor = torch.tensor(labels)

    # Build a DataLoader that yields (view_dict, label) batches.
    # NOTE: TensorDataset does not support dict inputs natively;
    # use a list of tensors and reconstruct the dict in a wrapper.
    view_list = [view_tensors[n] for n in BRANCH_NAMES]
    tensor_dataset = TensorDataset(*view_list, label_tensor)

    def collate_views(batch):
        """Reconstruct the view dict from a flat TensorDataset batch."""
        stacked = [torch.stack([b[i] for b in batch]) for i in range(len(batch[0]))]
        views = {name: stacked[idx] for idx, name in enumerate(BRANCH_NAMES)}
        labels_batch = stacked[-1]
        return views, labels_batch

    loader = DataLoader(
        tensor_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_views,
    )

    return loader, raw_images, labels


# ==============================================================================
# 7. MAIN EVALUATION PIPELINE
# ==============================================================================

def evaluate() -> None:
    """Run the full evaluation pipeline.

    Steps:
        1. Load the trained MultiViewNet (or initialise a fresh one as
           placeholder).
        2. Run inference and collect per-branch and fusion predictions.
        3. Train and evaluate the VanillaBaselineCNN on the same data.
        4. Compute all metrics.
        5. Generate and save all visualisations.
        6. Print a final comparison table.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}")
    print("  Multi-View Classifier — Full Evaluation Pipeline")
    print(f"{'=' * 60}")
    print(f"  Device : {DEVICE}")
    print(f"  Output : {RESULTS_DIR.resolve()}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Step 1: Initialise / load MultiViewNet
    # ------------------------------------------------------------------
    print("► Loading MultiViewNet...")

    model = MultiViewNet(view_configuration=MODEL_CONFIGURATION).to(DEVICE)

    # PLACEHOLDER: replace with your checkpoint loading logic.
    # Example:
    #   checkpoint = torch.load("checkpoints/best_model.pt", map_location=DEVICE)
    #   model.load_state_dict(checkpoint["model_state_dict"])
    print("  ⚠️  No checkpoint loaded — using randomly initialised weights.")

    # ------------------------------------------------------------------
    # Step 2: Load data
    # ------------------------------------------------------------------
    print("\n► Loading evaluation data...")
    test_loader, raw_images, true_labels = load_evaluation_data(model)

    # ------------------------------------------------------------------
    # Step 3: MultiViewNet inference
    # ------------------------------------------------------------------
    print("\n► Running MultiViewNet inference...")
    t0 = time.time()
    labels, fusion_probs, branch_probs = run_inference(
        model, test_loader, DEVICE, is_multiview=True
    )
    mvl_inference_time = time.time() - t0
    print(f"  Inference time: {mvl_inference_time:.2f}s")

    fusion_preds = (fusion_probs >= 0.5).astype(int)

    # Per-branch predictions.
    branch_preds = {
        name: (probs >= 0.5).astype(int)
        for name, probs in branch_probs.items()
    }

    # ------------------------------------------------------------------
    # Step 4: Baseline training and inference
    # ------------------------------------------------------------------
    print("\n► Training VanillaBaselineCNN baseline...")
    t0 = time.time()
    _, baseline_preds, baseline_probs = train_baseline(
        raw_images=raw_images,
        labels=true_labels,
        epochs=10,
        batch_size=32,
    )
    baseline_time = time.time() - t0
    print(f"  Baseline training + inference time: {baseline_time:.2f}s")

    # ------------------------------------------------------------------
    # Step 5: Compute all metrics
    # ------------------------------------------------------------------
    print("\n► Computing metrics...")

    all_metrics = []
    roc_data = {}
    pr_data = {}

    # Fusion head metrics.
    fusion_metrics = compute_metrics(
        labels, fusion_preds, fusion_probs, model_name="MVL Fusion"
    )
    all_metrics.append(fusion_metrics)
    roc_data["MVL Fusion"] = {
        "labels": labels,
        "probs": fusion_probs,
        "auc": fusion_metrics["roc_auc"],
    }
    pr_data["MVL Fusion"] = {
        "labels": labels,
        "probs": fusion_probs,
        "ap": fusion_metrics["avg_precision"],
    }

    # Per-branch metrics.
    branch_metrics = {}
    for name in BRANCH_NAMES:
        if name not in branch_probs:
            print(f"  ⚠️  Branch '{name}' had no predictions — skipping.")
            continue
        bm = compute_metrics(
            labels,
            branch_preds[name],
            branch_probs[name],
            model_name=f"Branch: {name.upper()}",
        )
        branch_metrics[name] = bm
        all_metrics.append(bm)
        roc_data[f"Branch {name.upper()}"] = {
            "labels": labels,
            "probs": branch_probs[name],
            "auc": bm["roc_auc"],
        }
        pr_data[f"Branch {name.upper()}"] = {
            "labels": labels,
            "probs": branch_probs[name],
            "ap": bm["avg_precision"],
        }

    # Baseline metrics.
    baseline_metrics = compute_metrics(
        true_labels,
        baseline_preds,
        baseline_probs,
        model_name="Vanilla CNN Baseline",
    )
    all_metrics.append(baseline_metrics)
    roc_data["Vanilla CNN"] = {
        "labels": true_labels,
        "probs": baseline_probs,
        "auc": baseline_metrics["roc_auc"],
    }
    pr_data["Vanilla CNN"] = {
        "labels": true_labels,
        "probs": baseline_probs,
        "ap": baseline_metrics["avg_precision"],
    }

    # ------------------------------------------------------------------
    # Step 6: Visualisations
    # ------------------------------------------------------------------
    print("\n► Generating visualisations...")

    # Confusion matrices.
    plot_confusion_matrix(
        labels, fusion_preds, "MVL Fusion", RESULTS_DIR
    )
    plot_confusion_matrix(
        true_labels, baseline_preds, "Vanilla CNN Baseline", RESULTS_DIR
    )
    for name in branch_metrics:
        plot_confusion_matrix(
            labels,
            branch_preds[name],
            f"Branch {name.upper()}",
            RESULTS_DIR,
        )

    # Curve plots.
    plot_roc_curves(roc_data, RESULTS_DIR)
    plot_precision_recall_curves(pr_data, RESULTS_DIR)

    # Metric bar chart.
    plot_metrics_bar(all_metrics, RESULTS_DIR)

    # Branch heatmap (only branches, not fusion or baseline).
    if branch_metrics:
        plot_branch_heatmap(branch_metrics, RESULTS_DIR)

    # Radar overview.
    radar_models = [m for m in all_metrics if m["model"] in
                    ("MVL Fusion", "Vanilla CNN Baseline")]
    if radar_models:
        plot_performance_overview(radar_models, RESULTS_DIR)

    # ------------------------------------------------------------------
    # Step 7: Final comparison table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  FINAL PERFORMANCE SUMMARY")
    print(f"{'=' * 60}")
    header = f"{'Model':<30} {'Acc':>7} {'F1':>7} {'AUC':>7} {'AP':>7}"
    print(header)
    print("─" * len(header))
    for m in all_metrics:
        print(
            f"{m['model']:<30} "
            f"{m['accuracy']:>7.4f} "
            f"{m['f1']:>7.4f} "
            f"{m['roc_auc']:>7.4f} "
            f"{m['avg_precision']:>7.4f}"
        )
    print(f"{'=' * 60}")

    # Improvement of fusion over baseline.
    fusion_acc = fusion_metrics["accuracy"]
    baseline_acc = baseline_metrics["accuracy"]
    delta = fusion_acc - baseline_acc
    symbol = "📈" if delta > 0 else "📉"
    print(
        f"\n{symbol}  MVL Fusion vs Baseline accuracy: "
        f"{delta:+.4f}  "
        f"({'outperforms' if delta > 0 else 'underperforms'} baseline)"
    )

    print(f"\n✅  All results saved to: {RESULTS_DIR.resolve()}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    evaluate()
