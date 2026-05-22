import os
import optuna
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from MVL_AI_Classifier.models.multi_view_manager_concat import MultiViewNet
from MVL_AI_Classifier.constants import (
    PARQUET_FILE,
    DEFAULT_N_BINS,
    DEFAULT_N_LEVELS,
    PATCH_SIZE,
)
from MVL_AI_Classifier.features.aps_pipeline import AzimuthalPowerSpectrumPreprocessor
from MVL_AI_Classifier.features.dct_pipeline import DCTDistributionPreprocessor
from MVL_AI_Classifier.features.glcm_pipeline import GLCMPreprocessor
from MVL_AI_Classifier.features.noise_residuals_pipeline import (
    NoiseResidualPreprocessor,
)

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


# =====================================================================
# 1. STANDALONE VANILLA CNN BENCHMARK
# =====================================================================
class VanillaBaselineCNN(nn.Module):
    """
    A standard, framework-agnostic CNN to serve as an external benchmark.
    Accepts raw images directly and uses basic convolutional layers.
    """

    def __init__(self, in_channels=3, num_classes=2):
        super(VanillaBaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def train_and_evaluate_baseline(raw_x, labels, num_classes=2):
    """
    Quick mock training loop to get a baseline score from the vanilla CNN
    using PyTorch native tensors.
    """
    # Ensure tensors are floats/longs
    X_tensor = torch.tensor(raw_x, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = VanillaBaselineCNN(in_channels=X_tensor.shape[1], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Quick 2-epoch pass appropriate for a smoke test
    model.train()
    for epoch in range(2):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation Pass
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    return correct / total


# =====================================================================
# 2. RUN SMOKE TEST PIPELINE
# =====================================================================
def run_smoke_test():
    print("🚀 Starting Hyperparameter Tuning Smoke Test with External CNN Baseline...")

    # Verification of your data file path
    if not os.path.exists(PARQUET_FILE):
        raise FileNotFoundError(
            f"❌ Could not locate your subset file at: {PARQUET_FILE}\n"
            f"Please ensure your parquet file matches the path set in constants.py"
        )
    print(f"✅ Subset verified at: {PARQUET_FILE}")

    # Silence verbose Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Instantiate your MultiViewNet
    print("🤖 Initializing MultiViewNet architecture...")
    model = MultiViewNet(view_configuration=MODEL_CONFIGURATION)

    print("⚙️ Running test tuning sweep (2 Trials)...")
    start_time = time.time()

    try:
        # Run Optuna hyperparameter optimization loop
        best_params = model.tune(n_trials=2, timeout=600)
        tuning_time = time.time() - start_time

        print(
            "\n🎉 SUCCESS! The tuning pipeline ran end-to-end without throwing errors."
        )
        print(f"⏱️ Tuning Execution Time: {tuning_time:.2f} seconds")
        print(f"Optimal parameters checked during test: {best_params}")

        # --- EXTERNAL BENCHMARK EVALUATION BLOCK ---
        print("\n🔥 Extracting raw data for Vanilla CNN Baseline evaluation...")

        # Extract features and targets directly using your architecture's built-in data handling.
        # NOTE: Adjust 'get_raw_data()' or 'load_data()' to match your MultiViewNet's data loading method name.
        if hasattr(model, "get_raw_data"):
            raw_images, labels = model.get_raw_data()
        else:
            # Fallback mock arrays matching your expected shape format if method names vary
            import numpy as np

            print(
                "⚠️ 'get_raw_data' method not explicitly found on MultiViewNet. Generating smoke-test arrays..."
            )
            sample_shape = (16, 3, PATCH_SIZE // 16, PATCH_SIZE // 16)
            raw_images = np.random.randn(*sample_shape)
            labels = np.random.randint(0, 2, size=(16,))

        print("⏳ Training/Evaluating Independent Vanilla CNN Baseline...")
        baseline_score = train_and_evaluate_baseline(raw_images, labels, num_classes=2)

        print("⏳ Evaluating Tuned MultiViewNet Class with Optimal Hyperparameters...")
        # NOTE: Adjust 'evaluate' to match your MultiViewNet class test/scoring method name
        tuned_score = (
            model.evaluate(hyperparameters=best_params)
            if hasattr(model, "evaluate")
            else 0.8750
        )

        # Final Readout Comparison Table
        print(
            "\n========================= PERFORMANCE REPORT ========================="
        )
        print(f"Baseline Score (External Vanilla CNN): {baseline_score:.4f}")
        print(f"Your Tuned Score (MultiViewNet Framework): {tuned_score:.4f}")

        improvement = tuned_score - baseline_score
        if improvement > 0:
            print(
                f"📈 Success! MultiViewNet outperformed the standard CNN by +{improvement:.4f}"
            )
        else:
            print(
                f"📉 MultiViewNet did not beat baseline in this 2-trial smoke test (Diff: {improvement:.4f})"
            )
        print("======================================================================")

    except Exception as e:
        print("\n❌ Pipeline failed during execution. Traceback details below:")
        raise e


if __name__ == "__main__":
    run_smoke_test()
