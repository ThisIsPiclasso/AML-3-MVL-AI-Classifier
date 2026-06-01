# Side length of the square input patch in pixels.
# Must be a power of 2 (efficient FFT) and divisible by JPEG_BLOCK_SIZE.
PATCH_SIZE = 256

# JPEG standard mandates 8x8 pixel blocks for DCT.
JPEG_BLOCK_SIZE = 8

# Number of non-overlapping JPEG blocks along each spatial dimension.
# 256 / 8 = 32 blocks per row and 32 blocks per column.
JPEG_BLOCKS_PER_DIM = PATCH_SIZE // JPEG_BLOCK_SIZE

# JPEG level shift: maps unsigned pixel values [0, 255] to
# signed values [-128, 127] for zero-centered DCT computation.
JPEG_RECENTER_VALUE = 128.0

# Small constant added to denominators and log arguments to prevent
# division-by-zero and log(0)
DEFAULT_EPSILON = 1e-8
# Default number of bins for histogram-based features (e.g., DCT coefficient histograms).
DEFAULT_N_BINS = 64

DEFAULT_N_LEVELS = 32

# Data frame constants
DEFAULT_AI_PATH = "./data/subset/ai"
DEFAULT_NATURE_PATH = "./data/subset/nature"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
NAME_MAP = {
    "adm": "ADM",
    "glide": "Glide",
    "midjourney": "Midjourney",
    "sdv4": "Stable Diffusion v1.4",
    "sdv5": "Stable Diffusion v1.5",
    "vqdm": "VQDM",
    "wukong": "Wukong",
}
SAMPLE_SIZE = 1000

# Default seed used for reproductibility
DEFAULT_SEED = 42

# Default training/validation/testing split
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.1

# Defaults for training
PARQUET_FILE = "./data/data_parquet/dataset.parquet"
NUM_WORKERS = 2
NUM_EPOCHS = 15

# hyperparameters for multi-view loss
DEFAULT_ALPHA = 0.1
DEFAULT_BETA = 0.20749566933161628
DEFAULT_TEMPERATURE = 3.5
DEFAULT_LEARNING_RATE = 0.00023462467904408212
BATCH_SIZE = 64

# Defaults for Optuna hyperparameter tuning
N_TRIALS = 100
MAX_TRAINING_TIME = 86400  # in seconds (24 hour)
MAX_TUNE_EPOCHS = 4
CACHE_DIR = "/workspace/AML-3-MVL-AI-Classifier/data/data_cache"
TRAIN_CACHE = "/workspace/AML-3-MVL-AI-Classifier/data/data_cache/train_features.h5"
VAL_CACHE = "/workspace/AML-3-MVL-AI-Classifier/data/data_cache/val_features.h5"
TEST_CACHE = "/workspace/AML-3-MVL-AI-Classifier/data/data_cache/test_features.h5"

# defauls for baseline CNN
BASELINE_EPOCHS = 10
BASELINE_BATCH_SIZE = 32
BASELINE_LR = 1e-3
