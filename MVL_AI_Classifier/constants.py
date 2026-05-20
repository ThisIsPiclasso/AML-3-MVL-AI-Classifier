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

# Default seed used for reproductibility
DEFAULT_SEED = 42

# Default training/validation/testing split
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.1
