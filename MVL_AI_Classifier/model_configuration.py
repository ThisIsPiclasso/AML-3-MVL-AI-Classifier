from MVL_AI_Classifier.features.aps_pipeline import AzimuthalPowerSpectrumPreprocessor
from MVL_AI_Classifier.features.dct_pipeline import DCTDistributionPreprocessor
from MVL_AI_Classifier.features.glcm_pipeline import GLCMPreprocessor
from MVL_AI_Classifier.features.noise_residuals_pipeline import (
    NoiseResidualPreprocessor,
)
from MVL_AI_Classifier.constants import DEFAULT_N_BINS, DEFAULT_N_LEVELS, PATCH_SIZE

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
