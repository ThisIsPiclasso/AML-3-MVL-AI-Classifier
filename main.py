# This is a sample Python script.
from MVL_AI_Classifier.features.aps_pipeline import AzimuthalPowerSpectrumPreprocessor
from MVL_AI_Classifier.constants import DEFAULT_N_BINS

MODEL_CONFIGURATION = {
    "aps": {
        "preprocessor": AzimuthalPowerSpectrumPreprocessor(),
        "model_type": "cnn",  # or mlp
        "input_shape": (DEFAULT_N_BINS,),
    }
}


def hello_world():
    return "Hello, World!"


if __name__ == "__main__":
    hello_world()
