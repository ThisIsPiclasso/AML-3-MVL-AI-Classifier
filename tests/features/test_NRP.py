import numpy as np

from features.noise_residual_pipeline import NoiseResidualPreprocessor
from MVL_AI_Classifier.constants import PATCH_SIZE


class TestNoiseResidualPreprocessor:
    """
    Tests for NoiseResidualPreprocessor.
    """

    def __init__(self):
        self.window_size = 16

        self.preprocessor = NoiseResidualPreprocessor(
            sigma=1.0,
            window_size=self.window_size,
        )

        self.windows_per_dim = PATCH_SIZE // self.window_size

        self.expected_shape = (
            3,
            self.windows_per_dim,
            self.windows_per_dim,
        )


    def _random_patch(self) -> np.ndarray:
        return np.random.randint(
            0,
            256,
            size=(PATCH_SIZE, PATCH_SIZE, 3),
            dtype=np.uint8,
        )



    def test_initialization(self):
        assert self.preprocessor.window_size == self.window_size
        assert self.preprocessor.windows_per_dim == self.windows_per_dim

        print("test_initialization passed")

    def test_invalid_window_size(self):
        try:
            NoiseResidualPreprocessor(window_size=15)

            raise AssertionError(
                "Expected ValueError for invalid window size"
            )

        except ValueError:
            pass

        print("test_invalid_window_size passed")

    def test_compute_residuals_shape(self):
        image = self._random_patch().astype(np.float32)

        residuals = self.preprocessor._compute_residuals(image)

        assert residuals.shape == image.shape

        print("test_compute_residuals_shape passed")

    def test_compute_residuals_dtype(self):
        image = self._random_patch().astype(np.float32)

        residuals = self.preprocessor._compute_residuals(image)

        assert residuals.dtype == np.float32

        print("test_compute_residuals_dtype passed")

    def test_reshape_to_windows_shape(self):
        channel = np.random.rand(
            PATCH_SIZE,
            PATCH_SIZE,
        ).astype(np.float32)

        windows = self.preprocessor._reshape_to_windows(channel)

        expected_shape = (
            self.windows_per_dim,
            self.windows_per_dim,
            self.window_size * self.window_size,
        )

        assert windows.shape == expected_shape

        print("test_reshape_to_windows_shape passed")

    def test_pearson_output_shape(self):
        shape = (
            self.windows_per_dim,
            self.windows_per_dim,
            self.window_size * self.window_size,
        )

        a = np.random.randn(*shape).astype(np.float32)
        b = np.random.randn(*shape).astype(np.float32)

        corr = self.preprocessor._pearson_from_windows(a, b)

        expected_shape = (
            self.windows_per_dim,
            self.windows_per_dim,
        )

        assert corr.shape == expected_shape

        print("test_pearson_output_shape passed")

    def test_pearson_output_range(self):
        shape = (
            self.windows_per_dim,
            self.windows_per_dim,
            self.window_size * self.window_size,
        )

        a = np.random.randn(*shape).astype(np.float32)
        b = np.random.randn(*shape).astype(np.float32)

        corr = self.preprocessor._pearson_from_windows(a, b)

        assert np.all(corr >= -1.0)
        assert np.all(corr <= 1.0)

        print("test_pearson_output_range passed")

    def test_output_shape(self):
        image = self._random_patch()

        features = self.preprocessor(image)

        assert features.shape == self.expected_shape

        print("test_output_shape passed")

    def test_output_dtype(self):
        image = self._random_patch()

        features = self.preprocessor(image)

        assert features.dtype == np.float32

        print("test_output_dtype passed")

    def test_output_is_finite(self):
        image = self._random_patch()

        features = self.preprocessor(image)

        assert np.all(np.isfinite(features))

        print("test_output_is_finite passed")

    def test_deterministic_output(self):
        image = self._random_patch()

        features_1 = self.preprocessor(image)
        features_2 = self.preprocessor(image)

        np.testing.assert_allclose(
            features_1,
            features_2,
        )

        print("test_deterministic_output passed")

    def test_black_image(self):
        image = np.zeros(
            (PATCH_SIZE, PATCH_SIZE, 3),
            dtype=np.uint8,
        )

        features = self.preprocessor(image)

        assert features.shape == self.expected_shape
        assert np.all(np.isfinite(features))

        print("test_black_image passed")

    def test_white_image(self):
        image = np.ones(
            (PATCH_SIZE, PATCH_SIZE, 3),
            dtype=np.uint8,
        ) * 255

        features = self.preprocessor(image)

        assert features.shape == self.expected_shape
        assert np.all(np.isfinite(features))

        print("test_white_image passed")

    def test_random_noise_image(self):
        image = self._random_patch()

        features = self.preprocessor(image)

        assert np.std(features) > 0.0

        print("test_random_noise_image passed")

    def test_degenerate_windows(self):
        """
        Constant windows should produce zero correlation.
        """

        shape = (
            self.windows_per_dim,
            self.windows_per_dim,
            self.window_size * self.window_size,
        )

        a = np.ones(shape, dtype=np.float32)
        b = np.ones(shape, dtype=np.float32)

        corr = self.preprocessor._pearson_from_windows(a, b)

        assert np.allclose(corr, 0.0)

        print("test_degenerate_windows passed")


    def run_all(self):
        self.test_initialization()
        self.test_invalid_window_size()
        self.test_compute_residuals_shape()
        self.test_compute_residuals_dtype()
        self.test_reshape_to_windows_shape()
        self.test_pearson_output_shape()
        self.test_pearson_output_range()
        self.test_output_shape()
        self.test_output_dtype()
        self.test_output_is_finite()
        self.test_deterministic_output()
        self.test_black_image()
        self.test_white_image()
        self.test_random_noise_image()
        self.test_degenerate_windows()

        print("\nAll tests passed successfully")


if __name__ == "__main__":
    tester = TestNoiseResidualPreprocessor()
    tester.run_all()