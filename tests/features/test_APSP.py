import numpy as np

from features.azimuthal_power_spectrum_pipeline import (
    AzimuthalPowerSpectrumPreprocessor,
)
from features.constants import PATCH_SIZE


class TestAzimuthalPowerSpectrumPreprocessor:
    """
    tests for AzimuthalPowerSpectrumPreprocessor.
    """

    def __init__(self):
        self.n_bins = 64
        self.preprocessor = AzimuthalPowerSpectrumPreprocessor(
            n_bins=self.n_bins
        )



    def _random_patch(self) -> np.ndarray:
        return np.random.randint(
            0,
            256,
            size=(PATCH_SIZE, PATCH_SIZE, 3),
            dtype=np.uint8,
        )


    def test_initialization(self):
        assert self.preprocessor.n_bins == self.n_bins

        assert self.preprocessor._window_2d.shape == (
            PATCH_SIZE,
            PATCH_SIZE,
        )

        print("test_initialization passed")

    def test_invalid_n_bins(self):
        try:
            AzimuthalPowerSpectrumPreprocessor(n_bins=0)
            raise AssertionError("Expected ValueError for n_bins=0")
        except ValueError:
            pass

        try:
            AzimuthalPowerSpectrumPreprocessor(n_bins=-1)
            raise AssertionError("Expected ValueError for negative n_bins")
        except ValueError:
            pass

        try:
            AzimuthalPowerSpectrumPreprocessor(n_bins=32.5)
            raise AssertionError("Expected ValueError for float n_bins")
        except ValueError:
            pass

        print("test_invalid_n_bins passed")

    def test_bin_index_shapes(self):
        bin_idx, bin_counts = self.preprocessor._build_bin_index()

        assert bin_idx.shape == (PATCH_SIZE * PATCH_SIZE,)
        assert bin_counts.shape == (self.n_bins,)

        print("test_bin_index_shapes passed")

    def test_bin_counts_positive(self):
        _, bin_counts = self.preprocessor._build_bin_index()

        assert np.all(bin_counts > 0)

        print("test_bin_counts_positive passed")

    def test_output_shape(self):
        patch = self._random_patch()

        features = self.preprocessor(patch)

        assert features.shape == (self.n_bins,)

        print("test_output_shape passed")

    def test_output_dtype(self):
        patch = self._random_patch()

        features = self.preprocessor(patch)

        assert features.dtype == np.float32

        print("test_output_dtype passed")

    def test_output_is_finite(self):
        patch = self._random_patch()

        features = self.preprocessor(patch)

        assert np.all(np.isfinite(features))

        print("test_output_is_finite passed")

    def test_deterministic_output(self):
        patch = self._random_patch()

        features_1 = self.preprocessor(patch)
        features_2 = self.preprocessor(patch)

        np.testing.assert_allclose(features_1, features_2)

        print("test_deterministic_output passed")

    def test_black_image(self):
        black_patch = np.zeros(
            (PATCH_SIZE, PATCH_SIZE, 3),
            dtype=np.uint8,
        )

        features = self.preprocessor(black_patch)

        assert np.all(np.isfinite(features))

        print("test_black_image passed")

    def test_white_image(self):
        white_patch = np.ones(
            (PATCH_SIZE, PATCH_SIZE, 3),
            dtype=np.uint8,
        ) * 255

        features = self.preprocessor(white_patch)

        assert np.all(np.isfinite(features))

        print("test_white_image passed")

    def test_random_noise_image(self):
        noise_patch = self._random_patch()

        features = self.preprocessor(noise_patch)

        assert np.std(features) > 0.0

        print("test_random_noise_image passed")


    def run_all(self):
        """
        Execute all tests manually.
        """

        self.test_initialization()
        self.test_invalid_n_bins()
        self.test_bin_index_shapes()
        self.test_bin_counts_positive()
        self.test_output_shape()
        self.test_output_dtype()
        self.test_output_is_finite()
        self.test_deterministic_output()
        self.test_black_image()
        self.test_white_image()
        self.test_random_noise_image()

        print("\nAll tests passed")


if __name__ == "__main__":
    tester = TestAzimuthalPowerSpectrumPreprocessor()
    tester.run_all()