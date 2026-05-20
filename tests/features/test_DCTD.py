import numpy as np

from features.dct_distribution_pipeline import DCTDistributionPreprocessor
from MVL_AI_Classifier.constants import (
    JPEG_BLOCK_SIZE,
    JPEG_BLOCKS_PER_DIM,
)


class TestDCTDistributionPreprocessor:
    """
    Tests for DCTDistributionPreprocessor.
    """

    def __init__(self):
        self.preprocessor = DCTDistributionPreprocessor()

        self.expected_shape = (
            3,  # RGB channels
            4,  # statistics
            JPEG_BLOCK_SIZE,
            JPEG_BLOCK_SIZE,
        )

        self.image_size = JPEG_BLOCK_SIZE * JPEG_BLOCKS_PER_DIM



    def _random_patch(self) -> np.ndarray:
        return np.random.randint(
            0,
            256,
            size=(self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )



    def test_initialization(self):
        assert self.preprocessor.sparsity_threshold > 0
        assert isinstance(
            self.preprocessor.log_compress_dispersion,
            bool,
        )

        print("test_initialization passed")

    def test_invalid_sparsity_threshold(self):
        try:
            DCTDistributionPreprocessor(sparsity_threshold=0)
            raise AssertionError(
                "Expected ValueError for sparsity_threshold=0"
            )
        except ValueError:
            pass

        try:
            DCTDistributionPreprocessor(sparsity_threshold=-1)
            raise AssertionError(
                "Expected ValueError for negative sparsity_threshold"
            )
        except ValueError:
            pass

        print("test_invalid_sparsity_threshold passed")

    def test_extract_blocks_shape(self):
        image = self._random_patch()

        blocks = self.preprocessor._extract_blocks(image)

        expected_shape = (
            3,
            JPEG_BLOCKS_PER_DIM * JPEG_BLOCKS_PER_DIM,
            JPEG_BLOCK_SIZE,
            JPEG_BLOCK_SIZE,
        )

        assert blocks.shape == expected_shape

        print("test_extract_blocks_shape passed")

    def test_extract_blocks_dtype(self):
        image = self._random_patch()

        blocks = self.preprocessor._extract_blocks(image)

        assert blocks.dtype == np.float32

        print("test_extract_blocks_dtype passed")

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

        np.testing.assert_allclose(features_1, features_2)

        print("test_deterministic_output passed")

    def test_black_image(self):
        image = np.zeros(
            (self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )

        features = self.preprocessor(image)

        assert features.shape == self.expected_shape
        assert np.all(np.isfinite(features))

        print("test_black_image passed")

    def test_white_image(self):
        image = np.ones(
            (self.image_size, self.image_size, 3),
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

    def test_unbiased_excess_kurtosis_shape(self):
        dummy = np.random.randn(
            3,
            JPEG_BLOCKS_PER_DIM * JPEG_BLOCKS_PER_DIM,
            JPEG_BLOCK_SIZE,
            JPEG_BLOCK_SIZE,
        ).astype(np.float32)

        kurt = self.preprocessor._unbiased_excess_kurtosis(dummy)

        expected_shape = (
            3,
            JPEG_BLOCK_SIZE,
            JPEG_BLOCK_SIZE,
        )

        assert kurt.shape == expected_shape

        print("test_unbiased_excess_kurtosis_shape passed")

    def test_unbiased_excess_kurtosis_finite(self):
        dummy = np.random.randn(
            3,
            JPEG_BLOCKS_PER_DIM * JPEG_BLOCKS_PER_DIM,
            JPEG_BLOCK_SIZE,
            JPEG_BLOCK_SIZE,
        ).astype(np.float32)

        kurt = self.preprocessor._unbiased_excess_kurtosis(dummy)

        assert np.all(np.isfinite(kurt))

        print("test_unbiased_excess_kurtosis_finite passed")

    def test_log_compression_enabled(self):
        processor = DCTDistributionPreprocessor(
            log_compress_dispersion=True
        )

        image = self._random_patch()

        features = processor(image)

        assert np.all(np.isfinite(features))

        print("test_log_compression_enabled passed")

    def test_log_compression_disabled(self):
        processor = DCTDistributionPreprocessor(
            log_compress_dispersion=False
        )

        image = self._random_patch()

        features = processor(image)

        assert np.all(np.isfinite(features))

        print("test_log_compression_disabled passed")



    def run_all(self):
        """
        Execute all tests manually.
        """

        self.test_initialization()
        self.test_invalid_sparsity_threshold()
        self.test_extract_blocks_shape()
        self.test_extract_blocks_dtype()
        self.test_output_shape()
        self.test_output_dtype()
        self.test_output_is_finite()
        self.test_deterministic_output()
        self.test_black_image()
        self.test_white_image()
        self.test_random_noise_image()
        self.test_unbiased_excess_kurtosis_shape()
        self.test_unbiased_excess_kurtosis_finite()
        self.test_log_compression_enabled()
        self.test_log_compression_disabled()

        print("\nAll tests passed")


if __name__ == "__main__":
    tester = TestDCTDistributionPreprocessor()
    tester.run_all()