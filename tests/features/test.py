import numpy as np

from features.glcm_pipeline import GLCMPreprocessor


class TestGLCMPreprocessor:
    """
    Tests for GLCMPreprocessor.
    """

    def __init__(self):
        self.n_levels = 32

        self.preprocessor = GLCMPreprocessor(
            n_levels=self.n_levels,
            symmetric=True,
        )

        self.image_size = 256

        self.expected_shape = (
            4,
            self.n_levels,
            self.n_levels,
        )



    def _random_patch(self) -> np.ndarray:
        return np.random.randint(
            0,
            256,
            size=(self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )


    def test_initialization(self):
        assert self.preprocessor.n_levels == self.n_levels
        assert self.preprocessor.symmetric is True

        print("test_initialization passed")

    def test_invalid_n_levels(self):
        try:
            GLCMPreprocessor(n_levels=0)
            raise AssertionError(
                "Expected ValueError for n_levels=0"
            )
        except ValueError:
            pass

        try:
            GLCMPreprocessor(n_levels=-1)
            raise AssertionError(
                "Expected ValueError for negative n_levels"
            )
        except ValueError:
            pass

        try:
            GLCMPreprocessor(n_levels=16.5)
            raise AssertionError(
                "Expected ValueError for float n_levels"
            )
        except ValueError:
            pass

        print("test_invalid_n_levels passed")

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

    def test_glcm_normalization(self):
        image = self._random_patch()

        features = self.preprocessor(image)

        # Each GLCM should sum to 1
        sums = features.sum(axis=(1, 2))

        np.testing.assert_allclose(
            sums,
            np.ones(4, dtype=np.float32),
            atol=1e-5,
        )

        print("test_glcm_normalization passed")

    def test_symmetric_glcm(self):
        image = self._random_patch()

        features = self.preprocessor(image)

        for i in range(features.shape[0]):
            np.testing.assert_allclose(
                features[i],
                features[i].T,
                atol=1e-6,
            )

        print("test_symmetric_glcm passed")

    def test_non_symmetric_glcm(self):
        processor = GLCMPreprocessor(
            n_levels=self.n_levels,
            symmetric=False,
        )

        image = self._random_patch()

        features = processor(image)

        assert features.shape == self.expected_shape

        print("test_non_symmetric_glcm passed")

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

    def test_compute_glcm_shape(self):
        qimg = np.random.randint(
            0,
            self.n_levels,
            size=(256, 256),
            dtype=np.int32,
        )

        glcm = self.preprocessor._compute_glcm(
            qimg=qimg,
            dx=1,
            dy=0,
        )

        assert glcm.shape == (
            self.n_levels,
            self.n_levels,
        )

        print("test_compute_glcm_shape passed")

    def test_compute_glcm_normalized(self):
        qimg = np.random.randint(
            0,
            self.n_levels,
            size=(256, 256),
            dtype=np.int32,
        )

        glcm = self.preprocessor._compute_glcm(
            qimg=qimg,
            dx=1,
            dy=0,
        )

        np.testing.assert_allclose(
            glcm.sum(),
            1.0,
            atol=1e-6,
        )

        print("test_compute_glcm_normalized passed")

  
    def run_all(self):
        """
        Execute all tests manually.
        """

        self.test_initialization()
        self.test_invalid_n_levels()
        self.test_output_shape()
        self.test_output_dtype()
        self.test_output_is_finite()
        self.test_glcm_normalization()
        self.test_symmetric_glcm()
        self.test_non_symmetric_glcm()
        self.test_deterministic_output()
        self.test_black_image()
        self.test_white_image()
        self.test_random_noise_image()
        self.test_compute_glcm_shape()
        self.test_compute_glcm_normalized()

        print("\nAll tests passed")


if __name__ == "__main__":
    tester = TestGLCMPreprocessor()
    tester.run_all()