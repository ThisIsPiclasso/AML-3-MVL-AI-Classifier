import numpy as np

from features.noise_residuals_pipeline import NoiseResidualPreprocessor
from MVL_AI_Classifier.constants import DEFAULT_EPSILON, PATCH_SIZE


class TestNoiseResidualPreprocessor:
    """Test suite for NoiseResidualPreprocessor."""

    def __init__(self):
        self.sigma = 1.0
        self.window_size = 16
        self.windows_per_dim = PATCH_SIZE // self.window_size  # 16
        self.preprocessor = NoiseResidualPreprocessor(
            sigma=self.sigma,
            window_size=self.window_size,
        )
        self.expected_shape = (3, self.windows_per_dim, self.windows_per_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_uint8_patch(self, seed: int = 42) -> np.ndarray:
        """Generate a reproducible random uint8 RGB patch.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Array of shape (PATCH_SIZE, PATCH_SIZE, 3), dtype uint8.
        """
        rng = np.random.default_rng(seed)
        return rng.integers(
            0, 256,
            size=(PATCH_SIZE, PATCH_SIZE, 3),
            dtype=np.uint8,
        )

    def _constant_patch(self, value: int) -> np.ndarray:
        """Generate a constant-valued RGB patch.

        Args:
            value: Pixel value for all channels (0-255).

        Returns:
            Array of shape (PATCH_SIZE, PATCH_SIZE, 3), dtype uint8.
        """
        return np.full(
            (PATCH_SIZE, PATCH_SIZE, 3),
            fill_value=value,
            dtype=np.uint8,
        )

    def _correlated_noise_patch(self, seed: int = 42) -> np.ndarray:
        """Generate a patch where all three channels share the same noise.

        When all channels have identical noise patterns, the inter-channel
        Pearson correlation should be very high (near +1.0).

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Array of shape (PATCH_SIZE, PATCH_SIZE, 3), dtype uint8.
        """
        rng = np.random.default_rng(seed)
        single_channel = rng.integers(
            0, 256,
            size=(PATCH_SIZE, PATCH_SIZE),
            dtype=np.uint8,
        )
        return np.stack([single_channel] * 3, axis=-1)

    def _uncorrelated_noise_patch(self, seed: int = 42) -> np.ndarray:
        """Generate a patch where each channel has independent noise.

        When channels have independent noise, inter-channel Pearson
        correlation should be near zero.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Array of shape (PATCH_SIZE, PATCH_SIZE, 3), dtype uint8.
        """
        rng = np.random.default_rng(seed)
        return rng.integers(
            0, 256,
            size=(PATCH_SIZE, PATCH_SIZE, 3),
            dtype=np.uint8,
        )

    def _two_channel_correlated_patch(
        self, correlated_channels: tuple[int, int], seed: int = 42
    ) -> np.ndarray:
        """Generate a patch where two specific channels are correlated.

        The two specified channels share the same noise. The third
        channel has independent noise.

        Args:
            correlated_channels: Tuple of two channel indices (0-2)
                that should be correlated.
            seed: Random seed for reproducibility.

        Returns:
            Array of shape (PATCH_SIZE, PATCH_SIZE, 3), dtype uint8.
        """
        rng = np.random.default_rng(seed)
        shared_noise = rng.integers(
            0, 256, size=(PATCH_SIZE, PATCH_SIZE), dtype=np.uint8
        )
        independent_noise = rng.integers(
            0, 256, size=(PATCH_SIZE, PATCH_SIZE), dtype=np.uint8
        )

        patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
        all_channels = {0, 1, 2}
        for ch in correlated_channels:
            patch[:, :, ch] = shared_noise
        third_channel = (all_channels - set(correlated_channels)).pop()
        patch[:, :, third_channel] = independent_noise

        return patch

    def _smooth_with_noise_patch(self, seed: int = 42) -> np.ndarray:
        """Generate a smooth gradient with added per-channel noise.

        The smooth gradient is removed by the high-pass filter,
        leaving only the noise component in the residual.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Array of shape (PATCH_SIZE, PATCH_SIZE, 3), dtype uint8.
        """
        rng = np.random.default_rng(seed)
        # Smooth horizontal gradient.
        gradient = np.linspace(50, 200, PATCH_SIZE, dtype=np.float32)
        gradient_2d = np.tile(gradient, (PATCH_SIZE, 1))
        # Add independent noise per channel.
        noise = rng.normal(0, 10, size=(PATCH_SIZE, PATCH_SIZE, 3))
        image = np.stack([gradient_2d] * 3, axis=-1) + noise
        return np.clip(image, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Group 1: Construction and validation
    # ------------------------------------------------------------------

    def test_default_parameters(self):
        """Verify default constructor stores correct values."""
        p = NoiseResidualPreprocessor()
        assert p.sigma == 1.0, f"Expected sigma=1.0, got {p.sigma}"
        assert p.window_size == 16, (
            f"Expected window_size=16, got {p.window_size}"
        )
        assert p.windows_per_dim == 16, (
            f"Expected windows_per_dim=16, got {p.windows_per_dim}"
        )
        assert p.epsilon == np.float32(DEFAULT_EPSILON), (
            f"Expected epsilon={DEFAULT_EPSILON}, got {p.epsilon}"
        )
        print("test_default_parameters passed")

    def test_custom_parameters(self):
        """Verify custom parameters are stored correctly."""
        p = NoiseResidualPreprocessor(
            sigma=2.0, window_size=32, epsilon=1e-6
        )
        assert p.sigma == 2.0, f"Expected sigma=2.0, got {p.sigma}"
        assert p.window_size == 32, (
            f"Expected window_size=32, got {p.window_size}"
        )
        assert p.windows_per_dim == 8, (
            f"Expected windows_per_dim=8, got {p.windows_per_dim}"
        )
        print("test_custom_parameters passed")

    def test_invalid_window_size_not_divisor(self):
        """Verify non-divisor window_size raises ValueError."""
        try:
            NoiseResidualPreprocessor(window_size=15)
            assert False, "Expected ValueError for window_size=15"
        except ValueError:
            pass
        print("test_invalid_window_size_not_divisor passed")

    def test_invalid_window_size_other(self):
        """Verify another non-divisor window_size raises ValueError."""
        try:
            NoiseResidualPreprocessor(window_size=7)
            assert False, "Expected ValueError for window_size=7"
        except ValueError:
            pass
        print("test_invalid_window_size_other passed")

    def test_valid_window_sizes(self):
        """Verify all valid divisors of PATCH_SIZE are accepted."""
        valid_sizes = [s for s in [1, 2, 4, 8, 16, 32, 64, 128, 256]
                       if PATCH_SIZE % s == 0]
        for ws in valid_sizes:
            p = NoiseResidualPreprocessor(window_size=ws)
            assert p.windows_per_dim == PATCH_SIZE // ws, (
                f"window_size={ws}: expected windows_per_dim="
                f"{PATCH_SIZE // ws}, got {p.windows_per_dim}"
            )
        print("test_valid_window_sizes passed")

    def test_invalid_input_shape(self):
        """Verify wrong input shape raises ValueError."""
        wrong_shape = np.zeros((128, 128, 3), dtype=np.uint8)
        try:
            self.preprocessor(wrong_shape)
            assert False, "Expected ValueError for wrong shape"
        except ValueError:
            pass
        print("test_invalid_input_shape passed")

    def test_invalid_input_channels(self):
        """Verify wrong channel count raises ValueError."""
        wrong_channels = np.zeros(
            (PATCH_SIZE, PATCH_SIZE, 1), dtype=np.uint8
        )
        try:
            self.preprocessor(wrong_channels)
            assert False, "Expected ValueError for 1 channel"
        except ValueError:
            pass
        print("test_invalid_input_channels passed")

    # ------------------------------------------------------------------
    # Group 2: Internal methods
    # ------------------------------------------------------------------

    def test_residual_shape(self):
        """Verify noise residual has same shape as input."""
        image = np.zeros(
            (PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32
        )
        residual = self.preprocessor._compute_noise_residuals(image)
        assert residual.shape == (PATCH_SIZE, PATCH_SIZE, 3), (
            f"Expected residual shape ({PATCH_SIZE}, {PATCH_SIZE}, 3), "
            f"got {residual.shape}"
        )
        print("test_residual_shape passed")

    def test_residual_dtype(self):
        """Verify noise residual is float32."""
        image = np.zeros(
            (PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32
        )
        residual = self.preprocessor._compute_noise_residuals(image)
        assert residual.dtype == np.float32, (
            f"Expected float32, got {residual.dtype}"
        )
        print("test_residual_dtype passed")

    def test_residual_of_constant_is_zero(self):
        """Verify constant image produces zero residual.

        A constant image is its own Gaussian blur, so the
        residual (image - blur) should be zero everywhere.
        """
        image = np.full(
            (PATCH_SIZE, PATCH_SIZE, 3), 128.0, dtype=np.float32
        )
        residual = self.preprocessor._compute_noise_residuals(image)
        max_residual = float(np.abs(residual).max())
        assert max_residual < 1e-5, (
            f"Constant image residual should be ≈0, "
            f"but max is {max_residual:.6f}"
        )
        print("test_residual_of_constant_is_zero passed")

    def test_window_reshape_shape(self):
        """Verify window reshaping produces correct output shape."""
        channel = np.zeros(
            (PATCH_SIZE, PATCH_SIZE), dtype=np.float32
        )
        windowed = self.preprocessor._reshape_channel_to_windows(channel)
        expected_shape = (
            self.windows_per_dim,
            self.windows_per_dim,
            self.window_size * self.window_size,
        )
        assert windowed.shape == expected_shape, (
            f"Expected windowed shape {expected_shape}, "
            f"got {windowed.shape}"
        )
        print("test_window_reshape_shape passed")

    def test_window_reshape_preserves_values(self):
        """Verify window reshaping correctly maps pixel positions.

        Set a known value at a specific pixel location and verify
        it appears in the correct window at the correct position.
        """
        channel = np.zeros(
            (PATCH_SIZE, PATCH_SIZE), dtype=np.float32
        )
        # Pixel at row=20, col=35.
        # Window row = 20 // 16 = 1, window col = 35 // 16 = 2.
        # Within-window row = 20 % 16 = 4, col = 35 % 16 = 3.
        # Flattened index = 4 * 16 + 3 = 67.
        channel[20, 35] = 99.0
        windowed = self.preprocessor._reshape_channel_to_windows(channel)

        win_row = 20 // self.window_size
        win_col = 35 // self.window_size
        pixel_row = 20 % self.window_size
        pixel_col = 35 % self.window_size
        flat_idx = pixel_row * self.window_size + pixel_col

        extracted = windowed[win_row, win_col, flat_idx]
        assert extracted == 99.0, (
            f"Expected 99.0 at window[{win_row}, {win_col}, {flat_idx}], "
            f"got {extracted}"
        )
        print("test_window_reshape_preserves_values passed")

    def test_window_reshape_total_elements(self):
        """Verify no pixels lost or duplicated during reshaping."""
        channel = np.arange(
            PATCH_SIZE * PATCH_SIZE, dtype=np.float32
        ).reshape(PATCH_SIZE, PATCH_SIZE)
        windowed = self.preprocessor._reshape_channel_to_windows(channel)
        assert windowed.size == channel.size, (
            f"Windowed has {windowed.size} elements, "
            f"channel has {channel.size}."
        )
        print("test_window_reshape_total_elements passed")

    # ------------------------------------------------------------------
    # Group 3: Output contract
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """Verify output has correct shape (3, wpd, wpd)."""
        patch = self._random_uint8_patch()
        features = self.preprocessor(patch)
        assert features.shape == self.expected_shape, (
            f"Expected shape {self.expected_shape}, got {features.shape}"
        )
        print("test_output_shape passed")

    def test_output_dtype(self):
        """Verify output is float32."""
        patch = self._random_uint8_patch()
        features = self.preprocessor(patch)
        assert features.dtype == np.float32, (
            f"Expected float32, got {features.dtype}"
        )
        print("test_output_dtype passed")

    def test_output_all_finite(self):
        """Verify no NaN or Inf values in output."""
        patch = self._random_uint8_patch()
        features = self.preprocessor(patch)
        assert np.all(np.isfinite(features)), (
            f"Found {(~np.isfinite(features)).sum()} non-finite values."
        )
        print("test_output_all_finite passed")

    def test_output_range(self):
        """Verify all correlation values are in [-1, 1]."""
        patch = self._random_uint8_patch()
        features = self.preprocessor(patch)
        assert float(features.min()) >= -1.0 - 1e-6, (
            f"Minimum correlation is {features.min()}, expected >= -1."
        )
        assert float(features.max()) <= 1.0 + 1e-6, (
            f"Maximum correlation is {features.max()}, expected <= 1."
        )
        print("test_output_range passed")

    def test_deterministic_same_input(self):
        """Verify identical inputs produce identical outputs."""
        patch = self._random_uint8_patch(seed=123)
        features_first = self.preprocessor(patch)
        features_second = self.preprocessor(patch)
        np.testing.assert_array_equal(
            features_first, features_second,
            err_msg="Same input produced different outputs."
        )
        print("test_deterministic_same_input passed")

    def test_different_inputs_different_outputs(self):
        """Verify distinct images produce distinct correlation maps."""
        patch_a = self._random_uint8_patch(seed=1)
        patch_b = self._random_uint8_patch(seed=2)
        features_a = self.preprocessor(patch_a)
        features_b = self.preprocessor(patch_b)
        assert not np.array_equal(features_a, features_b), (
            "Two different patches produced identical outputs."
        )
        print("test_different_inputs_different_outputs passed")

    def test_uint8_and_float32_equivalent(self):
        """Verify uint8 and float32 inputs produce the same output."""
        patch_uint8 = self._random_uint8_patch(seed=77)
        patch_float32 = patch_uint8.astype(np.float32)
        features_uint8 = self.preprocessor(patch_uint8)
        features_float32 = self.preprocessor(patch_float32)
        np.testing.assert_allclose(
            features_uint8, features_float32,
            rtol=1e-4,
            err_msg="uint8 and float32 inputs differ."
        )
        print("test_uint8_and_float32_equivalent passed")

    def test_different_window_size_changes_shape(self):
        """Verify changing window_size changes output spatial dims."""
        preprocessor_32 = NoiseResidualPreprocessor(window_size=32)
        preprocessor_8 = NoiseResidualPreprocessor(window_size=8)
        patch = self._random_uint8_patch()

        features_32 = preprocessor_32(patch)
        features_8 = preprocessor_8(patch)

        assert features_32.shape == (3, 8, 8), (
            f"Expected (3, 8, 8), got {features_32.shape}"
        )
        assert features_8.shape == (3, 32, 32), (
            f"Expected (3, 32, 32), got {features_8.shape}"
        )
        print("test_different_window_size_changes_shape passed")

    # ------------------------------------------------------------------
    # Group 4: Edge case inputs
    # ------------------------------------------------------------------

    def test_black_image_finite(self):
        """Verify black image produces finite output."""
        black = self._constant_patch(0)
        features = self.preprocessor(black)
        assert np.all(np.isfinite(features)), (
            "Black image produced non-finite values."
        )
        print("test_black_image_finite passed")

    def test_white_image_finite(self):
        """Verify white image produces finite output."""
        white = self._constant_patch(255)
        features = self.preprocessor(white)
        assert np.all(np.isfinite(features)), (
            "White image produced non-finite values."
        )
        print("test_white_image_finite passed")

    def test_constant_image_zero_correlation(self):
        """Verify constant image produces zero correlation.

        A constant image has zero residual, which means zero variance
        in every window. The Pearson correlation is undefined and
        should be set to 0.0 (degenerate window handling).
        """
        constant = self._constant_patch(128)
        features = self.preprocessor(constant)
        max_abs_corr = float(np.abs(features).max())
        assert max_abs_corr < 1e-5, (
            f"Constant image should produce zero correlation, "
            f"but max |corr| = {max_abs_corr:.6f}"
        )
        print("test_constant_image_zero_correlation passed")

    def test_constant_images_same_output(self):
        """Verify all constant images produce the same output.

        Regardless of brightness level, a constant image has zero
        residual and zero correlation.
        """
        features_0 = self.preprocessor(self._constant_patch(0))
        features_128 = self.preprocessor(self._constant_patch(128))
        features_255 = self.preprocessor(self._constant_patch(255))
        np.testing.assert_allclose(
            features_0, features_128, atol=1e-5,
            err_msg="Constant images at different levels should match."
        )
        np.testing.assert_allclose(
            features_128, features_255, atol=1e-5,
            err_msg="Constant images at different levels should match."
        )
        print("test_constant_images_same_output passed")

    # ------------------------------------------------------------------
    # Group 5: Semantic correctness
    # ------------------------------------------------------------------

    def test_correlated_channels_high_correlation(self):
        """Verify identical channel noise produces high correlation.

        When all three channels have the same pixel values, their
        noise residuals are identical. Pearson correlation of
        identical vectors is exactly 1.0.
        """
        patch = self._correlated_noise_patch(seed=42)
        features = self.preprocessor(patch)

        # All three pairs (RG, RB, GB) should have high correlation.
        for pair_idx, pair_name in enumerate(["RG", "RB", "GB"]):
            mean_corr = float(features[pair_idx].mean())
            assert mean_corr > 0.8, (
                f"Correlated channels should have high {pair_name} "
                f"correlation, got mean {mean_corr:.4f}"
            )
        print("test_correlated_channels_high_correlation passed")

    def test_uncorrelated_channels_low_correlation(self):
        """Verify independent channel noise produces near-zero correlation.

        When each channel has independent random noise, the inter-channel
        correlation should be near zero (not exactly zero due to finite
        sample size).
        """
        patch = self._uncorrelated_noise_patch(seed=42)
        features = self.preprocessor(patch)

        for pair_idx, pair_name in enumerate(["RG", "RB", "GB"]):
            mean_abs_corr = float(np.abs(features[pair_idx]).mean())
            assert mean_abs_corr < 0.3, (
                f"Uncorrelated channels should have low {pair_name} "
                f"correlation, got mean |corr| = {mean_abs_corr:.4f}"
            )
        print("test_uncorrelated_channels_low_correlation passed")

    def test_selective_channel_correlation(self):
        """Verify correlation detects which specific channels are correlated.

        When R and G share noise but B is independent, the RG correlation
        should be much higher than RB and GB correlations.
        """
        patch = self._two_channel_correlated_patch(
            correlated_channels=(0, 1), seed=42
        )
        features = self.preprocessor(patch)

        # Pair index 0 = RG, 1 = RB, 2 = GB
        mean_rg = float(features[0].mean())
        mean_rb = float(np.abs(features[1]).mean())
        mean_gb = float(np.abs(features[2]).mean())

        assert mean_rg > mean_rb + 0.2, (
            f"RG correlation ({mean_rg:.4f}) should be much higher "
            f"than RB ({mean_rb:.4f}) when only R and G are correlated."
        )
        assert mean_rg > mean_gb + 0.2, (
            f"RG correlation ({mean_rg:.4f}) should be much higher "
            f"than GB ({mean_gb:.4f}) when only R and G are correlated."
        )
        print("test_selective_channel_correlation passed")

    def test_correlated_higher_than_uncorrelated(self):
        """Verify correlated image has higher correlation than uncorrelated.

        Direct comparison of the same metric on two different inputs.
        """
        correlated = self._correlated_noise_patch(seed=10)
        uncorrelated = self._uncorrelated_noise_patch(seed=10)

        features_corr = self.preprocessor(correlated)
        features_uncorr = self.preprocessor(uncorrelated)

        mean_corr = float(np.abs(features_corr).mean())
        mean_uncorr = float(np.abs(features_uncorr).mean())

        assert mean_corr > mean_uncorr, (
            f"Correlated image should have higher mean |correlation| "
            f"({mean_corr:.4f}) than uncorrelated ({mean_uncorr:.4f})."
        )
        print("test_correlated_higher_than_uncorrelated passed")

    def test_three_correlation_pairs_present(self):
        """Verify output has exactly 3 correlation maps (RG, RB, GB)."""
        patch = self._random_uint8_patch()
        features = self.preprocessor(patch)
        assert features.shape[0] == 3, (
            f"Expected 3 channel pairs, got {features.shape[0]}"
        )
        print("test_three_correlation_pairs_present passed")

    def test_sigma_affects_residual_magnitude(self):
        """Verify larger sigma produces larger residual magnitudes.

        A larger sigma removes more low-frequency content, making
        the high-pass residual retain more energy from medium
        frequencies. This should produce residuals with different
        statistical properties.
        """
        patch = self._random_uint8_patch(seed=55)

        preprocessor_small = NoiseResidualPreprocessor(sigma=0.5)
        preprocessor_large = NoiseResidualPreprocessor(sigma=3.0)

        features_small = preprocessor_small(patch)
        features_large = preprocessor_large(patch)

        # The two different sigmas should produce detectably different outputs.
        assert not np.allclose(features_small, features_large, atol=0.01), (
            "Different sigma values should produce different correlation maps."
        )
        print("test_sigma_affects_residual_magnitude passed")

    def test_smooth_image_with_noise_has_low_correlation(self):
        """Verify smooth gradient with independent per-channel noise
        produces low inter-channel correlation.

        The gradient is removed by the high-pass filter. Only the
        independent noise remains in the residual.
        """
        patch = self._smooth_with_noise_patch(seed=42)
        features = self.preprocessor(patch)

        mean_abs_corr = float(np.abs(features).mean())
        assert mean_abs_corr < 0.4, (
            f"Smooth image with independent noise should have low "
            f"correlation, got mean |corr| = {mean_abs_corr:.4f}"
        )
        print("test_smooth_image_with_noise_has_low_correlation passed")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------

    def run_all(self):
        """Execute all tests and report results."""
        test_methods = [
            method_name
            for method_name in dir(self)
            if method_name.startswith("test_")
        ]

        passed = 0
        failed = 0
        errors = []

        for test_name in sorted(test_methods):
            try:
                getattr(self, test_name)()
                passed += 1
            except (AssertionError, Exception) as error:
                failed += 1
                errors.append((test_name, str(error)))
                print(f"FAILED: {test_name} — {error}")

        print(f"\n{'=' * 60}")
        print(
            f"Results: {passed} passed, {failed} failed, "
            f"{passed + failed} total"
        )
        if errors:
            print("\nFailed tests:")
            for test_name, error_msg in errors:
                print(f"  {test_name}: {error_msg}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    tester = TestNoiseResidualPreprocessor()
    tester.run_all()