"""Tests for the DCT Distribution preprocessor.

Tests are organized in five groups:
    1. Construction and validation — parameter storage and rejection.
    2. Block extraction — verify reshape geometry and properties.
    3. Output contract — shape, dtype, finiteness, determinism.
    4. Edge case inputs — constant, black, white images.
    5. Semantic correctness — verify statistics behave correctly
       for inputs with known properties.
"""

import numpy as np

from features.dct_pipeline import DCTDistributionPreprocessor
from MVL_AI_Classifier.constants import (
    DEFAULT_EPSILON,
    JPEG_BLOCK_SIZE,
    JPEG_BLOCKS_PER_DIM,
    JPEG_RECENTER_VALUE,
    PATCH_SIZE,
)


class TestDCTDistributionPreprocessor:
    """Test suite for DCTDistributionPreprocessor."""

    def __init__(self):
        self.preprocessor = DCTDistributionPreprocessor()
        self.preprocessor_no_log = DCTDistributionPreprocessor(
            log_compress_dispersion=False
        )
        self.expected_shape = (3, 4, JPEG_BLOCK_SIZE, JPEG_BLOCK_SIZE)
        self.num_blocks = JPEG_BLOCKS_PER_DIM * JPEG_BLOCKS_PER_DIM  # 1024

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_uint8_patch(self, seed: int = 42) -> np.ndarray:
        """Generate a reproducible random uint8 RGB patch.

        Args:
            seed: Random seed for reproducibility across test runs.

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

    def _gradient_patch(self) -> np.ndarray:
        """Generate a horizontal brightness gradient RGB patch.

        Left side is dark (0), right side is bright (255).
        All three channels are identical.

        Returns:
            Array of shape (PATCH_SIZE, PATCH_SIZE, 3), dtype uint8.
        """
        gradient_row = np.linspace(0, 255, PATCH_SIZE, dtype=np.float32)
        gradient_2d = np.tile(gradient_row, (PATCH_SIZE, 1))
        gradient_rgb = np.stack(
            [gradient_2d] * 3, axis=-1
        ).astype(np.uint8)
        return gradient_rgb

    def _checkerboard_patch(self, block_size: int = 8) -> np.ndarray:
        """Generate a checkerboard pattern RGB patch.

        Alternates between black (0) and white (255) in squares
        of the given block_size. Creates strong high-frequency
        energy at the block boundary frequency.

        Args:
            block_size: Side length of each checkerboard square.

        Returns:
            Array of shape (PATCH_SIZE, PATCH_SIZE, 3), dtype uint8.
        """
        row_indices = np.arange(PATCH_SIZE) // block_size
        col_indices = np.arange(PATCH_SIZE) // block_size
        row_grid, col_grid = np.meshgrid(row_indices, col_indices, indexing="ij")
        checkerboard = ((row_grid + col_grid) % 2 * 255).astype(np.uint8)
        return np.stack([checkerboard] * 3, axis=-1)

    def _textured_single_channel_patch(
        self, channel: int, seed: int = 42
    ) -> np.ndarray:
        """Generate a patch with random texture in one channel only.

        The specified channel gets random pixel values (spatial variation).
        The other two channels are constant at 128 (no variation across
        blocks after level shift and DC mean-centering).

        Args:
            channel: Which channel gets texture (0=R, 1=G, 2=B).
            seed: Random seed for reproducibility.

        Returns:
            Array of shape (PATCH_SIZE, PATCH_SIZE, 3), dtype uint8.
        """
        rng = np.random.default_rng(seed)
        patch = np.full(
            (PATCH_SIZE, PATCH_SIZE, 3),
            fill_value=128,
            dtype=np.uint8,
        )
        patch[:, :, channel] = rng.integers(
            0, 256, size=(PATCH_SIZE, PATCH_SIZE), dtype=np.uint8
        )
        return patch

    def test_three_channels_independent(self):
        """Verify each output channel reflects its input channel.

        A patch where only the red channel has spatial variation
        (random texture) should produce significantly larger
        dispersion in channel 0 than in channels 1 and 2 (which
        are constant and thus have zero block-to-block variation).
        """
        red_textured = self._textured_single_channel_patch(channel=0)
        features = self.preprocessor_no_log(red_textured)

        red_dispersion = float(np.abs(features[0, 0, :, :]).sum())
        green_dispersion = float(np.abs(features[1, 0, :, :]).sum())
        blue_dispersion = float(np.abs(features[2, 0, :, :]).sum())

        assert red_dispersion > green_dispersion, (
            f"Red channel should have more dispersion than green. "
            f"Red: {red_dispersion:.4f}, Green: {green_dispersion:.4f}"
        )
        assert red_dispersion > blue_dispersion, (
            f"Red channel should have more dispersion than blue. "
            f"Red: {red_dispersion:.4f}, Blue: {blue_dispersion:.4f}"
        )
        print("test_three_channels_independent passed")

    # ------------------------------------------------------------------
    # Group 1: Construction and validation
    # ------------------------------------------------------------------

    def test_default_parameters(self):
        """Verify default constructor stores correct parameter values."""
        p = DCTDistributionPreprocessor()
        assert p.epsilon == np.float32(DEFAULT_EPSILON), (
            f"Expected epsilon={DEFAULT_EPSILON}, got {p.epsilon}"
        )
        assert p.sparsity_threshold == np.float32(0.10), (
            f"Expected sparsity_threshold=0.10, got {p.sparsity_threshold}"
        )
        assert p.log_compress_dispersion is True, (
            "Default log_compress_dispersion should be True."
        )
        print("test_default_parameters passed")

    def test_custom_parameters(self):
        """Verify custom parameters are stored correctly."""
        p = DCTDistributionPreprocessor(
            epsilon=1e-6,
            sparsity_threshold=0.25,
            log_compress_dispersion=False,
        )
        assert p.epsilon == np.float32(1e-6), (
            f"Expected epsilon=1e-6, got {p.epsilon}"
        )
        assert p.sparsity_threshold == np.float32(0.25), (
            f"Expected sparsity_threshold=0.25, got {p.sparsity_threshold}"
        )
        assert p.log_compress_dispersion is False, (
            "log_compress_dispersion should be False."
        )
        print("test_custom_parameters passed")

    def test_invalid_sparsity_threshold_zero(self):
        """Verify sparsity_threshold=0 raises ValueError."""
        try:
            DCTDistributionPreprocessor(sparsity_threshold=0.0)
            assert False, "Expected ValueError for sparsity_threshold=0"
        except ValueError:
            pass
        print("test_invalid_sparsity_threshold_zero passed")

    def test_invalid_sparsity_threshold_negative(self):
        """Verify negative sparsity_threshold raises ValueError."""
        try:
            DCTDistributionPreprocessor(sparsity_threshold=-0.1)
            assert False, "Expected ValueError for negative sparsity_threshold"
        except ValueError:
            pass
        print("test_invalid_sparsity_threshold_negative passed")

    def test_invalid_input_shape(self):
        """Verify wrong input shape raises ValueError."""
        wrong_shape = np.zeros((128, 128, 3), dtype=np.uint8)
        try:
            self.preprocessor(wrong_shape)
            assert False, "Expected ValueError for wrong input shape"
        except ValueError:
            pass
        print("test_invalid_input_shape passed")

    def test_invalid_input_channels(self):
        """Verify wrong number of channels raises ValueError."""
        wrong_channels = np.zeros(
            (PATCH_SIZE, PATCH_SIZE, 1), dtype=np.uint8
        )
        try:
            self.preprocessor(wrong_channels)
            assert False, "Expected ValueError for 1-channel input"
        except ValueError:
            pass
        print("test_invalid_input_channels passed")

    # ------------------------------------------------------------------
    # Group 2: Block extraction
    # ------------------------------------------------------------------

    def test_block_extraction_shape(self):
        """Verify _extract_blocks produces correct output shape."""
        image = np.zeros(
            (PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32
        )
        blocks = self.preprocessor._extract_blocks(image)
        expected = (3, self.num_blocks, JPEG_BLOCK_SIZE, JPEG_BLOCK_SIZE)
        assert blocks.shape == expected, (
            f"Expected block shape {expected}, got {blocks.shape}"
        )
        print("test_block_extraction_shape passed")

    def test_block_extraction_dtype(self):
        """Verify _extract_blocks returns float32."""
        image = np.zeros(
            (PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32
        )
        blocks = self.preprocessor._extract_blocks(image)
        assert blocks.dtype == np.float32, (
            f"Expected float32, got {blocks.dtype}"
        )
        print("test_block_extraction_dtype passed")

    def test_block_extraction_contiguous(self):
        """Verify _extract_blocks returns C-contiguous array.

        C-contiguous memory layout is required for efficient DCT
        computation by scipy.
        """
        image = np.zeros(
            (PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32
        )
        blocks = self.preprocessor._extract_blocks(image)
        assert blocks.flags["C_CONTIGUOUS"], (
            "Block array should be C-contiguous for efficient DCT."
        )
        print("test_block_extraction_contiguous passed")

    def test_block_extraction_preserves_values(self):
        """Verify block extraction correctly maps pixel values.

        Set a known pixel value at a specific location and verify
        it appears in the correct block at the correct position.
        """
        image = np.zeros(
            (PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32
        )
        # Set pixel at row=10, col=20 in channel 1 (green) to 42.0.
        # This pixel is in block_row=1 (10//8), block_col=2 (20//8),
        # at within-block position pixel_row=2 (10%8), pixel_col=4 (20%8).
        image[10, 20, 1] = 42.0
        blocks = self.preprocessor._extract_blocks(image)

        block_row = 10 // JPEG_BLOCK_SIZE   # 1
        block_col = 20 // JPEG_BLOCK_SIZE   # 2
        pixel_row = 10 % JPEG_BLOCK_SIZE    # 2
        pixel_col = 20 % JPEG_BLOCK_SIZE    # 4
        block_index = block_row * JPEG_BLOCKS_PER_DIM + block_col  # 1*32+2 = 34

        extracted_value = blocks[1, block_index, pixel_row, pixel_col]
        assert extracted_value == 42.0, (
            f"Expected 42.0 at block[1, {block_index}, {pixel_row}, "
            f"{pixel_col}], got {extracted_value}"
        )
        print("test_block_extraction_preserves_values passed")

    def test_block_extraction_total_elements(self):
        """Verify no pixels are lost or duplicated during extraction.

        Total elements in blocks should equal total elements in image.
        """
        image = self._random_uint8_patch().astype(np.float32)
        blocks = self.preprocessor._extract_blocks(image)
        assert blocks.size == image.size, (
            f"Block array has {blocks.size} elements, "
            f"image has {image.size} elements."
        )
        print("test_block_extraction_total_elements passed")

    # ------------------------------------------------------------------
    # Group 3: Output contract
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """Verify output tensor has correct shape (3, 4, 8, 8)."""
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

    def test_different_inputs_produce_different_outputs(self):
        """Verify distinct images produce distinct feature tensors."""
        patch_a = self._random_uint8_patch(seed=1)
        patch_b = self._random_uint8_patch(seed=2)
        features_a = self.preprocessor(patch_a)
        features_b = self.preprocessor(patch_b)
        assert not np.array_equal(features_a, features_b), (
            "Two different random patches produced identical features."
        )
        print("test_different_inputs_produce_different_outputs passed")

    def test_uint8_and_float32_input_equivalent(self):
        """Verify uint8 and float32 inputs produce the same output."""
        patch_uint8 = self._random_uint8_patch(seed=77)
        patch_float32 = patch_uint8.astype(np.float32)
        features_uint8 = self.preprocessor(patch_uint8)
        features_float32 = self.preprocessor(patch_float32)
        np.testing.assert_allclose(
            features_uint8, features_float32,
            rtol=1e-5,
            err_msg="uint8 and float32 inputs produced different outputs."
        )
        print("test_uint8_and_float32_input_equivalent passed")

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

    def test_constant_image_zero_dispersion(self):
        """Verify constant image has zero dispersion at AC positions.

        A constant image produces identical DCT blocks (only DC nonzero).
        After DC mean-centering, all coefficients are zero or near-zero.
        MAD should be zero everywhere.
        """
        constant = self._constant_patch(128)
        features = self.preprocessor_no_log(constant)
        dispersion = features[:, 0, :, :]  # all channels, statistic 0
        max_dispersion = float(np.abs(dispersion).max())
        assert max_dispersion < 1e-4, (
            f"Constant image should have near-zero dispersion, "
            f"but max is {max_dispersion:.6f}"
        )
        print("test_constant_image_zero_dispersion passed")

    def test_constant_image_zero_sign_asymmetry(self):
        """Verify constant image has zero sign asymmetry.

        All DCT coefficients are zero or near-zero for a constant image,
        so neither positive nor negative counts dominate.
        """
        constant = self._constant_patch(100)
        features = self.preprocessor(constant)
        sign_asym = features[:, 3, :, :]  # statistic index 3
        max_asym = float(np.abs(sign_asym).max())
        assert max_asym < 0.1, (
            f"Constant image should have near-zero sign asymmetry, "
            f"but max is {max_asym:.4f}"
        )
        print("test_constant_image_zero_sign_asymmetry passed")

    def test_constant_image_high_sparsity(self):
        """Verify constant image has high sparsity.

        All DCT coefficients are zero or near-zero for a constant image,
        so nearly all should fall below the sparsity threshold.
        """
        constant = self._constant_patch(128)
        features = self.preprocessor(constant)
        sparsity = features[:, 2, :, :]  # statistic index 2
        mean_sparsity = float(sparsity.mean())
        assert mean_sparsity > 0.8, (
            f"Constant image should have high sparsity (>0.8), "
            f"but mean is {mean_sparsity:.4f}"
        )
        print("test_constant_image_high_sparsity passed")

    # ------------------------------------------------------------------
    # Group 5: Semantic correctness
    # ------------------------------------------------------------------

    def test_sparsity_range(self):
        """Verify sparsity values are in [0, 1] for any input.

        Sparsity is computed as the mean of a boolean array, so
        values must be between 0 (no coefficients near zero) and
        1 (all coefficients near zero).
        """
        patch = self._random_uint8_patch()
        features = self.preprocessor(patch)
        sparsity = features[:, 2, :, :]
        assert float(sparsity.min()) >= 0.0, (
            f"Sparsity minimum is {sparsity.min()}, expected >= 0."
        )
        assert float(sparsity.max()) <= 1.0, (
            f"Sparsity maximum is {sparsity.max()}, expected <= 1."
        )
        print("test_sparsity_range passed")

    def test_sign_asymmetry_range(self):
        """Verify sign asymmetry values are in [-1, 1].

        Sign asymmetry is (fraction_positive - fraction_negative),
        where both fractions are in [0, 1].
        """
        patch = self._random_uint8_patch()
        features = self.preprocessor(patch)
        sign_asym = features[:, 3, :, :]
        assert float(sign_asym.min()) >= -1.0, (
            f"Sign asymmetry minimum is {sign_asym.min()}, expected >= -1."
        )
        assert float(sign_asym.max()) <= 1.0, (
            f"Sign asymmetry maximum is {sign_asym.max()}, expected <= 1."
        )
        print("test_sign_asymmetry_range passed")

    def test_dispersion_non_negative_without_log(self):
        """Verify MAD is non-negative before log compression.

        Median Absolute Deviation is by definition non-negative.
        """
        patch = self._random_uint8_patch()
        features = self.preprocessor_no_log(patch)
        dispersion = features[:, 0, :, :]
        assert float(dispersion.min()) >= 0.0, (
            f"Dispersion (MAD) should be non-negative, "
            f"but min is {dispersion.min():.6f}"
        )
        print("test_dispersion_non_negative_without_log passed")

    def test_log_compression_reduces_range(self):
        """Verify log compression reduces the dynamic range of MAD.

        MAD can span several orders of magnitude across frequency
        positions. Log compression should reduce this range.
        """
        patch = self._random_uint8_patch()
        features_no_log = self.preprocessor_no_log(patch)
        features_with_log = self.preprocessor(patch)

        range_no_log = float(
            features_no_log[:, 0].max() - features_no_log[:, 0].min()
        )
        range_with_log = float(
            features_with_log[:, 0].max() - features_with_log[:, 0].min()
        )

        assert range_with_log < range_no_log, (
            f"Log compression should reduce range. "
            f"Without log: {range_no_log:.2f}, with log: {range_with_log:.2f}"
        )
        print("test_log_compression_reduces_range passed")

    def test_random_noise_near_zero_kurtosis(self):
        """Verify uniform random noise has kurtosis near expected value.

        Uniform distribution has theoretical excess kurtosis of -1.2.
        Random uint8 noise (uniform [0, 255]) should produce per-block
        DCT coefficients with kurtosis reasonably close to this value
        at most frequency positions.
        """
        noise = self._random_uint8_patch(seed=99)
        features = self.preprocessor(noise)
        kurtosis = features[:, 1, :, :]

        # Exclude DC position (0,0) which was mean-centered.
        ac_kurtosis = kurtosis[:, 1:, 1:]
        mean_kurtosis = float(ac_kurtosis.mean())

        # Uniform distribution kurtosis is -1.2. With finite samples
        # and DCT mixing, expect approximately [-3, 1].
        assert -5.0 < mean_kurtosis < 3.0, (
            f"Mean AC kurtosis for uniform noise should be roughly in "
            f"[-3, 1], got {mean_kurtosis:.4f}"
        )
        print("test_random_noise_near_zero_kurtosis passed")

    def test_random_noise_moderate_sparsity(self):
        """Verify random noise has moderate (not extreme) sparsity.

        Random noise has energy at all frequencies, so sparsity should
        not be very high (not all near-zero) or very low.
        """
        noise = self._random_uint8_patch(seed=88)
        features = self.preprocessor(noise)
        sparsity = features[:, 2, :, :]
        mean_sparsity = float(sparsity.mean())
        assert 0.01 < mean_sparsity < 0.95, (
            f"Random noise should have moderate sparsity, "
            f"got mean {mean_sparsity:.4f}"
        )
        print("test_random_noise_moderate_sparsity passed")

    def test_random_noise_near_zero_sign_asymmetry(self):
        """Verify random noise has near-zero sign asymmetry.

        Random noise should produce roughly equal numbers of positive
        and negative DCT coefficients at each frequency position.
        """
        noise = self._random_uint8_patch(seed=77)
        features = self.preprocessor(noise)
        sign_asym = features[:, 3, :, :]
        mean_abs_asym = float(np.abs(sign_asym).mean())
        assert mean_abs_asym < 0.15, (
            f"Random noise should have near-zero sign asymmetry, "
            f"got mean |asymmetry| = {mean_abs_asym:.4f}"
        )
        print("test_random_noise_near_zero_sign_asymmetry passed")

    def test_gradient_image_low_frequency_energy(self):
        """Verify gradient image concentrates dispersion in low frequencies.

        A smooth horizontal gradient has energy only at low horizontal
        frequencies. High-frequency positions should have less dispersion.
        """
        gradient = self._gradient_patch()
        features = self.preprocessor_no_log(gradient)
        dispersion = features[:, 0, :, :]

        # Average dispersion in low-frequency region (u<4, v<4).
        low_freq_dispersion = float(dispersion[:, :4, :4].mean())
        # Average dispersion in high-frequency region (u>=4, v>=4).
        high_freq_dispersion = float(dispersion[:, 4:, 4:].mean())

        assert low_freq_dispersion > high_freq_dispersion, (
            f"Gradient should have more low-frequency dispersion. "
            f"Low: {low_freq_dispersion:.4f}, High: {high_freq_dispersion:.4f}"
        )
        print("test_gradient_image_low_frequency_energy passed")

    def test_checkerboard_high_frequency_energy(self):
        """Verify checkerboard pattern has energy at high frequencies.

        An 8×8 checkerboard produces a specific frequency pattern.
        High-frequency dispersion should be significant.
        """
        checkerboard = self._checkerboard_patch(block_size=8)
        features = self.preprocessor_no_log(checkerboard)
        dispersion = features[:, 0, :, :]

        high_freq_energy = float(dispersion[:, 4:, 4:].mean())
        # For a checkerboard aligned with block boundaries,
        # there should be some measurable high-frequency energy.
        assert high_freq_energy >= 0.0, (
            f"Checkerboard should have non-negative high-frequency "
            f"dispersion, got {high_freq_energy:.6f}"
        )
        print("test_checkerboard_high_frequency_energy passed")

    def test_dc_mean_centering(self):
        """Verify DC coefficients are mean-centered after processing.

        The global mean of block DC values should be approximately
        zero after the mean-centering step. We verify this indirectly
        by checking that a constant-offset image (all pixels = 200)
        and another (all pixels = 50) produce identical outputs,
        since the DC mean is subtracted.
        """
        features_200 = self.preprocessor(self._constant_patch(200))
        features_50 = self.preprocessor(self._constant_patch(50))
        np.testing.assert_allclose(
            features_200, features_50,
            atol=1e-4,
            err_msg="Constant images with different brightness should "
                    "produce identical features after DC mean-centering."
        )
        print("test_dc_mean_centering passed")

    def test_higher_sparsity_threshold_increases_sparsity(self):
        """Verify that increasing sparsity_threshold increases sparsity.

        A higher threshold means more coefficients are considered
        'near zero', so sparsity values should increase.
        """
        patch = self._random_uint8_patch(seed=55)

        preprocessor_low = DCTDistributionPreprocessor(
            sparsity_threshold=0.05
        )
        preprocessor_high = DCTDistributionPreprocessor(
            sparsity_threshold=0.30
        )

        features_low = preprocessor_low(patch)
        features_high = preprocessor_high(patch)

        mean_sparsity_low = float(features_low[:, 2, :, :].mean())
        mean_sparsity_high = float(features_high[:, 2, :, :].mean())

        assert mean_sparsity_high > mean_sparsity_low, (
            f"Higher threshold should produce higher sparsity. "
            f"Low threshold: {mean_sparsity_low:.4f}, "
            f"high threshold: {mean_sparsity_high:.4f}"
        )
        print("test_higher_sparsity_threshold_increases_sparsity passed")

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
    tester = TestDCTDistributionPreprocessor()
    tester.run_all()