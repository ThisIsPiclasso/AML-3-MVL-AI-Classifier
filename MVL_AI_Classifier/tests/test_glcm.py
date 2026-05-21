"""Tests for the GLCM preprocessor.

Tests are organized in five groups:
    1. Construction and validation — parameter storage and rejection.
    2. GLCM computation internals — single-direction matrix properties.
    3. Output contract — shape, dtype, finiteness, determinism.
    4. Edge case inputs — constant, black, white images.
    5. Semantic correctness — verify co-occurrence statistics behave
       correctly for inputs with known texture properties.
"""

import numpy as np

from features.glcm_pipeline import GLCMPreprocessor
from MVL_AI_Classifier.constants import PATCH_SIZE


class TestGLCMPreprocessor:
    """Test suite for GLCMPreprocessor."""

    def __init__(self):
        self.n_levels = 32
        self.preprocessor = GLCMPreprocessor(n_levels=self.n_levels)
        self.expected_shape = (4, self.n_levels, self.n_levels)

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

    def _horizontal_stripe_patch(self, stripe_width: int = 8) -> np.ndarray:
        """Generate horizontal stripes alternating between dark and bright.

        Creates strong vertical co-occurrence (same intensity in adjacent
        rows within a stripe) and strong transitions at stripe boundaries.

        Args:
            stripe_width: Height of each stripe in pixels.

        Returns:
            Array of shape (PATCH_SIZE, PATCH_SIZE, 3), dtype uint8.
        """
        row_indices = np.arange(PATCH_SIZE)
        stripe_values = ((row_indices // stripe_width) % 2 * 200 + 27).astype(np.uint8)
        # stripe_values alternates between 27 and 227
        stripe_2d = np.tile(stripe_values[:, np.newaxis], (1, PATCH_SIZE))
        return np.stack([stripe_2d] * 3, axis=-1)

    def _vertical_stripe_patch(self, stripe_width: int = 8) -> np.ndarray:
        """Generate vertical stripes alternating between dark and bright.

        Creates strong horizontal co-occurrence transitions at stripe
        boundaries.

        Args:
            stripe_width: Width of each stripe in pixels.

        Returns:
            Array of shape (PATCH_SIZE, PATCH_SIZE, 3), dtype uint8.
        """
        col_indices = np.arange(PATCH_SIZE)
        stripe_values = ((col_indices // stripe_width) % 2 * 200 + 27).astype(np.uint8)
        stripe_2d = np.tile(stripe_values[np.newaxis, :], (PATCH_SIZE, 1))
        return np.stack([stripe_2d] * 3, axis=-1)

    def _gradient_patch(self) -> np.ndarray:
        """Generate a smooth horizontal gradient from 0 to 255.

        Returns:
            Array of shape (PATCH_SIZE, PATCH_SIZE, 3), dtype uint8.
        """
        gradient_row = np.linspace(0, 255, PATCH_SIZE, dtype=np.float32)
        gradient_2d = np.tile(gradient_row, (PATCH_SIZE, 1))
        return np.stack([gradient_2d] * 3, axis=-1).astype(np.uint8)

    def _checkerboard_patch(self, block_size: int = 1) -> np.ndarray:
        """Generate a pixel-level checkerboard (maximally textured).

        Args:
            block_size: Size of each checker square.

        Returns:
            Array of shape (PATCH_SIZE, PATCH_SIZE, 3), dtype uint8.
        """
        rows = np.arange(PATCH_SIZE) // block_size
        cols = np.arange(PATCH_SIZE) // block_size
        row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
        checker = ((row_grid + col_grid) % 2 * 255).astype(np.uint8)
        return np.stack([checker] * 3, axis=-1)

    # ------------------------------------------------------------------
    # Group 1: Construction and validation
    # ------------------------------------------------------------------

    def test_default_parameters(self):
        """Verify default constructor stores correct values."""
        p = GLCMPreprocessor()
        assert p.n_levels == 32, (
            f"Expected default n_levels=32, got {p.n_levels}"
        )
        assert p.symmetric is True, (
            "Default symmetric should be True."
        )
        print("test_default_parameters passed")

    def test_custom_parameters(self):
        """Verify custom parameters are stored correctly."""
        p = GLCMPreprocessor(n_levels=64, symmetric=False)
        assert p.n_levels == 64, (
            f"Expected n_levels=64, got {p.n_levels}"
        )
        assert p.symmetric is False, (
            "symmetric should be False."
        )
        print("test_custom_parameters passed")

    def test_invalid_n_levels_zero(self):
        """Verify n_levels=0 raises ValueError."""
        try:
            GLCMPreprocessor(n_levels=0)
            assert False, "Expected ValueError for n_levels=0"
        except ValueError:
            pass
        print("test_invalid_n_levels_zero passed")

    def test_invalid_n_levels_negative(self):
        """Verify negative n_levels raises ValueError."""
        try:
            GLCMPreprocessor(n_levels=-8)
            assert False, "Expected ValueError for negative n_levels"
        except ValueError:
            pass
        print("test_invalid_n_levels_negative passed")

    def test_invalid_n_levels_float(self):
        """Verify float n_levels is handled."""
        raised = False
        try:
            GLCMPreprocessor(n_levels=16.5)
        except (ValueError, TypeError):
            raised = True

        if not raised:
            p = GLCMPreprocessor(n_levels=16.5)
            assert p.n_levels == 16, (
                f"Float n_levels should truncate to 16, got {p.n_levels}"
            )
            print("test_invalid_n_levels_float passed (truncated)")
        else:
            print("test_invalid_n_levels_float passed (ValueError raised)")

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
            (PATCH_SIZE, PATCH_SIZE, 4), dtype=np.uint8
        )
        try:
            self.preprocessor(wrong_channels)
            assert False, "Expected ValueError for 4-channel input"
        except ValueError:
            pass
        print("test_invalid_input_channels passed")

    def test_quantization_scale(self):
        """Verify quantization scale factor is correctly computed."""
        expected_scale = np.float32(self.n_levels / 255.0)
        assert self.preprocessor._scale == expected_scale, (
            f"Expected scale {expected_scale}, "
            f"got {self.preprocessor._scale}"
        )
        print("test_quantization_scale passed")

    # ------------------------------------------------------------------
    # Group 2: GLCM computation internals
    # ------------------------------------------------------------------

    def test_glcm_matrix_shape(self):
        """Verify single GLCM has correct shape (n_levels, n_levels)."""
        quantized = np.zeros(
            (PATCH_SIZE, PATCH_SIZE), dtype=np.int32
        )
        glcm = self.preprocessor._compute_glcm(quantized, dx=1, dy=0)
        assert glcm.shape == (self.n_levels, self.n_levels), (
            f"Expected GLCM shape ({self.n_levels}, {self.n_levels}), "
            f"got {glcm.shape}"
        )
        print("test_glcm_matrix_shape passed")

    def test_glcm_probabilities_sum_to_one(self):
        """Verify normalized GLCM probabilities sum to 1.0.

        After normalization, the GLCM is a probability distribution
        and all entries must sum to approximately 1.0.
        """
        patch = self._random_uint8_patch()
        features = self.preprocessor(patch)

        for direction_idx in range(4):
            glcm_sum = float(features[direction_idx].sum())
            assert abs(glcm_sum - 1.0) < 1e-5, (
                f"GLCM direction {direction_idx} sums to {glcm_sum}, "
                f"expected 1.0"
            )
        print("test_glcm_probabilities_sum_to_one passed")

    def test_glcm_non_negative(self):
        """Verify all GLCM entries are non-negative.

        Probabilities cannot be negative.
        """
        patch = self._random_uint8_patch()
        features = self.preprocessor(patch)
        assert float(features.min()) >= 0.0, (
            f"GLCM contains negative values: min = {features.min()}"
        )
        print("test_glcm_non_negative passed")

    def test_symmetric_glcm_is_symmetric(self):
        """Verify GLCMs are symmetric when symmetric=True.

        With symmetric=True, G[i,j] should equal G[j,i] for all
        (i, j) in each directional GLCM.
        """
        patch = self._random_uint8_patch()
        features = self.preprocessor(patch)

        for direction_idx in range(4):
            glcm = features[direction_idx]
            np.testing.assert_allclose(
                glcm, glcm.T,
                atol=1e-6,
                err_msg=f"GLCM direction {direction_idx} is not symmetric."
            )
        print("test_symmetric_glcm_is_symmetric passed")

    def test_asymmetric_glcm_not_necessarily_symmetric(self):
        """Verify GLCMs are not forced symmetric when symmetric=False.

        For a non-symmetric image (e.g. gradient), the GLCM should
        differ from its transpose.
        """
        asymmetric_preprocessor = GLCMPreprocessor(
            n_levels=self.n_levels, symmetric=False
        )
        gradient = self._gradient_patch()
        features = asymmetric_preprocessor(gradient)

        # At least one direction should have a non-symmetric GLCM
        # for a gradient image.
        found_asymmetric = False
        for direction_idx in range(4):
            glcm = features[direction_idx]
            if not np.allclose(glcm, glcm.T, atol=1e-6):
                found_asymmetric = True
                break

        assert found_asymmetric, (
            "Gradient image with symmetric=False should produce at "
            "least one asymmetric GLCM."
        )
        print("test_asymmetric_glcm_not_necessarily_symmetric passed")

    # ------------------------------------------------------------------
    # Group 3: Output contract
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """Verify output has correct shape (4, n_levels, n_levels)."""
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
        """Verify distinct images produce distinct GLCMs."""
        patch_a = self._random_uint8_patch(seed=1)
        patch_b = self._random_uint8_patch(seed=2)
        features_a = self.preprocessor(patch_a)
        features_b = self.preprocessor(patch_b)
        assert not np.array_equal(features_a, features_b), (
            "Two different random patches produced identical GLCMs."
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

    def test_different_n_levels_changes_shape(self):
        """Verify changing n_levels changes GLCM matrix dimensions."""
        preprocessor_16 = GLCMPreprocessor(n_levels=16)
        preprocessor_64 = GLCMPreprocessor(n_levels=64)
        patch = self._random_uint8_patch()

        features_16 = preprocessor_16(patch)
        features_64 = preprocessor_64(patch)

        assert features_16.shape == (4, 16, 16), (
            f"Expected (4, 16, 16), got {features_16.shape}"
        )
        assert features_64.shape == (4, 64, 64), (
            f"Expected (4, 64, 64), got {features_64.shape}"
        )
        print("test_different_n_levels_changes_shape passed")

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

    def test_constant_image_single_cell(self):
        """Verify constant image concentrates probability in one cell.

        A constant image has only one intensity level. All pixel pairs
        have the same (level, level) co-occurrence. The GLCM should
        have all probability mass in a single diagonal cell.
        """
        constant = self._constant_patch(128)
        features = self.preprocessor(constant)

        for direction_idx in range(4):
            glcm = features[direction_idx]
            max_val = float(glcm.max())
            # The maximum cell should contain (nearly) all probability.
            assert max_val > 0.99, (
                f"Constant image GLCM direction {direction_idx}: "
                f"max cell is {max_val}, expected > 0.99"
            )
        print("test_constant_image_single_cell passed")

    def test_constant_image_diagonal_concentration(self):
        """Verify constant image GLCM has mass only on the diagonal.

        For a constant image, all pairs are (level, level), so only
        diagonal entries G[i,i] should be nonzero.
        """
        constant = self._constant_patch(100)
        features = self.preprocessor(constant)

        for direction_idx in range(4):
            glcm = features[direction_idx]
            # Zero out the diagonal and check remaining mass.
            off_diagonal = glcm.copy()
            np.fill_diagonal(off_diagonal, 0.0)
            off_diag_sum = float(off_diagonal.sum())
            assert off_diag_sum < 1e-5, (
                f"Constant image GLCM direction {direction_idx}: "
                f"off-diagonal sum is {off_diag_sum}, expected ≈ 0"
            )
        print("test_constant_image_diagonal_concentration passed")

    def test_black_and_white_different_glcm(self):
        """Verify black and white images have different GLCMs.

        Black (value 0) maps to bin 0. White (value 255) maps to
        the highest bin. Their GLCMs should have the single nonzero
        cell at different diagonal positions.
        """
        black_features = self.preprocessor(self._constant_patch(0))
        white_features = self.preprocessor(self._constant_patch(255))

        # Find which diagonal cell has the mass.
        black_peak = int(np.argmax(np.diag(black_features[0])))
        white_peak = int(np.argmax(np.diag(white_features[0])))

        assert black_peak != white_peak, (
            f"Black and white should map to different bins. "
            f"Black peak bin: {black_peak}, white peak bin: {white_peak}"
        )
        assert black_peak < white_peak, (
            f"Black should map to lower bin than white. "
            f"Black: {black_peak}, White: {white_peak}"
        )
        print("test_black_and_white_different_glcm passed")

    # ------------------------------------------------------------------
    # Group 5: Semantic correctness
    # ------------------------------------------------------------------

    def test_random_image_uses_multiple_cells(self):
        """Verify random image populates many GLCM cells.

        A random image should have co-occurrences spread across
        many (i, j) pairs, not concentrated in a few cells.
        """
        patch = self._random_uint8_patch()
        features = self.preprocessor(patch)

        for direction_idx in range(4):
            glcm = features[direction_idx]
            nonzero_cells = int((glcm > 0).sum())
            total_cells = self.n_levels * self.n_levels
            fill_ratio = nonzero_cells / total_cells
            assert fill_ratio > 0.05, (
                f"GLCM direction {direction_idx}: only {nonzero_cells} "
                f"of {total_cells} cells are nonzero (ratio {fill_ratio:.3f}). "
                f"Random image should populate more cells."
            )
        print("test_random_image_uses_multiple_cells passed")

    def test_checkerboard_off_diagonal_concentration(self):
        """Verify pixel-level checkerboard has strong off-diagonal mass.

        A 1-pixel checkerboard alternates between 0 and 255. Adjacent
        pixels always have different intensities, so co-occurrence is
        concentrated off-diagonal (G[low, high] and G[high, low]).
        """
        checker = self._checkerboard_patch(block_size=1)
        features = self.preprocessor(checker)

        for direction_idx in range(4):
            glcm = features[direction_idx]
            diagonal_mass = float(np.trace(glcm))
            off_diagonal_mass = float(glcm.sum() - diagonal_mass)
            assert off_diagonal_mass > diagonal_mass, (
                f"Checkerboard GLCM direction {direction_idx}: "
                f"off-diagonal mass ({off_diagonal_mass:.4f}) should "
                f"exceed diagonal mass ({diagonal_mass:.4f})"
            )
        print("test_checkerboard_off_diagonal_concentration passed")

    def test_gradient_diagonal_spread(self):
        """Verify gradient image has spread along the diagonal.

        A smooth gradient has many adjacent pixel pairs with similar
        (but not identical) intensity levels. The GLCM should show
        probability concentrated near (but not only on) the main
        diagonal.
        """
        gradient = self._gradient_patch()
        features = self.preprocessor(gradient)

        # Check horizontal GLCM (direction 0, dx=1, dy=0).
        # Adjacent horizontal pixels in a gradient differ by ~1 level.
        glcm = features[0]

        # Compute mass within ±2 of the diagonal.
        near_diagonal_mass = 0.0
        for i in range(self.n_levels):
            for j in range(max(0, i - 2), min(self.n_levels, i + 3)):
                near_diagonal_mass += float(glcm[i, j])

        assert near_diagonal_mass > 0.8, (
            f"Gradient horizontal GLCM should have >80% mass near "
            f"diagonal, got {near_diagonal_mass:.4f}"
        )
        print("test_gradient_diagonal_spread passed")

    def test_horizontal_stripes_vertical_direction(self):
        """Verify horizontal stripes produce strong diagonal mass
        in the vertical GLCM.

        Within each stripe, vertically adjacent pixels have the same
        intensity. The vertical GLCM (direction 1, dx=0, dy=1) should
        have most mass on the diagonal (same-level co-occurrence).
        """
        stripes = self._horizontal_stripe_patch(stripe_width=16)
        features = self.preprocessor(stripes)

        # Vertical GLCM is index 1.
        vertical_glcm = features[1]
        diagonal_mass = float(np.trace(vertical_glcm))

        # Most pairs are within a stripe (same level) → high diagonal mass.
        # A few pairs cross stripe boundaries → small off-diagonal mass.
        # With stripe_width=16, 255/256 ≈ 99.6% of pairs are within-stripe.
        assert diagonal_mass > 0.9, (
            f"Horizontal stripes should have >90% diagonal mass in "
            f"vertical GLCM, got {diagonal_mass:.4f}"
        )
        print("test_horizontal_stripes_vertical_direction passed")

    def test_four_directions_differ_for_directional_texture(self):
        """Verify directional texture produces different GLCMs per direction.

        Vertical stripes have different co-occurrence patterns
        horizontally vs vertically. The four directional GLCMs
        should not all be identical.
        """
        stripes = self._vertical_stripe_patch(stripe_width=16)
        features = self.preprocessor(stripes)

        # Horizontal GLCM (index 0) should differ from vertical (index 1)
        # because stripes are vertical.
        horizontal_glcm = features[0]
        vertical_glcm = features[1]
        assert not np.allclose(horizontal_glcm, vertical_glcm, atol=1e-4), (
            "Vertical stripes should produce different horizontal "
            "and vertical GLCMs."
        )
        print("test_four_directions_differ_for_directional_texture passed")

    def test_all_four_directions_present(self):
        """Verify all four directional GLCMs are computed and differ
        for a random image.
        """
        patch = self._random_uint8_patch()
        features = self.preprocessor(patch)

        assert features.shape[0] == 4, (
            f"Expected 4 directional GLCMs, got {features.shape[0]}"
        )

        # For a random image, all four should differ (extremely unlikely
        # to be identical by chance).
        for i in range(4):
            for j in range(i + 1, 4):
                assert not np.array_equal(features[i], features[j]), (
                    f"Directions {i} and {j} produced identical GLCMs "
                    f"for a random image."
                )
        print("test_all_four_directions_present passed")

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
    tester = TestGLCMPreprocessor()
    tester.run_all()
    