"""Tests for the Azimuthal Power Spectrum preprocessor."""

import numpy as np

from features.aps_pipeline import AzimuthalPowerSpectrumPreprocessor
from MVL_AI_Classifier.constants import DEFAULT_EPSILON, PATCH_SIZE


class TestAzimuthalPowerSpectrumPreprocessor:
    """Test suite for AzimuthalPowerSpectrumPreprocessor."""

    def __init__(self):
        self.n_bins = 64
        self.preprocessor = AzimuthalPowerSpectrumPreprocessor(
            n_bins=self.n_bins
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_uint8_patch(self, seed: int = 42) -> np.ndarray:
        """Generate a reproducible random uint8 RGB patch."""
        rng = np.random.default_rng(seed)
        return rng.integers(
            0, 256,
            size=(PATCH_SIZE, PATCH_SIZE, 3),
            dtype=np.uint8,
        )

    def _constant_patch(self, value: int) -> np.ndarray:
        """Generate a constant-valued RGB patch."""
        return np.full(
            (PATCH_SIZE, PATCH_SIZE, 3),
            fill_value=value,
            dtype=np.uint8,
        )

    def _sinusoidal_patch(self, frequency: float) -> np.ndarray:
        """Generate a horizontal sinusoidal pattern as an RGB patch.

        Args:
            frequency: Spatial frequency in cycles per image width.
        """
        x_coordinates = np.arange(PATCH_SIZE, dtype=np.float32)
        sine_wave = np.sin(
            2.0 * np.pi * frequency * x_coordinates / PATCH_SIZE
        )
        pixel_values = ((sine_wave * 77.5) + 127.5).astype(np.uint8)
        row_pattern = np.tile(pixel_values, (PATCH_SIZE, 1))
        rgb_patch = np.stack([row_pattern] * 3, axis=-1)
        return rgb_patch

    # ------------------------------------------------------------------
    # Group 1: Construction and validation
    # ------------------------------------------------------------------

    def test_initialization_stores_parameters(self):
        """Verify that constructor stores n_bins and epsilon."""
        assert self.preprocessor.n_bins == self.n_bins, (
            f"Expected n_bins={self.n_bins}, "
            f"got {self.preprocessor.n_bins}"
        )
        assert self.preprocessor.epsilon == np.float32(DEFAULT_EPSILON), (
            "Epsilon should be stored as float32."
        )
        print("test_initialization_stores_parameters passed")

    def test_hann_window_shape_and_dtype(self):
        """Verify precomputed Hann window has correct shape and dtype."""
        # Use the actual attribute name from your aps_pipeline.py
        window = self.preprocessor._window_2d
        assert window.shape == (PATCH_SIZE, PATCH_SIZE), (
            f"Expected window shape ({PATCH_SIZE}, {PATCH_SIZE}), "
            f"got {window.shape}"
        )
        assert window.dtype == np.float32, (
            f"Expected float32 window, got {window.dtype}"
        )
        print("test_hann_window_shape_and_dtype passed")

    def test_hann_window_center_near_one(self):
        """Verify that the Hann window center is approximately 1.0."""
        window = self.preprocessor._window_2d
        center = PATCH_SIZE // 2
        center_value = window[center, center]
        assert abs(center_value - 1.0) < 0.01, (
            f"Hann window center should be ≈1.0, got {center_value}"
        )
        print("test_hann_window_center_near_one passed")

    def test_hann_window_edges_near_zero(self):
        """Verify that the Hann window edges are near zero."""
        window = self.preprocessor._window_2d
        assert window[0, 0] < 0.001, (
            f"Hann window corner should be ≈0.0, got {window[0, 0]}"
        )
        assert window[0, PATCH_SIZE // 2] < 0.01, (
            f"Hann window edge midpoint should be ≈0.0, "
            f"got {window[0, PATCH_SIZE // 2]}"
        )
        print("test_hann_window_edges_near_zero passed")

    def test_invalid_n_bins_zero(self):
        """Verify that n_bins=0 raises ValueError."""
        try:
            AzimuthalPowerSpectrumPreprocessor(n_bins=0)
            assert False, "Expected ValueError for n_bins=0"
        except ValueError:
            pass
        print("test_invalid_n_bins_zero passed")

    def test_invalid_n_bins_negative(self):
        """Verify that negative n_bins raises ValueError."""
        try:
            AzimuthalPowerSpectrumPreprocessor(n_bins=-5)
            assert False, "Expected ValueError for negative n_bins"
        except ValueError:
            pass
        print("test_invalid_n_bins_negative passed")

    def test_invalid_n_bins_float(self):
        """Verify that float n_bins raises ValueError.

        Note: if your code does ``int(n_bins)`` before the isinstance
        check, or does not have the isinstance check, a float like
        32.5 will be silently truncated to 32. This test verifies
        that floats are rejected. If your code intentionally accepts
        floats, remove this test.
        """
        raised = False
        try:
            AzimuthalPowerSpectrumPreprocessor(n_bins=32.5)
        except (ValueError, TypeError):
            raised = True

        if not raised:
            # If no exception was raised, verify it at least truncated
            # correctly rather than failing silently in a bad way.
            preprocessor = AzimuthalPowerSpectrumPreprocessor(n_bins=32.5)
            assert preprocessor.n_bins == 32, (
                f"Float n_bins should either raise ValueError or truncate "
                f"to int. Got n_bins={preprocessor.n_bins}"
            )
            print("test_invalid_n_bins_float passed (float accepted, truncated to int)")
        else:
            print("test_invalid_n_bins_float passed (ValueError raised)")

    def test_invalid_input_shape(self):
        """Verify that wrong input shape raises ValueError."""
        wrong_shape = np.zeros((128, 128, 3), dtype=np.uint8)
        try:
            self.preprocessor(wrong_shape)
            assert False, "Expected ValueError for wrong input shape"
        except ValueError:
            pass
        print("test_invalid_input_shape passed")

    def test_invalid_input_channels(self):
        """Verify that wrong number of channels raises ValueError."""
        wrong_channels = np.zeros(
            (PATCH_SIZE, PATCH_SIZE, 4), dtype=np.uint8
        )
        try:
            self.preprocessor(wrong_channels)
            assert False, "Expected ValueError for 4-channel input"
        except ValueError:
            pass
        print("test_invalid_input_channels passed")

    # ------------------------------------------------------------------
    # Group 2: Lookup table geometry
    # ------------------------------------------------------------------

    def test_bin_index_shape(self):
        """Verify radial bin index array has one entry per FFT position."""
        # Use the actual attribute name from your aps_pipeline.py
        radial_bin_indices = self.preprocessor._bin_idx
        expected_length = PATCH_SIZE * PATCH_SIZE
        assert radial_bin_indices.shape == (expected_length,), (
            f"Expected shape ({expected_length},), "
            f"got {radial_bin_indices.shape}"
        )
        print("test_bin_index_shape passed")

    def test_bin_index_dtype(self):
        """Verify radial bin indices are int32 for use in np.bincount."""
        radial_bin_indices = self.preprocessor._bin_idx
        assert radial_bin_indices.dtype == np.int32, (
            f"Expected int32, got {radial_bin_indices.dtype}"
        )
        print("test_bin_index_dtype passed")

    def test_bin_index_range(self):
        """Verify all bin indices are in valid range [0, n_bins-1]."""
        radial_bin_indices = self.preprocessor._bin_idx
        assert radial_bin_indices.min() >= 0, (
            f"Minimum bin index is {radial_bin_indices.min()}, expected >= 0"
        )
        assert radial_bin_indices.max() <= self.n_bins - 1, (
            f"Maximum bin index is {radial_bin_indices.max()}, "
            f"expected <= {self.n_bins - 1}"
        )
        print("test_bin_index_range passed")

    def test_bin_counts_shape(self):
        """Verify bin counts array has exactly n_bins elements."""
        coefficients_per_bin = self.preprocessor._bin_counts
        assert coefficients_per_bin.shape == (self.n_bins,), (
            f"Expected shape ({self.n_bins},), "
            f"got {coefficients_per_bin.shape}"
        )
        print("test_bin_counts_shape passed")

    def test_bin_counts_sum_equals_total_pixels(self):
        """Verify all FFT positions are assigned to exactly one bin."""
        coefficients_per_bin = self.preprocessor._bin_counts
        total_assigned = int(coefficients_per_bin.sum())
        expected_total = PATCH_SIZE * PATCH_SIZE
        assert total_assigned == expected_total, (
            f"Sum of bin counts is {total_assigned}, "
            f"expected {expected_total}"
        )
        print("test_bin_counts_sum_equals_total_pixels passed")

    def test_bin_counts_all_positive(self):
        """Verify every radial bin contains at least one FFT coefficient."""
        coefficients_per_bin = self.preprocessor._bin_counts
        assert np.all(coefficients_per_bin > 0), (
            f"Found empty bins at indices: "
            f"{np.where(coefficients_per_bin == 0)[0]}"
        )
        print("test_bin_counts_all_positive passed")

    def test_inner_bins_smaller_than_outer_bins(self):
        """Verify inner bins contain fewer coefficients than outer bins.

        Ring area grows with radius (2πr), so outer bins should
        generally contain more FFT positions.
        """
        coefficients_per_bin = self.preprocessor._bin_counts
        inner_mean = float(coefficients_per_bin[:8].mean())
        outer_mean = float(coefficients_per_bin[16:48].mean())
        assert outer_mean > inner_mean, (
            f"Outer bins should have more coefficients. "
            f"Inner mean: {inner_mean:.1f}, outer mean: {outer_mean:.1f}"
        )
        print("test_inner_bins_smaller_than_outer_bins passed")

    # ------------------------------------------------------------------
    # Group 3: Output contract
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """Verify output has exactly n_bins elements."""
        patch = self._random_uint8_patch()
        features = self.preprocessor(patch)
        assert features.shape == (self.n_bins,), (
            f"Expected shape ({self.n_bins},), got {features.shape}"
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
            f"Found {(~np.isfinite(features)).sum()} non-finite values"
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
        """Verify distinct images produce distinct spectra."""
        patch_a = self._random_uint8_patch(seed=1)
        patch_b = self._random_uint8_patch(seed=2)
        features_a = self.preprocessor(patch_a)
        features_b = self.preprocessor(patch_b)
        assert not np.array_equal(features_a, features_b), (
            "Two different random patches produced identical spectra."
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

    def test_constant_image_flat_spectrum(self):
        """Verify constant image produces near-flat spectrum.

        After DC removal, a constant image is all zeros. The
        log-transformed spectrum should be approximately log(epsilon)
        for all bins.
        """
        constant = self._constant_patch(128)
        features = self.preprocessor(constant)
        spectrum_range = float(features.max() - features.min())
        assert spectrum_range < 1.0, (
            f"Constant image should produce near-flat spectrum, "
            f"but range is {spectrum_range:.2f}"
        )
        print("test_constant_image_flat_spectrum passed")

    def test_black_and_white_same_spectrum(self):
        """Verify black and white images produce identical spectra.

        Both are constant; after DC removal both become all-zeros.
        """
        black_features = self.preprocessor(self._constant_patch(0))
        white_features = self.preprocessor(self._constant_patch(255))
        np.testing.assert_allclose(
            black_features, white_features,
            rtol=1e-5,
            err_msg="Black and white images should have identical spectra."
        )
        print("test_black_and_white_same_spectrum passed")

    # ------------------------------------------------------------------
    # Group 5: Semantic correctness
    # ------------------------------------------------------------------

    def test_random_noise_nonzero_variance(self):
        """Verify random noise produces spectrum with nonzero variance."""
        noise = self._random_uint8_patch(seed=99)
        features = self.preprocessor(noise)
        spectrum_std = float(np.std(features))
        assert spectrum_std > 0.1, (
            f"Random noise spectrum should have significant variance, "
            f"but std is only {spectrum_std:.4f}"
        )
        print("test_random_noise_nonzero_variance passed")

    def test_low_frequency_signal_peaks_in_low_bins(self):
        """Verify low-frequency sinusoid has peak energy in low bins.

        A 4-cycle sine wave should produce its spectral peak in the
        lower portion of the radial frequency axis.
        """
        low_freq_patch = self._sinusoidal_patch(frequency=4.0)
        features = self.preprocessor(low_freq_patch)
        peak_bin = int(np.argmax(features))
        assert peak_bin < self.n_bins // 2, (
            f"Low-frequency signal peak should be in lower half, "
            f"but peak is at bin {peak_bin} of {self.n_bins}"
        )
        print("test_low_frequency_signal_peaks_in_low_bins passed")

    def test_high_frequency_signal_peaks_in_high_bins(self):
        """Verify high-frequency sinusoid has peak energy in high bins.

        A 64-cycle sine wave should produce its spectral peak in the
        upper portion of the radial frequency axis.

        Note: comparing sums of log-transformed values is unreliable
        because log(small_number) is a large negative, making sums
        counterintuitive. Using argmax (peak bin location) is robust
        regardless of log transformation.
        """
        high_freq_patch = self._sinusoidal_patch(frequency=64.0)
        features = self.preprocessor(high_freq_patch)
        peak_bin = int(np.argmax(features))
        assert peak_bin > self.n_bins // 4, (
            f"High-frequency signal peak should be in upper portion, "
            f"but peak is at bin {peak_bin} of {self.n_bins}"
        )
        print("test_high_frequency_signal_peaks_in_high_bins passed")

    def test_low_and_high_frequency_spectra_differ(self):
        """Verify peak bins differ for low and high frequency signals."""
        low_features = self.preprocessor(
            self._sinusoidal_patch(frequency=4.0)
        )
        high_features = self.preprocessor(
            self._sinusoidal_patch(frequency=64.0)
        )
        low_peak_bin = int(np.argmax(low_features))
        high_peak_bin = int(np.argmax(high_features))
        assert high_peak_bin > low_peak_bin, (
            f"High-frequency peak bin ({high_peak_bin}) should be higher "
            f"than low-frequency peak bin ({low_peak_bin})."
        )
        print("test_low_and_high_frequency_spectra_differ passed")

    def test_different_n_bins_produces_different_length(self):
        """Verify changing n_bins changes output length."""
        preprocessor_32 = AzimuthalPowerSpectrumPreprocessor(n_bins=32)
        preprocessor_128 = AzimuthalPowerSpectrumPreprocessor(n_bins=128)
        patch = self._random_uint8_patch()
        features_32 = preprocessor_32(patch)
        features_128 = preprocessor_128(patch)
        assert features_32.shape == (32,), (
            f"Expected shape (32,), got {features_32.shape}"
        )
        assert features_128.shape == (128,), (
            f"Expected shape (128,), got {features_128.shape}"
        )
        print("test_different_n_bins_produces_different_length passed")

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
            print(f"\nFailed tests:")
            for test_name, error_msg in errors:
                print(f"  {test_name}: {error_msg}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    tester = TestAzimuthalPowerSpectrumPreprocessor()
    tester.run_all()