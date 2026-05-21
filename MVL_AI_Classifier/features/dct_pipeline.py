import numpy as np
from scipy.fftpack import dctn

from MVL_AI_Classifier.features.base_processor import BasePreprocessor
from MVL_AI_Classifier.constants import (
    DEFAULT_EPSILON,
    JPEG_BLOCK_SIZE,
    JPEG_BLOCKS_PER_DIM,
    JPEG_RECENTER_VALUE,
)
from MVL_AI_Classifier.features.rgb_normalization_pipeline import (
    RGBNormalizationPreprocessor,
)


class DCTDistributionPreprocessor(BasePreprocessor):
    """Compute per-channel distributional statistics of block DCT coefficients.

    The image is partitioned into 1024 non-overlapping 8x8 blocks (the
    JPEG block structure). The DCT-II is applied to each block, and four
    statistics are computed at each frequency position across all blocks.
    This view captures higher-order statistics (dispersion,
    kurtosis, sparsity, sign asymmetry) per individual frequency
    position ``(u, v)``, resolved per color channel. The DC
    coefficient is mean-centered to remove the globally-redundant
    component while preserving spatial variation of block means.
    """

    def __init__(
        self,
        epsilon: float = DEFAULT_EPSILON,
        sparsity_threshold: float = 0.10,
        log_compress_dispersion: bool = True,
    ):
        """Initialize the DCT distribution preprocessor.

        Args:
            epsilon: Numerical stability constant for division and log.
            sparsity_threshold: Fraction of the channel's RMS energy below
                which a coefficient is considered "near zero". Value 0.10
                means below 10% of typical coefficient magnitude. Must be
                positive.
            log_compress_dispersion: If True, apply ``log(x + epsilon)``
                to the MAD statistic. MAD has wide dynamic range across
                frequency positions; log compression normalizes scale.
                Not applied to other statistics because they are already
        """

        self.epsilon = np.float32(epsilon)
        self.sparsity_threshold = np.float32(sparsity_threshold)
        self.log_compress_dispersion = bool(log_compress_dispersion)
        self._normalization = RGBNormalizationPreprocessor()

    def _extract_blocks(self, image: np.ndarray) -> np.ndarray:
        """Partition an RGB image into non-overlapping 8x8 blocks.

        Uses array reshaping to partition the
        image into a channel-first block array suitable for vectorized
        DCT computation.

        Args:
            image: Level-shifted float32 image of shape
                ``(PATCH_SIZE, PATCH_SIZE, 3)``.

        Returns:
            C-contiguous float32 array of shape
            ``(3, num_blocks, 8, 8)`` where ``num_blocks = 1024``.
        """
        num_blocks_per_dim = JPEG_BLOCKS_PER_DIM  # 32
        block_size = JPEG_BLOCK_SIZE  # 8

        blocks = (
            image
            # Split each spatial dimension into (block_grid, within_block):
            # (256, 256, 3) → (32, 8, 32, 8, 3)
            # Axes: (block_row, pixel_row, block_col, pixel_col, channel)
            .reshape(
                num_blocks_per_dim,
                block_size,
                num_blocks_per_dim,
                block_size,
                3,
            )
            # Rearrange to channel-first, then block-grid, then block-pixels:
            # (32, 8, 32, 8, 3) → (3, 32, 32, 8, 8)
            # Axes: (channel, block_row, block_col, pixel_row, pixel_col)
            .transpose(4, 0, 2, 1, 3)
            # Flatten block grid into single block index:
            # (3, 32, 32, 8, 8) → (3, 1024, 8, 8)
            # Axes: (channel, block_index, pixel_row, pixel_col)
            .reshape(3, num_blocks_per_dim * num_blocks_per_dim, block_size, block_size)
        )

        # Transpose creates a non-contiguous view (data remains in original
        # memory layout). Copy to contiguous memory so dctn can operate on
        #  sequential data.
        return np.ascontiguousarray(blocks, dtype=np.float32)

    def _unbiased_excess_kurtosis(self, dct_coefficients: np.ndarray) -> np.ndarray:
        """Compute unbiased Fisher excess kurtosis along the block axis.

        Kurtosis measures how heavy-tailed a distribution is:
            - Gaussian distribution → excess kurtosis = 0
            - Heavier tails (more outliers) → positive kurtosis
            - Lighter tails (tighter clustering) → negative kurtosis

        Uses float64 internally for the ``m4/m2² - 3`` subtraction
        to avoid errors when kurtosis is near zero.

        Args:
            dct_coefficients: Array of shape ``(3, num_blocks, 8, 8)``
                containing DCT coefficients for all blocks and channels.

        Returns:
            Excess kurtosis per channel and frequency position,
            shape ``(3, 8, 8)`` as float32.
        """
        num_blocks = dct_coefficients.shape[1]  # 1024

        # Mean coefficient at each frequency position across all blocks.
        mean_per_position = dct_coefficients.mean(axis=1, keepdims=True)
        # Shape: (3, 1, 8, 8)

        # Deviation of every block's coefficient from the position mean.
        deviations = dct_coefficients - mean_per_position
        # Shape: (3, 1024, 8, 8)

        # Second and fourth central moments, computed in float64 to
        # preserve precision during the kurtosis subtraction.
        # m2 = E[d²] (variance), m4 = E[d⁴] (fourth central moment)
        second_moment = (deviations**2).mean(axis=1).astype(np.float64)
        fourth_moment = (deviations**4).mean(axis=1).astype(np.float64)
        # Shape: (3, 8, 8) each, float64

        # Biased excess kurtosis: g2 = m4/m2² - 3
        # The subtraction near 3 (not near 3069) minimizes cancellation.
        # Positions with near-zero variance get kurtosis = 0 (undefined).
        biased_excess_kurtosis = np.where(
            second_moment > self.epsilon,
            fourth_moment / (second_moment**2 + self.epsilon) - 3.0,
            0.0,
        )
        # Shape: (3, 8, 8), float64

        # Bias correction: convert biased g2 to unbiased G2.
        # Formula: G2 = (n-1)/((n-2)(n-3)) × ((n+1)×g2 + 6)
        bias_correction_factor = (num_blocks - 1) / (
            (num_blocks - 2) * (num_blocks - 3)
        )
        unbiased_kurtosis = bias_correction_factor * (
            (num_blocks + 1) * biased_excess_kurtosis + 6.0
        )
        # Shape: (3, 8, 8), float64

        result = unbiased_kurtosis.astype(np.float32)
        return result

    def __call__(self, image_patch: np.ndarray) -> np.ndarray:
        """Compute four distributional statistics of block DCT coefficients.

        The four statistics per frequency position are:
            - Index 0: Robust dispersion (MAD) — spread of coefficient values.
            - Index 1: Excess kurtosis — tail weight of the distribution.
            - Index 2: Sparsity — fraction of near-zero coefficients.
            - Index 3: Sign asymmetry — ratio of positive to negative.

        Args:
            image_patch: RGB image of shape ``(PATCH_SIZE, PATCH_SIZE, 3)``,
                uint8 or float32 in [0, 255].

        Returns:
            Feature tensor of shape ``(3, 4, 8, 8)`` as float32 where:
                - axis 0: color channel (R, G, B)
                - axis 1: statistic index [MAD, kurtosis, sparsity, sign_asym]
                - axes 2-3: DCT frequency position (u, v)
        """
        image = self._normalization(image_patch)

        # Maps [0, 255] → [-128, 127].
        image_centered = image - np.float32(JPEG_RECENTER_VALUE)

        blocks = self._extract_blocks(image_centered)
        # Shape: (3, 1024, 8, 8), float32, C-contiguous

        # axes=(-2, -1) transforms only the 8×8 pixel dimensions,
        # leaving channel and block axes untouched.
        # norm="ortho" preserves energy: sum(pixels²) = sum(DCT²).
        # Processes all 3 × 1024 = 3072 blocks in one vectorized call.
        dct_coefficients = np.asarray(
            dctn(blocks, axes=(-2, -1), norm="ortho"),
            dtype=np.float32,
        )  # Shape: (3, 1024, 8, 8), float32

        # The DC at (0,0) encodes each block's mean brightness.
        # Subtracting it removes the redundancy with APS while preserving
        # the SPATIAL VARIATION of block means (how brightness
        # varies from block to block)
        global_dc_mean_per_channel = dct_coefficients[:, :, 0, 0].mean(
            axis=1, keepdims=True
        )
        # Shape: (3, 1) — one scalar per color channel
        dct_coefficients[:, :, 0, 0] -= global_dc_mean_per_channel
        # After: mean of all block DCs ≈ 0; variation preserved.

        absolute_coefficients = np.abs(dct_coefficients)
        # Shape: (3, 1024, 8, 8

        # --- Statistic A: Robust dispersion (Median Absolute Deviation). ---
        # MAD = median(|x - median(x)|) across all 1024 blocks.
        # More robust to outlier blocks (containing sharp edges) than
        # standard deviation. Captures how "spread out" the coefficient
        # values are at each frequency position.
        median_per_position = np.median(dct_coefficients, axis=1, keepdims=True)
        # Shape: (3, 1, 8, 8)
        robust_dispersion = np.median(
            np.abs(dct_coefficients - median_per_position), axis=1
        ).astype(np.float32)
        # Shape: (3, 8, 8)

        # --- Statistic B: Unbiased excess kurtosis. ---
        kurtosis = self._unbiased_excess_kurtosis(dct_coefficients)
        # Shape: (3, 8, 8)

        # --- Statistic C: Scale-invariant sparsity. ---
        # RMS (Root Mean Square) = typical coefficient magnitude per channel.
        # Used as reference scale: "near zero" means below 10% of this RMS.
        channel_rms = np.sqrt(
            (dct_coefficients**2).mean(axis=(1, 2, 3), keepdims=True) + self.epsilon
        )
        # Shape: (3, 1, 1, 1) — one scalar per channel, broadcastable.

        # Fraction of blocks where |coefficient| < threshold at each position.
        sparsity_threshold = self.sparsity_threshold * channel_rms
        sparsity = (
            (absolute_coefficients < sparsity_threshold).mean(axis=1).astype(np.float32)
        )
        # Shape: (3, 8, 8)

        # --- Statistic D: Sign asymmetry. ---
        # (fraction positive) - (fraction negative) across blocks.
        # Range [-1, 1]. Near 0 = symmetric.
        # Note: APS discards sign via |F|², so this captures a completely different signal.
        fraction_positive = (dct_coefficients > 0).mean(axis=1)
        fraction_negative = (dct_coefficients < 0).mean(axis=1)
        sign_asymmetry = (fraction_positive - fraction_negative).astype(np.float32)
        # Shape: (3, 8, 8)

        features = np.stack(
            [robust_dispersion, kurtosis, sparsity, sign_asymmetry],
            axis=1,
        ).astype(np.float32)
        # Shape: (3, 4, 8, 8)
        # features[channel, 0, u, v] = robust dispersion (MAD)
        # features[channel, 1, u, v] = excess kurtosis
        # features[channel, 2, u, v] = sparsity fraction
        # features[channel, 3, u, v] = sign asymmetry

        # MAD can have awide dynamic range across the frequency positions
        # Log maps this to a more uniform scale
        # Not applied to other statistics because:
        #   - Kurtosis can be negative (log undefined).
        #   - Sparsity is in [0, 1] (already bounded).
        #   - Sign asymmetry is in [-1, 1] (bounded and signed).
        if self.log_compress_dispersion:
            features[:, 0] = np.log(features[:, 0] + self.epsilon)
        features = features.reshape(12, 8, 8)
        return features
