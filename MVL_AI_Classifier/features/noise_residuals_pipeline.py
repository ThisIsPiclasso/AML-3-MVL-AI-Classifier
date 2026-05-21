import numpy as np
from scipy.ndimage import gaussian_filter

from features.base_processor import BasePreprocessor
from MVL_AI_Classifier.constants import DEFAULT_EPSILON, PATCH_SIZE
from features.rgb_normalization_pipeline import RGBNormalizationPreprocessor


class NoiseResidualPreprocessor(BasePreprocessor):
    """Compute inter-channel noise correlation structure from an RGB patch.

    For each spatial window of size ``(window_size × window_size)``,
    computes the Pearson correlation between the noise residuals of
    each pair of color channels (R-G, R-B, G-B).

    The noise residual is the high-frequency component obtained by
    subtracting a Gaussian-blurred version of each channel:

        ``residual_c = channel_c - gaussian_blur(channel_c)``
    """

    # Channel pairs for inter-channel correlation.
    # Indices into the 3-channel (R=0, G=1, B=2) axis.
    _CHANNEL_PAIRS = [(0, 1), (0, 2), (1, 2)]  # R-G, R-B, G-B

    def __init__(
        self,
        sigma: float = 1.0,
        window_size: int = 16,
        epsilon: float = DEFAULT_EPSILON,
    ):
        """Initialize the noise residual preprocessor.

        Args:
            sigma: Standard deviation of the Gaussian low-pass filter
                used to extract the noise residual.
            window_size: Side length of spatial windows for local
                correlation computation. Must divide ``PATCH_SIZE``
                exactly. Default 16 gives 256 pixels per window
                (Pearson standard error ≈ 0.06).
            epsilon: Numerical stability constant for Pearson
                denominator and degenerate window detection.

        Raises:
            ValueError: If ``window_size`` does not divide ``PATCH_SIZE``
                exactly.
        """
        if PATCH_SIZE % window_size != 0:
            raise ValueError(
                f"window_size={window_size} must divide "
                f"PATCH_SIZE={PATCH_SIZE} exactly."
            )

        self.sigma = sigma
        self.window_size = window_size
        self.epsilon = epsilon
        self.windows_per_dim = PATCH_SIZE // window_size

        self._normalization = RGBNormalizationPreprocessor()

    def _compute_noise_residuals(self, image: np.ndarray) -> np.ndarray:
        """Extract high-frequency noise via Gaussian high-pass filtering.

        Each color channel is filtered independently: the Gaussian
        blur is applied only to the two spatial dimensions, not across
        color channels.

        Args:
            image: Float32 RGB image of shape
                ``(PATCH_SIZE, PATCH_SIZE, 3)`` in [0, 255].

        Returns:
            Noise residual of shape ``(PATCH_SIZE, PATCH_SIZE, 3)``
            as float32. Values can be positive or negative.
        """
        # sigma=(σ, σ, 0) blurs spatial dimensions but NOT color channels.
        # mode="reflect" mirrors pixels at boundaries (abcd|dcba),
        lowpass = gaussian_filter(
            image,
            sigma=(self.sigma, self.sigma, 0.0),
            mode="reflect",
        )
        return (image - lowpass).astype(np.float32)

    def _reshape_channel_to_windows(self, channel: np.ndarray) -> np.ndarray:
        """Reshape a single-channel 2D array into non-overlapping windows.

        Each window's pixels are flattened into a vector suitable for
        Pearson correlation computation.

        Args:
            channel: Single-channel residual of shape
                ``(PATCH_SIZE, PATCH_SIZE)``, float32.

        Returns:
            Windowed array of shape
            ``(windows_per_dim, windows_per_dim, window_size²)``
            as float32, where the last axis contains the flattened
            pixel values within each spatial window.
        """
        window_size = self.window_size
        windows_per_dim = self.windows_per_dim

        return (
            channel
            # Split spatial dims into (window_grid, pixels_within_window):
            # (256, 256) → (16, 16, 16, 16)
            # Axes: (win_row, pixel_y, win_col, pixel_x)
            .reshape(windows_per_dim, window_size, windows_per_dim, window_size)
            # Group window indices together and pixel indices together:
            # → (16, 16, 16, 16)
            # Axes: (win_row, win_col, pixel_y, pixel_x)
            .transpose(0, 2, 1, 3)
            # Flatten each window's pixels into a single vector:
            # → (16, 16, 256)
            # Axes: (win_row, win_col, flattened_pixels)
            .reshape(windows_per_dim, windows_per_dim, window_size * window_size)
        )

    def _pearson_correlation_from_windows(
        self,
        channel_a_windows: np.ndarray,
        channel_b_windows: np.ndarray,
    ) -> np.ndarray:
        """Compute Pearson correlation between two windowed channel residuals.

        For each spatial window, computes the Pearson correlation
        coefficient between the noise residual vectors of two color
        channels. Degenerate windows (constant residual, zero variance)
        are assigned a correlation of 0.0.

        Args:
            channel_a_windows: Windowed residual of shape
                ``(windows_per_dim, windows_per_dim, window_size²)``.
            channel_b_windows: Same shape as ``channel_a_windows``.

        Returns:
            Pearson correlation map of shape
            ``(windows_per_dim, windows_per_dim)`` as float32.
            Values in [-1, 1]. Degenerate windows are 0.0.
        """
        # Mean-center each window independently to remove local luminance.
        # This ensures correlation measures noise pattern similarity,
        # not brightness similarity.
        centered_a = channel_a_windows - channel_a_windows.mean(
            axis=-1, keepdims=True
        )
        centered_b = channel_b_windows - channel_b_windows.mean(
            axis=-1, keepdims=True
        )
        # Shape: (windows_per_dim, windows_per_dim, window_size²) each

        # Pearson formula: r = Σ(a·b) / sqrt(Σa² · Σb²)
        covariance = (centered_a * centered_b).sum(axis=-1)
        sum_sq_a = (centered_a ** 2).sum(axis=-1)
        sum_sq_b = (centered_b ** 2).sum(axis=-1)
        # Shape: (windows_per_dim, windows_per_dim) each

        # Detect degenerate windows where one or both channels have
        # zero variance (constant residual within the window).
        # This is common in saturated regions or uniform backgrounds.
        has_variance = (sum_sq_a > self.epsilon) & (sum_sq_b > self.epsilon)

        denominator = np.sqrt(sum_sq_a * sum_sq_b)

        # Safe division: only compute correlation where both channels
        # have meaningful variance. Degenerate windows get 0.0.
        correlation = np.where(
            has_variance,
            covariance / (denominator + self.epsilon),
            np.float32(0.0),
        )

        return correlation.astype(np.float32)

    def __call__(self, image_patch: np.ndarray) -> np.ndarray:
        """Compute local inter-channel noise correlation maps.

        Args:
            image_patch: RGB image of shape ``(PATCH_SIZE, PATCH_SIZE, 3)``,
                uint8 or float32.

        Returns:
            Inter-channel correlation tensor of shape
            ``(3, windows_per_dim, windows_per_dim)`` as float32.
            Axis 0 order: [corr_RG, corr_RB, corr_GB].
            Each value is a Pearson correlation in [-1, 1].
            Default shape with window_size=16: ``(3, 16, 16)``.
        """

        image = self._normalization(image_patch)
        # Extract noise residuals per channel.
        noise_residuals = self._compute_noise_residuals(image)
        # Shape: (256, 256, 3), float32

        # Reshape each channel's residual into windows ONCE to avoid redundancy
        windowed_channels = [
            self._reshape_channel_to_windows(noise_residuals[..., channel_idx])
            for channel_idx in range(3)
        ]
        # List of 3 arrays, each shape: (16, 16, 256)

        # Compute Pearson correlation for each channel pair.
        correlation_maps = np.stack(
            [
                self._pearson_correlation_from_windows(
                    windowed_channels[channel_i],
                    windowed_channels[channel_j],
                )
                for channel_i, channel_j in self._CHANNEL_PAIRS
            ],
            axis=0,
        )
        # Shape: (3, 16, 16), float32
        # Index 0: R-G correlation, Index 1: R-B, Index 2: G-B

        return correlation_maps